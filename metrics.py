import gc
import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

try:
    import psutil
except ImportError as e:
    raise ImportError("This script requires psutil. Please `pip install psutil`.") from e


np.random.seed(42)
n_bootstrap = 1000


def round_percent(value):
    """Round a float to 2 decimal places and convert to percent scale.
    
    Args:
        value (float): Input value
        
    Returns:
        float: Rounded percentage value
    """
    return round(float(value) * 100, 2)


def compute_confidence_interval(scores, confidence=0.95):
    """
    Compute confidence interval using bootstrapping.
    
    This function computes the confidence interval for a set of scores using
    the bootstrap percentile method.

    Parameters:
        scores (np.ndarray): Array of metric scores from bootstrap samples
        confidence (float): Confidence level (default: 0.95 for 95% CI)

    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval
    """
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    return lower, upper


def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000):
    """
    Compute metric with bootstrapping for confidence intervals.
    
    This function computes a metric with bootstrapping to estimate confidence intervals.

    Parameters:
        y_true (np.ndarray): Ground truth values
        y_pred (np.ndarray): Predicted values
        metric_func (callable): Metric function to compute
        n_bootstrap (int): Number of bootstrap samples

    Returns:
        tuple: (metric_value, lower_ci, upper_ci)
    """
    n_samples = len(y_true)
    bootstrap_scores = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Generate bootstrap indices
        indices = np.random.randint(0, n_samples, n_samples)
        # Compute metric on bootstrap sample
        bootstrap_scores[i] = metric_func(y_true[indices], y_pred[indices])

    # Compute confidence interval
    lower, upper = compute_confidence_interval(bootstrap_scores)
    return np.mean(bootstrap_scores), lower, upper


class Profiler:
    """Measure wall time, CPU time, Python-heap peak (tracemalloc), and process RSS delta.
    
    A profiler class that measures various performance metrics including execution time,
    memory usage, and resource consumption.
    """
    
    def __init__(self, label=""):
        """Initialize the profiler.
        
        Args:
            label (str): Label for the profiling session
        """
        self.label = label
        self.proc = psutil.Process(os.getpid())
        self.started = False
        self.result = {}

    def __enter__(self):
        """Enter the profiling context.
        
        Returns:
            Profiler: The profiler instance
        """
        gc.collect()
        self.rss_before = self.proc.memory_info().rss
        self.t_wall0 = time.perf_counter()
        self.t_cpu0 = time.process_time()
        # memory peak
        import tracemalloc
        self._tracemalloc = tracemalloc
        self._tracemalloc.start()
        self.started = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the profiling context and compute metrics.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        if not self.started:
            return
        current_py, peak_py = self._tracemalloc.get_traced_memory()
        self._tracemalloc.stop()

        t_cpu = time.process_time() - self.t_cpu0
        t_wall = time.perf_counter() - self.t_wall0
        rss_after = self.proc.memory_info().rss
        delta_rss = max(0, rss_after - self.rss_before)

        self.result = {
            "label": self.label,
            "wall_time_s": t_wall,
            "cpu_time_s": t_cpu,
            "peak_py_MB": peak_py / (1024 ** 2),
            "delta_rss_MB": delta_rss / (1024 ** 2),
            "rss_after_MB": rss_after / (1024 ** 2),
        }


def compute_pixel_metrics(total_gt_pixel_scores, total_pred_pixel_scores, use_ci=False):
    """
    Computes pixel-level metrics including AUROC, AP, max F1 score,
    and determines the threshold based on precision + recall.
    
    Parameters:
        total_gt_pixel_scores (np.ndarray): Ground truth (binary).
        total_pred_pixel_scores (np.ndarray): Predictions (continuous).
        use_ci (bool): Whether to compute confidence intervals.

    Returns:
        dict: Pixel-level metrics.
        float: Threshold for best F1.
        dict: Profiling information.
    """
    with Profiler("pixel_metrics") as prof:
        if use_ci:
            # Compute metrics with confidence intervals
            auroc_pixel, auroc_lower, auroc_upper = bootstrap_metric(
                total_gt_pixel_scores,
                total_pred_pixel_scores,
                roc_auc_score,
                n_bootstrap,
            )
            ap_pixel, ap_lower, ap_upper = bootstrap_metric(
                total_gt_pixel_scores,
                total_pred_pixel_scores,
                average_precision_score,
                n_bootstrap,
            )
        else:
            auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pred_pixel_scores)
            ap_pixel = average_precision_score(
                total_gt_pixel_scores,
                total_pred_pixel_scores,
            )

        precision, recall, thresholds = precision_recall_curve(
            total_gt_pixel_scores,
            total_pred_pixel_scores,
        )
        if thresholds.size == 0:
            threshold = 0.5
        else:
            threshold = thresholds[np.argmax(precision + recall)]

        binarized_pred = (total_pred_pixel_scores >= threshold).astype(np.uint8)

        if use_ci:
            f1_pixel, f1_lower, f1_upper = bootstrap_metric(
                total_gt_pixel_scores,
                binarized_pred,
                f1_score,
                n_bootstrap,
            )
        else:
            f1_pixel = f1_score(total_gt_pixel_scores, binarized_pred)

        pixel_metrics = {
            "pixel_auroc": round_percent(auroc_pixel),
            "pixel_ap": round_percent(ap_pixel),
            "pixel_f1": round_percent(f1_pixel),
        }

        if use_ci:
            pixel_metrics.update(
                {
                    "pixel_auroc_ci": (
                        round_percent(auroc_lower),
                        round_percent(auroc_upper),
                    ),
                    "pixel_ap_ci": (
                        round_percent(ap_lower),
                        round_percent(ap_upper),
                    ),
                    "pixel_f1_ci": (
                        round_percent(f1_lower),
                        round_percent(f1_upper),
                    ),
                }
            )

    # return pixel_metrics, threshold
    profile = prof.result
    profile["n_pixels"] = int(total_gt_pixel_scores.size)
    profile["throughput_pixels_per_s"] = (
        profile["n_pixels"] / profile["wall_time_s"] if profile["wall_time_s"] > 0 else np.nan
    )
    return pixel_metrics, threshold, profile


def compute_image_metrics(ep_score_gt, ep_score_pred, use_ci=False):
    """
    Computes image-level AUROC, AP, accuracy, F1, and best threshold.

    Parameters:
        ep_score_gt (np.ndarray): Ground truth (binary).
        ep_score_pred (np.ndarray): Predictions (continuous).
        use_ci (bool): Whether to compute confidence intervals.

    Returns:
        dict: Image-level metrics.
        float: Threshold for best F1.
        dict: Profiling information.
    """
    with Profiler("image_metrics") as prof:
        if use_ci:
            # Compute metrics with confidence intervals
            auroc, auroc_lower, auroc_upper = bootstrap_metric(
                ep_score_gt,
                ep_score_pred,
                roc_auc_score,
                n_bootstrap,
            )
            ap, ap_lower, ap_upper = bootstrap_metric(
                ep_score_gt,
                ep_score_pred,
                average_precision_score,
                n_bootstrap,
            )
        else:
            auroc = roc_auc_score(ep_score_gt, ep_score_pred)
            ap = average_precision_score(ep_score_gt, ep_score_pred)

        precision, recall, thresholds = precision_recall_curve(ep_score_gt, ep_score_pred)
        if thresholds.size == 0:
            threshold = 0.5
        else:
            threshold = thresholds[np.argmax(precision + recall)]

        pred = (ep_score_pred >= threshold).astype(np.uint8)
        gt = ep_score_gt.astype(np.uint8)

        acc = np.mean(pred == gt)
        f1 = f1_score(gt, pred)

        image_metrics = {
            "image_auroc": round_percent(auroc),
            "image_ap": round_percent(ap),
            "image_f1": round_percent(f1),
            "image_acc": round_percent(acc),
        }

        if use_ci:
            # Compute CIs for accuracy and F1 using bootstrapping
            n_samples = len(gt)
            acc_scores = np.zeros(n_bootstrap)
            f1_scores = np.zeros(n_bootstrap)

            for i in range(n_bootstrap):
                indices = np.random.randint(0, n_samples, n_samples)
                acc_scores[i] = np.mean(pred[indices] == gt[indices])
                f1_scores[i] = f1_score(gt[indices], pred[indices])

            acc_lower, acc_upper = compute_confidence_interval(acc_scores)
            f1_lower, f1_upper = compute_confidence_interval(f1_scores)

            image_metrics.update(
                {
                    "image_auroc": round_percent(auroc),
                    "image_auroc_ci": (
                        round_percent(auroc_lower),
                        round_percent(auroc_upper),
                    ),
                    "image_ap": round_percent(ap),
                    "image_ap_ci": (round_percent(ap_lower), round_percent(ap_upper)),
                    "image_f1": round_percent(f1),
                    "image_f1_ci": (round_percent(f1_lower), round_percent(f1_upper)),
                    "image_acc": round_percent(acc),
                    "image_acc_ci": (round_percent(acc_lower), round_percent(acc_upper)),
                }
            )

    # return image_metrics, threshold
    profile = prof.result
    profile["n_images"] = int(ep_score_gt.size)
    profile["throughput_images_per_s"] = (
        profile["n_images"] / profile["wall_time_s"] if profile["wall_time_s"] > 0 else np.nan
    )
    return image_metrics, threshold, profile


def compute_ad_metric(data, save_path=None, only_img=False, use_ci=True):
    """
    Compute image and pixel-level metrics from dataset.

    Parameters:
        data (list[dict]): Each dict with keys: "img_label", "img_output", "pixel_label", "pixel_output"
        save_path (str): Path to save npz file.
        only_img (bool): Whether to skip pixel metrics.
        use_ci (bool): Whether to compute confidence intervals.

    Returns:
        dict: Combined metrics.
    """
    num_samples = len(data)
    print(f"Computing metrics for {num_samples} samples...")

    ep_score_gt = np.empty(num_samples, dtype=np.uint8)
    ep_score_pred = np.empty(num_samples, dtype=np.float16)

    gt_pixel_list = []
    pred_pixel_list = []

    for i, sample in enumerate(tqdm(data)):
        if not all(
            k in sample
            for k in ["img_label", "img_output", "pixel_label", "pixel_output"]
        ):
            raise KeyError(f"Missing required keys in sample {i}")

        ep_score_gt[i] = sample["img_label"]
        ep_score_pred[i] = sample["img_output"]

        if not only_img:
            gt_pixel_list.append(sample["pixel_label"].astype(np.uint8).ravel())
            pred_pixel_list.append(sample["pixel_output"].astype(np.float16).ravel())

    if not only_img:
        total_gt_pixel_scores = np.concatenate(gt_pixel_list)
        total_pred_pixel_scores = np.concatenate(pred_pixel_list)

    del data

    if save_path is not None:
        np.savez_compressed(
            os.path.join(save_path, "ep_score_gt.npz"),
            data=ep_score_gt,
        )
        np.savez_compressed(
            os.path.join(save_path, "ep_score_pred.npz"),
            data=ep_score_pred,
        )

        del gt_pixel_list, pred_pixel_list

        if not only_img:
            np.savez_compressed(
                os.path.join(save_path, "total_gt_pixel_scores.npz"),
                data=total_gt_pixel_scores,
            )
            np.savez_compressed(
                os.path.join(save_path, "total_pred_pixel_scores.npz"),
                data=total_pred_pixel_scores,
            )

    if not only_img:
        pixel_metrics, pixel_threshold, pixel_profile = compute_pixel_metrics(
            total_gt_pixel_scores, total_pred_pixel_scores, use_ci=use_ci
        )
        del total_gt_pixel_scores, total_pred_pixel_scores
    else:
        pixel_metrics = {}
        pixel_threshold = None
        pixel_profile = {}

    image_metrics, image_threshold, image_profile = compute_image_metrics(
        ep_score_gt, ep_score_pred, use_ci=use_ci
    )
    del ep_score_gt, ep_score_pred

    return {
        **image_metrics,
        **pixel_metrics,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "profile_image": image_profile,
        "profile_pixel": pixel_profile,
    }
