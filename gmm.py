import json, os, random, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


file_cfgs = {
    "Dataset A": {
        "path": "results/dataset_a/case_as.json",
        "random_state": 42,
        "init": "random",
    },
    "Dataset B": {
        "path": "results/dataset_b/case_as.json",
        "random_state": 42,
        "init": "random",
    },
    "Dataset C": {
        "path": "results/dataset_c/case_as.json",
        "random_state": 42,
        "init": "random",
    },
    "Dataset D": {
        "path": "results/dataset_d/case_as.json",
        "random_state": 42,
        "init": "random",
    },
}

stage_names = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
classes = sorted(stage_names)
mean_fpr = np.linspace(0, 1, 101)
results = {}

n_bootstrap = 1000
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

for name, cfg in file_cfgs.items():
    print(f"\n=== Processing {name} ===")

    with open(cfg["path"], "r") as f:
        data = json.load(f)
    df = pd.DataFrame(
        [
            {"id": k, "pred_as": v["pred_as"], "stage": v["stage"]}
            for k, v in data.items()
        ]
    )

    # ---------- GMM ----------
    gmm = GaussianMixture(
        n_components=4, random_state=cfg["random_state"], init_params=cfg["init"]
    )
    df["cluster"] = gmm.fit_predict(df[["pred_as"]])

    mapping = {
        cl: df.loc[df["cluster"] == cl, "stage"].mode().iloc[0]
        for cl in range(4)
        if not df.loc[df["cluster"] == cl, "stage"].mode().empty
    }
    df["mapped"] = df["cluster"].map(mapping)
    stage2cluster = {st: cl for cl, st in mapping.items()}

    # ------------------------------------------------------------------
    # —— GMM Beeswarm / Swarm Plot
    # ------------------------------------------------------------------
    import seaborn as sns, matplotlib.pyplot as plt, matplotlib.patches as mpatches
    import numpy as np, pandas as pd

    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={"font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "legend.title_fontsize": 18}
    )

    pred_vals = np.array([p if np.isscalar(p) else p[0] for p in df["pred_as"]])
    df_plot = pd.DataFrame(
        {"pred_as": pred_vals,
        "stage_idx": df["mapped"]}
    )
    stage_order   = ["None", "Mild", "Moderate", "Severe"]
    palette_base  = sns.color_palette("Set2", 4)        
    stage_palette = dict(zip(stage_order, palette_base))
    df_plot["stage_name"] = df_plot["stage_idx"].map(stage_names)

    # 2) Swarm plot
    fig_swarm, ax_swarm = plt.subplots(figsize=(8, 6))
    sns.swarmplot(
        data=df_plot,
        x="pred_as",
        y="stage_name",
        order=stage_order,          
        hue="stage_name",
        palette=stage_palette,
        size=6, edgecolor="k", linewidth=0.6, alpha=0.8,
        dodge=False, ax=ax_swarm
    )

    for idx, st in enumerate(stage_order):
        vals = df_plot.loc[df_plot["stage_name"] == st, "pred_as"]
        if len(vals) == 0:
            continue
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax_swarm.plot(med, idx, marker="D", color="k", ms=8, zorder=3)
        ax_swarm.hlines(idx, q1, q3, color="k", lw=3, zorder=2)

    ax_swarm.set_yticklabels(stage_order, rotation=45, ha="right")
    ax_swarm.set_xlabel(""); ax_swarm.set_ylabel("")

    handles = [mpatches.Patch(color=stage_palette[st], label=st)
            for st in stage_order if st in df_plot["stage_name"].unique()]
    ax_swarm.legend(handles=handles, title="", loc="best")

    sns.despine(ax=ax_swarm, left=False, bottom=True)
    ax_swarm.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig_swarm.savefig(f"images/{name}_beeswarm.svg", dpi=600)
    plt.close(fig_swarm)

    probs_raw = gmm.predict_proba(df[["pred_as"]])  # (n,4)
    probs_aligned = np.zeros_like(probs_raw)
    for st, cl in stage2cluster.items():
        probs_aligned[:, st] = probs_raw[:, cl]

    y_true, y_pred = df["stage"].values, df["mapped"].values

    # ---------- Bootstrap ----------
    strata_idx = {st: df[df["stage"] == st].index.to_numpy() for st in classes}
    tprs_micro, tprs_macro, metric_rows = [], [], []

    for _ in tqdm(range(n_bootstrap), desc=f"{name} bootstrap", leave=False):
        idxs = np.concatenate(
            [
                resample(strata_idx[st], replace=True, n_samples=len(strata_idx[st]))
                for st in classes
            ]
        )
        y_t, y_p = y_true[idxs], y_pred[idxs]
        probs_bs = probs_aligned[idxs]
        y_bin = label_binarize(y_t, classes=classes)

        metric_rows.append(
            {
                "auc_micro": roc_auc_score(
                    pd.get_dummies(y_t), probs_bs, average="micro", multi_class="ovr"
                ),
                "auc_macro": roc_auc_score(
                    pd.get_dummies(y_t), probs_bs, average="macro", multi_class="ovr"
                ),
                "acc_all": accuracy_score(y_t, y_p),
                "f1_micro": f1_score(y_t, y_p, average="micro", zero_division=0),
                "f1_macro": f1_score(y_t, y_p, average="macro", zero_division=0),
            }
        )

        # -- micro ROC
        fpr_mi, tpr_mi, _ = roc_curve(y_bin.ravel(), probs_bs.ravel())
        tprs_micro.append(np.interp(mean_fpr, fpr_mi, tpr_mi, left=0, right=1))

        # -- macro ROC
        all_fpr = np.unique(
            np.concatenate(
                [roc_curve(y_bin[:, i], probs_bs[:, i])[0] for i in range(4)]
            )
        )
        mean_tpr = (
            sum(
                np.interp(
                    all_fpr,
                    roc_curve(y_bin[:, i], probs_bs[:, i])[0],
                    roc_curve(y_bin[:, i], probs_bs[:, i])[1],
                )
                for i in range(4)
            )
            / 4
        )
        tprs_macro.append(np.interp(mean_fpr, all_fpr, mean_tpr, left=0, right=1))

    df_bs = pd.DataFrame(metric_rows)
    tprs_micro = np.vstack(tprs_micro)
    tprs_macro = np.vstack(tprs_macro)

    res = {
        "mean_tpr_mi": tprs_micro.mean(axis=0),
        "low_mi": np.percentile(tprs_micro, 2.5, axis=0),
        "hi_mi": np.percentile(tprs_micro, 97.5, axis=0),
        "mean_tpr_ma": tprs_macro.mean(axis=0),
        "low_ma": np.percentile(tprs_macro, 2.5, axis=0),
        "hi_ma": np.percentile(tprs_macro, 97.5, axis=0),
    }
    for m in ["auc_micro", "auc_macro", "acc_all", "f1_micro", "f1_macro"]:
        res[f"{m}_mean"] = df_bs[m].mean()
        res[f"{m}_lo"], res[f"{m}_hi"] = df_bs[m].quantile([0.025, 0.975])

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    res["cm"] = cm
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)  
    os.makedirs("images", exist_ok=True)
    fig_cm, ax_cm = plt.subplots(figsize=(5.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=[stage_names[c] for c in classes]).plot(
        cmap="Blues", ax=ax_cm, colorbar=False, values_format="d"
    )
    plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig_cm.savefig(f"images/{safe_name}_confusion.svg", dpi=600)
    plt.close(fig_cm)

    results[name] = res


colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
linestyles = ["-", "--", "-.", ":"]
markers = ["o", "s", "D", "^"]
mark_every = 12


def plot_family(mean_key: str, auc_key: str, outfile: str):
    plt.figure(figsize=(12, 10))
    for idx, (name, res) in enumerate(results.items()):
        auc_val = res[auc_key] * 100 
        plt.plot(
            mean_fpr,
            res[mean_key],
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=3,
            marker=markers[idx],
            markevery=mark_every,
            markersize=5,
            label=f"{name} ({auc_val:.2f}%)",
        )
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("False Positive Rate", fontsize=28)
    plt.ylabel("True Positive Rate", fontsize=28)
    plt.legend(loc="lower right", fontsize=28)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    plt.close()


plot_family("mean_tpr_mi", "auc_micro_mean", "images/micro_roc_all.svg")
plot_family("mean_tpr_ma", "auc_macro_mean", "images/macro_roc_all.svg")

print("\n=== Summary (mean ± 95% CI) ===")
for name, res in results.items():
    print(f"\n{name}")
    for key, label in [
        ("auc_micro", "Micro-AUROC"),
        ("auc_macro", "Macro-AUROC"),
        ("acc_all", "Accuracy"),
        ("f1_micro", "F1-micro"),
        ("f1_macro", "F1-macro"),
    ]:
        print(
            f"  {label:<12}: {res[f'{key}_mean']*100:.2f}% "
            f"({res[f'{key}_lo']*100:.2f}–{res[f'{key}_hi']*100:.2f}%)"
        )

print("\nAll figures saved to ./images/")
