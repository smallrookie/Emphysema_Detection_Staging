import random
import numpy as np

from typing import Dict, Hashable, Mapping

import monai
import imgaug.augmenters as iaa

from monai.config.type_definitions import NdarrayOrTensor
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform, RandomizableTransform
from scipy.spatial import cKDTree
from scripy.stats import truncnorm


class EmphysemaGenerated(RandomizableTransform, MapTransform):
    """Emphysema Generation Transform
    
    A MONAI transform that generates synthetic emphysema patterns in lung CT scans
    using Worley noise and truncated Gaussian distributions.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        type: str = "uniform",
        thr: float = 0.3,
        size: tuple = (384, 384),
        seed: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        """Initialize the emphysema generation transform.
        
        Args:
            keys (KeysCollection): Keys to apply the transform to
            prob (float): Probability of applying the transform
            type (str): Type of noise distribution ("uniform" or "guassian")
            thr (float): Threshold for emphysema pattern generation
            size (tuple): Size of the input images
            seed (int): Random seed for reproducibility
            allow_missing_keys (bool): Whether to allow missing keys in the data dictionary
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.type = type

        self.size = size
        self.thr = thr
        self.transform = self._get_transform()
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def _get_transform(self):
        """Get the base transform pipeline.
        
        Returns:
            Compose: MONAI transform pipeline
        """
        base_transforms = [
            monai.transforms.EnsureChannelFirstd(
                keys=["img", "aug_img", "seg"], channel_dim="no_channel"
            ),
            monai.transforms.ScaleIntensityRanged(
                keys=["img", "aug_img"],
                a_min=-1000,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            monai.transforms.RandFlipd(
                keys=["img", "aug_img", "seg"], prob=0.3, spatial_axis=[0, 1]
            ),
            monai.transforms.RandRotate90d(keys=["img", "aug_img", "seg"], prob=0.3),
        ]

        return monai.transforms.Compose(base_transforms)

    def generate_emphysema(self, image, mask):
        """Generate synthetic emphysema patterns in the input image.
        
        Args:
            image: Input CT image
            mask: Lung mask
            
        Returns:
            tuple: Augmented image and emphysema mask
        """
        img = image.get_array()
        mask = mask.get_array()

        worley_scale = 12
        min_worley_scale = 8
        num_points = 2 ** np.random.randint(min_worley_scale, worley_scale)

        worley_noise = self.rand_worley_2d_np(self.size, num_points)
        worley_noise = (worley_noise - worley_noise.min()) / (
            worley_noise.max() - worley_noise.min()
        )
        worley_noise = self.rot(image=worley_noise)

        worley_thr = worley_noise > self.thr
        
        if self.type == "uniform":
            ep_img = np.random.randint(-1000, -951, size=self.size)
        elif self.type == "guassian":
            a = -1000
            b = -951
            mean = -975
            std = 5
            a_, b_ = (a - mean) / std, (b - mean) / std
            trunc_gauss = truncnorm(a_, b_, loc=mean, scale=std)
            ep_img = trunc_gauss.rvs(self.size).astype(np.int32)
        else:
            raise ValueError("Unknown prior type")


        img_thr = ep_img * worley_thr
        ep_mask = (mask != 0) & (img < -600)

        aug_img = np.where(worley_thr, img_thr, img)
        aug_img = np.where(ep_mask, aug_img, img)

        ep_mask = (ep_mask & (aug_img < -950)).astype(np.float32)

        aug_img_tensor = monai.data.MetaTensor(aug_img)
        ep_mask_tensor = monai.data.MetaTensor(ep_mask)
        return aug_img_tensor, ep_mask_tensor

    def rand_worley_2d_np(self, shape, num_points):
        """Generate 2D Worley noise.
        
        Args:
            shape (tuple): Shape of the output noise
            num_points (int): Number of random points to generate
            
        Returns:
            np.ndarray: Generated Worley noise
        """
        height, width = shape
        points = np.random.rand(num_points, 2) * [width, height]
        tree = cKDTree(points)
        x_coords = np.arange(width)
        y_coords = np.arange(height)
        xx, yy = np.meshgrid(x_coords, y_coords)
        pixel_coords = np.column_stack((xx.ravel(), yy.ravel()))
        min_dists, _ = tree.query(pixel_coords, k=1)
        min_dists = min_dists.reshape((height, width))
        max_dist = np.hypot(width, height)
        noise = min_dists / max_dist
        return noise

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """Apply the emphysema generation transform.
        
        Args:
            data (Mapping): Input data dictionary
            
        Returns:
            Dict: Transformed data dictionary
        """
        data_dict = dict(data)
        self.randomize(None)

        img = data_dict["img"][0]
        seg = data_dict["seg"][0]
        mask = data_dict["mask"][0]

        if self._do_transform and seg.max() != 1:
            ep_img, ep_mask = self.generate_emphysema(img, mask)
            return self.transform({"img": img, "aug_img": ep_img, "seg": ep_mask})
        else:
            return self.transform({"img": img, "aug_img": img, "seg": seg})