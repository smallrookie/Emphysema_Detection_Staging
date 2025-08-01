import random
import numpy as np

from typing import Dict, Hashable, Mapping

import monai
import imgaug.augmenters as iaa

from monai.config.type_definitions import NdarrayOrTensor
from monai.config import KeysCollection
from monai.transforms.transform import MapTransform, RandomizableTransform
from scipy.spatial import cKDTree


class EmphysemaGenerated(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ep_prior_file: str = None,
        prob: float = 0.1,
        thr: float = 0.3,
        size: tuple = (384, 384),
        seed: int = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        if ep_prior_file is not None:
            ep_prior_data = np.load(ep_prior_file, allow_pickle=True)
            self.ep_pixel_val = ep_prior_data["values"]
            self.ep_pixel_val_prob = ep_prior_data["prob"]
        else:
            self.ep_pixel_val = None
            self.ep_pixel_val_prob = None

        self.size = size
        self.thr = thr
        self.transform = self._get_transform()
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def _get_transform(self):
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
        
        ep_img = np.random.randint(-1000, -951, size=self.size)

        img_thr = ep_img * worley_thr
        ep_mask = (mask != 0) & (img < -600)

        aug_img = np.where(worley_thr, img_thr, img)
        aug_img = np.where(ep_mask, aug_img, img)

        ep_mask = (ep_mask & (aug_img < -950)).astype(np.float32)

        aug_img_tensor = monai.data.MetaTensor(aug_img)
        ep_mask_tensor = monai.data.MetaTensor(ep_mask)
        return aug_img_tensor, ep_mask_tensor

    def rand_worley_2d_np(self, shape, num_points):
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
