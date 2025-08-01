import argparse
import logging
import os
import sys

from datetime import datetime, timedelta

import numpy as np
import monai
import torch
import torch.distributed as dist

from monai.data import Dataset, list_data_collate, DistributedSampler
from monai.transforms import Compose
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import AUROC

from ep_generator import EmphysemaGenerated
from utils import config_cpu_num, set_random_seed, setup_logging, log_config_details
from lr_scheduler import LinearWarmupCosineAnnealingLR
from edlnet import EDLNet


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="EDLNet")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cpu_num", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--input_size", default=[384, 384])
parser.add_argument("--prob", type=float, default=0.9)
parser.add_argument("--thr", type=float, default=0.3)
parser.add_argument("--gate", action="store_true", default=False)
parser.add_argument("--fup", action="store_true", default=False)
parser.add_argument("--scrb", action="store_true", default=False)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument(
    "--base_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--tr_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--save",
    type=str,
    default=None,
)

args = parser.parse_args()


save_path = os.path.join(
    args.save,
    args.model + "_" + str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")),
)
os.makedirs(save_path, exist_ok=True)

os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
dist.init_process_group(backend="nccl", timeout=timedelta(seconds=10800))

rank = dist.get_rank()
device_id = rank % torch.cuda.device_count()
torch.cuda.set_device(device_id)

dist.barrier(device_ids=[device_id])

device = torch.device(device_id)

set_random_seed(args.seed)
config_cpu_num(args.cpu_num)


def main():
    setup_logging("training.log", save_path)

    tr_files = np.load(args.tr_path, allow_pickle=True)["arr_0"]
    val_files = np.load(args.val_path, allow_pickle=True)["arr_0"]

    tr_trans = Compose(
        [
            monai.transforms.LoadImaged(keys=["img", "seg", "mask"], image_only=True),
            monai.transforms.EnsureChannelFirstd(keys=["img", "seg", "mask"]),
            monai.transforms.ResizeWithPadOrCropd(
                keys=["img", "seg", "mask"],
                mode=["constant", "constant", "constant"],
                spatial_size=args.input_size,
            ),
            EmphysemaGenerated(
                keys=["img", "seg", "mask"],
                prob=args.prob,
                thr=args.thr,
                seed=args.seed,
            ),
        ]
    )

    tr_ds = Dataset(data=tr_files, transform=tr_trans)
    tr_sampler = DistributedSampler(dataset=tr_ds, even_divisible=True, shuffle=True)
    tr_dl = DataLoader(
        dataset=tr_ds,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        sampler=tr_sampler,
        pin_memory=True,
    )

    model = EDLNet(
        n_channels=1,
        n_classes=2,
        alpha=args.alpha,
        use_gate=args.gate,
        use_scrb=args.scrb,
    ).to(device)

    model = DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=args.fup,
    )

    mse_loss = torch.nn.MSELoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-5,
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
    )

    if rank == 0:
        log_config_details(args)

    best_auroc_metric = -1
    val_interval = args.val_interval
    for epoch in range(args.max_epochs):
        model.train()
        for i_batch, batch_data in enumerate(tr_dl):
            imgs = batch_data["img"].to(device)  
            inputs = batch_data["aug_img"].to(device)  
            labels = batch_data["seg"].to(device).to(torch.bfloat16)  

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # ---- Forward Pass ----
                recon_image = model(inputs)
                recon_img = recon_image[:, 0].unsqueeze(1)
                pred_seg = recon_image[:, 1].unsqueeze(1)

                # ---- Loss Calculation ----
                recon_loss = mse_loss(imgs, recon_img)
                seg_loss = mse_loss(labels, pred_seg)

                total_loss = recon_loss + seg_loss

            total_loss.backward()
            optimizer.step()

            if rank == 0:
                logging.info(
                    f"epoch: {epoch + 1}/{args.max_epochs}, "
                    f"batch: {i_batch + 1}/{len(tr_dl)}, "
                    f"total_loss: {total_loss:.4f}, "
                    f"recon_loss: {recon_loss:.4f}, "
                    f"seg_loss: {seg_loss:.4f}"
                )

        scheduler.step()

        if rank == 0:
            state = {"model_state_dict": model.module.state_dict()}
            torch.save(state, os.path.join(save_path, f"auroc_model.pth"))

    dist.destroy_process_group()


main()
