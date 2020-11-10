"""Model test."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 02日 星期一 17:52:14 CST
# ***
# ************************************************************************************/
#
import os
import argparse
import torch
from data import get_data
from model import get_model, model_load, valid_epoch, model_setenv

if __name__ == "__main__":
    """Test model."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/VideoColor.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # get model
    model_r = get_model("modelR")
    model_load(model_r, args.checkpoint)
    model_r.to(device)

    model_c = get_model("modelR")
    model_load(model_c, args.checkpoint)
    model_c.to(device)

    if os.environ["ENABLE_APEX"] == "YES":
        from apex import amp
        model_r = amp.initialize(model_r, opt_level="O1")
        model_c = amp.initialize(model_c, opt_level="O1")

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model_r, device, tag='test')
