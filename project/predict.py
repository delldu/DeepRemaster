"""Model predict."""
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
import glob
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model, model_load, model_setenv
from data import Video, get_reference, rgb2lab, lab2rgb
from tqdm import tqdm

import pdb

if __name__ == "__main__":
    """Predict."""

    model_setenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="models/VideoColor.pth", help="checkpint file")
    parser.add_argument('--input', type=str, default="dataset/predict/input", help="input image folder")
    parser.add_argument('--output', type=str, default="dataset/predict/output", help="output image folder")

    args = parser.parse_args()

    # CPU or GPU ?
    device = torch.device(os.environ["DEVICE"])

    # Restoration
    model_r = get_model("modelR")
    model_load(model_r, "modelR", args.checkpoint)
    model_r.to(device)
    model_r.eval()

    model_c = get_model("modelC")
    model_load(model_c, "modelC", args.checkpoint)
    model_c.to(device)
    model_c.eval()


    # if os.environ["ENABLE_APEX"] == "YES":
    #     from apex import amp
    #     model_r = amp.initialize(model_r, opt_level="O1")
    #     model_c = amp.initialize(model_c, opt_level="O1")

    totensor = transforms.ToTensor()
    toimage = transforms.ToPILImage()

    # notice seqlen must be greater than 1, for an example: 5
    seqlen = 5

    video = Video(seqlen=seqlen)
    video.reset(args.input)
    progress_bar = tqdm(total=len(video))
    refimgs = get_reference(os.path.dirname(args.input) + "/reference")
    refimgs = refimgs.unsqueeze(0)
    reference_tensor = refimgs.to(device)
    # (Pdb) refimgs.size()
    # torch.Size([1, 4, 3, 256, 341]), [B, T, C, H , W] OK

    # len(video)
    for index in range(0, len(video), seqlen):
        progress_bar.update(seqlen)

        input_tensor = rgb2lab(video[index]).unsqueeze(0)

        # [B,T,C,H,W] --> [B,C,T,H,W]
        input_tensor = input_tensor.transpose(2, 1).contiguous()
        input_tensor = input_tensor.to(device)
        # (Pdb) input_tensor.size()
        # torch.Size([1, 3, 1, 320, 432]) ==> [B,C,T,H,W] OK
        input_tensor_l = input_tensor[:, [0], :, :, :]
        with torch.no_grad():
            output_tensor_l = model_r(input_tensor_l)  # [B, C, T, H, W]
            output_tensor_ab = model_c(output_tensor_l, reference_tensor)

        # [B,C,T,H,W] -> [B,T,C,H,W]
        output_tensor_l = output_tensor_l.transpose(2, 1).contiguous()
        output_tensor_ab = output_tensor_ab.transpose(2, 1).contiguous()
        # (Pdb) output_tensor_l.size(), output_tensor_ab.size()
        # (torch.Size([1, 1, 1, 320, 432]), torch.Size([1, 1, 2, 320, 432]))

        output_tensor_l.squeeze_(0)
        output_tensor_ab.squeeze_(0)
        output_tensor_lab = torch.cat((output_tensor_l, output_tensor_ab), dim=1)
        output_tensor = lab2rgb(output_tensor_lab)

        # (Pdb) output_tensor_l.size(), output_tensor_ab.size()
        # (torch.Size([1, 1, 320, 432]), torch.Size([1, 2, 320, 432]))
        # (Pdb) output_tensor.size()
        # torch.Size([1, 3, 320, 432])

        output_tensor = output_tensor.clamp(0, 1.0)
        for j in range(seqlen):
            toimage(output_tensor[j].cpu()).save("{}/{:06d}.png".format(args.output, index + 1 + j))
