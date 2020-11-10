"""Create model."""
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
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from data import VIDEO_SEQUENCE_LENGTH

def PSNR(img1, img2):
    """PSNR."""
    difference = (1.*img1-img2)**2
    mse = torch.sqrt(torch.mean(difference)) + 0.000001
    return 20*torch.log10(1./mse)

class TempConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)):
        super(TempConv, self).__init__()
        self.conv3d = nn.Conv3d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(x)), inplace=True)


class Upsample(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=(1, 2, 2)):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv3d = nn.Conv3d(in_planes, out_planes, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        return F.elu(self.bn(self.conv3d(F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False))), inplace=True)


class UpsampleConcat(nn.Module):
    def __init__(self, in_planes_up, in_planes_flat, out_planes):
        super(UpsampleConcat, self).__init__()
        self.conv3d = TempConv(in_planes_up + in_planes_flat, out_planes,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=(1, 2, 2),
                           mode='trilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)

        del x1, x2
        torch.cuda.empty_cache()

        return self.conv3d(x)


class SourceReferenceAttention(nn.Module):
    """
    Source-Reference Attention Layer
    """

    def __init__(self, in_planes_s, in_planes_r):
        """
        Parameters
        ----------
            in_planes_s: int
                Number of input source feature vector channels.
            in_planes_r: int
                Number of input reference feature vector channels.
        """
        super(SourceReferenceAttention, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_planes_s,
                                    out_channels=in_planes_s//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_planes_r,
                                  out_channels=in_planes_r//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_planes_r,
                                    out_channels=in_planes_r,    kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source, reference):
        """
        Parameters
        ----------
            source : torch.Tensor
                Source feature maps (B x Cs x Ts x Hs x Ws)
            reference : torch.Tensor
                Reference feature maps (B x Cr x Tr x Hr x Wr )
         Returns :
            torch.Tensor
                Source-reference attention value added to the input source features
            torch.Tensor
                Attention map (B x Ns x Nt) (Ns=Ts*Hs*Ws, Nr=Tr*Hr*Wr)
        """
        s_batchsize, sC, sT, sH, sW = source.size()
        r_batchsize, rC, rT, rH, rW = reference.size()
        proj_query = self.query_conv(source).view(
            s_batchsize, -1, sT*sH*sW).permute(0, 2, 1)
        proj_key = self.key_conv(reference).view(r_batchsize, -1, rT*rW*rH)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(reference).view(r_batchsize, -1, rT*rH*rW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(s_batchsize, sC, sT, sH, sW)
        out = self.gamma*out + source

        del proj_query, proj_key, energy, proj_value
        torch.cuda.empty_cache()

        return out, attention


class VideoRestoreModel(nn.Module):
    def __init__(self):
        super(VideoRestoreModel, self).__init__()

        self.layers = nn.Sequential(
            nn.ReplicationPad3d((1, 1, 1, 1, 1, 1)),
            TempConv(1,  64, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(0, 0, 0)),
            TempConv(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(128, 256, kernel_size=(3, 3, 3),
                     stride=(1, 2, 2), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Upsample(256, 128),
            TempConv(128,  64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            TempConv(64,  64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            Upsample(64, 16),
            nn.Conv3d(16, 1, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1))
        )

    def forward(self, x):
        return (x + torch.tanh(self.layers(x.clone()-0.4462414))).clamp_(0, 1)


class VideoColorModel(nn.Module):
    def __init__(self):
        super(VideoColorModel, self).__init__()

        self.down1 = nn.Sequential(
            nn.ReplicationPad3d((1, 1, 1, 1, 0, 0)),
            TempConv(1,  64, stride=(1, 2, 2), padding=(0, 0, 0)),
            TempConv(64, 128),
            TempConv(128, 128),
            TempConv(128, 256, stride=(1, 2, 2)),
            TempConv(256, 256),
            TempConv(256, 256),
            TempConv(256, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
            TempConv(512, 512)
        )
        self.flat = nn.Sequential(
            TempConv(512, 512),
            TempConv(512, 512)
        )
        self.down2 = nn.Sequential(
            TempConv(512, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
        )
        self.stattn1 = SourceReferenceAttention(
            512, 512)  # Source-Reference Attention
        self.stattn2 = SourceReferenceAttention(
            512, 512)  # Source-Reference Attention
        self.selfattn1 = SourceReferenceAttention(512, 512)  # Self Attention
        self.conv1 = TempConv(512, 512)
        self.up1 = UpsampleConcat(512, 512, 512)  # 1/8
        self.selfattn2 = SourceReferenceAttention(512, 512)  # Self Attention
        self.conv2 = TempConv(512, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.up2 = nn.Sequential(
            Upsample(256, 128),  # 1/4
            TempConv(128, 64, kernel_size=(3, 3, 3),
                     stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.up3 = nn.Sequential(
            Upsample(64, 32),  # 1/2
            TempConv(32, 16, kernel_size=(3, 3, 3),
                     stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.up4 = nn.Sequential(
            Upsample(16, 8),  # 1/1
            nn.Conv3d(8, 2, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1))
        )
        self.reffeatnet1 = nn.Sequential(
            TempConv(3,  64, stride=(1, 2, 2)),
            TempConv(64, 128),
            TempConv(128, 128),
            TempConv(128, 256, stride=(1, 2, 2)),
            TempConv(256, 256),
            TempConv(256, 256),
            TempConv(256, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
            TempConv(512, 512),
        )
        self.reffeatnet2 = nn.Sequential(
            TempConv(512, 512, stride=(1, 2, 2)),
            TempConv(512, 512),
            TempConv(512, 512),
        )

    def forward(self, x, x_refs=None):
        x1 = self.down1(x - 0.4462414)

        if x_refs is not None:
            # [B,T,C,H,W] --> [B,C,T,H,W]
            x_refs = x_refs.transpose(2, 1).contiguous()
            reffeat = self.reffeatnet1(x_refs-0.48)
            x1, _ = self.stattn1(x1, reffeat)

        x2 = self.flat(x1)
        out = self.down2(x1)

        del x1, _
        torch.cuda.empty_cache()

        if x_refs is not None:
            reffeat2 = self.reffeatnet2(reffeat)
            out, _ = self.stattn2(out, reffeat2)

        out = self.conv1(out)
        out, _ = self.selfattn1(out, out)
        out = self.up1(out, x2)
        out, _ = self.selfattn2(out, out)
        out = self.conv2(out)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)

        del x2, reffeat, reffeat2
        torch.cuda.empty_cache()

        return torch.sigmoid(out)

def model_load(model, subname, path):
    """Load model."""
    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    # subname: modelC, modelR
    state_dict = state_dict[subname]
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""
    torch.save(model.state_dict(), path)

def model_export():
    """Export model to onnx."""

    import onnx
    from onnx import optimizer

    # xxxx--modify here
    onnx_file = "model.onnx"
    weight_file = "checkpoint/weight.pth"

    # 1. Load model
    print("Loading model ...")
    model = VideoColorModel()
    model_load(model, "modelC", weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    # xxxx--modify here
    dummy_input = torch.randn(1, 3, 512, 512)
    input_names = [ "input" ]
    output_names = [ "output" ]
    torch.onnx.export(model, dummy_input, onnx_file,
                    input_names=input_names, 
                    output_names=output_names,
                    verbose=True,
                    opset_version=11,
                    keep_initializers_as_inputs=True,
                    export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    model = onnx.load(onnx_file)
    onnx.checker.check_model(model)

    print("Optimizing model ...")
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, onnx_file)

    # 4. Visual model
    # python -c "import netron; netron.start('model.onnx')"


def get_model(subname):
    """Create model."""
    if subname == "modelR":
        model = VideoRestoreModel()
    else:
        model = VideoColorModel()
    return model


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""
        self.reset()

    def reset(self):
        """Reset average."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # xxxx--modify here
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            # xxxx--modify here
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default environment variables to avoid exceptions
    if os.environ.get("ONLY_USE_CPU") != "YES" and os.environ.get("ONLY_USE_CPU") != "NO":
        os.environ["ONLY_USE_CPU"] = "NO"

    if os.environ.get("ENABLE_APEX") != "YES" and os.environ.get("ENABLE_APEX") != "NO":
        os.environ["ENABLE_APEX"] = "YES"

    if os.environ.get("DEVICE") != "YES" and os.environ.get("DEVICE") != "NO":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Is there GPU ?
    if not torch.cuda.is_available():
        os.environ["ONLY_USE_CPU"] = "YES"

    # export ONLY_USE_CPU=YES ?
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["ENABLE_APEX"] = "NO"
    else:
        try:
            from apex import amp
        except:
            os.environ["ENABLE_APEX"] = "NO"

    # Running on GPU if available
    if os.environ.get("ONLY_USE_CPU") == "YES":
        os.environ["DEVICE"] = 'cpu'
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])
    print("  ONLY_USE_CPU: ", os.environ["ONLY_USE_CPU"])
    print("  ENABLE_APEX: ", os.environ["ENABLE_APEX"])


def infer_perform():
    """Model infer performance ..."""

    model_setenv()
    device = os.environ["DEVICE"]

    model = VideoColorModel()
    model.eval()
    model = model.to(device)

    with tqdm(total=len(1000)) as t:
        t.set_description(tag)

        # xxxx--modify here
        input = torch.randn(64, 3, 512, 512)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)

        t.update(1)


if __name__ == '__main__':
    """Test model ..."""

    # model_export()
    infer_perform()
