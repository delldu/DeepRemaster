"""Data loader."""
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
import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.utils as utils
from PIL import Image

train_dataset_rootdir = "dataset/train/"
test_dataset_rootdir = "dataset/test/"

# !!! For VideoColor, this const must be > 1 !!!
VIDEO_SEQUENCE_LENGTH = 5


def rgb2xyz(rgb):  # rgb from [0,1]
    # [0.412453, 0.357580, 0.180423],
    # [0.212671, 0.715160, 0.072169],
    # [0.019334, 0.119193, 0.950227]

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:, 0, :, :]+.357580*rgb[:, 1, :, :]+.180423*rgb[:, 2, :, :]
    y = .212671*rgb[:, 0, :, :]+.715160*rgb[:, 1, :, :]+.072169*rgb[:, 2, :, :]
    z = .019334*rgb[:, 0, :, :]+.119193*rgb[:, 1, :, :]+.950227*rgb[:, 2, :, :]

    out = torch.cat(
        (x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)

    return out


def xyz2rgb(xyz):
    # [ 3.24048134, -1.53715152, -0.49853633],
    # [-0.96925495,  1.87599   ,  0.04155593],
    # [ 0.05564664, -0.20404134,  1.05731107]

    r = 3.24048134*xyz[:, 0, :, :]-1.53715152 * \
        xyz[:, 1, :, :]-0.49853633*xyz[:, 2, :, :]
    g = -0.96925495*xyz[:, 0, :, :]+1.87599 * \
        xyz[:, 1, :, :]+.04155593*xyz[:, 2, :, :]
    b = .05564664*xyz[:, 0, :, :]-.20404134 * \
        xyz[:, 1, :, :]+1.05731107*xyz[:, 2, :, :]

    rgb = torch.cat(
        (r[:, None, :, :], g[:, None, :, :], b[:, None, :, :]), dim=1)
    # Some times reaches a small negative number, which causes NaNs
    rgb = torch.max(rgb, torch.zeros_like(rgb))

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    return rgb


def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    # sc.size() torch.Size([1, 3, 1, 1])

    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:, 1, :, :]-16.
    a = 500.*(xyz_int[:, 0, :, :]-xyz_int[:, 1, :, :])
    b = 200.*(xyz_int[:, 1, :, :]-xyz_int[:, 2, :, :])
    out = torch.cat(
        (L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)

    return out


def lab2xyz(lab):
    y_int = (lab[:, 0, :, :]+16.)/116.
    x_int = (lab[:, 1, :, :]/500.) + y_int
    z_int = y_int - (lab[:, 2, :, :]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat(
        (x_int[:, None, :, :], y_int[:, None, :, :], z_int[:, None, :, :]), dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    sc = sc.to(out.device)

    out = out*sc

    return out


def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    # xyz2lab(rgb2xyz(rgb)) parameters:
    # input: rgb in [0, 1.0]
    # output: l in [0, 100], ab in [-110, 110]

    l_rs = lab[:, [0], :, :]/100.0
    ab_rs = (lab[:, 1:, :, :] + 110.0)/220.0
    out = torch.cat((l_rs, ab_rs), dim=1)

    # return: tensor space: [0.0, 1.0]
    return out


def lab2rgb(lab_rs):
    l = lab_rs[:, [0], :, :] * 100.0
    ab = (lab_rs[:, 1:, :, :]) * 220.0 - 110.0
    lab = torch.cat((l, ab), dim=1)

    # lab range: l->[0, 100], ab in [-110, 110] ==> rgb: [0, 1.0]
    out = xyz2rgb(lab2xyz(lab))

    return out

def multiple_crop(data, mult=16, HWmax=[4096, 4096]):
    # crop image to a multiple
    H, W = data.shape[1:]
    Hnew = min(int(H/mult)*mult, HWmax[0])
    Wnew = min(int(W/mult)*mult, HWmax[1])
    h = (H-Hnew)//2
    w = (W-Wnew)//2
    return data[:, h:h+Hnew, w:w+Wnew]

def get_transform(train=True):
    """Transform images."""
    ts = []
    # if train:
    #     ts.append(T.RandomHorizontalFlip(0.5))

    ts.append(T.ToTensor())
    return T.Compose(ts)

def get_reference(root):
    """Get all frames of video."""

    filelist = list(sorted(os.listdir(root)))
    sequence = []
    transforms = get_transform(train = False)
    for filename in filelist:
        img = Image.open(os.path.join(root, filename)).convert("RGB")
        img = transforms(img)
        img = multiple_crop(img)
        C, H, W = img.size()
        img = img.view(1, C, H, W)
        sequence.append(img)

    # General, return Tensor: T x C x H x W], T = self.seqlen
    return torch.cat(sequence, dim=0)

class Video(data.Dataset):
    """Define Video Frames Class."""

    def __init__(self, seqlen=VIDEO_SEQUENCE_LENGTH, transforms=get_transform()):
        """Init dataset."""
        super(Video, self).__init__()
        self.seqlen = seqlen
        self.transforms = transforms
        self.root = ""
        self.images = []

    def reset(self, root):
        # print("Video Reset Root: ", root)
        self.root = root
        self.images = list(sorted(os.listdir(root)))

    def __getitem__(self, idx):
        """Load images."""
        n = len(self.images)
        filelist = []
        for k in range(-(self.seqlen//2), (self.seqlen//2) + 1):
            if (idx + k < 0):
                filename = self.images[0]
            elif (idx + k >= n):
                filename = self.images[n - 1]
            else:
                filename = self.images[idx + k]
            filelist.append(os.path.join(self.root, filename))
        # print("filelist: ", filelist)
        sequence = []
        for filename in filelist:
            img = Image.open(filename).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
                img = multiple_crop(img)
                C, H, W = img.size()
                img = img.view(1, C, H, W)
            sequence.append(img)

        if self.transforms is not None:
            # General, return Tensor: T x C x H x W], T = self.seqlen
            return torch.cat(sequence, dim=0)

        return sequence

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)


class VideoColorDataset(data.Dataset):
    """Define dataset."""

    def __init__(self, root, seqlen=VIDEO_SEQUENCE_LENGTH, transforms=get_transform()):
        """Init dataset."""
        super(VideoColorDataset, self).__init__()

        self.root = root
        self.seqlen = seqlen
        self.transforms = transforms

        # load all images, sorting for alignment
        self.images = []
        # index start offset
        self.indexs = []
        offset = 0
        ds = list(sorted(os.listdir(root)))
        for d in ds:
            fs = sorted(os.listdir(root + "/" + d))
            for f in fs:
                self.images.append(d + "/" + f)
                self.indexs.append(offset)
            offset += len(fs)
        self.video_cache = Video(seqlen=seqlen, transforms=transforms)

    def __getitem__(self, idx):
        """Load images."""
        # print("dataset index:", idx)
        image_path = os.path.join(self.root, self.images[idx])
        if (self.video_cache.root != os.path.dirname(image_path)):
            self.video_cache.reset(os.path.dirname(image_path))
        return self.video_cache[idx - self.indexs[idx]]

    def __len__(self):
        """Return total numbers of images."""
        return len(self.images)

    def __repr__(self):
        """
        Return printable representation of the dataset object.
        """
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms: '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def train_data(bs):
    """Get data loader for trainning & validating, bs means batch_size."""

    train_ds = VideoColorDataset(
        train_dataset_rootdir, VIDEO_SEQUENCE_LENGTH, get_transform(train=True))
    print(train_ds)

    # Split train_ds in train and valid set
    # xxxx--modify here
    valid_len = int(0.2 * len(train_ds))
    indices = [i for i in range(len(train_ds) - valid_len, len(train_ds))]

    valid_ds = data.Subset(train_ds, indices)
    indices = [i for i in range(len(train_ds) - valid_len)]
    train_ds = data.Subset(train_ds, indices)

    # Define training and validation data loaders
    n_threads = min(4,  bs)
    train_dl = data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=n_threads)
    valid_dl = data.DataLoader(
        valid_ds, batch_size=bs, shuffle=False, num_workers=n_threads)

    return train_dl, valid_dl


def test_data(bs):
    """Get data loader for test, bs means batch_size."""

    test_ds = VideoColorDataset(
        test_dataset_rootdir, VIDEO_SEQUENCE_LENGTH,  get_transform(train=False))
    test_dl = data.DataLoader(test_ds, batch_size=bs,
                              shuffle=False, num_workers=4)

    return test_dl


def get_data(trainning=True, bs=4):
    """Get data loader for trainning & validating, bs means batch_size."""

    return train_data(bs) if trainning else test_data(bs)


def VideoColorDatasetTest():
    """Test dataset ..."""

    ds = VideoColorDataset(train_dataset_rootdir)
    print(ds)
    vs = Video()
    vs.reset("dataset/predict/input")

    refimgs = get_reference("dataset/predict/reference")
    pdb.set_trace()


if __name__ == '__main__':
    VideoColorDatasetTest()
