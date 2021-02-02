import torch
import cv2
import numpy as np
import random
import torchvision.transforms.functional as F
from torchvision.transforms import RandomAffine
from torch.utils.data import DataLoader
from scipy import ndimage
from PIL import Image
from skimage import img_as_ubyte


def to_numpy_img(img, is_mask=False):
    if is_mask:
        img = img_as_ubyte(img.detach().squeeze().cpu().numpy())
    else:
        img = img_as_ubyte(simple_denorm(img.squeeze(dim=0)).detach().permute(1, 2, 0).cpu().numpy())
    return img


def simple_denorm(img, shift=1, scale=2, min=0, max=1):
    out = (img + shift) / scale
    out = out.clamp(min, max)
    return out


# ------------------------------------------------------------------
# ------------      DataLoader       ----------------
class DataProcessor(object):
    def __init__(self, opt):
        self.opt = opt
        self._data_setting(opt['flist'])
        self.index = 0

        self.img_loc = None
        self.mask_loc = None
        self.result_loc = None

    def _data_setting(self, flist):
        self.img_list = []
        self.parent_mask_list = []
        self.result_list = []

        with open(flist, mode='r') as f:
            lines = f.read().splitlines()

        for line in lines:
            image, mask, out = line.split(',')
            self.img_list.append(image)
            self.parent_mask_list.append(mask)
            self.result_list.append(out)

    def __len__(self):
        return len(self.img_list)

    def next(self):
        # training session
        self.img_loc = self.img_list[self.index]
        self.mask_loc = self.parent_mask_list[self.index]
        self.result_loc = self.result_list[self.index]
        self.index += 1

        dataset = BaseDataset(self.img_loc, self.mask_loc, self.opt['mask_type'],
                              self.opt['iterations']*self.opt['batch_size'],
                              self.opt['reuse_parent_mask'])

        data_loader = DataLoader(dataset,
                                 batch_size=self.opt['batch_size'],
                                 shuffle=False,
                                 #num_workers=self.opt['num_workers'],
                                 drop_last=True)
        return data_loader

    def get_test_data(self):
        # get image
        img = Image.open(self.img_loc).convert("RGB")
        mask = Image.open(self.mask_loc).convert("L")

        # to tensor, normalization
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).unsqueeze(dim=0)
        mask = F.to_tensor(mask).unsqueeze(dim=0)

        # masking
        img = img * (1 - mask) + mask

        return img, mask, self.result_loc


class BaseDataset(object):
    def __init__(self, img_loc, parent_mask_loc, mask_type, total_data_number, reuse_parent_mask):
        self.img_loc = img_loc
        self.parent_mask_loc = parent_mask_loc
        self.total_data_number = total_data_number
        self.reuse_parent_mask = reuse_parent_mask
        self.augmentation = RandomAffine(degrees=180, scale=[0.7, 1.2])
        self.mask_generator = self._mask_fn(mask_type)

    def _mask_fn(self, mask_type):
        functions = {
            'ff': self._get_ff_mask,
            'box': self._get_box_mask,
            'ca': self._get_ca_mask,
            'random': self._get_random_mask
        }
        return functions[mask_type]

    def __len__(self):
        return self.total_data_number

    def __getitem__(self, idx):
        # get image, parent_mask
        img = Image.open(self.img_loc).convert("RGB")
        parent_mask = Image.open(self.parent_mask_loc).convert("L")

        # normalize and convert image to tensor
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        # get child_mask
        if self.reuse_parent_mask and (np.random.rand() < 0.3):
            # use augmented parent mask as child_mask
            child_mask = self.augmentation(parent_mask)
            child_mask = F.to_tensor(child_mask)
        else:
            child_mask = self.mask_generator(img.size(1), img.size(2))
            child_mask = torch.from_numpy(child_mask).unsqueeze(dim=0).float()

        # convert parent mask to tensor
        parent_mask = F.to_tensor(parent_mask)

        # parent masking to img
        img = img * (1 - parent_mask) + parent_mask
        return img, parent_mask, child_mask

    @staticmethod
    def _get_ff_mask(h, w, num_v=None):
        # Source: Generative Inpainting https://github.com/JiahuiYu/generative_inpainting

        mask = np.zeros((h, w))
        if num_v is None:
            num_v = 15 + np.random.randint(9)  # 5

        for i in range(num_v):
            start_x = np.random.randint(w)
            start_y = np.random.randint(h)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(4.0)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(60)  # 40
                brush_w = 10 + np.random.randint(15)  # 10
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)

                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y

        return mask.astype(np.float32)

    @staticmethod
    def _get_box_mask(h, w):
        height, width = h, w

        mask = np.zeros((height, width))

        mask_width = random.randint(int(0.3 * width), int(0.7 * width))
        mask_height = random.randint(int(0.3 * height), int(0.7 * height))

        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)

        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    @staticmethod
    def _get_ca_mask(h, w, scale=None, r=None):

        if scale is None:
            scale = random.choice([1, 2, 4, 8])
        if r is None:
            r = random.randint(2, 6)  # repeat median filter r times

        height = h
        width = w
        mask = np.random.randint(2, size=(height // scale, width // scale))

        for _ in range(r):
            mask = ndimage.median_filter(mask, size=3, mode='constant')

        # Todo modify resize because scipy.misc has no attribute "imresize"
        mask = np.array(Image.fromarray(mask.astype('uint8')).resize(size=(h, w), resample=Image.NEAREST))
        # Todo Original is
        # mask = misc.imresize(mask, (h, w), interp='nearest')

        if scale > 1:
            struct = ndimage.generate_binary_structure(2, 1)
            mask = ndimage.morphology.binary_dilation(mask, struct)
        elif scale > 3:
            struct = np.array([[0., 0., 1., 0., 0.],
                               [0., 1., 1., 1., 0.],
                               [1., 1., 1., 1., 1.],
                               [0., 1., 1., 1., 0.],
                               [0., 0., 1., 0., 0.]])

        return mask.transpose((1, 0))

    def _get_random_mask(self, h, w):
        f = random.choice([self._get_box_mask, self._get_ca_mask, self._get_ff_mask])
        return f(h, w)



