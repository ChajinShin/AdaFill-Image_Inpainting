import os
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from PIL import Image


def get_dataloader(opt, training_set=True):
    dataset = DataBase(opt, training_set)
    data_loader = DataLoader(dataset,
                             batch_size=opt['batch_size'],
                             num_workers=opt['num_workers'],
                             drop_last=True if training_set else False,
                             shuffle=True if training_set else False)
    return data_loader


class DataBase(Dataset):
    def __init__(self, opt, train_set=True):
        self.opt = opt
        self.train_set = train_set
        self.mask_func = self.mask_prepare()
        self.img_list, self.mask_list = self._scan()
        self.preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
        )

    def _scan(self):
        # Image scan
        if self.train_set:
            self.path_to_img = os.path.join(self.opt['dataset_dir'], self.opt['train_img_list'])
        else:
            self.path_to_img = os.path.join(self.opt['dataset_dir'], self.opt['val_img_list'])

        with open(self.path_to_img, mode='r', encoding='utf8') as f:
            img_list = f.read().split('\n')

        input_list = []
        for img in img_list:
            input_list.append(os.path.join(self.opt['dataset_dir'], img))

        # user mask scan
        if self.opt['mask_type'] == 'user':
            assert self.opt['mask_folder_dir'] is not None, "Please set the mask folder directory"
            mask_list = []
            for mask_name in os.listdir(self.opt['mask_folder_dir']):
                mask_list.append(os.path.join(self.opt['mask_folder_dir'], mask_name))
        else:
            mask_list = None

        return input_list, mask_list

    def mask_prepare(self):
        mask_generator = Masks()
        mask_func = {"ff": mask_generator.get_ff_mask,
                     "ca": mask_generator.get_ca_mask,
                     "box": mask_generator.get_box_mask,
                     "fixed_box": mask_generator.get_fixed_mask,
                     "random": mask_generator.get_random_mask,
                     "user": mask_generator.get_user_mask}
        return mask_func

    def __getitem__(self, idx):
        image = self._get_image(idx)

        # preprocessing
        image = self.preprocessing(image)

        # get mask
        if self.opt['mask_type'] == 'user':
            mask = self.mask_func[self.opt['mask_type']](self.mask_list[idx])
        else:
            mask = self.mask_func[self.opt['mask_type']](image.size(1), image.size(2))
        mask = torch.from_numpy(mask).float().unsqueeze(dim=0)

        return image, mask

    def __len__(self):
        return len(self.img_list)

    def _get_image(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        return img






class Masks():
    @staticmethod
    def get_ff_mask(h, w, num_v=None):
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
    def get_box_mask(h, w):
        height, width = h, w

        mask = np.zeros((height, width))

        mask_width = random.randint(int(0.3 * width), int(0.7 * width))
        mask_height = random.randint(int(0.3 * height), int(0.7 * height))

        mask_x = random.randint(0, width - mask_width)
        mask_y = random.randint(0, height - mask_height)

        mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
        return mask

    @staticmethod
    def get_ca_mask(h, w, scale=None, r=None):

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

    @staticmethod
    def get_random_mask(h, w):
        f = random.choice([Masks.get_box_mask, Masks.get_ca_mask, Masks.get_ff_mask])
        return f(h, w)

    @staticmethod
    def get_fixed_mask(h, w):
        mask_h = h//2
        mask_w = w//2
        mask = np.zeros(shape=[h, w])
        mask[h//4: h//4 + mask_h, w//4: w//4 + mask_w] = 1.0
        return mask

    @staticmethod
    def get_user_mask(dir):
        mask = Image.open(dir).convert('L')
        mask = np.array(mask)
        return mask




