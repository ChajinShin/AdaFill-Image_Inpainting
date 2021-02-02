import re
import sys
import torch
import time
import logging
from torchvision import models
from collections import OrderedDict


class ElapsedTimeProcess(object):
    def __init__(self, max_iter: int, start_iter: int = 0, output_type: str = 'summary_with_str'):
        self.max_iter = max_iter
        self.current_iter = start_iter
        if output_type not in ['seconds', 'summary', 'summary_with_str']:
            raise ValueError("Unknown type '{}'.".format(self.output_type))
        self.output_type = output_type
        self.t1 = 0
        self.t2 = 0

    def start(self):
        self.t1 = time.time()

    def end(self):
        self.t2 = time.time()

        eta = (self.t2 - self.t1) * (self.max_iter - self.current_iter)
        self.current_iter += 1

        if self.output_type == 'seconds':
            return eta
        elif self.output_type == 'summary':
            return self._summary(eta, with_str=False)
        elif self.output_type == 'summary_with_str':
            return self._summary(eta, with_str=True)
        else:
            raise ValueError("Unknown type '{}'.".format(self.output_type))

    def _summary(self, eta, with_str=True):
        elapsed_time_dict = self._calculate_summary(eta)
        if with_str:
            return self._to_string(elapsed_time_dict)
        else:
            return elapsed_time_dict

    @staticmethod
    def _calculate_summary(eta):
        elapsed_time_dict = OrderedDict()

        # days
        eta_days = int(eta // (24 * 3600))
        if eta_days != 0:
            elapsed_time_dict['eta_days'] = eta_days

        # hours
        eta_hours = int((eta // 3600) % 24)
        if eta_hours != 0:
            elapsed_time_dict['eta_hours'] = eta_hours

        # minutes
        eta_minutes = int((eta // 60) % 60)
        if eta_minutes != 0:
            elapsed_time_dict['eta_minutes'] = eta_minutes

        # seconds
        elapsed_time_dict['eta_seconds'] = int(eta % 60)

        return elapsed_time_dict

    @staticmethod
    def _to_string(elapsed_time_dict):
        output = ''
        for key, value in elapsed_time_dict.items():
            if key == 'eta_days':
                output += '{} days '.format(value)
            elif key == 'eta_hours':
                output += '{} h '.format(value)
            elif key == 'eta_minutes':
                output += '{} m '.format(value)
            elif key == 'eta_seconds':
                output += '{} s'.format(value)
            else:
                raise KeyError('Some key has mismatched name')
        return output


class ProcessBar(object):
    def __init__(self, max_iter, prefix='', suffix='', bar_length=50):
        self.max_iter = max_iter
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.iteration = 0

    def step(self, iteration=None, other_info: str=None):
        if iteration is None:
            self.iteration += 1

        percent = 100 * self.iteration / self.max_iter
        filled_length = int(round(self.bar_length * self.iteration) / self.max_iter)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
        msg = '\r{} [{}] {:.1f}% {}'.format(self.prefix, bar, percent, self.suffix)
        if other_info is not None:
            msg = msg + "  |   " + other_info
        sys.stdout.write(msg)
        if self.iteration == self.max_iter:
            sys.stdout.write('\n')
        sys.stdout.flush()


class Logger(object):
    def __init__(self, logging_file_dir):
        self.logger = logging.getLogger('log')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(logging_file_dir)
        self.logger.addHandler(handler)

    def __call__(self, msg):
        self.logger.info(msg)
        print(msg)


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for index in range(2):
            self.relu1_1.add_module(str(index), features[index])

        for index in range(2, 4):
            self.relu1_2.add_module(str(index), features[index])

        for index in range(4, 7):
            self.relu2_1.add_module(str(index), features[index])

        for index in range(7, 9):
            self.relu2_2.add_module(str(index), features[index])

        for index in range(9, 12):
            self.relu3_1.add_module(str(index), features[index])

        for index in range(12, 14):
            self.relu3_2.add_module(str(index), features[index])

        for index in range(14, 16):
            self.relu3_3.add_module(str(index), features[index])

        for index in range(16, 18):
            self.relu3_4.add_module(str(index), features[index])

        for index in range(18, 21):
            self.relu4_1.add_module(str(index), features[index])

        for index in range(21, 23):
            self.relu4_2.add_module(str(index), features[index])

        for index in range(23, 25):
            self.relu4_3.add_module(str(index), features[index])

        for index in range(25, 27):
            self.relu4_4.add_module(str(index), features[index])

        for index in range(27, 30):
            self.relu5_1.add_module(str(index), features[index])

        for index in range(30, 32):
            self.relu5_2.add_module(str(index), features[index])

        for index in range(32, 34):
            self.relu5_3.add_module(str(index), features[index])

        for index in range(34, 36):
            self.relu5_4.add_module(str(index), features[index])

        # Don't need to gradient update
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        relu1_1 = self.relu1_1(X)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def loss_str_split(loss_str: str):
    delete_space = re.sub(r'\s', '', loss_str)
    add_mark = re.sub(r'\+', r'^+', delete_space)
    add_mark = re.sub(r'-', r'^-', add_mark)
    return re.split(r'\^', add_mark)

