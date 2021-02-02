import os
import torch
import yaml
import datetime
import skimage.io as io
from .src import DataSetting, module
from .src.utils import model_setting
from solver import Solver
from Metric import MetricCenter
from Tools.Utils import denorm, ProcessBar, loss_str_split, Logger, ElapsedTimeProcess
from Tools.Loss import L1, NSGAN, PerceptualLoss, StyleLoss


def get_options():
    with open('./Model/Pretraining/options.yml', 'r', encoding='utf-8') as f:
        opt = yaml.load(f, Loader=yaml.Loader)
    return opt


class Policy(Solver):
    def __init__(self, opt, dev):
        super(Policy, self).__init__(opt, dev)
        self.step = 0
        self.log = None
        self.metric_calculator = MetricCenter(opt, dev)

        # model, optimizer, scheduler setting
        self.inpaint_net, self.disc_net, self.inpaint_optim, self.disc_optim, self.inpaint_scheduler, self.disc_scheduler = model_setting(opt, dev)

    def config_init(self):
        # elapsed timer
        if not self.opt['test_only']:
            self.eta_timer = ElapsedTimeProcess((len(self.train_loader) * self.opt['epochs']) // self.opt['log_step'], self.step)

        # Experiment several configuration
        # make main folder
        if not os.path.exists(self.opt['experiment_name']):
            os.makedirs(self.opt['experiment_name'])

        # make sub folder
        self.save_root = self.opt['experiment_name'] + '/save'
        self.ckpt_root = self.opt['experiment_name'] + '/ckpt'
        os.makedirs(self.save_root, exist_ok=True)
        os.makedirs(self.ckpt_root, exist_ok=True)

        with open(self.opt['experiment_name'] + '/config.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
            for arg, val in self.opt.items():
                f.write('{}: {}\n'.format(arg, val))
            f.write('\n')

        # logger
        self.logger = Logger(self.opt['experiment_name'] + '/log')

    def _load_dataloader(self):
        # data loader setting
        if not self.opt['test_only']:
            self.train_loader = DataSetting.get_dataloader(self.opt, training_set=True)
        self.test_loader = DataSetting.get_dataloader(self.opt, training_set=False)

    def save_model(self):
        os.makedirs(self.ckpt_root, exist_ok=True)

        inpaint_network_dir = self.ckpt_root + '/step_' + str(self.step) + '_inpaint.pth'
        disc_network_dir = self.ckpt_root + '/step_' + str(self.step) + '_disc.pth'

        torch.save({'step': self.step+1, 'state_dict': self.inpaint_net.state_dict(),
                    'optimizer_state_dict': self.inpaint_optim.state_dict(),
                    'scheduler_state_dict': self.inpaint_scheduler.state_dict()},
                   inpaint_network_dir)
        torch.save({'state_dict': self.disc_net.state_dict(),
                    'optimizer_state_dict': self.disc_optim.state_dict(),
                    'scheduler_state_dict': self.disc_scheduler.state_dict()},
                   disc_network_dir)
        self.logger("Model save completed")

    def load_model(self):
        if os.path.exists(str(self.opt['inpaint_parameters'])):
            gen_state_dict = torch.load(self.opt['inpaint_parameters'], map_location='cpu')
            self.inpaint_net.load_state_dict(gen_state_dict['state_dict'])
            self.inpaint_optim.load_state_dict(gen_state_dict['optimizer_state_dict'])
            self.inpaint_scheduler.load_state_dict(gen_state_dict['scheduler_state_dict'])
            self.step = gen_state_dict['step']
            print("Inpaint model load completed")

        if os.path.exists(str(self.opt['disc_parameters'])):
            disc_state_dict = torch.load(self.opt['disc_parameters'], map_location='cpu')
            self.disc_net.load_state_dict(disc_state_dict['state_dict'])
            self.disc_optim.load_state_dict(disc_state_dict['optimizer_state_dict'])
            self.disc_scheduler.load_state_dict(disc_state_dict['scheduler_state_dict'])
            print("Discriminator model load completed")

    def _change_mode(self, mode='train'):
        if mode == 'train':
            self.inpaint_net.train()
            self.disc_net.train()
        elif mode == 'eval':
            self.inpaint_net.eval()
            self.disc_net.eval()

    def _get_last_lr(self) -> dict:
        gen_lr = self.inpaint_scheduler.get_last_lr()[0]
        disc_lr = self.disc_scheduler.get_last_lr()[0]

        last_lr = {
            'gen_LR': gen_lr,
            'disc_LR': disc_lr
        }
        return last_lr

    def lr_step(self):
        self.inpaint_scheduler.step()
        self.disc_scheduler.step()

    def _loss_setting(self):
        print("Preparing loss function: ")

        # Inpaint loss
        self.inpaint_loss = dict()
        for r_l in loss_str_split(self.opt['inpaint_loss']):
            weight, loss_type = r_l.split('*')
            if loss_type == 'L1':
                self.inpaint_loss[loss_type] = L1(float(weight))
            elif loss_type == 'NSGAN':
                self.inpaint_loss[loss_type] = NSGAN(float(weight), is_disc=False)
            elif loss_type == 'PerceptualLoss':
                self.inpaint_loss[loss_type] = PerceptualLoss(self.dev, weight=float(weight))
            elif loss_type == 'StyleLoss':
                self.inpaint_loss[loss_type] = StyleLoss(self.dev, weight=float(weight))
            else:
                raise ValueError("{} is not a valid loss type".format(loss_type))

        # Discriminator loss
        self.disc_loss = dict()
        for d_l in loss_str_split(self.opt['disc_loss']):
            weight, loss_type = d_l.split('*')
            if loss_type == 'NSGAN':
                self.disc_loss[loss_type] = NSGAN(float(weight), is_disc=True)
            else:
                raise ValueError("{} is not a valid loss type".format(loss_type))

    def set(self):
        if not self.opt['test_only']:
            self._loss_setting()
        self.load_model()
        self._load_dataloader()
        self.config_init()
        print('Setting complete')

    def fit(self):
        initial_step = self.step
        self.eta_timer.start()
        for step in range(initial_step, len(self.train_loader) * self.opt['epochs']):
            self.step = step

            try:
                inputs = list([next(iters)])
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = list([next(iters)])

            log = self._fit_one_step(inputs)
            self._log_policy(log)

            if (self.step + 1) % self.opt['log_step'] == 0:
                self.logging()
            if (self.step + 1) % self.opt['eval_step'] == 0:
                metric = self.evaluate()
                self.print_metric(metric)
            if (self.step + 1) % self.opt['save_step'] == 0:
                self.save_model()

        metric = self.evaluate()
        self.print_metric(metric)
        self.save_model()

    def _log_policy(self, log):
        if self.log is None:
            self.log = log.copy()
        else:
            for key, value in log.items():
                self.log[key] += value

    def logging(self):
        if self.log is None:
            pass
        else:
            eta = self.eta_timer.end()
            msg = ''
            total_iterations = len(self.train_loader) * self.opt['epochs']
            msg += '{} / {} iter  |'.format(total_iterations, self.step+1)

            for key, value in self.log.items():
                value /= self.opt['log_step']
                msg += '{} : {:.4f}   |   '.format(key, value)
            self.logger(msg + 'ETA: ' + eta)

            self.log = None
            self.eta_timer.start()

    def print_metric(self, metric):
        msg = ''
        for m_name, m_value in metric.items():
            msg += m_name + ": {:.4f} ".format(m_value)
        self.logger(msg)

    def _fit_one_step(self, inputs: list) -> dict:
        log = {
            'd_loss': 0,
            'g_loss': 0,
            'real_score': 0,
            'fake_score': 0
        }

        self.inpaint_optim.zero_grad()
        self.disc_optim.zero_grad()

        img, mask = inputs[0][0].to(self.dev), inputs[0][1].to(self.dev)
        condition_img = img * (1-mask) + mask
        input_data = torch.cat([condition_img, mask], dim=1)

        fake_img = self.inpaint_net(input_data)
        complete_img = fake_img * mask + img * (1 - mask)

        # Discriminator
        for _ in range(self.opt['gan_k']):
            self.disc_optim.zero_grad()
            d_fake = self.disc_net(complete_img.detach())
            d_real = self.disc_net(img)

            d_loss = 0
            for loss_type, loss_fn in self.disc_loss.items():
                if loss_type == 'NSGAN':
                    d_loss += loss_fn(d_fake, d_real)
                else:
                    raise ValueError("{} is not a valid loss type".format(loss_type))
            d_loss.backward()
            self.disc_optim.step()
            log['d_loss'] += d_loss.item()
            log['fake_score'] += d_fake.mean().item()
            log['real_score'] += d_real.mean().item()
        log['d_loss'] /= self.opt['gan_k']
        log['fake_score'] /= self.opt['gan_k']
        log['real_score'] /= self.opt['gan_k']

        # Generator
        self.inpaint_optim.zero_grad()
        d_fake = self.disc_net(complete_img)

        # inpaint loss
        i_loss = 0
        for loss_type, loss_fn in self.inpaint_loss.items():
            if loss_type == 'NSGAN':
                l = loss_fn(d_fake)
                i_loss += l
                log['g_loss'] = l.item()
            elif loss_type == 'L1':
                l = loss_fn(fake_img, img)
                i_loss += l
                log['L1'] = l.item()
            elif loss_type == 'PerceptualLoss':
                l = loss_fn(fake_img, img)
                i_loss += l
                log['p_loss'] = l.item()
            elif loss_type == 'StyleLoss':
                l = loss_fn(fake_img, img)
                i_loss += l
                log['s_loss'] = l.item()
            else:
                raise ValueError("{} is not a valid loss type".format(loss_type))
        i_loss.backward()
        self.inpaint_optim.step()
        return log

    def forward_prop_for_eval(self, inputs):
        img, mask = inputs[0][0].to(self.dev), inputs[0][1].to(self.dev)
        condition_img = img * (1 - mask) + mask
        input_data = torch.cat([condition_img, mask], dim=1)
        fake_img = self.inpaint_net(input_data)
        complete_img = fake_img * mask + img * (1 - mask)

        real_img = denorm(img)
        complete_img = denorm(complete_img)
        mask_img = denorm(condition_img)
        return complete_img, real_img, mask_img

    @torch.no_grad()
    def evaluate(self):
        self._change_mode(mode='eval')
        progress_bar = ProcessBar(max_iter=len(self.test_loader), prefix='Evaluation Process : ', suffix=' Complete')

        iters = iter(self.test_loader)
        for step in range(len(self.test_loader)):
            inputs = list([next(iters)])

            pred, gt, mask = self.forward_prop_for_eval(inputs)
            self.metric_calculator.forward(pred, gt)
            self.result_save(pred, gt, mask, step)
            progress_bar.step()
        metric = self.metric_calculator.get_result()

        self._change_mode(mode='train')
        return metric

    def result_save(self, pred, gt, mask, step):
        iteration = gt.size(0)
        for idx in range(iteration):
            gt_img = gt[idx].mul(255).round().cpu().byte().permute(1, 2, 0).numpy()
            pred_img = pred[idx].mul(255).round().cpu().byte().permute(1, 2, 0).numpy()
            mask_img = mask[idx].mul(255).round().cpu().byte().permute(1, 2, 0).numpy()

            save_path = os.path.join(self.save_root, "{:04d}.png".format(step * self.opt['batch_size'] + idx))
            io.imsave(save_path, pred_img)

            save_path_gt = os.path.join(self.save_root, "{:04d}_GT.png".format(step * self.opt['batch_size'] + idx))
            save_path_mask = os.path.join(self.save_root, "{:04d}_mask.png".format(step * self.opt['batch_size'] + idx))
            io.imsave(save_path_gt, gt_img)
            io.imsave(save_path_mask, mask_img)








