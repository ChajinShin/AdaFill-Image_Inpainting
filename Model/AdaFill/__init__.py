import os
import torch
import datetime
import torch.optim as optim
from solver import Solver
from skimage.io import imsave
from .src import network
from Tools.Utils import ProcessBar, ElapsedTimeProcess, loss_str_split
from Tools.Loss import L1
from .src.dataprocess import to_numpy_img, DataProcessor


class Policy(Solver):
    def __init__(self, opt, dev):
        super(Policy, self).__init__(opt, dev)

    def set(self):
        self._config_setting()
        self._dataloader_setting()
        self._loss_setting()

    def _config_setting(self):
        # Experiment several configuration
        # make main folder
        if not os.path.exists(self.opt['experiment_name']):
            os.makedirs(self.opt['experiment_name'], exist_ok=True)

        # make sub folder
        self.save_root = self.opt['experiment_name'] + '/results'
        os.makedirs(self.save_root, exist_ok=True)

        with open(self.opt['experiment_name'] + '/config.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
            for arg, val in self.opt.items():
                f.write('{}: {}\n'.format(arg, val))
            f.write('\n')

    def _dataloader_setting(self):
        self.data_processor = DataProcessor(self.opt)

    def _network_setting(self):
        # network
        self.inpaint_net = network.InpaintNet(self.opt).to(self.dev)

        # optimizer
        self.inpaint_optim = optim.Adam(params=self.inpaint_net.parameters(), lr=float(self.opt['inpaint_lr']), betas=self.opt['betas'])

    def _loss_setting(self):
        # inpaint loss
        self.inpaint_loss = dict()
        for l in loss_str_split(self.opt['inpaint_loss']):
            weight, loss_type = l.split('*')
            if loss_type == 'L1':
                self.inpaint_loss[loss_type] = L1(float(weight))
            else:
                raise ValueError("{} is not a valid loss type".format(loss_type))

    def _training_setting(self, total_iterations):
        # process bar setting
        self.Pb = ProcessBar(max_iter=total_iterations, prefix='Process: ')

        # elapsed time setting
        self.Eta = ElapsedTimeProcess(max_iter=total_iterations)

    def load_parameter(self):
        if os.path.exists(self.opt['inpaint_parameters']):
            state_dict = torch.load(self.opt['inpaint_parameters'], map_location='cpu')
            self.inpaint_net.load_state_dict(state_dict['state_dict'])
            print("inpaint network state dict load complete!")

    def initialization(self):
        self._network_setting()
        self.load_parameter()
        data_loader = self.data_processor.next()
        self._training_setting(len(data_loader))
        return data_loader

    def fit(self):
        for img_idx in range(len(self.data_processor)):
            data_loader = self.initialization()

            # training
            self.Eta.start()
            for iteration, (img, parent_mask, child_mask) in enumerate(data_loader):
                log = dict()

                img = img.to(self.dev)
                parent_mask = parent_mask.to(self.dev)
                child_mask = child_mask.to(self.dev)

                # mask preprocessing
                # img is already masked with parent mask in data_loader
                masked_img = img * (1 - child_mask) + child_mask
                input_data = torch.cat([masked_img, child_mask], dim=1)

                # forward
                self.inpaint_optim.zero_grad()
                tmp_predict = self.inpaint_net(input_data)
                predict = tmp_predict * (1-parent_mask) + parent_mask

                # inpaint network update
                total_loss = 0
                for loss_type, loss_fn in self.inpaint_loss.items():
                    if loss_type == 'L1':
                        l = loss_fn(predict, img)
                        total_loss += l
                        log['L1'] = l.item()

                total_loss.backward()
                self.inpaint_optim.step()

                # process printing
                elapsed_time = self.Eta.end()
                info = '{:04d}th_image,   iter:   {},  '.format(img_idx, iteration)
                for log_type, value in log.items():
                    info = info + '{}:  {:.4f}   '.format(log_type, value)
                info = info + 'ETA:  ' + elapsed_time

                self.Pb.step(other_info=info)
                self.Eta.start()

            self.test()
            print('')

    @torch.no_grad()
    def test(self):
        # eval mode
        self.inpaint_net.eval()

        # here img is already degraded.
        img, mask, result_loc = self.data_processor.get_test_data()
        img = img.to(self.dev)
        mask = mask.to(self.dev)

        input_data = torch.cat([img, mask], dim=1)
        out = self.inpaint_net(input_data)
        complete_result = out * mask + img * (1 - mask)

        # show image
        img = to_numpy_img(complete_result)

        # save image
        imsave(result_loc, img)





