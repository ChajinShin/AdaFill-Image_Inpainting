import yaml
from importlib import import_module


def get_options():
    with open('options.yml', 'r', encoding='utf8') as f:
        opt = yaml.load(f, Loader=yaml.Loader)

    if opt['task'] == 'Pretraining':
        del opt['AdaFill']
        opt = dict(opt, **opt['Pretraining'])
        del opt['Pretraining']
    elif opt['task'] == 'AdaFill':
        del opt['Pretraining']
        opt = dict(opt, **opt['AdaFill'])
        del opt['AdaFill']
    else:
        raise ValueError("task option is only available with 'Pretraining' or 'AdaFill'.")
    return opt


def get_solver(opt, dev):
    if opt['task'] == 'Pretraining':
        module = import_module('Model.Pretraining')
        solver = module.Policy(opt, dev)
        return solver
    elif opt['task'] == 'AdaFill':
        module = import_module('Model.AdaFill')
        solver = module.Policy(opt, dev)
        return solver
    else:
        raise ValueError("task option is only available with 'Pretraining' or 'AdaFill'.")


class Solver(object):
    def __init__(self, opt, dev):
        self.opt = opt
        self.dev = dev

    def set(self):
        raise NotImplementedError()

    def fit(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()
