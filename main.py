import torch
import os
from solver import get_options, get_solver


def main():
    # fetch option and set device
    opt = get_options()
    torch.manual_seed(opt['seed'])
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if opt['use_cuda'] else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu_idx']

    solver = get_solver(opt, dev)
    solver.set()

    if opt['task'] == 'Pretraining':
        if opt['test_only']:
            print("Evaluating...")
            metrics = solver.evaluate()
            solver.print_metric(metrics)
        else:
            solver.fit()
    elif opt['task']:
        solver.fit()
    else:
        raise ValueError("task option is only available with 'Pretraining' or 'AdaFill'.")


if __name__ == "__main__":
    main()
