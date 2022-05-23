import os
import subprocess
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        "-d",
                        default="PACS",
                        type=str,
                        help="Dataset")
    parser.add_argument("--method", "-m", default="ERM")
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, nargs="+", default=None)
    parser.add_argument('--wd',
                        type=float,
                        nargs="+",
                        default=[1e-6])
    parser.add_argument('--fda_mode', type=str, default="mix")
    args = parser.parse_args()

    data_root = "/home/zhaoxin/data/DG/domainbed"

    if args.dataset == 'PACS':
        exp_name = 'PACS'
        checkpoint_freq = 100
    elif args.dataset == 'VLCS':
        exp_name = 'VLCS'
        checkpoint_freq = 50
    elif args.dataset == 'OfficeHome':
        exp_name = 'OH'
        checkpoint_freq = 100
    elif args.dataset == 'TerraIncognita':
        exp_name = 'TR'
        checkpoint_freq = 100
    elif args.dataset == 'DomainNet':
        exp_name = 'DN'
        checkpoint_freq = 500

    assert args.lr is not None, "set lr"

    method = args.method
    if args.method == "FDA":
        method = method + '_' + args.fda_mode

    for lr in args.lr:
        for weight_decay in args.wd:
            exp_name_all = '_'.join([
                exp_name, method, f'lr={lr}_wd={weight_decay}',
                str(args.seed)
            ])
            subprocess.run(
                f'CUDA_VISIBLE_DEVICES={args.gpu} '
                f'python train_all.py '
                f'{exp_name_all} '
                f'--dataset {args.dataset} '
                f'--algorithm {args.method} '
                f'--deterministic '
                f'--trial_seed {args.seed} '
                f'--checkpoint_freq {checkpoint_freq} '
                f'--data_dir {data_root} '
                f'--fda_mode {args.fda_mode} '
                f'--lr {lr} '
                f'--weight_decay {weight_decay}',
                shell=True,
                check=True)
