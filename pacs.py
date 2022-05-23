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

    exp_name = exp_name + '_' + args.method + '_' + str(args.seed)

    subprocess.run(
        f'CUDA_VISIBLE_DEVICES={args.gpu} '
        f'python train_all.py '
        f'{exp_name} '
        f'--dataset {args.dataset} '
        f'--algorithm {args.method} '
        f'--deterministic '
        f'--trial_seed {args.seed} '
        f'--checkpoint_freq {checkpoint_freq} '
        f'--data_dir {data_root}',
        shell=True,
        check=True)
