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
    parser.add_argument("--gpu", "-g", default=0, type=int, help="Gpu ID")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    args = parser.parse_args()

    data_root = "/home/zhaoxin/data/DG/domainbed"
    exp_name = 'PACS' + str(args.seed)

    subprocess.run(
        f'CUDA_VISIBLE_DEVICES={args.gpu} '
        f'python train_all.py '
        f'{exp_name} '
        f'--dataset {args.dataset} '
        f'--deterministic '
        f'--trial_seed {args.seed} '
        f'--checkpoint_freq {args.checkpoint_freq} '
        f'--data_dir {data_root}',
        shell=True,
        check=True
    )
