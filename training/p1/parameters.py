import argparse
import torch
import numpy as np
import random
from numpy.random import choice

def arg_parse():
    parser = argparse.ArgumentParser()


    parser.add_argument('--num_workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")

    # good idea than put hw2_data in each floder
    parser.add_argument('--train_data', default="../hw2_data/face/train/", type=str)
    parser.add_argument('--test_data', default="../hw2_data/face/test/", type=str)
    parser.add_argument('--ckpts_dir', type=str, default='ckpts')
    parser.add_argument('--save_train_result_dir', type=str, default='results')
    parser.add_argument('--test', type=str, default='500_G.pth',
                        help="path to the trained model")
    parser.add_argument('--ckpt_g', type=str, default='',
                        help="path to the trained G model")
    parser.add_argument('--ckpt_d', type=str, default='',
                        help="path to the trained D model")
    parser.add_argument('--save_test_result_dir', type=str, default='inference',
                        help="path to the 1000 output images")


    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--epochs', default=500, type=int,
                        help="num of training iterations")
    parser.add_argument('--g_iter', default=1, type=int,
                        help="num of training g iterations")
    parser.add_argument('--d_iter', default=1, type=int,
                        help="num of training d iterations")
    parser.add_argument('--beta1', default=0.5, type=int,
                        help="beta 1 of optimizer")
    parser.add_argument('--val_epoch', default=10, type=int,
                        help="num of validation iterations")
    parser.add_argument('--train_batch', default=128, type=int,
                        help="train batch size")
    parser.add_argument('--test_batch', default=128, type=int,
                        help="test batch size")
    parser.add_argument('--lr_d', default=0.0001, type=float,
                        help="initial learning rate")
    parser.add_argument('--lr_g', default=0.0002, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', default=0.00008, type=float,
                        help="initial weight decay")
    parser.add_argument('--log_interval', default=5, type=int,
                        help="print in log interval iterations")
    parser.add_argument('--random_seed', type=int, default=123)

    args = parser.parse_args()
    return args

# just let all seed become fixed :)

def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
