import argparse
import torch
from utils import int_tuple, float_tuple, str_tuple, bool_flag
import os
CHECKPOINT_DIR = './checkpoints'
TEST_DIR = './layouts_out'
SUNCG_DIR = '/data/vision/billf/jwu-phys/dataset/billf-10/SceneRGBD/SUNCG'
os.environ["SUNCG_DIR"] = SUNCG_DIR
class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        parser = self.parser

        parser.add_argument('--dataset', default='suncg', choices=['suncg'])
        parser.add_argument('--suncg_train_dir', default="metadata/data_rot_train.json")
        parser.add_argument('--suncg_val_dir', default="metadata/data_rot_val.json")
        parser.add_argument('--suncg_data_dir', default=SUNCG_DIR)

        parser.add_argument('--loader_num_workers', default=8, type=int)
        parser.add_argument('--embedding_dim', default=64, type=int)
        parser.add_argument('--gconv_mode', default='feedforward')
        parser.add_argument('--gconv_dim', default=128, type=int)
        parser.add_argument('--gconv_hidden_dim', default=512, type=int)
        parser.add_argument('--gconv_num_layers', default=5, type=int)
        parser.add_argument('--mlp_normalization', default='batch', type=str)

        parser.add_argument('--vec_noise_dim', default=0, type=int)
        parser.add_argument('--layout_noise_dim', default=32, type=int)

        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--num_iterations', default=600000, type=int)
        parser.add_argument('--eval_mode_after', default=-1, type=int)
        parser.add_argument('--learning_rate', default=1e-4, type=float)

        parser.add_argument('--print_every', default=100, type=int)
        parser.add_argument('--checkpoint_every', default=1000, type=int)
        parser.add_argument('--snapshot_every', default=10000, type=int)

        parser.add_argument('--output_dir', default=CHECKPOINT_DIR)
        parser.add_argument('--checkpoint_name', default='latest_checkpoint')
        parser.add_argument('--timing', default=False, type=bool_flag)
        parser.add_argument('--multigpu', default=False, type=bool_flag)
        parser.add_argument('--restore_from_checkpoint', default=False, type=bool_flag)
        parser.add_argument('--checkpoint_start_from', default=None)

        # newly added arguments
        parser.add_argument('--test_dir', default=TEST_DIR)
        parser.add_argument('--gpu_id', default=0, type=int)
        parser.add_argument('--KL_loss_weight', default=0.1, type=float)
        parser.add_argument('--use_AE', default=False, type=bool_flag)
        parser.add_argument('--decoder_cat', default=True, type=bool_flag)
        parser.add_argument('--train_3d', default=True, type=bool_flag)
        parser.add_argument('--KL_linear_decay', default=False, type=bool_flag)
        parser.add_argument('--use_attr_30', default=True, type=bool_flag)
        parser.add_argument('--manual_seed', default=42, type=int)

        # Testing modes
        parser.add_argument('--batch_gen', action='store_true')
        parser.add_argument('--measure_acc_l1_std', action='store_true')
        parser.add_argument('--heat_map', action='store_true')
        parser.add_argument('--draw_2d', action='store_true')
        parser.add_argument('--draw_3d', action='store_true')
        parser.add_argument('--fine_tune', action='store_true')
        parser.add_argument('--gan_shade', action='store_true')
        parser.add_argument('--blender_path', default="/data/vision/billf/mooncam/code/yonglong/blender")



    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        if self.opt.manual_seed is not None:
            torch.manual_seed(self.opt.manual_seed)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt
