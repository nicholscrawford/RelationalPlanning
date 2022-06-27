import numpy as np
import argparse
import sys
import os

import torch

sys.path.append(os.getcwd())

bashCommand = "echo $PYTHONPATH"
os.system(bashCommand)

#from robot_learning.logger.tensorboardx_logger import TensorboardXLogger
from relational_precond.utils.tensorboardx_logger import TensorboardXLogger 
from utils.colors import bcolors
from utils.torch_utils import get_weight_norm_for_network
from vae.config.base_config import BaseVAEConfig

def create_log_dirs(config):
    args = config.args
    # Create logger directory if required
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(config.get_logger_dir()):
        os.makedirs(config.get_logger_dir())
    if not os.path.exists(config.get_model_checkpoint_dir()):
        os.makedirs(config.get_model_checkpoint_dir())

def add_common_args_to_parser(parser,
                              cuda=False,
                              result_dir=False,
                              checkpoint_path=False,
                              num_epochs=False,
                              batch_size=False,
                              lr=False,
                              save_freq_iters=False,
                              log_freq_iters=False,
                              print_freq_iters=False,
                              test_freq_iters=False):
    if cuda:
        parser.add_argument('--cuda', type=int, default=0, help="Use cuda")
    if result_dir:
        parser.add_argument('--result_dir', type=str, required=True,
                            help='Directory to save logsa and config in.')
    if checkpoint_path:
        parser.add_argument('--checkpoint_path', type=str, default='',
                            help='Checkpoint to test on.')

    if num_epochs:
        parser.add_argument('--num_epochs', type=int, default=100,
                            help='Number of epochs to run')
    if batch_size:
        parser.add_argument('--batch_size', type=int, default=8,
                            help='Batch size for each step')
    if lr:
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate for the forward model.')

    if save_freq_iters:
        parser.add_argument('--save_freq_iters', type=int, default=500,
                            help='Frequency at which to save models.')
    if log_freq_iters:
        parser.add_argument('--log_freq_iters', type=int, default=1,
                            help='Frequency to save logs.')
    if print_freq_iters:
        parser.add_argument('--print_freq_iters', type=int, default=1,
                            help='Frequency to save logs.')

    if test_freq_iters:
        parser.add_argument('--test_freq_iters', type=int, default=1000,
                            help='Frequency to test during training.')

class BaseVAETrainer(object):
    def __init__(self, config):
        self.config = config
        self.logger = TensorboardXLogger(self.config.get_logger_dir())
        self.model = None
        self.train_step_count = 0

    def log_model_to_tensorboard(self):
        '''Log weights and gradients of network to Tensorboard.'''
        if self.model is None:
            print(bcolors.c_red("Warning: Did get None model. Will not log."))
        model_l2_norm, model_grad_l2_norm = \
                get_weight_norm_for_network(self.model)
        self.logger.summary_writer.add_histogram(
                'histogram/mu_linear_layer_weights',
                self.model.mu_linear.weight,
                self.train_step_count)
        self.logger.summary_writer.add_histogram(
                'histogram/logvar_linear_layer_weights',
                self.model.logvar_linear.weight,
                self.train_step_count)
        self.logger.summary_writer.add_scalar(
                'weight/lstm_spatial_relation_model',
                model_l2_norm,
                self.train_step_count)
        self.logger.summary_writer.add_scalar(
                'grad/lstm_spatial_relation_model',
                model_grad_l2_norm,
                self.train_step_count)

    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return self.config.get_model_checkpoint_dir()

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    def save_checkpoint(self, epoch, data=None):
        if data is None:
            data = { 'graph_model': self.model.state_dict() }
        cp_filepath = self.model_checkpoint_filename(epoch)
        torch.save(data, cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        '''Load models from checkpoint.'''
        checkpoint_models = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint_models['graph_model'])

    def get_model_data_to_save(self):
        raise ValueError("Should be overriden.")