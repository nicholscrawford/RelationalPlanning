import argparse
import copy
import json
import os
import pickle
import pprint
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import functional as F
from tqdm import tqdm

#import open3d

class Trainer:
    def __init__():
        pass

    
def main(args):
    
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    

if __name__ == '__main__':
    #Review needed args
    parser = argparse.ArgumentParser(
        description='Train relational predictor model.'
        )

    parser.add_argument('--train_dir', required=True, action='append',
                        help='Path to training data.')


    main(args)
