import argparse

import os

import sys

sys.path.append(os.getcwd())

from relational_precond.model.GNN_pytorch_geometry import GNNTrainer, GNNModel, MLPModel, GNNModelOptionalEdge, MLPModelOptionalEdge
from relational_precond.trainer.multi_object_precond.trainer import Trainer
from relational_precond.dataloader.real_robot_dataloader import AllPairVoxelDataloaderPointCloud3stack

from vae.config.base_config import BaseVAEConfig


def main(args):
    args.result_dir = args.train_dir
    args.test_dir = args.train_dir

    t = Trainer()
    config = BaseVAEConfig(args, dtype=t.dtype)

    # TODO: Review and edit
    dataloader = AllPairVoxelDataloaderPointCloud3stack(
                        config,
                        use_multiple_train_dataset = False,
                        use_multiple_test_dataset = False, 
                        pick_place = False, 
                        pushing = True,
                        stacking = False, 
                        set_max = True, 
                        max_objects = 8,
                        voxel_datatype_to_use=0,
                        load_all_object_pair_voxels=True,
                        test_end_relations = False,
                        real_data = False, 
                        start_id = 0, 
                        max_size = 2, 
                        start_test_id = 0, 
                        test_max_size = 2,
                        updated_behavior_params = True,
                        pointconv_baselines = True,
                        save_data_path = "",
                        evaluate_end_relations = False,
                        using_multi_step_statistics = False,
                        total_multi_steps = 2
                        )

    t.train( 1000, 1, dataloader)
    

if __name__ == '__main__':
    #Limiting arguments. Will add config file if needed.
    parser = argparse.ArgumentParser(
        description='Train relational predictor model.'
        )

    parser.add_argument('--train_dir', required=True, action='append',
                        help='Path to training data.')

    args = parser.parse_args()
    main(args)
