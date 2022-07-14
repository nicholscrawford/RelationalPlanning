import numpy as np
#import open3d

import argparse
import pickle
import h5py
import sys
import os
import pprint
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import torch.optim as optim
import time
import json
import copy
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append(os.getcwd())

from relational_precond.utils.torch_utils import get_weight_norm_for_network, to_numpy
from relational_precond.utils.colors import bcolors
from relational_precond.utils.data_utils import str2bool
from relational_precond.utils.data_utils import recursively_save_dict_contents_to_group
from relational_precond.utils.data_utils import convert_list_of_array_to_dict_of_array_for_hdf5
from relational_precond.utils.image_utils import get_image_tensor_mask_for_bb

from relational_precond.dataloader.real_robot_dataloader import AllPairVoxelDataloaderPointCloud3stack

from relational_precond.model.contact_model import PointConv, SigmoidRelations, SigmoidRelations_1, PointConv_planar

from relational_precond.model.contact_model import SigmoidRelations, SpatialClassifier, Contrasive ,SpatialClassifierHorizon, SpatialClassifierLeft, SpatialClassifierRight, SpatialClassifierVertical, SpatialClassifierFront, SpatialClassifierBehind, SpatialClassifierStack, SpatialClassifierTop, SpatialClassifierBelow

from relational_precond.utils.data_utils import get_euclid_dist_matrix_for_data
from relational_precond.utils import math_util
from relational_precond.utils import torch_util

from vae.config.base_config import BaseVAEConfig
from vae.trainer.base_train import create_log_dirs, add_common_args_to_parser
from vae.trainer.base_train import BaseVAETrainer

import numpy as np
from collections import OrderedDict
from itertools import chain, filterfalse, permutations, product
from numbers import Number

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import SubsetRandomSampler


ALL_OBJ_PAIRS_GNN = 'all_object_pairs_gnn'
ALL_OBJ_PAIRS_GNN_NEW = 'all_object_pairs_gnn_new'
ALL_OBJ_PAIRS_GNN_RAW_INFO = 'all_object_pairs_gnn_raw_obj_info'


def save_emb_data_to_h5(result_dir, result_dict):
    emb_h5_path = os.path.join(result_dir, 'train_result_emb.h5')
    emb_h5_f = h5py.File(emb_h5_path, 'w')

    # Create a new dictionary so that each scene emb which can have different
    # number of objects as compared to other scenes will be stored separately.
    result_h5_dict = {'emb': {}}
    for k, v in result_dict['emb'].items():
        if k != 'train_img_emb' and k != 'test_img_emb':
            result_h5_dict['emb'][k] = v
        else:
            assert type(v) is list
            result_h5_dict['emb'][k] = dict()
            for scene_i, scene_emb in enumerate(v):
                result_h5_dict['emb'][k][f'{scene_i:05d}'] = np.copy(scene_emb)

    result_h5_dict = {'emb': result_h5_dict['emb'] }
    recursively_save_dict_contents_to_group(emb_h5_f, '/', result_h5_dict)
    emb_h5_f.flush()
    emb_h5_f.close()
    pkl_path = os.path.join(result_dir, 'train_result_info.pkl')
    with open(pkl_path, 'wb') as pkl_f:
        pkl_output_dict = {'output': result_dict['output']}
        pickle.dump(pkl_output_dict, pkl_f, protocol=2)

    print(bcolors.c_blue("Did save emb data: {}".format(emb_h5_path)))


def create_model_with_checkpoint_1(model_name,
                                 checkpoint_path, 
                                 args=None, 
                                 cuda=False):
    '''Create model with checkpoint path and args loaded from the saved model
    dir.
    args: None if the args are none will try to load args from the emb cp dir.
    '''
    assert model_name in ['resnet10', 'resnet18', 'simple_model', 
                          'small_simple_model', 'bb_only', 'pointcloud'], \
        "Invalid model type"

    if args is None:
        result_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        args_path = os.path.join(result_dir, 'config.pkl')
        with open(args_path, 'rb') as args_f:
            args = pickle.load(args_f)
            print(bcolors.c_green("Did load emb args: {}".format(args_path)))

    action_size = 7
    
    if args.test_spatial == True:
        model = PointConv(normal_channel=False)
        model_planar = PointConv_planar(normal_channel=False)
        print('enter model')
    elif model_name == 'pointcloud':
        model = PointConv(normal_channel=False)
        model_planar = PointConv_planar(normal_channel=False)
        print('enter model')
    elif model_name == 'resnet10':
        model = get_unscaled_resnet10(
            args.z_dim, action_size, args,
            use_spatial_softmax=args.use_spatial_softmax,
            pred_contacts_in_output=args.pred_contacts_in_output,
            )
    elif model_name == 'resnet18':
        model = get_unscaled_resnet18(
            args.z_dim, action_size, args,
            use_spatial_softmax=args.use_spatial_softmax,
            pred_contacts_in_output=args.pred_contacts_in_output,
            )
    elif model_name == 'simple_model':
        model = UnscaledVoxelModel(
            args.z_dim, action_size, args, n_classes=2*args.classif_num_classes,
            use_spatial_softmax=args.use_spatial_softmax,
            pred_contacts_in_output=args.pred_contacts_in_output,
            )
    elif model_name == 'small_simple_model':
        model = SmallEmbUnscaledVoxelModel(
            args.z_dim, action_size, args, n_classes=2*args.classif_num_classes,
            use_spatial_softmax=args.use_spatial_softmax,
            pred_contacts_in_output=args.pred_contacts_in_output,
            )
    else:
        raise ValueError("Invalid model: {}".format(model))

    
    if checkpoint_path is not None and len(checkpoint_path) > 0:
        checkpoint_models = torch.load(checkpoint_path,
                                       map_location=torch.device('cpu'))
        #print(checkpoint_models['model'])
        model.load_state_dict(checkpoint_models['model'])
        if checkpoint_models.get('model_planar') is not None:
            model_planar.load_state_dict(checkpoint_models['model_planar'])
        #model_planar.load_state_dict(checkpoint_models['model_planar'])
        #model_dict['model_planar'] = self.model_planar
        spatial_classifier_horizon = SpatialClassifierHorizon(args.z_dim, args)
        spatial_classifier_left = SpatialClassifierLeft(args.z_dim, args)
        spatial_classifier_right = SpatialClassifierRight(args.z_dim, args)
        spatial_classifier_vertical = SpatialClassifierVertical(args.z_dim, args)
        spatial_classifier_front = SpatialClassifierFront(args.z_dim, args)
        spatial_classifier_behind = SpatialClassifierBehind(args.z_dim, args)
        spatial_classifier_stack = SpatialClassifierStack(args.z_dim, args)
        spatial_classifier_top = SpatialClassifierTop(args.z_dim, args)
        spatial_classifier_below = SpatialClassifierBelow(args.z_dim, args)
        # sigmoid_relations
        sigmoid_relations = SigmoidRelations(args.z_dim, args)
        if checkpoint_models.get('sigmoid_relations') is not None:
            sigmoid_relations.load_state_dict(checkpoint_models['sigmoid_relations'])
        if checkpoint_models.get('spatial_modelp_horizon') is not None:
            spatial_classifier_horizon.load_state_dict(checkpoint_models['spatial_modelp_horizon'])
        if checkpoint_models.get('spatial_modelp_left') is not None:
            spatial_classifier_left.load_state_dict(checkpoint_models['spatial_modelp_left'])
        if checkpoint_models.get('spatial_modelp_right') is not None:
            spatial_classifier_right.load_state_dict(checkpoint_models['spatial_modelp_right'])
        if checkpoint_models.get('spatial_modelp_vertical') is not None:
            spatial_classifier_vertical.load_state_dict(checkpoint_models['spatial_modelp_vertical'])
        if checkpoint_models.get('spatial_modelp_front') is not None:
            spatial_classifier_front.load_state_dict(checkpoint_models['spatial_modelp_front'])
        if checkpoint_models.get('spatial_modelp_behind') is not None:
            spatial_classifier_behind.load_state_dict(checkpoint_models['spatial_modelp_behind'])
        if checkpoint_models.get('spatial_modelp_stack') is not None:
            spatial_classifier_stack.load_state_dict(checkpoint_models['spatial_modelp_stack'])
        if checkpoint_models.get('spatial_modelp_top') is not None:
            spatial_classifier_top.load_state_dict(checkpoint_models['spatial_modelp_top'])
        if checkpoint_models.get('spatial_modelp_below') is not None:
            spatial_classifier_below.load_state_dict(checkpoint_models['spatial_modelp_below'])


    return model, model_planar, sigmoid_relations, spatial_classifier_horizon ,spatial_classifier_left, spatial_classifier_right, spatial_classifier_vertical ,spatial_classifier_front, spatial_classifier_behind, spatial_classifier_stack, spatial_classifier_top, spatial_classifier_below



class MultiObjectVoxelPrecondTrainerE2E(BaseVAETrainer):
    def __init__(self, config):
        super(MultiObjectVoxelPrecondTrainerE2E, self).__init__(config)


        
        args = config.args
        self.timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

        

        # there are a couple of methods in AllPairVoxelDataloader that should be
        # implemented by any dataloader that needs to be used for multi-object
        # precond learning.
        self.stacking = True
        self.enable_shape = False
        self.pushing = args.pushing
        self.pushing_3 = False
        self.seperate = True
        

        self.use_point_cloud_embedding = True
        self.e2e = True
        
        
        self.relations_only = False

        self.use_centroid_estimation = False

        self.all_classifier = False
        self.all_gt = False

        self.all_gt_sigmoid = True # use sigmoid version of edge classifier

        self.new_relational_classifier = True 
        self.use_graph_dynamics = False
        self.use_dynamics_action_embed = True
        self.graph_dynamics_graph_relations = False
        # use newly trained relational classifer combination of gnn and mlp

        self.gt_edge = False

        self.data_sequence = False
        
        
        self.pick_place = args.pick_place
        # if self.pick_place:
        #     self.pushing = True
        if self.pushing:
            self.stacking = True

        self.save_all_planning_info = args.save_all_planning_info
        self.real_data = args.real_data
        self.manual_relations = args.manual_relations
        if self.real_data:
            self.manual_relations = True
        self.pointconv_baselines = args.pointconv_baselines
        self.bounding_box_baselines = args.bounding_box_baselines
        self.mlp = args.mlp
        self.set_max = args.set_max
        self.max_objects = args.max_objects
        self.execute_planning = args.execute_planning
        self.using_sub_goal = args.using_sub_goal

        self.execute_planning = args.execute_planning

        save_points = False
        sampling_points = args.sampling_points

        save_sampling_points = args.save_sampling_points
        recursive_saving = args.recursive_saving

        self.using_multi_step = args.using_multi_step

        self.test_end_relations = args.test_end_relations

        self.evaluate_end_relations = args.evaluate_end_relations

        self.consider_current_relations = args.consider_current_relations

        self.consider_end_relations = args.consider_end_relations

        self.seperate_range = args.seperate_range

        self.random_sampling_relations = args.random_sampling_relations

        self.using_delta_training = args.using_delta_training

        self.cem_planning = args.cem_planning

        self.graph_search = args.graph_search
        
        self.using_multi_step_latent = args.using_multi_step_latent

        self.rcpe = args.rcpe

        self.test_next_step = args.test_next_step

        self.using_latent_regularization = args.using_latent_regularization

        self.save_many_data = args.save_many_data

        self.test_next_step = args.test_next_step

        self.using_multi_step_statistics = args.using_multi_step_statistics

        self.total_sub_step = args.total_sub_step

        self.sampling_once = args.sampling_once
        
        self.contrasive = False
        
        # AllPairVoxelDataloaderPointCloud3stack
        

        
        
        self.classify_data = False
        self.four_data = False
        

        self.previous_threshold = -100
        self.node_pose_list = []
        self.action_list = []
        self.goal_relation_list = []
        self.gt_extents_range_list = []
        self.gt_pose_list = []

        self.node_pose_list_planning = []  # only save it if planning success
        self.action_list_planning = []
        self.goal_relation_list_planning = []
        self.gt_extents_range_list_planning = []
        self.gt_pose_list_planning = []

        if True:
            if self.stacking:
                if self.set_max:
                    self.dataloader = AllPairVoxelDataloaderPointCloud3stack(
                        config,
                        use_multiple_train_dataset = args.use_multiple_train_dataset,
                        use_multiple_test_dataset = args.use_multiple_test_dataset, 
                        pick_place = self.pick_place, 
                        pushing = self.pushing,
                        stacking = self.stacking, 
                        set_max = self.set_max, 
                        max_objects = self.max_objects,
                        voxel_datatype_to_use=args.voxel_datatype,
                        load_all_object_pair_voxels=('all_object_pairs' in args.train_type),
                        load_scene_voxels=(args.train_type == 'unfactored_scene' or 
                                        args.train_type == 'unfactored_scene_resnet18'),
                        test_end_relations = self.test_end_relations,
                        real_data = self.real_data, 
                        start_id = args.start_id, 
                        max_size = args.max_size, 
                        start_test_id = args.start_test_id, 
                        test_max_size = args.test_max_size,
                        updated_behavior_params = args.updated_behavior_params,
                        pointconv_baselines = args.pointconv_baselines,
                        save_data_path = args.save_data_path,
                        evaluate_end_relations = args.evaluate_end_relations,
                        using_multi_step_statistics = args.using_multi_step_statistics,
                        total_multi_steps = args.total_sub_step
                        )
                else:
                    self.dataloader = AllPairVoxelDataloaderPointCloud3stack(
                        config,
                        pick_place = self.pick_place, 
                        pushing = self.pushing,
                        stacking = self.stacking, 
                        set_max = self.set_max, 
                        voxel_datatype_to_use=args.voxel_datatype,
                        load_all_object_pair_voxels=('all_object_pairs' in args.train_type),
                        load_scene_voxels=(args.train_type == 'unfactored_scene' or 
                                        args.train_type == 'unfactored_scene_resnet18'),
                        test_end_relations = self.test_end_relations
                        )
                if self.test_end_relations:
                    self.fail_mp_num, self.total_num = self.dataloader.get_fail_motion_planner_num()
                    self.fail_exe_num = 0
                    self.success_exe_num = 0
                
        args = config.args
        
        # TODO: Use the arguments saved in the emb_checkpoint_dir to create 

        
        if args.train_type == 'all_object_pairs_gnn_new':
            from relational_precond.model.GNN_pytorch_geometry import GNNTrainer, GNNModel, MLPModel, GNNModelOptionalEdge, MLPModelOptionalEdge
            #if args.train_type == 'pointcloud':
            self.emb_model = PointConv(normal_channel=False)
            self.emb_model_planar = PointConv(normal_channel=False)
            if True:
                if self.e2e:
                    # self.emb_model_pretrained, self.emb_model_planar ,self.sigmoid_relations, self.spatial_classifier_horizon, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_vertical, self.spatial_classifier_front, self.spatial_classifier_behind, self.spatial_classifier_stack, self.spatial_classifier_top, self.spatial_classifier_below = create_model_with_checkpoint_1(
                    # # 'simple_model',
                    # # 'small_simple_model',
                    # 'pointcloud',
                    # args.emb_checkpoint_path,
                    # )
                    self.emb_model = PointConv(normal_channel=False)
                    if self.pointconv_baselines:
                        self.pointconv_sigmoid_relations = SigmoidRelations_1()
                    # self.sigmoid_relations = SigmoidRelations(args.z_dim, args) # begin about the sigmoid relations stuff. 
                
                
            self.enable_orientation = False
            if self.four_data:
                self.num_nodes = 4
            elif self.pick_place:
                self.num_nodes = 3
            elif self.pushing:
                if self.pushing_3:
                    self.num_nodes = 3
                else:
                    self.num_nodes = 4
            else:
                self.num_nodes = 3
            if self.set_max:
                node_inp_size, edge_inp_size = self.max_objects + 3, args.z_dim
            else:
                node_inp_size, edge_inp_size = self.num_nodes + 3, args.z_dim
            if(self.enable_orientation):
                node_inp_size = 10
            if self.enable_shape:
                node_inp_size += 512
            self.node_inp_size = node_inp_size
            
            self.edge_inp_size = edge_inp_size

            
            
            edge_classifier = True
            self.edge_classifier = edge_classifier
            node_emb_size, edge_emb_size = 128, 128
            self.node_emb_size = node_emb_size
            self.edge_emb_size = edge_emb_size
            if self.use_point_cloud_embedding:
                self.node_inp_size = self.node_emb_size
            
                self.edge_inp_size = self.edge_emb_size
            if self.graph_dynamics_graph_relations:
                self.dynamics_model = GNNModel(
                                node_inp_size, 
                                edge_inp_size,
                                self.node_inp_size, 
                                predict_edge_output = True,
                                edge_output_size = edge_inp_size,
                                graph_output_emb_size=16, 
                                node_emb_size=node_emb_size, 
                                edge_emb_size=edge_emb_size,
                                message_output_hidden_layer_size=128,  
                                message_output_size=128, 
                                node_output_hidden_layer_size=64,
                                predict_obj_masks=False,
                                predict_graph_output=False,
                                use_edge_embedding = True,
                            )
            if self.mlp:
                self.classif_model = MLPModelOptionalEdge(
                            self.node_inp_size, 
                            2*self.node_inp_size,
                            relation_output_size = args.z_dim, 
                            node_output_size = node_emb_size, 
                            predict_edge_output = True,
                            edge_output_size = edge_emb_size,
                            graph_output_emb_size=16, 
                            node_emb_size=node_emb_size, 
                            edge_emb_size=edge_emb_size,
                            message_output_hidden_layer_size=128,  
                            message_output_size=128, 
                            node_output_hidden_layer_size=64,
                            all_classifier = self.all_classifier,
                            predict_obj_masks=False,
                            predict_graph_output=False,
                            use_edge_embedding = False,
                            use_edge_input = True, 
                            max_objects = self.max_objects,
                            use_one_hot_embedding = True
                        )
                self.classif_model_decoder = MLPModelOptionalEdge(
                            self.node_emb_size, 
                            self.edge_emb_size,
                            relation_output_size = args.z_dim, 
                            node_output_size = self.node_inp_size, 
                            predict_edge_output = True,
                            edge_output_size = edge_inp_size,
                            graph_output_emb_size=16, 
                            node_emb_size=node_emb_size, 
                            edge_emb_size=edge_emb_size,
                            message_output_hidden_layer_size=128,  
                            message_output_size=128, 
                            node_output_hidden_layer_size=64,
                            all_classifier = self.all_classifier,
                            predict_obj_masks=False,
                            predict_graph_output=False,
                            use_edge_embedding = False,
                            use_edge_input = True, 
                            max_objects = self.max_objects,
                            use_one_hot_embedding = False
                        )
                    
            elif self.bounding_box_baselines or self.rcpe:
                node_emb_size, edge_emb_size = 128, 128
                self.node_emb_size = node_emb_size
                self.edge_emb_size = edge_emb_size
                #self.node_inp_size = 24 ## which is 8*3 8 points as the bounding box of an object
                #self.node_inp_size = 9 if use position + 6DOF orinetation representation. 
                node_output_dim = 9
                self.classif_model = GNNModelOptionalEdge(
                            self.node_inp_size, 
                            self.edge_inp_size,
                            relation_output_size = args.z_dim, 
                            node_output_size = node_emb_size, 
                            predict_edge_output = True,
                            edge_output_size = edge_emb_size,
                            graph_output_emb_size=16, 
                            node_emb_size=node_emb_size, 
                            edge_emb_size=edge_emb_size,
                            message_output_hidden_layer_size=128,  
                            message_output_size=128, 
                            node_output_hidden_layer_size=64,
                            all_classifier = self.all_classifier,
                            predict_obj_masks=False,
                            predict_graph_output=False,
                            use_edge_embedding = False,
                            use_edge_input = False, 
                            max_objects = self.max_objects
                        )
                self.classif_model_decoder = GNNModelOptionalEdge(
                            self.node_emb_size, 
                            self.edge_emb_size,
                            relation_output_size = args.z_dim, 
                            node_output_size = node_output_dim, 
                            predict_edge_output = True,
                            edge_output_size = edge_inp_size,
                            graph_output_emb_size=16, 
                            node_emb_size=node_emb_size, 
                            edge_emb_size=edge_emb_size,
                            message_output_hidden_layer_size=128,  
                            message_output_size=128, 
                            node_output_hidden_layer_size=64,
                            all_classifier = self.all_classifier,
                            predict_obj_masks=False,
                            predict_graph_output=False,
                            use_edge_embedding = False,
                            use_edge_input = True, 
                            max_objects = self.max_objects
                        )
            else:
                if self.use_point_cloud_embedding:
                    self.classif_model = GNNModelOptionalEdge(
                                self.node_inp_size, 
                                self.edge_inp_size,
                                relation_output_size = args.z_dim, 
                                node_output_size = node_emb_size, 
                                predict_edge_output = True,
                                edge_output_size = edge_emb_size,
                                graph_output_emb_size=16, 
                                node_emb_size=node_emb_size, 
                                edge_emb_size=edge_emb_size,
                                message_output_hidden_layer_size=128,  
                                message_output_size=128, 
                                node_output_hidden_layer_size=64,
                                all_classifier = self.all_classifier,
                                predict_obj_masks=False,
                                predict_graph_output=False,
                                use_edge_embedding = False,
                                use_edge_input = False, 
                                max_objects = self.max_objects
                            )
                    self.classif_model_decoder = GNNModelOptionalEdge(
                                self.node_emb_size, 
                                self.edge_emb_size,
                                relation_output_size = args.z_dim, 
                                node_output_size = self.node_inp_size, 
                                predict_edge_output = True,
                                edge_output_size = edge_inp_size,
                                graph_output_emb_size=16, 
                                node_emb_size=node_emb_size, 
                                edge_emb_size=edge_emb_size,
                                message_output_hidden_layer_size=128,  
                                message_output_size=128, 
                                node_output_hidden_layer_size=64,
                                all_classifier = self.all_classifier,
                                predict_obj_masks=False,
                                predict_graph_output=False,
                                use_edge_embedding = False,
                                use_edge_input = True, 
                                max_objects = self.max_objects
                            )
                    if self.use_graph_dynamics:
                        self.classif_model_dynamics = GNNModelOptionalEdge(
                                    self.node_emb_size, 
                                    self.edge_emb_size,
                                    relation_output_size = args.z_dim, 
                                    node_output_size = self.node_emb_size, 
                                    predict_edge_output = True,
                                    edge_output_size = self.edge_emb_size,
                                    graph_output_emb_size=16, 
                                    node_emb_size=node_emb_size, 
                                    edge_emb_size=edge_emb_size,
                                    message_output_hidden_layer_size=128,  
                                    message_output_size=128, 
                                    node_output_hidden_layer_size=64,
                                    all_classifier = self.all_classifier,
                                    predict_obj_masks=False,
                                    predict_graph_output=False,
                                    use_edge_embedding = False,
                                    use_edge_input = True, 
                                    max_objects = self.max_objects
                                )
                else:
                    self.classif_model = GNNModelOptionalEdge(
                                self.node_inp_size, 
                                self.edge_inp_size,
                                node_output_size = node_emb_size, 
                                predict_edge_output = True,
                                edge_output_size = edge_emb_size,
                                graph_output_emb_size=16, 
                                node_emb_size=node_emb_size, 
                                edge_emb_size=edge_emb_size,
                                message_output_hidden_layer_size=128,  
                                message_output_size=128, 
                                node_output_hidden_layer_size=64,
                                all_classifier = self.all_classifier,
                                predict_obj_masks=False,
                                predict_graph_output=False,
                                use_edge_embedding = False,
                                use_edge_input = False
                            )
                    self.classif_model_decoder = GNNModelOptionalEdge(
                                self.node_emb_size, 
                                self.edge_emb_size,
                                node_output_size = self.node_inp_size, 
                                predict_edge_output = True,
                                edge_output_size = edge_inp_size,
                                graph_output_emb_size=16, 
                                node_emb_size=node_emb_size, 
                                edge_emb_size=edge_emb_size,
                                message_output_hidden_layer_size=128,  
                                message_output_size=128, 
                                node_output_hidden_layer_size=64,
                                all_classifier = self.all_classifier,
                                predict_obj_masks=False,
                                predict_graph_output=False,
                                use_edge_embedding = False,
                                use_edge_input = True
                            )
                    if self.use_graph_dynamics:
                        self.classif_model_dynamics = GNNModelOptionalEdge(
                                    self.node_emb_size, 
                                    self.edge_emb_size,
                                    node_output_size = self.node_emb_size, 
                                    predict_edge_output = True,
                                    edge_output_size = self.edge_emb_size,
                                    graph_output_emb_size=16, 
                                    node_emb_size=node_emb_size, 
                                    edge_emb_size=edge_emb_size,
                                    message_output_hidden_layer_size=128,  
                                    message_output_size=128, 
                                    node_output_hidden_layer_size=64,
                                    all_classifier = self.all_classifier,
                                    predict_obj_masks=False,
                                    predict_graph_output=False,
                                    use_edge_embedding = False,
                                    use_edge_input = True
                                )
                
                
            
            
             
            self.stable_obj_pred_loss = nn.CrossEntropyLoss()
        

        self.opt_emb = optim.Adam(self.emb_model.parameters(), lr=args.emb_lr)
        self.opt_classif = optim.Adam(self.classif_model.parameters(), lr=1e-4) 
        self.opt_classif_decoder = optim.Adam(self.classif_model_decoder.parameters(), lr=1e-4) 
        if self.pointconv_baselines:
            self.opt_pointconv_sigmoid = optim.Adam(self.pointconv_sigmoid_relations.parameters(), lr=1e-4) 
        if self.use_graph_dynamics:
            self.opt_classif_dynamics = optim.Adam(self.classif_model_dynamics.parameters(), lr=1e-4) 
        if self.graph_dynamics_graph_relations:
            self.opt_dynamics = optim.Adam(self.dynamics_model.parameters(), lr=1e-4) 
        
        self.precond_loss = nn.BCELoss()

        self.dynamics_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def get_model_list(self):
        if self.graph_dynamics_graph_relations:
            return [self.dynamics_model ,self.emb_model, self.emb_model_planar, self.sigmoid_relations ,self.classif_model,self.classif_model_decoder,self.classif_model_dynamics,self.spatial_classifier_horizon, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_vertical ,self.spatial_classifier_front, self.spatial_classifier_behind, self.spatial_classifier_stack, self.spatial_classifier_top, self.spatial_classifier_below]
        elif self.use_graph_dynamics:
            return [self.emb_model, self.emb_model_planar, self.sigmoid_relations ,self.classif_model,self.classif_model_decoder,self.classif_model_dynamics,self.spatial_classifier_horizon, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_vertical ,self.spatial_classifier_front, self.spatial_classifier_behind, self.spatial_classifier_stack, self.spatial_classifier_top, self.spatial_classifier_below]
        elif self.contrasive:
            return [self.emb_model, self.emb_model_planar,self.classif_model,self.classif_model_decoder,self.spatial_classifier_contrasive, self.spatial_classifier_horizon, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_vertical ,self.spatial_classifier_front, self.spatial_classifier_behind, self.spatial_classifier_stack, self.spatial_classifier_top, self.spatial_classifier_below]
        elif self.new_relational_classifier:
            if self.pointconv_baselines:
                return [self.pointconv_sigmoid_relations, self.emb_model, self.emb_model_planar, self.sigmoid_relations ,self.classif_model,self.classif_model_decoder,self.spatial_classifier_horizon, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_vertical ,self.spatial_classifier_front, self.spatial_classifier_behind, self.spatial_classifier_stack, self.spatial_classifier_top, self.spatial_classifier_below]
            else:
                return [self.emb_model, self.classif_model,self.classif_model_decoder]
        else:
            return [self.emb_model, self.emb_model_planar, self.classif_model,self.classif_model_decoder,self.spatial_classifier_horizon, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_vertical ,self.spatial_classifier_front, self.spatial_classifier_behind, self.spatial_classifier_stack, self.spatial_classifier_top, self.spatial_classifier_below]
        #return [self.emb_model, self.classif_model, self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_front, self.spatial_classifier_behind]
        #return [self.emb_model, self.classif_model] # self.spatial_classifier_left, self.spatial_classifier_right, self.spatial_classifier_front, self.spatial_classifier_behind

    def get_state_dict(self):
        if self.contrasive:
                return {
                'emb_model': self.emb_model.state_dict(),
                'classif_model': self.classif_model.state_dict(),
                'classif_model_decoder': self.classif_model_decoder.state_dict()
            }
        else:
            if self.pointconv_baselines:
                return {
                    'pointconv_sigmoid_relations':self.pointconv_sigmoid_relations.state_dict(),
                    'emb_model': self.emb_model.state_dict(),
                    'classif_model': self.classif_model.state_dict(),
                    'classif_model_decoder': self.classif_model_decoder.state_dict()
                }
            else:
                return {
                    'emb_model': self.emb_model.state_dict(),
                    'classif_model': self.classif_model.state_dict(),
                    'classif_model_decoder': self.classif_model_decoder.state_dict()
                    
                }

    def set_model_device(self, device=torch.device("cpu")):
        model_list = self.get_model_list()
        for m in model_list:
            m.to(device)
    
    def set_model_to_train(self):
        model_list = self.get_model_list()
        for m in model_list:
            m.train()

    def set_model_to_eval(self):
        model_list = self.get_model_list()
        for m in model_list:
            m.eval()

    def save_checkpoint(self, epoch):
        cp_filepath = self.model_checkpoint_filename(epoch)
        torch.save(self.get_state_dict(), cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        cp_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # print(cp_models.keys())
        # print('cp_models', cp_models['spatial_classifier_left'])
        self.emb_model.load_state_dict(cp_models['emb_model'])
        self.classif_model.load_state_dict(cp_models['classif_model'])
        self.classif_model_decoder.load_state_dict(cp_models['classif_model_decoder'])
        if self.pointconv_baselines:
            self.pointconv_sigmoid_relations.load_state_dict(cp_models['pointconv_sigmoid_relations'])

    def process_raw_batch_data_point_cloud(self, batch_data):
        '''Process raw batch data and collect relevant objects in a dict.'''
        proc_batch_dict = {
            # Save scene index starting from 0
            'batch_scene_index_list': [],
            # Save input image
            'batch_voxel_list': [],
            'batch_voxel_list_single': [],
            'batch_voxel_anchor_list': [],
            'batch_voxel_other_list': [],
            # Save info for object pairs which are far apart?
            'batch_obj_pair_far_apart_list': [],
            # Save info for object positions
            'batch_obj_center_list': [],
            # Save precond output
            'batch_precond_label_list': [],
            # Save stable object ids
            'batch_precond_stable_obj_id_list': [],
            # Get object id being removed 
            'batch_box_stacking_remove_obj_id_list': [],
            # Get contact edges for box stacking
            'batch_contact_edge_list': [],
            # Save scene_path
            'batch_scene_path_list': [],
            # Save all object pair path in a scene
            'batch_scene_all_object_pair_path_list': [],
            # pos list
            'batch_all_obj_pair_pos': [],
            'batch_pc_center': [],
            'batch_this_one_hot_encoding': [],
            'batch_bounding_box': [],
            'batch_rotated_bounding_box': [],
            'batch_all_obj_pair_relation': [],
            'batch_all_obj_pair_orient': [],
            'batch_action': [],
            'batch_index_i': [],
            'batch_index_j': [],
            'batch_num_objects': [],
            'batch_select_obj_num_range': [],
            'batch_gt_pose_list': [],
            'batch_gt_orientation_list': [],
            'batch_gt_extents_list': [],
            'batch_gt_extents_range_list': [],
            'batch_goal_relations': [],
            'batch_predicted_relations': [],
        }

        args = self.config.args
        x_dict = proc_batch_dict

        for b, data in enumerate(batch_data):
            if 'all_object_pairs' in args.train_type:
                for voxel_idx, voxels in enumerate(data['all_object_pair_voxels']):
                    proc_voxels = voxels #self.process_raw_voxels(voxels)
                    x_dict['batch_voxel_list'].append(proc_voxels)
                    x_dict['batch_scene_index_list'].append(b)
                    x_dict['batch_obj_pair_far_apart_list'].append(
                        data['all_object_pair_far_apart_status'][voxel_idx]
                    )
                
                #print('all_object_pair_voxels_single shape', len(data['all_object_pair_voxels_single'])) # 
                for voxel_idx, voxels in enumerate(data['all_object_pair_voxels_single']):
                    proc_voxels = voxels #self.process_raw_voxels(voxels)
                    x_dict['batch_voxel_list_single'].append(proc_voxels)
                
                
                for voxel_idx, voxels in enumerate(data['all_object_pair_anchor_voxels']):
                    proc_voxels = voxels #self.process_raw_voxels(voxels)
                    x_dict['batch_voxel_anchor_list'].append(proc_voxels)
                for voxel_idx, voxels in enumerate(data['all_object_pair_other_voxels']):
                    proc_voxels = voxels #self.process_raw_voxels(voxels)
                    x_dict['batch_voxel_other_list'].append(proc_voxels)


                x_dict['batch_scene_all_object_pair_path_list'].append(
                    data['all_object_pair_path'])
                x_dict['batch_goal_relations'].append(
                    data['goal_relations'])
                if 'predicted_relations' in data:
                    x_dict['batch_predicted_relations'].append(
                        data['predicted_relations'])
                x_dict['batch_gt_pose_list'].append(
                    data['gt_pose_list'])
                x_dict['batch_gt_orientation_list'].append(
                    data['gt_orientation_list'])
                x_dict['batch_gt_extents_list'].append(
                    data['gt_extents_list'])
                x_dict['batch_gt_extents_range_list'].append(
                    data['gt_extents_range_list'])
                x_dict['batch_action'].append(
                    data['action'])
                x_dict['batch_index_i'].append(
                    data['index_i'])
                x_dict['batch_index_j'].append(
                    data['index_j'])
                x_dict['batch_num_objects'].append(
                    data['num_objects'])
                    
                x_dict['batch_select_obj_num_range'].append(
                    data['select_obj_num_range'])
                x_dict['batch_all_obj_pair_pos'].append(
                    data['all_obj_pair_pos'])
                x_dict['batch_pc_center'].append(
                    data['pc_center'])
                x_dict['batch_this_one_hot_encoding'].append(
                    data['this_one_hot_encoding'])
                x_dict['batch_bounding_box'].append(
                    data['bounding_box'])
                x_dict['batch_rotated_bounding_box'].append(
                    data['rotated_bounding_box'])
                x_dict['batch_all_obj_pair_relation'].append(
                    data['relation'])
                x_dict['batch_all_obj_pair_orient'].append(
                    data['all_obj_pair_orient'])
                #print(data)
                if data.get('obj_center_list') is not None:
                    x_dict['batch_obj_center_list'].append(torch.Tensor(data['obj_center_list']))

            elif args.train_type == 'unfactored_scene' or \
                 args.train_type == 'unfactored_scene_resnet18':
                voxels = data['scene_voxels']
                proc_voxels = voxels #self.process_raw_voxels(voxels)
                x_dict['batch_voxel_list'].append(proc_voxels)
            else:
                raise ValueError(f"Invalid train type {args.train_type}")
        
            if args.train_type == 'all_object_pairs_gnn' or \
               args.train_type == ALL_OBJ_PAIRS_GNN_NEW or \
               args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO or \
               args.train_type == 'all_object_pairs_g_f_ij_box_stacking_node_label':
            
                if data.get('contact_edges') is not None and args.use_contact_edges_only:
                    x_dict['batch_contact_edge_list'].append(
                        data['contact_edges'])
                else:
                    x_dict['batch_contact_edge_list'] = None
                if data.get('box_stacking_remove_obj_id') is not None:
                    x_dict['batch_box_stacking_remove_obj_id_list'].append(
                        data['box_stacking_remove_obj_id']) 

            x_dict['batch_precond_label_list'].append(data['precond_label'])
            x_dict['batch_scene_path_list'].append(data['scene_path'])

        return x_dict

    def collate_batch_data_to_tensors_point_cloud(self, proc_batch_dict):
        '''Collate processed batch into tensors.'''
        # Now collate the batch data together
        x_tensor_dict = {}
        x_dict = proc_batch_dict
        device = self.config.get_device()
        args = self.config.args

        #print('length', [len(x_dict['batch_voxel_list']), len(x_dict['batch_voxel_list_single'])])
        if len(x_dict['batch_voxel_list']) > 0:
            x_tensor_dict['batch_voxel'] = torch.stack(
                x_dict['batch_voxel_list']).to(device)
            x_tensor_dict['batch_voxel_single'] = torch.stack(
                x_dict['batch_voxel_list_single']).to(device)
            x_tensor_dict['batch_anchor_voxel'] = torch.stack(
                x_dict['batch_voxel_anchor_list']).to(device)
            x_tensor_dict['batch_other_voxel'] = torch.stack(
                x_dict['batch_voxel_other_list']).to(device)
            x_tensor_dict['batch_obj_pair_far_apart_list'] = torch.LongTensor(
                x_dict['batch_obj_pair_far_apart_list']).to(device)

        x_tensor_dict['batch_precond_label_list'] = torch.FloatTensor(
            x_dict['batch_precond_label_list']).to(device)

        x_tensor_dict['batch_all_obj_pair_pos'] = torch.FloatTensor(
            x_dict['batch_all_obj_pair_pos']).to(device)
        x_tensor_dict['batch_pc_center'] = torch.FloatTensor(
            x_dict['batch_pc_center']).to(device)
        x_tensor_dict['batch_this_one_hot_encoding'] = torch.FloatTensor(
            x_dict['batch_this_one_hot_encoding']).to(device)
        x_tensor_dict['batch_bounding_box'] = torch.FloatTensor(
            x_dict['batch_bounding_box']).to(device)
        x_tensor_dict['batch_rotated_bounding_box'] = torch.FloatTensor(
            x_dict['batch_rotated_bounding_box']).to(device)
        x_tensor_dict['batch_all_obj_pair_relation'] = torch.FloatTensor(
            x_dict['batch_all_obj_pair_relation']).to(device)
        x_tensor_dict['batch_all_obj_pair_orient'] = torch.FloatTensor(
            x_dict['batch_all_obj_pair_orient']).to(device)
        x_tensor_dict['batch_gt_pose_list'] = torch.FloatTensor(
            x_dict['batch_gt_pose_list']).to(device)
        x_tensor_dict['batch_gt_orientation_list'] = torch.FloatTensor(
            x_dict['batch_gt_orientation_list']).to(device)
        x_tensor_dict['batch_goal_relations'] = torch.FloatTensor(
            x_dict['batch_goal_relations']).to(device)
        if len(x_dict['batch_predicted_relations']) > 0:
            x_tensor_dict['batch_predicted_relations'] = torch.FloatTensor(
                x_dict['batch_predicted_relations']).to(device)
        #print(x_dict['batch_gt_extents_range_list'])
        if len(x_dict['batch_gt_extents_range_list'][0]) != 0:
            if x_dict['batch_gt_extents_range_list'][0][0] != None:
                x_tensor_dict['batch_gt_extents_range_list'] = torch.FloatTensor(
                    x_dict['batch_gt_extents_range_list']).to(device)
        x_tensor_dict['batch_gt_extents_list'] = torch.FloatTensor(
            x_dict['batch_gt_extents_list']).to(device)
        x_tensor_dict['batch_action'] = torch.FloatTensor(
            x_dict['batch_action']).to(device)
        x_tensor_dict['batch_index_i'] = torch.FloatTensor(
            x_dict['batch_index_i']).to(device)
        x_tensor_dict['batch_index_j'] = torch.FloatTensor(
            x_dict['batch_index_j']).to(device)
        x_tensor_dict['batch_num_objects'] = torch.FloatTensor(
            x_dict['batch_num_objects']).to(device)
        x_tensor_dict['batch_select_obj_num_range'] = torch.Tensor(
            x_dict['batch_select_obj_num_range']).to(device)
        x_tensor_dict['batch_scene_index_list'] = torch.LongTensor(
            x_dict['batch_scene_index_list']).to(device)
        
        if args.train_type == 'all_object_pairs_gnn' or \
           args.train_type == ALL_OBJ_PAIRS_GNN_NEW or \
           args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO or \
           args.train_type == 'all_object_pairs_g_f_ij_box_stacking_node_label':
            # This will be a list of tensors since each scene can have a variable
            # number of objects.
            x_tensor_dict['batch_precond_stable_obj_id_list'] = []
            for t in x_dict['batch_precond_stable_obj_id_list']:
                t = t.to(device)
                x_tensor_dict['batch_precond_stable_obj_id_list'].append(t)
        
            if x_dict.get('batch_obj_center_list') is not None:
                x_tensor_dict['batch_obj_center_list'] = x_dict['batch_obj_center_list']
            if x_dict.get('batch_contact_edge_list') is not None:
                x_tensor_dict['batch_contact_edge_list'] = x_dict['batch_contact_edge_list']
            if x_dict.get('batch_box_stacking_remove_obj_id_list') is not None:
                x_tensor_dict['batch_box_stacking_remove_obj_id_list'] = \
                    x_dict['batch_box_stacking_remove_obj_id_list']

        return x_tensor_dict

    def generate_edge_embed(self, node_pose):
        node_pose_numpy = node_pose.cpu().numpy()[:,-3:]
        #right_prediction, left_prediction, front_prediction, behind_prediction
        total_list = []

        block_size = 0.08*2 # corrspond to simple dataset 

        for i in range(node_pose_numpy.shape[0]):
            for j in range(node_pose_numpy.shape[0]):
                pred_list = []
                if(j != i):
                    max_noise = 0.01
                    rand_right = np.random.uniform(0, max_noise)
                    rand_left = np.random.uniform(0, max_noise)
                    rand_front = np.random.uniform(0, max_noise)
                    rand_behind = np.random.uniform(0, max_noise)
                    if(node_pose_numpy[i][0] > node_pose_numpy[j][0]):
                        pred_list.extend([0 + rand_right, 1 - rand_right, 1 - rand_left, 0 + rand_left])
                    else:
                        pred_list.extend([1 - rand_right, 0 + rand_right, 0 + rand_left, 1 - rand_left])
                    if(node_pose_numpy[i][1] > node_pose_numpy[j][1]):
                        pred_list.extend([0 + rand_front, 1 - rand_front, 1 - rand_behind, 0 + rand_behind])
                    else:
                        pred_list.extend([1 - rand_front, 0 + rand_front, 0 + rand_behind, 1 - rand_behind])
                    device = self.config.get_device()
                    total_list.append(torch.Tensor(pred_list).to(device))
        return total_list

    def CE_loss(self, input_tensor, goal):
        goal_tensor = torch.stack([goal, 1 - goal], axis = 0).T
        # print(goal_tensor)
        # print(goal_tensor.shape) # (8(bacth_size), 2)
        # loss = -torch.mean(goal_tensor * torch.log(input_tensor) + (1 - goal_tensor) * torch.log(1 - input_tensor))
        # return torch.mean(loss)
        loss = -torch.mean(goal_tensor * torch.log(input_tensor + 1e-6) + (1 - goal_tensor) * torch.log(1 - input_tensor + 1e-6))
        return torch.mean(loss)
   
    def create_graph(self, num_nodes, node_inp_size, node_pose, edge_size, edge_feature, action):
        nodes = list(range(num_nodes))
        # Create a completely connected graph
        edges = list(permutations(nodes, 2))
        edge_index = torch.LongTensor(np.array(edges).T)
        x = node_pose#torch.zeros((num_nodes, node_inp_size))#torch.eye(node_inp_size).float()
        edge_attr = edge_feature #torch.rand(len(edges), edge_size)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, action = action)
        # Recreate x as target
        data.y = x
        return data
 
    def run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e(self,
                           x_tensor_dict,
                           x_tensor_dict_next,
                           batch_size,
                           train=False,
                           save_preds=False,
                           save_emb=False, 
                           threshold = 0):

        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        voxel_data = x_tensor_dict['batch_voxel']
        voxel_data_single = x_tensor_dict['batch_voxel_single']

        voxel_data_anchor = x_tensor_dict['batch_anchor_voxel']

        voxel_data_other = x_tensor_dict['batch_other_voxel']

        


        voxel_data_next = x_tensor_dict_next['batch_voxel']
        voxel_data_next_single = x_tensor_dict_next['batch_voxel_single']
        voxel_data_anchor_next = x_tensor_dict_next['batch_anchor_voxel']
        voxel_data_other_next = x_tensor_dict_next['batch_other_voxel']

        voxel_data_edge_next = torch.cat([
                voxel_data_anchor_next,
                voxel_data_other_next], dim=1)

        select_obj_num_range = x_tensor_dict['batch_select_obj_num_range']
        select_obj_num_range_next = x_tensor_dict_next['batch_select_obj_num_range']

        gt_pose_list = x_tensor_dict['batch_gt_pose_list']
        if not train:
            if self.pushing and 'batch_gt_extents_range_list' in x_tensor_dict:
                gt_extents_range_list = x_tensor_dict['batch_gt_extents_range_list']
            else:
                gt_extents_range_list = torch.zeros((5,3)).to(device)

            self.gt_pose = x_tensor_dict['batch_gt_pose_list'].cpu().detach().numpy()
            if self.pushing and 'batch_gt_extents_range_list' in x_tensor_dict:
                self.gt_extents_range = x_tensor_dict['batch_gt_extents_range_list'].cpu().detach().numpy()
            else:
                self.gt_extents_range = np.zeros((5,3))
        
        action_fake = x_tensor_dict_next['batch_all_obj_pair_pos'] - x_tensor_dict['batch_all_obj_pair_pos']

        action = x_tensor_dict_next['batch_action']

        stacking = self.stacking

        self.num_nodes = x_tensor_dict['batch_num_objects'].cpu().numpy().astype(int)[0]
        
        # Now join the img_emb into a list for each scene.
        if  'all_object_pairs' in args.train_type:
            # Get the embeddings
     
            img_emb_anchor = self.emb_model(voxel_data_anchor)
            img_emb_other = self.emb_model(voxel_data_other)

            img_emb_single = self.emb_model(voxel_data_single)

            img_emb_next_single = self.emb_model(voxel_data_next_single)
            
            node_info_extra = torch.stack([img_emb_anchor[0], img_emb_anchor[2], img_emb_anchor[4]])

            img_emb = torch.cat([
                img_emb_anchor,
                img_emb_other], dim=1)
            
            img_emb = img_emb.to(device)
            

            img_emb_anchor_next = self.emb_model(voxel_data_anchor_next)
            img_emb_other_next = self.emb_model(voxel_data_other_next)
            img_emb_next = torch.cat([
                img_emb_anchor_next,
                img_emb_other_next], dim=1)

            img_emb_next = img_emb_next.to(device)


            
            
            one_hot_encoding = torch.eye(self.num_nodes).float().to(device)
            
            if self.enable_orientation:
                node_pose = torch.cat((one_hot_encoding, x_tensor_dict['batch_all_obj_pair_pos'][0], x_tensor_dict['batch_all_obj_pair_orient'][0]), 1)
                node_pose_goal = torch.cat((one_hot_encoding, x_tensor_dict_next['batch_all_obj_pair_pos'][0], x_tensor_dict_next['batch_all_obj_pair_orient'][0]), 1)
            else:
                node_pose = torch.cat((one_hot_encoding, x_tensor_dict['batch_all_obj_pair_pos'][0]), 1)
                node_pose_goal = torch.cat((one_hot_encoding, x_tensor_dict_next['batch_all_obj_pair_pos'][0]), 1)
            
            if self.enable_shape:
                node_pose = torch.cat((node_pose, node_info_extra), 1)
                node_pose_goal = torch.cat((node_pose_goal, node_info_extra_next), 1)

            action_list = []
            for _ in range(self.num_nodes):
                action_list.append(action[0][0][:])
            action_torch = torch.stack(action_list)

            self.debug_tools = False
            
            if self.debug_tools:
                print('node pose', node_pose)
                print('node pose goal', node_pose_goal)

            # print(generate_edge_embed_list)

            edge_feature = self.generate_edge_embed(node_pose)
            edge_feature_2 = self.generate_edge_embed(node_pose_goal)

            if True:
                x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'][0]
                x_tensor_dict_next['batch_all_obj_pair_relation'] = x_tensor_dict_next['batch_all_obj_pair_relation'][0]#print('edge shape', outs['edge_embed'].shape)


            # print('current relation shape', x_tensor_dict['batch_all_obj_pair_relation'].shape)
            # print('next relation shape', x_tensor_dict_next['batch_all_obj_pair_relation'].shape)
            # print('predict current and gt current', [inp_emb, x_tensor_dict['batch_all_obj_pair_relation'][:, -4:-2]])
            # print('predict next and gt next', [inp_emb_next, x_tensor_dict_next['batch_all_obj_pair_relation'][:, -4:-2]])
            # print('current relation difference', self.dynamics_loss(inp_emb,x_tensor_dict['batch_all_obj_pair_relation'][:, -4:-2]))
            # print('next relation difference', self.dynamics_loss(inp_emb_next,x_tensor_dict_next['batch_all_obj_pair_relation'][:, -4:-2]))
            # print('pred relations', torch.stack(scene_emb_list[0]))
            # print('ground_truth relations', x_tensor_dict['batch_all_obj_pair_relation'])
            #print([select_obj_num_range.shape, self.classif_model.one_hot_encoding_embed])
            select_obj_num_range = select_obj_num_range.cpu().numpy()[0]
            #print(select_obj_num_range)
            if self.set_max:
                one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))
            else:
                one_hot_encoding = np.zeros((self.num_nodes, self.num_nodes))
            
            for one_hot_i in range(len(select_obj_num_range)):
                one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
            one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
            latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
            print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
            node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)
            #print(node_pose.shape)
            
            select_obj_num_range_next = select_obj_num_range_next.cpu().numpy()[0]
            #print(select_obj_num_range)
            if self.set_max:
                one_hot_encoding_next = np.zeros((self.num_nodes, self.max_objects))
            else:
                one_hot_encoding_next = np.zeros((self.num_nodes, self.num_nodes))
            
            for one_hot_i in range(len(select_obj_num_range_next)):
                one_hot_encoding_next[one_hot_i][(int)(select_obj_num_range_next[one_hot_i])] = 1
            one_hot_encoding_next_tensor = torch.Tensor(one_hot_encoding_next).to(device)
            latent_one_hot_encoding_next = self.classif_model.one_hot_encoding_embed(one_hot_encoding_next_tensor)
            node_pose_goal = torch.cat([img_emb_next_single, latent_one_hot_encoding_next], dim = 1)
            #print('input shapa', node_pose.shape)
            
            if self.mlp:
                data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
        
                data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
            else:
                data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
        
                data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, 0, None, action_torch)
                
                
            
            
            if train or not train:
                if self.new_relational_classifier:
                    if self.use_graph_dynamics:
                        batch = Batch.from_data_list([data]).to(device)
                        #print(batch)
                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                        
                        
                        data_dynamics = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)
                        batch_dynamics = Batch.from_data_list([data_dynamics]).to(device)
                        outs_dynamics = self.classif_model_dynamics(batch_dynamics.x, batch_dynamics.edge_index, batch_dynamics.edge_attr, batch_dynamics.batch, batch_dynamics.action)
                        
                        
                        data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch) 
                        batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                        outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
                        
                        #outs_decoder = self.classif_model_decoder(outs_embed['pred'], batch.edge_index, outs_embed['pred_edge'], batch.batch, batch.action)

                        batch2 = Batch.from_data_list([data_next]).to(device)
                        #print(batch)
                        outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                        #print(outs['pred'].size())
                        
                        
                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs_2['pred'], self.edge_emb_size, outs_2['pred_edge'], action_torch)
                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)
                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                        
                        
                        data_2_decoder_edge = self.create_graph(self.num_nodes, self.node_emb_size, outs_dynamics['pred'], self.edge_emb_size, outs_dynamics['pred_edge'], action_torch)
                        batch_decoder_2_edge = Batch.from_data_list([data_2_decoder_edge]).to(device)
                        outs_decoder_2_edge = self.classif_model_decoder(batch_decoder_2_edge.x, batch_decoder_2_edge.edge_index, batch_decoder_2_edge.edge_attr, batch_decoder_2_edge.batch, batch_decoder_2_edge.action)
                        #outs_edge = self.classif_model.forward_decoder(outs['pred_embedding'], batch.edge_index, outs['pred_edge_embed'], batch.batch, batch.action)
                        
                        total_loss = 0
                        # total_loss += self.dynamics_loss(node_pose, outs_decoder['pred']) # node reconstruction loss
                        # total_loss += self.dynamics_loss(node_pose_goal, outs_decoder_2['pred'])
                        
                        
                        # print(outs_decoder['pred_sigmoid'][:].shape)
                        # print(x_tensor_dict['batch_all_obj_pair_relation'].shape)
                        total_loss += self.bce_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation'][:, :])
                        total_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
                        
                        # print('current', [outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']])
                        # print('pred', [outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation']])
                        
                        total_loss += self.dynamics_loss(outs_dynamics['current_embed'], outs_2['current_embed'])
                        total_loss += self.dynamics_loss(outs_dynamics['edge_embed'], outs_2['edge_embed'])
                        total_loss += self.bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
                        
                        
                        # total_loss += self.dynamics_loss(node_pose_goal, outs_decoder_2_edge['pred'])
                        
                        
                        print(total_loss)
                    else:
                        batch = Batch.from_data_list([data]).to(device)
                        #print(batch)
                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                        
                        
                        data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)
                        
                        batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                        
                        outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
                        
                        #outs_decoder = self.classif_model_decoder(outs_embed['pred'], batch.edge_index, outs_embed['pred_edge'], batch.batch, batch.action)

                        batch2 = Batch.from_data_list([data_next]).to(device)
                        #print(batch)
                        outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                        #print(outs['pred'].size())
                        
                        
                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs_2['pred'], self.edge_emb_size, outs_2['pred_edge'], action_torch)
                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)
                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                        
                        
                        data_2_decoder_edge = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred_embedding'], self.edge_emb_size, outs['pred_edge_embed'], action_torch)
                        batch_decoder_2_edge = Batch.from_data_list([data_2_decoder_edge]).to(device)
                        outs_decoder_2_edge = self.classif_model_decoder(batch_decoder_2_edge.x, batch_decoder_2_edge.edge_index, batch_decoder_2_edge.edge_attr, batch_decoder_2_edge.batch, batch_decoder_2_edge.action)
                        #outs_edge = self.classif_model.forward_decoder(outs['pred_embedding'], batch.edge_index, outs['pred_edge_embed'], batch.batch, batch.action)
                        
                        total_loss = 0
                        # total_loss += self.dynamics_loss(node_pose, outs_decoder['pred']) # node reconstruction loss
                        # total_loss += self.dynamics_loss(node_pose_goal, outs_decoder_2['pred'])
                        
                        
                        # print(outs_decoder['pred_sigmoid'][:].shape)
                        # print(x_tensor_dict['batch_all_obj_pair_relation'].shape)
                        total_loss += self.bce_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation'][:, :])
                        total_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
                        
                        # print('current', [outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']])
                        # print('pred', [outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation']])
                        
                        # print('self.using_latent_regularization', self.using_latent_regularization)
                        if self.using_latent_regularization:
                            print('enter self.using_latent_regularization')
                            total_loss += self.dynamics_loss(outs['pred_embedding'], outs_2['current_embed'])
                            total_loss += self.dynamics_loss(outs['pred_edge_embed'], outs_2['edge_embed'])
                        
                        total_loss += self.bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
                        
                        
                        # total_loss += self.dynamics_loss(node_pose_goal, outs_decoder_2_edge['pred'])
                        
                        
                        print(total_loss)
                 
        #print(total_loss)
        if train:
            self.opt_emb.zero_grad()
            self.opt_classif.zero_grad()
            self.opt_classif_decoder.zero_grad()
            if self.use_graph_dynamics:
                self.opt_classif_dynamics.zero_grad()

            total_loss.backward()
            if args.emb_lr >= 1e-5:
                #if 'all_object_pairs' in args.train_type:
                    # raise ValueError("Not frozen")
                    #print("Not frozen")
                self.opt_emb.step()
            self.opt_classif.step()
            self.opt_classif_decoder.step()
            if self.use_graph_dynamics:
                self.opt_classif_dynamics.step()
            leap = 0

        if self.pick_place:
            graph_latent = True
            multi_step_planning = True
        elif not stacking:
            graph_latent = False
            multi_step_planning = False
        else:
            graph_latent = True
            multi_step_planning = True
        edge_classifier = True
        cem_planning = True

        if self.previous_threshold != threshold:
            self.node_pose_list = []
            self.action_list = []
            self.goal_relation_list = []
            self.gt_extents_range_list = []
            self.gt_pose_list = []
            self.predicted_relations = []

            self.node_pose_list_planning = []
            self.action_list_planning = []
            self.goal_relation_list_planning = []
            self.gt_extents_range_list_planning = []
            self.gt_pose_list_planning = []
            self.predicted_relations_planning = []
            
        self.previous_threshold = threshold
        
        
        #Testing
        if not train:
            success_num = 0
            total_num = 1
            planning_success_num = 0
            planning_threshold = threshold

            for test_iter in range(total_num):

                num_nodes = self.num_nodes

                if self.mlp:
                    data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                    # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
                else:
                    data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                batch = Batch.from_data_list([data_1]).to(device)
                outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)



                data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
                batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

                print('current relations', outs_decoder['pred_sigmoid'][:])
                if self.real_data:
                    #part to set goal relations manually
                    print('current relations', outs_decoder['pred_sigmoid'][:])
                    goal_relations = np.array([[0., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                    [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                    [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                    [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                    [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                    [0., 0., 1., 0., 0., 0., 0.]])  # 3-2 # push object 2 minus direction correct 

                    # goal_relations = np.array([[0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                    # [0., 0., 1., 0., 0., 0.],   # 1-3
                    # [0., 0., 0., 1., 0., 0.],   # 2-1
                    # [0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                    # [0., 0., 0., 1., 0., 0.],   # 3-1
                    # [0., 0., 0., 1., 0., 0.]])  # 3-2 # push object 2 positive direction wrong 

                    # goal_relations = np.array([[0., 0., 0., 0., 1., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                    # [0., 0., 1., 0., 0., 0.],   # 1-3
                    # [0., 0., 0., 0., 0., 1.],   # 2-1
                    # [0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                    # [0., 0., 0., 1., 0., 0.],   # 3-1
                    # [0., 0., 0., 1., 0., 0.]])  # 3-2 # push object 3 positive direction correct

                    # goal_relations = np.array([[0., 0., 0., 0., 1., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                    # [0., 0., 0., 1., 0., 0.],   # 1-3
                    # [0., 0., 0., 0., 0., 1.],   # 2-1
                    # [0., 0., 0., 1., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                    # [0., 0., 1., 0., 0., 0.],   # 3-1
                    # [0., 0., 1., 0., 0., 0.]])  # 3-2 # push object 3 negative direction correct

                    # # action as push block 2  [0.6176837668646232, -0.12890155670030637, 0.9120449995994568]
                    # # [[0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]]
                    x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations).to(device)
                    print(x_tensor_dict_next['batch_all_obj_pair_relation'])
                    #time.sleep(5)
                
                min_cost = 1e5
                loss_list = []
                all_action_list = []
                print('actual action', action)
                for obj_mov in range(self.num_nodes):
                    print('mov obj', obj_mov)
                    action_selections = 500
                    action_mu = np.zeros((action_selections, 1, 2))
                    action_sigma = np.ones((action_selections, 1, 2))
                    for i_iter in range(5):
                        action_noise = np.zeros((action_selections, 1, 2))
                        action_noise[:,:,0] = (np.random.rand(action_selections, 1) - 0.5) * 0.1
                        action_noise[:,:,1] = (np.random.rand(action_selections, 1) - 0.5) * 0.6
                        #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                        act = action_mu + action_noise*action_sigma
                        costs = []
                        for j in range(action_selections):
                            action_numpy = np.zeros((num_nodes, 3))
                            action_numpy[obj_mov][0] = act[j, 0, 0]
                            action_numpy[obj_mov][1] = act[j, 0, 1]
                            action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                            
                            if self.set_max:
                                action = np.zeros((num_nodes, self.max_objects + 3))
                            else:
                                action = np.zeros((num_nodes, num_nodes + 3))
                            for i in range(action.shape[0]):
                                action[i][obj_mov] = 1
                                action[i][-3:] = action_numpy[obj_mov]
                            
                            sample_action = torch.Tensor(action).to(device)
                            
                            # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                            # for i in range(action_1.shape[0]):
                            #     action_1[i][obj_mov] = 1
                            #     action_1[i][3:] = action_numpy[obj_mov]
                            # sample_action = torch.Tensor(action_1)
                            #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                            # if(_ == 0):
                            #     sample_action = action
                            this_sequence = []
                            this_sequence.append(sample_action)
                            loss_func = nn.MSELoss()
                            test_loss = 0
                            current_latent = outs['current_embed']
                            egde_latent = outs['edge_embed']
                            #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                            for seq in range(len(this_sequence)):
                                #print([current_latent, this_sequence[seq]])
                                if self.use_graph_dynamics:
                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                else:
                                    current_action = self.classif_model.action_emb(this_sequence[seq])

                                graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                current_latent = self.classif_model.graph_dynamics(graph_node_action)
                            for seq in range(len(this_sequence)):
                                #print([current_latent, this_sequence[seq]])
                                #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                edge_num = egde_latent.shape[0]
                                edge_action_list = []
                                if self.use_graph_dynamics:
                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                else:
                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                for _ in range(edge_num):
                                    edge_action_list.append(current_action[0])
                                edge_action = torch.stack(edge_action_list)
                                graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                            #test_loss += loss_func(current_latent, outs_2['current_embed'])
                            #print(egde_latent.shape)
                            # print(current_latent)
                            # print(egde_latent)
                            # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                            data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
    
                            batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                            outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                            
                            
                            
                            if self.all_gt_sigmoid:
                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
            
                            costs.append(test_loss.detach().cpu().numpy())
                            # if(test_loss.detach().cpu().numpy() < min_cost):
                            #     min_action = this_sequence
                            #     min_cost = test_loss.detach().cpu().numpy()
                    
                            #     costs.append(test_loss)

                        index = np.argsort(costs)
                        elite = act[index,:,:]
                        elite = elite[:3, :, :]
                            # print('elite')
                            # print(elite)
                        act_mu = elite.mean(axis = 0)
                        act_sigma = elite.std(axis = 0)
                        print([act_mu, act_sigma])
                        # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                        #     break
                        #print(act_sigma)
                    # print('find_actions')
                    # print(act_mu)
                    chosen_action = act_mu
                    action_numpy = np.zeros((num_nodes, 3))
                    action_numpy[obj_mov][0] = chosen_action[0, 0]
                    action_numpy[obj_mov][1] = chosen_action[0, 1]
                    action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                    if self.set_max:
                        action = np.zeros((num_nodes, self.max_objects + 3))
                    else:
                        action = np.zeros((num_nodes, num_nodes + 3))
                    for i in range(action.shape[0]):
                        action[i][obj_mov] = 1
                        action[i][-3:] = action_numpy[obj_mov]
                            
                    sample_action = torch.Tensor(action).to(device)
                    # if(_ == 0):
                    #     sample_action = action
                    this_sequence = []
                    this_sequence.append(sample_action)
                    if True:
                        loss_func = nn.MSELoss()
                        test_loss = 0
                        current_latent = outs['current_embed']
                        egde_latent = outs['edge_embed']
                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                        for seq in range(len(this_sequence)):
                            #print([current_latent, this_sequence[seq]])
                            if self.use_graph_dynamics:
                                current_action = self.classif_model.action_emb(this_sequence[seq])
                            else:
                                current_action = self.classif_model.action_emb(this_sequence[seq])
                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                        for seq in range(len(this_sequence)):
                            #print([current_latent, this_sequence[seq]])
                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                            edge_num = egde_latent.shape[0]
                            edge_action_list = []
                            if self.use_graph_dynamics:
                                current_action = self.classif_model.action_emb(this_sequence[seq])
                            else:
                                current_action = self.classif_model.action_emb(this_sequence[seq])
                            for _ in range(edge_num):
                                edge_action_list.append(current_action[0])
                            edge_action = torch.stack(edge_action_list)
                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                        #test_loss += loss_func(current_latent, outs_2['current_embed'])
                        #print(egde_latent.shape)
                        # print(current_latent)
                        # print(egde_latent)
                        #outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
    
                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                        
                        
                        
                        if self.all_gt_sigmoid:
                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                
                        #sample_list.append(outs_edge['pred_edge'])
                        loss_list.append(test_loss)
                        if(test_loss.detach().cpu().numpy() < min_cost):
                            min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                            min_action = this_sequence
                            min_pose = outs_decoder_2['pred'][:, :]
                            min_cost = test_loss.detach().cpu().numpy()
                
                # print('initial_edge_embed', x_tensor_dict['batch_all_obj_pair_relation'][0][:])
                # print('min_prediction', min_prediction)
                # print('min node pose prediction', min_pose)
                pred_relations = min_prediction.cpu().detach().numpy()
                goal_relations = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()

                planning_success_num = 1
                for obj_id in range(pred_relations.shape[0]):
                    for relation_id in range(pred_relations.shape[1]):
                        if goal_relations[obj_id][relation_id] == 1:
                            if pred_relations[obj_id][relation_id] < planning_threshold:
                                planning_success_num = 0
                        elif goal_relations[obj_id][relation_id] == 0:
                            if pred_relations[obj_id][relation_id] > 1 - planning_threshold:
                                planning_success_num = 0

                print('pred_relations', pred_relations)
                print('goal_relations', goal_relations)
                print('planning_success_num', planning_success_num)
                node_pose_numpy = node_pose.detach().cpu().numpy()
                change_id_leap = 0
                if True: #for seq in range(len(min_action)):
                    this_seq_numpy = min_action[0].cpu().numpy()
                    change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                        #print(change_id)
                    if change_id == 0:
                        change_id_leap = 1
                        node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                    #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                    #     #if(this_seq_numpy[0])
                    # all_node_pose_list.append(node_pose_numpy)
                    # node_pose = torch.Tensor(node_pose_numpy)
                    # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                node_pose_numpy_goal = node_pose_goal.detach().cpu().numpy()
                choose_list = [0,-1]
                # print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                
                
                print("min_action", min_action)
                #min_action_numpy
                print("action_torch", action_torch)
                #print()
                print("min_cost", min_cost)
                # if change_id_leap == 1:
                #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                #     print(goal_loss)
                #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                #         success_num += 1
                min_action_numpy = min_action[0].cpu().numpy()
                action_numpy = action_torch.cpu().numpy()

                # for node_pose_iter in range(node_pose_numpy.shape[0]):
                #     self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                # for action_iter in range(1):
                #     self.action_list.append(min_action_numpy[action_iter])
                # for goal_relation_i in range(goal_relation.shape[0]):
                #     for goal_relation_j in range(goal_relation.shape[1]):
                #         self.goal_relation_list.append(goal_relation[goal_relation_i][goal_relation_j])
                self.node_pose_list.append(node_pose_numpy)
                self.action_list.append(min_action_numpy)
                self.goal_relation_list.append(goal_relations)
                self.gt_pose_list.append(self.gt_pose[0])
                self.gt_extents_range_list.append(self.gt_extents_range[0])
                self.predicted_relations.append(pred_relations)
                
                
                
            
                
                # simplied version of planned action success rate for pushing task
                success_num = 1
                for action_i in range(min_action_numpy.shape[1] - 3):
                    if(min_action_numpy[0][action_i] != action_numpy[0][action_i]):
                        success_num = 0
                if min_action_numpy[0][-2]*action_numpy[0][-2] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                    success_num = 0
                
            print("success_num", success_num)
            print("success_num/total_num", success_num/total_num)                    
            #print(action)

        # #print(batch_result_dict)
        planning_leap = 0
        if not train:
            leap  = success_num
            planning_leap = planning_success_num
        else:
            leap = 0
            planning_leap = 0
        return batch_result_dict, leap, planning_leap

    def run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e_manual_relations(self,
                           x_tensor_dict,
                           x_tensor_dict_next,
                           batch_size,
                           train=False,
                           save_preds=False,
                           save_emb=False, 
                           threshold = 0):
        batch_result_dict = {}
        device = self.config.get_device()
        args = self.config.args

        #print(x_tensor_dict)
        voxel_data = x_tensor_dict['batch_voxel']
        
        voxel_data_single = x_tensor_dict['batch_voxel_single']
        # print(voxel_data_single.shape)
        # print(voxel_data_single[4])
        
        voxel_data_anchor = x_tensor_dict['batch_anchor_voxel']

        
        voxel_data_other = x_tensor_dict['batch_other_voxel']

        


        voxel_data_next = x_tensor_dict_next['batch_voxel']
        voxel_data_next_single = x_tensor_dict_next['batch_voxel_single']
        # print(voxel_data_next_single.shape)
        # print(voxel_data_next_single[4])
        # print(x_tensor_dict_next['batch_this_one_hot_encoding'])
        this_one_hot_encoding_numpy = x_tensor_dict_next['batch_this_one_hot_encoding'].detach().cpu().numpy()[0,0]
        #print(this_one_hot_encoding_numpy.shape)
        if not self.real_data:
            for check_i in range(this_one_hot_encoding_numpy.shape[0]):
                if this_one_hot_encoding_numpy[check_i] == 0:
                    for check_i_again in range(check_i,this_one_hot_encoding_numpy.shape[0]):
                        if this_one_hot_encoding_numpy[check_i_again] == 1:
                            print(this_one_hot_encoding_numpy)
                            raise ValueError("invalide this one hot encoding")

        #time.sleep(10)
        voxel_data_anchor_next = x_tensor_dict_next['batch_anchor_voxel']
        voxel_data_other_next = x_tensor_dict_next['batch_other_voxel']

        voxel_data_edge_next = torch.cat([
                voxel_data_anchor_next,
                voxel_data_other_next], dim=1)

        select_obj_num_range = x_tensor_dict['batch_select_obj_num_range']
        select_obj_num_range_next = x_tensor_dict_next['batch_select_obj_num_range']

        gt_pose_list = x_tensor_dict['batch_gt_pose_list']
        gt_orientation_list = x_tensor_dict['batch_gt_orientation_list']
        self.pc_center = x_tensor_dict['batch_pc_center'].cpu().detach().numpy()
        if not train:
            #print(x_tensor_dict)
            if self.pushing and 'batch_gt_extents_range_list' in x_tensor_dict:
                gt_extents_range_list = x_tensor_dict['batch_gt_extents_range_list']
            else:
                gt_extents_range_list = torch.zeros((5,3)).to(device)

            # print('gt_pose_list', gt_pose_list)
            # print('gt_extents_range_list',gt_extents_range_list)
            self.gt_pose = x_tensor_dict['batch_gt_pose_list'].cpu().detach().numpy()
            self.gt_orientation = x_tensor_dict['batch_gt_orientation_list'].cpu().detach().numpy()
            if self.pushing and 'batch_gt_extents_range_list' in x_tensor_dict:
                self.gt_extents_range = x_tensor_dict['batch_gt_extents_range_list'].cpu().detach().numpy()
            else:
                self.gt_extents_range = np.zeros((5,3))

            if self.pushing and 'batch_gt_extents_list' in x_tensor_dict:
                self.gt_extents = x_tensor_dict['batch_gt_extents_list'].cpu().detach().numpy()
            else:
                self.gt_extents = np.zeros((5,3))

        # print('gt_pose_list', self.gt_pose_list)
        # print('gt_extents_range_list', self.gt_extents_range)

        # print(x_tensor_dict['batch_pc_center'].shape)
        # time.sleep(10)

        
        action_fake = x_tensor_dict_next['batch_all_obj_pair_pos'] - x_tensor_dict['batch_all_obj_pair_pos']
        # print(device)
        
        # print(action)
        # print(voxel_data.shape)
        #action = action.to(device)
        action = x_tensor_dict_next['batch_action']

        stacking = self.stacking

        self.num_nodes = x_tensor_dict['batch_num_objects'].cpu().numpy().astype(int)[0]
        #print(self.num_nodes)
        #time.sleep(10)
        #print(action)
        #print(x_tensor_dict)
        
        # Now join the img_emb into a list for each scene.
        if  'all_object_pairs' in args.train_type:
            # Get the embeddings
            # if args.emb_lr <= 1e-6:
            #     with torch.no_grad():
            #         img_emb = self.emb_model(voxel_data)
            # else:
            #     img_emb = self.emb_model(voxel_data)
            # print('single voxel_data shape', voxel_data_single.shape)
            # print('voxel_data shape', voxel_data.shape)
            # print('voxel_data_anchor shape', voxel_data_anchor.shape)
            # print('voxel_data_other shape', voxel_data_other.shape)
            img_emb_anchor = self.emb_model(voxel_data_anchor)
            img_emb_other = self.emb_model(voxel_data_other)
            # print(img_emb_anchor.shape)
            # print(img_emb_other.shape)
            #print('voxel data single shape', voxel_data_single.shape)
            img_emb_single = self.emb_model(voxel_data_single)
            #print('single voxel_data shape', img_emb_single.shape)

            img_emb_next_single = self.emb_model(voxel_data_next_single)
            #print('single voxel_data shape', img_emb_next_single.shape)
            
            node_info_extra = torch.stack([img_emb_anchor[0], img_emb_anchor[2], img_emb_anchor[4]])
            #print(node_info_extra.shape)
            #node_info
            img_emb = torch.cat([
                img_emb_anchor,
                img_emb_other], dim=1)

            # print(img_emb.shape)
            # time.sleep(10)


            
            img_emb = img_emb.to(device)
            
            img_emb_anchor_next = self.emb_model(voxel_data_anchor_next)
            img_emb_other_next = self.emb_model(voxel_data_other_next)
            img_emb_next = torch.cat([
                img_emb_anchor_next,
                img_emb_other_next], dim=1)

            # print('inp_emb shape', inp_emb.shape)
            # print('inp_emb', inp_emb)

            img_emb_next = img_emb_next.to(device)

            
            one_hot_encoding = torch.eye(self.num_nodes).float().to(device)
            
            # print(one_hot_encoding)
            # print(x_tensor_dict['batch_all_obj_pair_pos'][0])
            if self.enable_orientation:
                node_pose = torch.cat((one_hot_encoding, x_tensor_dict['batch_all_obj_pair_pos'][0], x_tensor_dict['batch_all_obj_pair_orient'][0]), 1)
                node_pose_goal = torch.cat((one_hot_encoding, x_tensor_dict_next['batch_all_obj_pair_pos'][0], x_tensor_dict_next['batch_all_obj_pair_orient'][0]), 1)
            else:
                node_pose = torch.cat((one_hot_encoding, x_tensor_dict['batch_all_obj_pair_pos'][0]), 1)
                node_pose_goal = torch.cat((one_hot_encoding, x_tensor_dict_next['batch_all_obj_pair_pos'][0]), 1)
            
            if self.enable_shape:
                node_pose = torch.cat((node_pose, node_info_extra), 1)
                node_pose_goal = torch.cat((node_pose_goal, node_info_extra_next), 1)

            #print('node pose', node_pose)
            #print(action.shape)
            #action_torch = torch.cat((action[:][0][0][:], action[:][0][0][:], action[:][0][0][:]), 0)
            
            # print(action[:][0][0][:])
            # print(action[:][0][0][:].shape)
            # print(action_torch)
            # print(action_torch.shape)
            #print(node_pose)
            # print(generate_edge_embed_list)

            edge_feature = self.generate_edge_embed(node_pose)
            edge_feature_2 = self.generate_edge_embed(node_pose_goal)

            if True:
                x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'][0]
                x_tensor_dict_next['batch_all_obj_pair_relation'] = x_tensor_dict_next['batch_all_obj_pair_relation'][0]#print('edge shape', outs['edge_embed'].shape)


            # print('current relation shape', x_tensor_dict['batch_all_obj_pair_relation'].shape)
            # print('next relation shape', x_tensor_dict_next['batch_all_obj_pair_relation'].shape)
            # print('predict current and gt current', [inp_emb, x_tensor_dict['batch_all_obj_pair_relation'][:, -4:-2]])
            # print('predict next and gt next', [inp_emb_next, x_tensor_dict_next['batch_all_obj_pair_relation'][:, -4:-2]])
            # print('current relation difference', self.dynamics_loss(inp_emb,x_tensor_dict['batch_all_obj_pair_relation'][:, -4:-2]))
            # print('next relation difference', self.dynamics_loss(inp_emb_next,x_tensor_dict_next['batch_all_obj_pair_relation'][:, -4:-2]))
            # print('pred relations', torch.stack(scene_emb_list[0]))
            # print('ground_truth relations', x_tensor_dict['batch_all_obj_pair_relation'])
            if self.use_point_cloud_embedding:
                #print([select_obj_num_range.shape, self.classif_model.one_hot_encoding_embed])
                if self.evaluate_end_relations:
                    new_total_objects = 0
                    for check_i in range(this_one_hot_encoding_numpy.shape[0]):
                        if this_one_hot_encoding_numpy[check_i] == 1:
                            new_total_objects += 1

                    self.num_nodes = new_total_objects
                    img_emb_single = img_emb_single[:self.num_nodes]
                    img_emb_next_single = img_emb_next_single[:self.num_nodes]
                    select_obj_num_range = select_obj_num_range.cpu().numpy()[0]
                    #print(select_obj_num_range)
                    if self.set_max:
                        one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))
                    else:
                        one_hot_encoding = np.zeros((self.num_nodes, self.num_nodes))
                    
                    print(select_obj_num_range)
                    for one_hot_i in range(self.num_nodes):
                        one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
                    one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
                    latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
                    #print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
                    node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)
                    #print(node_pose.shape)
                    
                    select_obj_num_range_next = select_obj_num_range_next.cpu().numpy()[0]
                    #print(select_obj_num_range)
                    if self.set_max:
                        one_hot_encoding_next = np.zeros((self.num_nodes, self.max_objects))
                    else:
                        one_hot_encoding_next = np.zeros((self.num_nodes, self.num_nodes))
                    
                    for one_hot_i in range(self.num_nodes):
                        one_hot_encoding_next[one_hot_i][(int)(select_obj_num_range_next[one_hot_i])] = 1
                    one_hot_encoding_next_tensor = torch.Tensor(one_hot_encoding_next).to(device)
                    latent_one_hot_encoding_next = self.classif_model.one_hot_encoding_embed(one_hot_encoding_next_tensor)
                    node_pose_goal = torch.cat([img_emb_next_single, latent_one_hot_encoding_next], dim = 1)
                else:
                    select_obj_num_range = select_obj_num_range.cpu().numpy()[0]
                    #print(select_obj_num_range)
                    if self.set_max:
                        one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))
                    else:
                        one_hot_encoding = np.zeros((self.num_nodes, self.num_nodes))
                    
                    for one_hot_i in range(len(select_obj_num_range)):
                        one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
                    one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
                    latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
                    print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
                    node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)
                    #print(node_pose.shape)
                    
                    select_obj_num_range_next = select_obj_num_range_next.cpu().numpy()[0]
                    #print(select_obj_num_range)
                    if self.set_max:
                        one_hot_encoding_next = np.zeros((self.num_nodes, self.max_objects))
                    else:
                        one_hot_encoding_next = np.zeros((self.num_nodes, self.num_nodes))
                    
                    for one_hot_i in range(len(select_obj_num_range_next)):
                        one_hot_encoding_next[one_hot_i][(int)(select_obj_num_range_next[one_hot_i])] = 1
                    one_hot_encoding_next_tensor = torch.Tensor(one_hot_encoding_next).to(device)
                    latent_one_hot_encoding_next = self.classif_model.one_hot_encoding_embed(one_hot_encoding_next_tensor)
                    node_pose_goal = torch.cat([img_emb_next_single, latent_one_hot_encoding_next], dim = 1)
            #print('input shapa', node_pose.shape)
            
            action_list = []
            for _ in range(self.num_nodes):
                action_list.append(action[0][0][:])
            action_torch = torch.stack(action_list)
            
            if self.mlp:
                data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
        
                data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
            else:
                data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
        
                data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, 0, None, action_torch)
                
                
            
            
            if train or not train:
                if not self.evaluate_end_relations:
                    batch = Batch.from_data_list([data]).to(device)
                    #print(batch)
                    outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                    
                    
                    data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)
                    
                    batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                    
                    outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
                    
                    #outs_decoder = self.classif_model_decoder(outs_embed['pred'], batch.edge_index, outs_embed['pred_edge'], batch.batch, batch.action)

                    batch2 = Batch.from_data_list([data_next]).to(device)
                    #print(batch)
                    outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                    #print(outs['pred'].size())
                    
                    
                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs_2['pred'], self.edge_emb_size, outs_2['pred_edge'], action_torch)
                    batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)
                    outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                    
                    
                    data_2_decoder_edge = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred_embedding'], self.edge_emb_size, outs['pred_edge_embed'], action_torch)
                    batch_decoder_2_edge = Batch.from_data_list([data_2_decoder_edge]).to(device)
                    outs_decoder_2_edge = self.classif_model_decoder(batch_decoder_2_edge.x, batch_decoder_2_edge.edge_index, batch_decoder_2_edge.edge_attr, batch_decoder_2_edge.batch, batch_decoder_2_edge.action)
                    #outs_edge = self.classif_model.forward_decoder(outs['pred_embedding'], batch.edge_index, outs['pred_edge_embed'], batch.batch, batch.action)
                    
                    total_loss = 0
                    # total_loss += self.dynamics_loss(node_pose, outs_decoder['pred']) # node reconstruction loss
                    # total_loss += self.dynamics_loss(node_pose_goal, outs_decoder_2['pred'])
                    
                    
                    # print(outs_decoder['pred_sigmoid'][:].shape)
                    # print(x_tensor_dict['batch_all_obj_pair_relation'].shape)
                    total_loss += self.bce_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation'][:, :])
                    total_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
                    
                    # print('current', [outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']])
                    # print('pred', [outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation']])
                    
                    if train:
                        total_loss += self.dynamics_loss(outs['pred_embedding'], outs_2['current_embed'])
                        total_loss += self.dynamics_loss(outs['pred_edge_embed'], outs_2['edge_embed'])
                        total_loss += self.bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
                        
                    
                    # total_loss += self.dynamics_loss(node_pose_goal, outs_decoder_2_edge['pred'])
                    
                    
                    print(total_loss)
                
        #print(total_loss)
        if train:
            self.opt_emb.zero_grad()
            self.opt_classif.zero_grad()
            self.opt_classif_decoder.zero_grad()
            if self.use_graph_dynamics:
                self.opt_classif_dynamics.zero_grad()

            total_loss.backward()
            if args.emb_lr >= 1e-5:
                #if 'all_object_pairs' in args.train_type:
                    # raise ValueError("Not frozen")
                    #print("Not frozen")
                self.opt_emb.step()
            self.opt_classif.step()
            self.opt_classif_decoder.step()
            if self.use_graph_dynamics:
                self.opt_classif_dynamics.step()
            leap = 0

        if self.pick_place:
            graph_latent = True
            multi_step_planning = True
        elif not stacking:
            graph_latent = False
            multi_step_planning = False
        else:
            graph_latent = True
            multi_step_planning = True
        edge_classifier = True
        cem_planning = True

        if self.previous_threshold != threshold:
            self.node_pose_list = []
            self.action_list = []
            self.action_variance_list = []
            self.goal_relation_list = []
            self.gt_extents_list = []
            self.gt_extents_range_list = []
            self.gt_pose_list = []
            self.pc_center_list = []
            self.gt_orientation_list = []
            self.predicted_relations = []
            self.all_index_i_list = []
            self.all_index_j_list = []

            self.node_pose_list_planning = []
            self.action_list_planning = []
            self.action_variance_list_planning = []
            self.goal_relation_list_planning = []
            self.gt_extents_list_planning = []
            self.gt_extents_range_list_planning = []
            self.gt_pose_list_planning = []
            self.pc_center_list_planning = []
            self.gt_orientation_list_planning = []
            self.predicted_relations_planning = []
            
        self.previous_threshold = threshold
        
        
        
        goal_relations_list = []
        expected_action_list = []
        total_objects = self.num_nodes
        #verify this part is correct before doing large-scale real experiments
        if self.manual_relations:
            if self.num_nodes == 3: 
                #part to set goal relations manually
                
                goal_relations_list.append(np.array([[0., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.]]))  # 3-2 # push object 2 minus direction correct 
                expected_action_list.append([2, -1])
                ##  I try to define some terms here left, right , behind, front, below, top, contact 

                goal_relations_list.append(np.array([[0., 0., 1., 0., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.]]))  # 3-2 # push object 2 positive direction wrong 
                expected_action_list.append([2, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.]]))  # 3-2 # push object 3 positive direction correct
                expected_action_list.append([3, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.]]))  # 3-2 # push object 3 negative direction correct
                expected_action_list.append([3, -1])

                # # action as push block 2  [0.6176837668646232, -0.12890155670030637, 0.9120449995994568]
                # # [[0, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 1, 1, 0, 0, 0], [0, 1, 1, 0, 0, 0]]
                # x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations).to(device)
                # print(x_tensor_dict_next['batch_all_obj_pair_relation'])
                #time.sleep(5)
            elif self.num_nodes == 4:
                #part to set goal relations manually
                
                goal_relations_list.append(np.array([[0., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.]]))     # 4-3
                expected_action_list.append([2, -1])

                goal_relations_list.append(np.array([[0., 0., 1., 0., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 2 positive direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.]]))     # 4-3
                expected_action_list.append([2, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 3 positive direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.]]))     # 4-3
                expected_action_list.append([3, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 3 negative direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.]]))     # 4-3
                expected_action_list.append([3, -1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 4 positive direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.]]))     # 4-3
                expected_action_list.append([4, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 4 positive direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.]]))     # 4-3
                expected_action_list.append([4, -1])        
            elif self.num_nodes == 5:
                #part to set goal relations manually
                
                goal_relations_list.append(np.array([[0., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([2, -1])

                goal_relations_list.append(np.array([[0., 0., 1., 0., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [0., 0., 0., 1., 0., 0., 0.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 2 positive direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([2, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 3 positive direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([3, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 3 negative direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([3, -1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 4 positive direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([4, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 4 negative direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([4, -1])   

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 5 positive direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([5, 1])    

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 5 negative direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.]]))       # 5-4
                expected_action_list.append([5, -1])     
            elif self.num_nodes == 6:
                #part to set goal relations manually

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [1., 0., 0., 1., 0., 0., 0.],     # 1-5
                [1., 0., 0., 1., 0., 0., 0.],     # 1-6
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [1., 0., 0., 1., 0., 0., 0.],     # 2-5
                [1., 0., 0., 1., 0., 0., 0.],     # 2-6
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [1., 0., 0., 1., 0., 0., 0.],     # 3-5
                [1., 0., 0., 1., 0., 0., 0.],     # 3-6
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],     # 4-3
                [1., 0., 0., 1., 0., 0., 0.],     # 4-5
                [1., 0., 0., 1., 0., 0., 0.],     # 4-6
                [0., 1., 1., 0., 0., 0., 0.],   # 5-1
                [0., 1., 1., 0., 0., 0., 0.],   # 5-2
                [0., 1., 1., 0., 0., 0., 0.],   # 5-3
                [0., 1., 1., 0., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 1., 0., 0., 0., 0.],   # 6-1
                [0., 1., 1., 0., 0., 0., 0.],   # 6-2
                [0., 1., 1., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([1, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [1., 0., 1., 0., 0., 0., 0.],     # 1-5
                [1., 0., 1., 0., 0., 0., 0.],     # 1-6
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [1., 0., 1., 0., 0., 0., 0.],     # 2-5
                [1., 0., 1., 0., 0., 0., 0.],     # 2-6
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [1., 0., 1., 0., 0., 0., 0.],     # 3-5
                [1., 0., 1., 0., 0., 0., 0.],     # 3-6
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],     # 4-3
                [1., 0., 1., 0., 0., 0., 0.],     # 4-5
                [1., 0., 1., 0., 0., 0., 0.],     # 4-6
                [0., 1., 0., 1., 0., 0., 0.],   # 5-1
                [0., 1., 0., 1., 0., 0., 0.],   # 5-2
                [0., 1., 0., 1., 0., 0., 0.],   # 5-3
                [0., 1., 0., 1., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 1., 0., 0., 0.],   # 6-1
                [0., 1., 0., 1., 0., 0., 0.],   # 6-2
                [0., 1., 0., 1., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([1, -1])
                
            
                
                goal_relations_list.append(np.array([[0., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [1., 0., 0., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [1., 0., 1., 0., 0., 0., 0.],   # 2-5
                [1., 0., 1., 0., 0., 0., 0.],   # 2-6
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [1., 0., 1., 0., 0., 0., 0.],   # 3-5
                [1., 0., 1., 0., 0., 0., 0.],   # 3-6
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],     # 4-3
                [1., 0., 1., 0., 0., 0., 0.],     # 4-5
                [1., 0., 1., 0., 0., 0., 0.],     # 4-6
                [0., 1., 0., 0., 0., 0., 0.],   # 5-1
                [0., 1., 0., 1., 0., 0., 0.],   # 5-2
                [0., 1., 0., 1., 0., 0., 0.],   # 5-3
                [0., 1., 0., 1., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 1., 0., 0., 0.],   # 6-2
                [0., 1., 0., 1., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([2, -1])

                goal_relations_list.append(np.array([[0., 0., 1., 0., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [1., 0., 0., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [0., 0., 0., 1., 0., 0., 0.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [1., 0., 0., 1., 0., 0., 0.],   # 2-5
                [1., 0., 0., 1., 0., 0., 0.],   # 2-6
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [1., 0., 0., 1., 0., 0., 0.],   # 3-5
                [1., 0., 0., 1., 0., 0., 0.],   # 3-6
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],     # 4-3
                [1., 0., 0., 1., 0., 0., 0.],     # 4-5
                [1., 0., 0., 1., 0., 0., 0.],     # 4-6
                [0., 1., 0., 0., 0., 0., 0.],   # 5-1
                [0., 1., 1., 0., 0., 0., 0.],   # 5-2
                [0., 1., 1., 0., 0., 0., 0.],   # 5-3
                [0., 1., 1., 0., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 1., 0., 0., 0., 0.],   # 6-2
                [0., 1., 1., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))     # 6-5
                expected_action_list.append([2, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [1., 0., 0., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [1., 0., 0., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [1., 0., 0., 1., 0., 0., 0.],   # 3-5
                [1., 0., 0., 1., 0., 0., 0.],   # 3-6
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],     # 4-3
                [1., 0., 0., 1., 0., 0., 0.],     # 4-5
                [1., 0., 0., 1., 0., 0., 0.],     # 4-6
                [0., 1., 0., 0., 0., 0., 0.],   # 5-1
                [0., 1., 0., 0., 0., 0., 0.],   # 5-2
                [0., 1., 1., 0., 0., 0., 0.],   # 5-3
                [0., 1., 1., 0., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 1., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([3, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [1., 0., 0., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [1., 0., 0., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [1., 0., 1., 0., 0., 0., 0.],   # 3-5
                [1., 0., 1., 0., 0., 0., 0.],   # 3-6
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],     # 4-3
                [1., 0., 1., 0., 0., 0., 0.],     # 4-5
                [1., 0., 1., 0., 0., 0., 0.],     # 4-6
                [0., 1., 0., 0., 0., 0., 0.],   # 5-1
                [0., 1., 0., 0., 0., 0., 0.],   # 5-2
                [0., 1., 0., 1., 0., 0., 0.],   # 5-3
                [0., 1., 0., 1., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 1., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([3, -1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [1., 0., 0., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [1., 0., 0., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [1., 0., 0., 0., 0., 0., 0.],   # 3-5
                [1., 0., 0., 0., 0., 0., 0.],   # 3-6
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],     # 4-3
                [1., 0., 0., 1., 0., 0., 0.],     # 4-5
                [1., 0., 0., 1., 0., 0., 0.],     # 4-6
                [0., 1., 0., 0., 0., 0., 0.],   # 5-1
                [0., 1., 0., 0., 0., 0., 0.],   # 5-2
                [0., 1., 0., 0., 0., 0., 0.],   # 5-3
                [0., 1., 1., 0., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([4, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [1., 0., 0., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [1., 0., 0., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [1., 0., 0., 0., 0., 0., 0.],   # 3-5
                [1., 0., 0., 0., 0., 0., 0.],   # 3-6
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],     # 4-3
                [1., 0., 1., 0., 0., 0., 0.],     # 4-5
                [1., 0., 1., 0., 0., 0., 0.],     # 4-6
                [0., 1., 0., 0., 0., 0., 0.],   # 5-1
                [0., 1., 0., 0., 0., 0., 0.],   # 5-2
                [0., 1., 0., 0., 0., 0., 0.],   # 5-3
                [0., 1., 0., 1., 0., 0., 0.],     # 5-4
                [0., 0., 0., 0., 1., 0., 1.],     # 5-6
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 0., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],     # 6-4
                [0., 0., 0., 0., 0., 1., 1.]]))       # 6-5
                expected_action_list.append([4, -1])        
            elif self.num_nodes == 7:
                #part to set goal relations manually
                
                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [0., 0., 0., 0., 1., 0., 0.],   # 1-5
                [1., 0., 0., 1., 0., 0., 0.],   # 1-6
                [1., 0., 0., 1., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [0., 0., 0., 0., 1., 0., 0.],   # 2-5
                [1., 0., 0., 1., 0., 0., 0.],   # 2-6
                [1., 0., 0., 1., 0., 0., 0.],   # 2-7
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [0., 0., 0., 0., 1., 0., 0.],     # 3-5
                [1., 0., 0., 1., 0., 0., 0.],   # 3-6
                [1., 0., 0., 1., 0., 0., 0.],   # 3-7
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],    # 4-3
                [0., 0., 0., 0., 1., 0., 1.],      # 4-5
                [1., 0., 0., 1., 0., 0., 0.],   # 4-6
                [1., 0., 0., 1., 0., 0., 0.],   # 4-7
                [0., 0., 0., 0., 0., 1., 0.],      # 5-1
                [0., 0., 0., 0., 0., 1., 0.],      # 5-2
                [0., 0., 0., 0., 0., 1., 0.],      # 5-3
                [0., 0., 0., 0., 0., 1., 1.],       # 5-4
                [1., 0., 0., 1., 0., 0., 0.],   # 5-6
                [1., 0., 0., 1., 0., 0., 0.],   # 5-7
                [0., 1., 1., 0., 0., 0., 0.],   # 6-1
                [0., 1., 1., 0., 0., 0., 0.],   # 6-2
                [0., 1., 1., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],   # 6-4
                [0., 1., 1., 0., 0., 0., 0.],   # 7-1
                [0., 1., 1., 0., 0., 0., 0.],   # 7-2
                [0., 1., 1., 0., 0., 0., 0.],   # 7-3
                [0., 1., 1., 0., 0., 0., 0.],   # 7-4
                [0., 1., 1., 0., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))   # 7-6
                expected_action_list.append([1, 1])  


                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [0., 0., 0., 0., 1., 0., 0.],   # 1-5
                [1., 0., 1., 0., 0., 0., 0.],   # 1-6
                [1., 0., 1., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [0., 0., 0., 0., 1., 0., 0.],   # 2-5
                [1., 0., 1., 0., 0., 0., 0.],   # 2-6
                [1., 0., 1., 0., 0., 0., 0.],   # 2-7
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [0., 0., 0., 0., 1., 0., 0.],     # 3-5
                [1., 0., 1., 0., 0., 0., 0.],   # 3-6
                [1., 0., 1., 0., 0., 0., 0.],   # 3-7
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],    # 4-3
                [0., 0., 0., 0., 1., 0., 1.],      # 4-5
                [1., 0., 1., 0., 0., 0., 0.],   # 4-6
                [1., 0., 1., 0., 0., 0., 0.],   # 4-7
                [0., 0., 0., 0., 0., 1., 0.],      # 5-1
                [0., 0., 0., 0., 0., 1., 0.],      # 5-2
                [0., 0., 0., 0., 0., 1., 0.],      # 5-3
                [0., 0., 0., 0., 0., 1., 1.],       # 5-4
                [1., 0., 1., 0., 0., 0., 0.],   # 5-6
                [1., 0., 1., 0., 0., 0., 0.],   # 5-7
                [0., 1., 0., 1., 0., 0., 0.],   # 6-1
                [0., 1., 0., 1., 0., 0., 0.],   # 6-2
                [0., 1., 0., 1., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],   # 6-4
                [0., 1., 0., 1., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 1., 0., 0., 0.],   # 7-1
                [0., 1., 0., 1., 0., 0., 0.],   # 7-2
                [0., 1., 0., 1., 0., 0., 0.],   # 7-3
                [0., 1., 0., 1., 0., 0., 0.],   # 7-4
                [0., 1., 0., 1., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))     # 7-6
                expected_action_list.append([1, -1])       
                
                goal_relations_list.append(np.array([[0., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [1., 0., 1., 0., 0., 0., 0.],   # 2-6
                [1., 0., 1., 0., 0., 0., 0.],   # 2-7
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [1., 0., 1., 0., 0., 0., 0.],   # 3-6
                [1., 0., 1., 0., 0., 0., 0.],   # 3-7
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [1., 0., 1., 0., 0., 0., 0.],   # 4-6
                [1., 0., 1., 0., 0., 0., 0.],   # 4-7
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.],       # 5-4
                [1., 0., 1., 0., 0., 0., 0.],   # 5-6
                [1., 0., 1., 0., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 1., 0., 0., 0.],   # 6-2
                [0., 1., 0., 1., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],   # 6-4
                [0., 1., 0., 1., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 1., 0., 0., 0.],   # 7-2
                [0., 1., 0., 1., 0., 0., 0.],   # 7-3
                [0., 1., 0., 1., 0., 0., 0.],   # 7-4
                [0., 1., 0., 1., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))   # 7-6
                expected_action_list.append([2, -1])

                goal_relations_list.append(np.array([[0., 0., 1., 0., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 1., 0., 0., 0.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 1., 0., 0., 0.],   # 2-6
                [1., 0., 0., 1., 0., 0., 0.],   # 2-7
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [1., 0., 0., 1., 0., 0., 0.],   # 3-6
                [1., 0., 0., 1., 0., 0., 0.],   # 3-7
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [1., 0., 0., 1., 0., 0., 0.],   # 4-6
                [1., 0., 0., 1., 0., 0., 0.],   # 4-7
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.],       # 5-4
                [1., 0., 0., 1., 0., 0., 0.],   # 5-6
                [1., 0., 0., 1., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 1., 0., 0., 0., 0.],   # 6-2
                [0., 1., 1., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],   # 6-4
                [0., 1., 1., 0., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 1., 0., 0., 0., 0.],   # 7-2
                [0., 1., 1., 0., 0., 0., 0.],   # 7-3
                [0., 1., 1., 0., 0., 0., 0.],   # 7-4
                [0., 1., 1., 0., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))   # 7-6
                expected_action_list.append([2, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 1., 0., 0., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 1., 0., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [1., 0., 0., 0., 0., 0., 0.],   # 2-7
                [0., 0., 0., 1., 0., 0., 0.],   # 3-1
                [0., 0., 0., 1., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [1., 0., 0., 1., 0., 0., 0.],   # 3-6
                [1., 0., 0., 1., 0., 0., 0.],   # 3-7
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [1., 0., 0., 1., 0., 0., 0.],   # 4-6
                [1., 0., 0., 1., 0., 0., 0.],   # 4-7
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.],       # 5-4
                [1., 0., 0., 1., 0., 0., 0.],   # 5-6
                [1., 0., 0., 1., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 1., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],   # 6-4
                [0., 1., 1., 0., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 0., 0., 0., 0.],   # 7-2
                [0., 1., 1., 0., 0., 0., 0.],   # 7-3
                [0., 1., 1., 0., 0., 0., 0.],   # 7-4
                [0., 1., 1., 0., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))   # 7-6
                expected_action_list.append([3, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 1., 0., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [1., 0., 0., 0., 0., 0., 0.],   # 2-7
                [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                [0., 0., 1., 0., 0., 0., 0.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [1., 0., 1., 0., 0., 0., 0.],   # 3-6
                [1., 0., 1., 0., 0., 0., 0.],   # 3-7
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [1., 0., 1., 0., 0., 0., 0.],   # 4-6
                [1., 0., 1., 0., 0., 0., 0.],   # 4-7
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.],       # 5-4
                [1., 0., 1., 0., 0., 0., 0.],   # 5-6
                [1., 0., 1., 0., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 1., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],   # 6-4
                [0., 1., 0., 1., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 0., 0., 0., 0.],   # 7-2
                [0., 1., 0., 1., 0., 0., 0.],   # 7-3
                [0., 1., 0., 1., 0., 0., 0.],   # 7-4
                [0., 1., 0., 1., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))     # 7-6
                expected_action_list.append([3, -1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 1., 0., 0., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 1., 0., 0., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [1., 0., 0., 0., 0., 0., 0.],   # 2-7
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 1., 0., 0., 0., 0.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [1., 0., 0., 0., 0., 0., 0.],   # 3-6
                [1., 0., 0., 0., 0., 0., 0.],   # 3-7
                [0., 0., 0., 1., 0., 0., 0.],     # 4-1
                [0., 0., 0., 1., 0., 0., 0.],     # 4-2
                [0., 0., 0., 1., 0., 0., 0.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [1., 0., 0., 1., 0., 0., 0.],   # 4-6
                [1., 0., 0., 1., 0., 0., 0.],   # 4-7
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.],       # 5-4
                [1., 0., 0., 1., 0., 0., 0.],   # 5-6
                [1., 0., 0., 1., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 0., 0., 0., 0.],   # 6-3
                [0., 1., 1., 0., 0., 0., 0.],   # 6-4
                [0., 1., 1., 0., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 0., 0., 0., 0.],   # 7-2
                [0., 1., 0., 0., 0., 0., 0.],   # 7-3
                [0., 1., 1., 0., 0., 0., 0.],   # 7-4
                [0., 1., 1., 0., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))   # 7-6
                expected_action_list.append([4, 1])

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 1., 0., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 1., 0., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [1., 0., 0., 0., 0., 0., 0.],   # 2-7
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 1., 0., 0., 0.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [1., 0., 0., 0., 0., 0., 0.],   # 3-6
                [1., 0., 0., 0., 0., 0., 0.],   # 3-7
                [0., 0., 1., 0., 0., 0., 0.],     # 4-1
                [0., 0., 1., 0., 0., 0., 0.],     # 4-2
                [0., 0., 1., 0., 0., 0., 0.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [1., 0., 1., 0., 0., 0., 0.],   # 4-6
                [1., 0., 1., 0., 0., 0., 0.],   # 4-7
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.],       # 5-4
                [1., 0., 1., 0., 0., 0., 0.],   # 5-6
                [1., 0., 1., 0., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 0., 0., 0., 0.],   # 6-3
                [0., 1., 0., 1., 0., 0., 0.],   # 6-4
                [0., 1., 0., 1., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 0., 0., 0., 0.],   # 7-2
                [0., 1., 0., 0., 0., 0., 0.],   # 7-3
                [0., 1., 0., 1., 0., 0., 0.],   # 7-4
                [0., 1., 0., 1., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))     # 7-6
                expected_action_list.append([4, -1])   

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [0., 0., 1., 0., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [0., 0., 1., 0., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [1., 0., 0., 0., 0., 0., 0.],   # 2-7
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [0., 0., 1., 0., 0., 0., 0.],     # 3-5
                [1., 0., 0., 0., 0., 0., 0.],   # 3-6
                [1., 0., 0., 0., 0., 0., 0.],   # 3-7
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],    # 4-3
                [0., 0., 1., 0., 0., 0., 0.],      # 4-5
                [1., 0., 0., 0., 0., 0., 0.],   # 4-6
                [1., 0., 0., 0., 0., 0., 0.],   # 4-7
                [0., 0., 0., 1., 0., 0., 0.],      # 5-1
                [0., 0., 0., 1., 0., 0., 0.],      # 5-2
                [0., 0., 0., 1., 0., 0., 0.],      # 5-3
                [0., 0., 0., 1., 0., 0., 0.],       # 5-4
                [1., 0., 0., 1., 0., 0., 0.],   # 5-6
                [1., 0., 0., 1., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 0., 0., 0., 0.],   # 6-3
                [0., 1., 0., 0., 0., 0., 0.],   # 6-4
                [0., 1., 1., 0., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 0., 0., 0., 0.],   # 7-2
                [0., 1., 0., 0., 0., 0., 0.],   # 7-3
                [0., 1., 0., 0., 0., 0., 0.],   # 7-4
                [0., 1., 1., 0., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))   # 7-6
                expected_action_list.append([5, 1])    

                goal_relations_list.append(np.array([[0., 0., 0., 0., 1., 0., 1.],  # 1-2  # (1,0) for z as below as anchor, other
                [0., 0., 0., 0., 1., 0., 0.],   # 1-3
                [0., 0., 0., 0., 1., 0., 0.],   # 1-4
                [0., 0., 0., 1., 0., 0., 0.],   # 1-5
                [1., 0., 0., 0., 0., 0., 0.],   # 1-6
                [1., 0., 0., 0., 0., 0., 0.],   # 1-7
                [0., 0., 0., 0., 0., 1., 1.],   # 2-1
                [0., 0., 0., 0., 1., 0., 1.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                [0., 0., 0., 0., 1., 0., 0.],   # 2-4
                [0., 0., 0., 1., 0., 0., 0.],   # 2-5
                [1., 0., 0., 0., 0., 0., 0.],   # 2-6
                [1., 0., 0., 0., 0., 0., 0.],   # 2-7
                [0., 0., 0., 0., 0., 1., 0.],   # 3-1
                [0., 0., 0., 0., 0., 1., 1.],  # 3-2 # push object 2 minus direction correct 
                [0., 0., 0., 0., 1., 0., 1.],     # 3-4
                [0., 0., 0., 1., 0., 0., 0.],     # 3-5
                [1., 0., 0., 0., 0., 0., 0.],   # 3-6
                [1., 0., 0., 0., 0., 0., 0.],   # 3-7
                [0., 0., 0., 0., 0., 1., 0.],     # 4-1
                [0., 0., 0., 0., 0., 1., 0.],     # 4-2
                [0., 0., 0., 0., 0., 1., 1.],    # 4-3
                [0., 0., 0., 1., 0., 0., 0.],      # 4-5
                [1., 0., 0., 0., 0., 0., 0.],   # 4-6
                [1., 0., 0., 0., 0., 0., 0.],   # 4-7
                [0., 0., 1., 0., 0., 0., 0.],      # 5-1
                [0., 0., 1., 0., 0., 0., 0.],      # 5-2
                [0., 0., 1., 0., 0., 0., 0.],      # 5-3
                [0., 0., 1., 0., 0., 0., 0.],       # 5-4
                [1., 0., 1., 0., 0., 0., 0.],   # 5-6
                [1., 0., 1., 0., 0., 0., 0.],   # 5-7
                [0., 1., 0., 0., 0., 0., 0.],   # 6-1
                [0., 1., 0., 0., 0., 0., 0.],   # 6-2
                [0., 1., 0., 0., 0., 0., 0.],   # 6-3
                [0., 1., 0., 0., 0., 0., 0.],   # 6-4
                [0., 1., 0., 1., 0., 0., 0.],   # 6-5
                [0., 0., 0., 0., 1., 0., 1.],   # 6-7
                [0., 1., 0., 0., 0., 0., 0.],   # 7-1
                [0., 1., 0., 0., 0., 0., 0.],   # 7-2
                [0., 1., 0., 0., 0., 0., 0.],   # 7-3
                [0., 1., 0., 0., 0., 0., 0.],   # 7-4
                [0., 1., 0., 1., 0., 0., 0.],   # 7-5
                [0., 0., 0., 0., 0., 1., 1.]]))     # 7-6
                expected_action_list.append([5, -1])     
            
        # sub_goal_list_i = []
        # sub_goal_list_j = []
        # if self.using_sub_goal:
        #     index_i = []
        #     index_j = []
        #     for obj_i in range(self.num_nodes - 2):
        #         for consider_i in range(3):
        #             index_i.append(self.num_nodes - 2 - obj_i) # = [3,3,3]
        #             index_j.append(-(3 - consider_i)) # = [-3,-2,-1]
        #         print(index_i)
        #         print(index_j)
        #         sub_goal_list_i.append(np.array(index_i))
        #         sub_goal_list_j.append(np.array(index_j))

        sub_goal_list_i = []
        sub_goal_list_j = []
        consider_end_range = 5
        if self.real_data:
            #self.sampling_relations_number_range = [5]
            self.sampling_relations_number_range = [17]
        else:
            if self.sampling_once:
                self.sampling_relations_number_range = [5]
            else:
                self.sampling_relations_number_range = [1,5,9,13,17]
        if self.using_sub_goal and not self.evaluate_end_relations:
            
            if self.random_sampling_relations:
                sampling_relations_number_range = self.sampling_relations_number_range
                total_range = self.num_nodes * (self.num_nodes - 1) * consider_end_range # front behind contact
                for sampling_relations_number in sampling_relations_number_range:
                    #sampling_relations_number = 5
                    A = np.random.permutation(np.arange(total_range))[:sampling_relations_number]
                    print(A)
                    ## row_num = A/consider_end_range, column_num = A%consider_end_range
                    index_i = []
                    index_j = []
                    for A_i in range(sampling_relations_number):
                        index_i.append((int)(A[A_i]/consider_end_range))
                        index_j.append(- (consider_end_range - A[A_i]%consider_end_range)) ## the value will be -consider_end_range,..,-2,-1
                    sub_goal_list_i.append(np.array(index_i))
                    sub_goal_list_j.append(np.array(index_j))
            else:
                index_i = []
                index_j = []
                index_i.append(2)
                index_j.append(-3)
                # sub_goal_list_i.append(np.array(index_i))
                # sub_goal_list_j.append(np.array(index_j))

                index_i.append(2)
                index_j.append(-4)
                index_i.append(2)
                index_j.append(-5)
                sub_goal_list_i.append(np.array(index_i))
                sub_goal_list_j.append(np.array(index_j))
                # for obj_i in range(self.num_nodes - 2):
                #     for consider_i in range(1):
                #         index_i.append(self.num_nodes - 2 - obj_i) # = [3,3,3]
                #         index_j.append(-3) # = [-3,-2,-1]
                #     print(index_i)
                #     print(index_j)
                #     sub_goal_list_i.append(np.array(index_i))
                #     sub_goal_list_j.append(np.array(index_j))


        
        #self.execute_planning = False
        
        if self.evaluate_end_relations:
            num_nodes = self.num_nodes


            if self.mlp:
                data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
            else:
                data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
            batch = Batch.from_data_list([data_1]).to(device)
            outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)



            data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
            batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
            outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

            
            goal_relations_numpy = x_tensor_dict['batch_goal_relations'].cpu().detach().numpy()[0]
            predicted_relations_numpy = x_tensor_dict['batch_predicted_relations'].cpu().detach().numpy()[0]
            
            if self.mlp:
                data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
        
                data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
            else:
                data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
        
                data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, 0, None, action_torch)
            
            #data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
            
            batch1 = Batch.from_data_list([data]).to(device)
            #print(batch1)
            outs_1 = self.classif_model(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch, batch1.action)
            #print(outs['pred'].size())
            
            
            data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs_1['pred'], self.edge_emb_size, outs_1['pred_edge'], action_torch)
            batch_decoder_1 = Batch.from_data_list([data_1_decoder]).to(device)
            outs_decoder_1 = self.classif_model_decoder(batch_decoder_1.x, batch_decoder_1.edge_index, batch_decoder_1.edge_attr, batch_decoder_1.batch, batch_decoder_1.action)

            
            pred_relations = outs_decoder_1['pred_sigmoid'][:]

            
            #data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, 0, None, action_torch)
            

            batch2 = Batch.from_data_list([data_next]).to(device)
            #print(batch)
            outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
            #print(outs['pred'].size())
            
            
            data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs_2['pred'], self.edge_emb_size, outs_2['pred_edge'], action_torch)
            batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)
            outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
            


            pred_next_relations = outs_decoder_2['pred_sigmoid'][:]
            
            
            
            # print('predicted relations current step shape', pred_relations.shape)
            # print('ground truth relations current step shape', x_tensor_dict['batch_all_obj_pair_relation'][0].shape)

            predicted_current_relations_numpy = pred_relations.cpu().detach().numpy()

            gt_current_relations_numpy = x_tensor_dict['batch_all_obj_pair_relation'].cpu().detach().numpy()
            
            
            # print('predicted relations next step shape', pred_next_relations.shape)
            # print('predicted relations next step', pred_next_relations)
            # print('ground truth relations next step shape', x_tensor_dict_next['batch_all_obj_pair_relation'][0].shape)
            # print('ground truth relations next step', x_tensor_dict_next['batch_all_obj_pair_relation'][0])
            
            predicted_relations_numpy = pred_next_relations.cpu().detach().numpy()

            gt_relations_numpy = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()
            
            print('all numpy shape', goal_relations_numpy.shape, predicted_relations_numpy.shape, gt_relations_numpy.shape)

            goal_leap = np.zeros((goal_relations_numpy.shape[0], 1))
            total_sequence = 0
            for this_i in range(this_one_hot_encoding_numpy.shape[0]):
                for this_j in range(this_one_hot_encoding_numpy.shape[0]):
                    if this_j != this_i:
                        if this_one_hot_encoding_numpy[this_i] == 1 and this_one_hot_encoding_numpy[this_j] == 1:
                            goal_leap[total_sequence] = 1
                        total_sequence += 1
            
            old_goal_relations_numpy = copy.deepcopy(goal_relations_numpy)

            old_gt_relations_numpy = copy.deepcopy(gt_relations_numpy)

            goal_relations_numpy = np.zeros((predicted_relations_numpy.shape[0], predicted_relations_numpy.shape[1]))

            gt_relations_numpy = np.zeros((predicted_relations_numpy.shape[0], predicted_relations_numpy.shape[1]))

            
            
            current_i = 0
            for this_i in range(old_goal_relations_numpy.shape[0]):
                if goal_leap[this_i] == 1:
                    goal_relations_numpy[current_i] = old_goal_relations_numpy[this_i]
                    gt_relations_numpy[current_i] = old_gt_relations_numpy[this_i]
                    current_i += 1

            if self.real_data:
                print('current relations', outs_decoder['pred_sigmoid'][:])
                print('ground truth relations', gt_relations_numpy)
                print('difference between current relations and GT relations', self.dynamics_loss(outs_decoder['pred_sigmoid'][:], torch.Tensor(gt_relations_numpy).to(device)))
                print('one_hot_encoding', x_tensor_dict['batch_this_one_hot_encoding'])
            
            
            print('new all numpy shape', goal_relations_numpy.shape, predicted_relations_numpy.shape, gt_relations_numpy.shape)
            
            thredhold = 0.5
            consider_range = 0
            consider_object = goal_relations_numpy.shape[0]
            using_sub_goal = self.using_sub_goal

            if self.using_sub_goal:
                # print('using sub goal')
                # time.sleep(10)
                current_index_i = x_tensor_dict['batch_index_i'].detach().cpu().numpy()[0]
                current_index_j = x_tensor_dict['batch_index_j'].detach().cpu().numpy()[0]
                print('previous index i j')
                print(current_index_i)
                print(current_index_j)
                relation_num_index = (int)(len(current_index_i)/4)
                if self.total_num[relation_num_index, 0] < 20:
                    total_objects_planning = this_one_hot_encoding_numpy.shape[0]
                    print(total_objects_planning)
                    print(this_one_hot_encoding_numpy)
                    total_objects_on_table = 0
                    for each_one_hot_bit in this_one_hot_encoding_numpy:
                        if each_one_hot_bit == 1:
                            total_objects_on_table += 1
                    new_index_i = []
                    new_index_j = []
                    
                    for old_index_i_ind in range(len(current_index_i)):
                        old_index_i = current_index_i[old_index_i_ind]
                        corres_anchor = (int)(old_index_i/(total_objects_planning - 1)) 
                        current_remaining_value = []
                        for each_value in range(total_objects_planning):
                            if each_value != corres_anchor:
                                current_remaining_value.append(each_value)
                        # print(old_index_i)
                        # print(total_objects_planning)
                        # print(current_remaining_value)
                        corres_other = current_remaining_value[(int)(old_index_i%(total_objects_planning - 1))]
                        if self.real_data and total_objects_planning == 7 and total_objects_on_table < 7:
                            if corres_anchor >= 5:
                                corres_anchor = corres_anchor - 1
                            if corres_other >= 5:
                                corres_other = corres_other - 1  ### check this for the 7 objects case. currently only for 7 objects case.
                        
                        if this_one_hot_encoding_numpy[corres_anchor] == 1 and this_one_hot_encoding_numpy[corres_other] == 1:
                            if corres_other > corres_anchor:
                                new_index = (corres_anchor*(total_objects_on_table - 1)) + corres_other - 1
                            else:
                                new_index = (corres_anchor*(total_objects_on_table - 1)) + corres_other
                            new_index_i.append(new_index)
                            new_index_j.append((int)(current_index_j[old_index_i_ind]))
                    print('new index i j')
                    print(new_index_i)
                    print(new_index_j)
                    #time.sleep(10)
                    # consider_range = 4
                    # consider_object = self.num_nodes - 1

                    planning_pr = np.zeros((1,4)) # TP, FP, TN, FN
                    
                    planning_leap = 1
                    # for i in range(goal_relations_numpy.shape[0]):
                    #     for j in range(goal_relations_numpy.shape[1]):
                    #         if j >= consider_range and i < consider_object:
                    #             if np.abs(predicted_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                    #                 planning_leap = 0
                    for each_index in range(len(new_index_i)): # i in range(goal_relations_numpy.shape[0]):
                        i = new_index_i[each_index]
                        j = new_index_j[each_index]
                        if np.abs(predicted_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                            planning_leap = 0
                    
                    # if planning_leap == 1:
                    #     self.success_planning_num += 1
                    # else:
                    #     self.fail_planning_num += 1
                    
                    leap = 1
                    print('goal_relations_numpy', goal_relations_numpy)
                    print('gt_relations_numpy', gt_relations_numpy)
                    #print('check whether the two timesteps relations are equal', gt_current_relations_numpy - gt_relations_numpy)
                    print('range, object', [consider_range, consider_object])
                    #time.sleep(5)
                    for each_index in range(len(new_index_i)): # i in range(goal_relations_numpy.shape[0]):
                        i = new_index_i[each_index]
                        j = new_index_j[each_index]
                        if np.abs(goal_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                            leap = 0
                    
                    if planning_leap == 1 and leap == 1:
                        planning_pr[0,0] += 1
                        self.planning_pr[relation_num_index,0,0] += 1
                    elif planning_leap == 1 and leap == 0:
                        planning_pr[0,1] += 1
                        self.planning_pr[relation_num_index,0,1] += 1
                    elif planning_leap == 0 and leap == 0:
                        planning_pr[0,2] += 1
                        self.planning_pr[relation_num_index,0,2] += 1
                    elif planning_leap == 0 and leap == 1:
                        planning_pr[0,3] += 1
                        self.planning_pr[relation_num_index,0,3] += 1
                    print(planning_pr[:,-1])
                    
                    # time.sleep(10)

                    if leap == 1:
                        self.success_exe_num[relation_num_index,0] += 1
                        print('success execution')
                    else:
                        self.fail_exe_num[relation_num_index,0] += 1
                        print('failed execution')


                    self.total_num[relation_num_index,0] += 1

                    leap = 1
                    for each_index in range(len(new_index_i)): # i in range(goal_relations_numpy.shape[0]):
                        i = new_index_i[each_index]
                        j = new_index_j[each_index]
                        if np.abs(predicted_relations_numpy[i][j] - goal_relations_numpy[i][j]) > thredhold:
                            leap = 0
                    if leap == 1:
                        self.success_pred_exe_num[relation_num_index,0] += 1
                        print('success predction execution')
                    else:
                        self.fail_pred_exe_num[relation_num_index,0] += 1
                        print('failed predction execution')

                    leap = 1
                    for each_index in range(len(new_index_i)): # i in range(goal_relations_numpy.shape[0]):
                        i = new_index_i[each_index]
                        j = new_index_j[each_index]
                        if np.abs(predicted_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                            leap = 0
                    if leap == 1:
                        self.success_pred_num[relation_num_index,0] += 1
                        print('success predction')
                    else:
                        self.fail_pred_num[relation_num_index,0] += 1
                        print('failed predction')
                
            else:
                # print('using sub goal')
                # time.sleep(10)
                consider_range = 4
                consider_object = self.num_nodes - 1

                planning_pr = np.zeros((1,4)) # TP, FP, TN, FN
                
                planning_leap = 1
                for i in range(goal_relations_numpy.shape[0]):
                    for j in range(goal_relations_numpy.shape[1]):
                        if j >= consider_range and i < consider_object:
                            if np.abs(predicted_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                                planning_leap = 0
                
                # if planning_leap == 1:
                #     self.success_planning_num += 1
                # else:
                #     self.fail_planning_num += 1
                
                leap = 1
                print('goal_relations_numpy', goal_relations_numpy)
                print('gt_relations_numpy', gt_relations_numpy)
                #print('check whether the two timesteps relations are equal', gt_current_relations_numpy - gt_relations_numpy)
                print('range, object', [consider_range, consider_object])
                #time.sleep(5)
                for i in range(goal_relations_numpy.shape[0]):
                    for j in range(goal_relations_numpy.shape[1]):
                        if j >= consider_range and i < consider_object:
                            if np.abs(goal_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                                leap = 0
                
                if planning_leap == 1 and leap == 1:
                    planning_pr[0,0] += 1
                    self.planning_pr[0,0] += 1
                elif planning_leap == 1 and leap == 0:
                    planning_pr[0,1] += 1
                    self.planning_pr[0,1] += 1
                elif planning_leap == 0 and leap == 0:
                    planning_pr[0,2] += 1
                    self.planning_pr[0,2] += 1
                elif planning_leap == 0 and leap == 1:
                    planning_pr[0,3] += 1
                    self.planning_pr[0,3] += 1
                print(planning_pr)
                
                # time.sleep(10)

                if leap == 1:
                    self.success_exe_num += 1
                    print('success execution')
                else:
                    self.fail_exe_num += 1
                    print('failed execution')


                leap = 1
                for i in range(goal_relations_numpy.shape[0]):
                    for j in range(goal_relations_numpy.shape[1]):
                        if j >= consider_range and i < consider_object:
                            if np.abs(predicted_relations_numpy[i][j] - goal_relations_numpy[i][j]) > thredhold:
                                leap = 0
                if leap == 1:
                    self.success_pred_exe_num += 1
                    print('success predction execution')
                else:
                    self.fail_pred_exe_num += 1
                    print('failed predction execution')

                leap = 1
                for i in range(goal_relations_numpy.shape[0]):
                    for j in range(goal_relations_numpy.shape[1]):
                        if j >= consider_range and i < consider_object:
                            if np.abs(predicted_relations_numpy[i][j] - gt_relations_numpy[i][j]) > thredhold:
                                leap = 0
                if leap == 1:
                    self.success_pred_num += 1
                    print('success predction')
                else:
                    self.fail_pred_num += 1
                    print('failed predction')
            
            each_relations_success_list = np.zeros((predicted_relations_numpy.shape[1], 1))
            each_relations_success_rate_list = np.zeros((predicted_relations_numpy.shape[1], 1))
            for i in range(goal_relations_numpy.shape[0]):
                for j in range(goal_relations_numpy.shape[1]):
                    if np.abs(predicted_relations_numpy[i][j] - gt_relations_numpy[i][j]) < thredhold:
                        each_relations_success_list[j,0] += 1
            


            #print(each_relations_success_list)
            each_relations_success_rate_list = each_relations_success_list/predicted_relations_numpy.shape[0]
            #print(each_relations_success_rate_list)
            
            each_relations_pr = np.zeros((predicted_relations_numpy.shape[1], 4)) # TP, FP, TN, FN
            precision_relations = np.zeros((predicted_relations_numpy.shape[1], 1)) # TP, FP, TN, FN
            recall_relations = np.zeros((predicted_relations_numpy.shape[1], 1)) # TP, FP, TN, FN
            f_relations = np.zeros((predicted_relations_numpy.shape[1], 1)) # TP, FP, TN, FN
            if self.consider_end_relations:
                print('evaluate end relations')
                for i in range(goal_relations_numpy.shape[0]):
                    for j in range(goal_relations_numpy.shape[1]):
                        if predicted_relations_numpy[i][j] > thredhold and gt_relations_numpy[i][j] > thredhold: # TP
                            each_relations_pr[j,0] += 1
                            self.detection_pr[j,0] += 1
                        elif predicted_relations_numpy[i][j] > thredhold and gt_relations_numpy[i][j] < thredhold: # FP
                            each_relations_pr[j,1] += 1
                            self.detection_pr[j,1] += 1
                        elif predicted_relations_numpy[i][j] < thredhold and gt_relations_numpy[i][j] < thredhold: # TN
                            each_relations_pr[j,2] += 1
                            self.detection_pr[j,2] += 1
                        elif predicted_relations_numpy[i][j] < thredhold and gt_relations_numpy[i][j] > thredhold: # FN
                            each_relations_pr[j,3] += 1
                            self.detection_pr[j,3] += 1
            
            if self.consider_current_relations:
                print('evaluate current relations')
                for i in range(goal_relations_numpy.shape[0]):
                    for j in range(goal_relations_numpy.shape[1]):
                        if predicted_current_relations_numpy[i][j] > thredhold and gt_current_relations_numpy[i][j] > thredhold: # TP
                            each_relations_pr[j,0] += 1
                            self.detection_pr[j,0] += 1
                        elif predicted_current_relations_numpy[i][j] > thredhold and gt_current_relations_numpy[i][j] < thredhold: # FP
                            each_relations_pr[j,1] += 1
                            self.detection_pr[j,1] += 1
                        elif predicted_current_relations_numpy[i][j] < thredhold and gt_current_relations_numpy[i][j] < thredhold: # TN
                            each_relations_pr[j,2] += 1
                            self.detection_pr[j,2] += 1
                        elif predicted_current_relations_numpy[i][j] < thredhold and gt_current_relations_numpy[i][j] > thredhold: # FN
                            each_relations_pr[j,3] += 1
                            self.detection_pr[j,3] += 1
            
            
            for i in range(each_relations_pr.shape[0]):
                precision_relations[i, 0] = each_relations_pr[i, 0]/(each_relations_pr[i, 0] + each_relations_pr[i, 1])
                recall_relations[i, 0] = each_relations_pr[i, 0]/(each_relations_pr[i, 0] + each_relations_pr[i, 3])
                f_relations[i, 0] = (2*precision_relations[i, 0]*recall_relations[i, 0])/(precision_relations[i, 0] + recall_relations[i, 0])
            
            print(each_relations_pr[:, -1])  # if there are some nans, use lots of samples then this problem will not be a problem anymore. 
            # print(precision_relations)
            # print(recall_relations)
            # print(f_relations)
            #time.sleep(10)
            
            if True:
                x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'][0]
                x_tensor_dict_next['batch_all_obj_pair_relation'] = x_tensor_dict_next['batch_all_obj_pair_relation'][0]#print('edge shape', outs['edge_embed'].shape)


            planning_leap = 0
            if False:
                leap  = success_num
                planning_leap = planning_success_num
            else:
                leap = 0
                planning_leap = 0
            
            batch_result_dict['success_exe_num'] = self.success_exe_num
            batch_result_dict['fail_exe_num'] = self.fail_exe_num

            batch_result_dict['success_planning_num'] = self.success_planning_num
            batch_result_dict['fail_planning_num'] = self.fail_planning_num

            batch_result_dict['planning_pr'] = self.planning_pr

            batch_result_dict['detection_pr'] = self.detection_pr

            batch_result_dict['success_pred_exe_num'] = self.success_pred_exe_num
            batch_result_dict['fail_pred_exe_num'] = self.fail_pred_exe_num

            batch_result_dict['success_pred_num'] = self.success_pred_num
            batch_result_dict['fail_pred_num'] = self.fail_pred_num


            
            batch_result_dict['total_num'] = self.total_num

            if self.using_sub_goal:
                self.fail_mp_num = 0
                self.point_cloud_not_complete = 0
                batch_result_dict['point_cloud_not_complete'] = self.point_cloud_not_complete
                batch_result_dict['fail_mp_num'] = 0
            else:
                batch_result_dict['fail_mp_num'] = self.fail_mp_num
                batch_result_dict['point_cloud_not_complete'] = self.total_num - self.success_exe_num - self.fail_exe_num - self.fail_mp_num
            print("sucess_exe, fail_exe, fail_mp, total_num", self.success_exe_num, self.fail_exe_num, self.fail_mp_num, self.total_num)
            
        #Testing/Planning
        elif not train:
            if self.pushing and not self.pick_place:
                if graph_latent:
                    if True:
                        if edge_classifier:
                            
                            if self.graph_search:

                                if cem_planning:
                                    total_num = 0
                                    total_succes_num = 0
                                    

                                    if not self.using_sub_goal:
                                        sub_goal_list_i.append(1)
                                        sub_goal_list_j.append(1)
                                    # if self.using_multi_step:
                                    #     sub_goal_list_i = []
                                    #     sub_goal_list_j = []

                                    
                                    if True:
                                        all_goal_list = []
                                        all_index_i_list = []
                                        all_index_j_list = []
                                        
                                        each_goal = 5
                                        index_i = 2
                                        index_j = 3

                                        all_goal_list.append(each_goal)
                                        all_index_i_list.append(index_i)
                                        all_index_j_list.append(index_j)

                                        each_goal = 5
                                        index_i = 1
                                        index_j = 3

                                        all_goal_list.append(each_goal)
                                        all_index_i_list.append(index_i)
                                        all_index_j_list.append(index_j)
                                    
                                        each_goal = 1
                                        index_i = 1
                                        index_j = 2

                                        all_goal_list.append(each_goal)
                                        all_index_i_list.append(index_i)
                                        all_index_j_list.append(index_j)
                                        
                                        subgoal_index_list_total = [[0], [1], [2], [0,1], [0,2], [1,2]]
                                        # subgoal_index_list_total = [[0,1]]
                                        for each_subgoal_index in range(len(subgoal_index_list_total)):
                                            this_round_sampled_action = []
                                            for each_index in range(2):
                                                if each_index == 0:
                                                    subgoal_index_list = subgoal_index_list_total[each_subgoal_index]
                                                else:
                                                    subgoal_index_list = [0,1,2]
                                                    for each_ii in subgoal_index_list_total[each_subgoal_index]:
                                                        subgoal_index_list.remove(each_ii)

                                                print('subgoal index list', subgoal_index_list)
                                                
                                                if self.all_gt_sigmoid:
                                                    success_num = 0
                                                    total_num += 1
                                                    planning_success_num = 0
                                                    planning_total_num = 0
                                                    planning_threshold = threshold

                                                    for test_iter in range(1):
                                                        print('enter')
                                                        plannning_sequence = 1
                                                        sample_sequence = 1
                                                        num_nodes = self.num_nodes


                                                        if self.mlp:
                                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                                                            # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
                                                        else:
                                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                                        if each_index == 0:
                                                            batch = Batch.from_data_list([data_1]).to(device)
                                                            outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)

                                                            this_time_step_embed = outs['current_embed']
                                                            this_time_step_edge_embed = outs['edge_embed']



                                                 
                                                        
                                                        if self.execute_planning:
                                                            min_cost = 1e5
                                                            loss_list = []
                                                            all_action_list = []
                                                            
                                                            for obj_mov in range(self.num_nodes):
                                                                if self.seperate_range:
                                                                    middle_point = [[1,0.3], [0,0.3]]
                                                                else:
                                                                    middle_point = [[0.5,0.6]]
                                                                for current_middle_point in middle_point:
                                                                    print('mov obj', obj_mov)
                                                                    action_selections = 100
                                                                    action_mu = np.zeros((action_selections, 1, 2))
                                                                    action_sigma = np.ones((action_selections, 1, 2))
                                                                    
                                                                    
                                                                    for i_iter in range(5):
                                                                        action_noise = np.zeros((action_selections, 1, 2))
                                                                        action_noise[:,:,0] = (np.random.rand(action_selections, 1) - 0.5) * 0.1
                                                                        action_noise[:,:,1] = (np.random.rand(action_selections, 1) - current_middle_point[0]) * current_middle_point[1]
                                                                        #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                                                                        act = action_mu + action_noise*action_sigma
                                                                        costs = []
                                                                        for j in range(action_selections):
                                                                            action_numpy = np.zeros((num_nodes, 3))
                                                                            action_numpy[obj_mov][0] = act[j, 0, 0]
                                                                            action_numpy[obj_mov][1] = act[j, 0, 1]
                                                                            action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                                            
                                                                            if self.set_max:
                                                                                action = np.zeros((num_nodes, self.max_objects + 3))
                                                                            else:
                                                                                action = np.zeros((num_nodes, num_nodes + 3))
                                                                            for i in range(action.shape[0]):
                                                                                action[i][obj_mov] = 1
                                                                                action[i][-3:] = action_numpy[obj_mov]
                                                                            
                                                                            sample_action = torch.Tensor(action).to(device)
                                                                            
                                                                            # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                                            # for i in range(action_1.shape[0]):
                                                                            #     action_1[i][obj_mov] = 1
                                                                            #     action_1[i][3:] = action_numpy[obj_mov]
                                                                            # sample_action = torch.Tensor(action_1)
                                                                            #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                                            # if(_ == 0):
                                                                            #     sample_action = action
                                                                            this_sequence = []
                                                                            this_sequence.append(sample_action)
                                                                            loss_func = nn.MSELoss()
                                                                            test_loss = 0
                                                                            current_latent = this_time_step_embed 
                                                                            egde_latent = this_time_step_edge_embed 
                                                                            #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                            for seq in range(len(this_sequence)):
                                                                                #print([current_latent, this_sequence[seq]])
                                                                                if self.use_graph_dynamics:
                                                                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                                else:
                                                                                    current_action = self.classif_model.action_emb(this_sequence[seq])

                                                                                graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                                current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                            for seq in range(len(this_sequence)):
                                                                                #print([current_latent, this_sequence[seq]])
                                                                                #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                                edge_num = egde_latent.shape[0]
                                                                                edge_action_list = []
                                                                                if self.use_graph_dynamics:
                                                                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                                else:
                                                                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                                for _ in range(edge_num):
                                                                                    edge_action_list.append(current_action[0])
                                                                                edge_action = torch.stack(edge_action_list)
                                                                                graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                                egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                            #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                            #print(egde_latent.shape)
                                                                            # print(current_latent)
                                                                            # print(egde_latent)
                                                                            # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                            data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                    
                                                                            batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                            outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                            
                                                                            
                                                                            
                                                                            if self.all_gt_sigmoid:
                                                                                for each_sub_goal_index in subgoal_index_list:
                                                                                    index_i = all_index_i_list[each_sub_goal_index]
                                                                                    index_j = all_index_j_list[each_sub_goal_index]
                                                                                    current_goal = all_goal_list[each_sub_goal_index]
                                                                                    
                                                                                    x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[current_goal]).to(device)
                                                                                    test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                                    
                                                                            
                                                                            costs.append(test_loss.detach().cpu().numpy())
                                                                            # if(test_loss.detach().cpu().numpy() < min_cost):
                                                                            #     min_action = this_sequence
                                                                            #     min_cost = test_loss.detach().cpu().numpy()
                                                                    
                                                                            #     costs.append(test_loss)

                                                                        index = np.argsort(costs)
                                                                        elite = act[index,:,:]
                                                                        elite = elite[:3, :, :]
                                                                            # print('elite')
                                                                            # print(elite)
                                                                        act_mu = elite.mean(axis = 0)
                                                                        act_sigma = elite.std(axis = 0)
                                                                        print([act_mu, act_sigma])
                                                                        # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                                        #     break
                                                                        #print(act_sigma)
                                                                    # print('find_actions')
                                                                    # print(act_mu)
                                                                    chosen_action = act_mu
                                                                    action_numpy = np.zeros((num_nodes, 3))
                                                                    action_numpy[obj_mov][0] = chosen_action[0, 0]
                                                                    action_numpy[obj_mov][1] = chosen_action[0, 1]
                                                                    action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)

                                                                    action_numpy_variance = np.zeros((num_nodes, 2))
                                                                    action_numpy_variance[obj_mov][0] = act_sigma[0, 0]
                                                                    action_numpy_variance[obj_mov][1] = act_sigma[0, 1]
                                                                    if self.set_max:
                                                                        action = np.zeros((num_nodes, self.max_objects + 3))
                                                                    else:
                                                                        action = np.zeros((num_nodes, num_nodes + 3))
                                                                    for i in range(action.shape[0]):
                                                                        action[i][obj_mov] = 1
                                                                        action[i][-3:] = action_numpy[obj_mov]
                                                                            
                                                                    sample_action = torch.Tensor(action).to(device)
                                                                    # if(_ == 0):
                                                                    #     sample_action = action
                                                                    this_sequence = []
                                                                    this_sequence.append(sample_action)

                                                                    this_sequence_variance = []
                                                                    this_sequence_variance.append(action_numpy_variance)
                                                                    if True:
                                                                        loss_func = nn.MSELoss()
                                                                        test_loss = 0
                                                                        current_latent = outs['current_embed']
                                                                        egde_latent = outs['edge_embed']
                                                                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                            edge_num = egde_latent.shape[0]
                                                                            edge_action_list = []
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            for _ in range(edge_num):
                                                                                edge_action_list.append(current_action[0])
                                                                            edge_action = torch.stack(edge_action_list)
                                                                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                        #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                        #print(egde_latent.shape)
                                                                        # print(current_latent)
                                                                        # print(egde_latent)
                                                                        #outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                    
                                                                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                        
                                                                        
                                                                        
                                                                        if self.all_gt_sigmoid:
                                                                            for each_sub_goal_index in subgoal_index_list:
                                                                                
                                                                                index_i = all_index_i_list[each_sub_goal_index]
                                                                                index_j = all_index_j_list[each_sub_goal_index]
                                                                                current_goal = all_goal_list[each_sub_goal_index]
                                                                                
                                                                                print([index_i, index_j, current_goal])
                                                                                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[current_goal]).to(device)
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                                print(outs_decoder_2['pred_sigmoid'][index_i, index_j])
                                                                                print(x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                        
                                                                        #sample_list.append(outs_edge['pred_edge'])
                                                                        print('test_loss', test_loss)
                                                                        
                                                                        loss_list.append(test_loss)
                                                                        if(test_loss.detach().cpu().numpy() < min_cost):
                                                                            min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                                                                            min_action = this_sequence
                                                                            min_action_variance = this_sequence_variance
                                                                            min_pose = outs_decoder_2['pred'][:, :]
                                                                            min_cost = test_loss.detach().cpu().numpy()
                                                                            this_time_step_embed = current_latent
                                                                            this_time_step_edge_embed = egde_latent

                                                            pred_relations = min_prediction.cpu().detach().numpy()
                                                            goal_relations = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()

                                                            planning_success_num = 1
                                                            for obj_id in range(pred_relations.shape[0]):
                                                                for relation_id in range(pred_relations.shape[1]):
                                                                    if goal_relations[obj_id][relation_id] == 1:
                                                                        if pred_relations[obj_id][relation_id] < planning_threshold:
                                                                            planning_success_num = 0
                                                                    elif goal_relations[obj_id][relation_id] == 0:
                                                                        if pred_relations[obj_id][relation_id] > 1 - planning_threshold:
                                                                            planning_success_num = 0

                                                            print('pred_relations', pred_relations)
                                                            print('goal_relations shape', goal_relations.shape)
                                                            print('goal_relations', goal_relations)
                                                            print('planning_success_num', planning_success_num)
                                                            node_pose_numpy = node_pose.detach().cpu().numpy()
                                                            change_id_leap = 0
                                                            if True: #for seq in range(len(min_action)):
                                                                this_seq_numpy = min_action[0].cpu().numpy()
                                                                change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                                    #print(change_id)
                                                                if change_id == 0:
                                                                    change_id_leap = 1
                                                                    node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                                                #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                                                #     #if(this_seq_numpy[0])
                                                                # all_node_pose_list.append(node_pose_numpy)
                                                                # node_pose = torch.Tensor(node_pose_numpy)
                                                                # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                                            node_pose_numpy_goal = node_pose_goal.detach().cpu().numpy()
                                                            choose_list = [0,-1]
                                                            # print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                                                                 
                                                            print(min_action)
                                                            print(min_action_variance)
                                                            #min_action_numpy
                                                            print(action_torch)
                                                            #print()
                                                            print(min_cost)
                                                            this_round_sampled_action.append(min_action)
                                                            if each_index == 1:
                                                                pred_relations_numpy = copy.deepcopy(pred_relations)

                                                                # time.sleep(10)
                                                                leap = 1
                                                                for each_sub_goal_index in subgoal_index_list:
                                                                    index_i = all_index_i_list[each_sub_goal_index]
                                                                    index_j = all_index_j_list[each_sub_goal_index]
                                                                    current_goal = all_goal_list[each_sub_goal_index]

                                                                    goal_relations_numpy = goal_relations_list[current_goal]
                                                                    # print(pred_relations_numpy[index_i][index_j])
                                                                    # print(goal_relations_numpy[index_i][index_j])
                                                                    # print('leap', leap)
                                                                    # print(np.abs(pred_relations_numpy[index_i][index_j] - goal_relations_numpy[index_i][index_j]))
                                                                    if np.abs(pred_relations_numpy[index_i][index_j] - goal_relations_numpy[index_i][index_j]) > 0.5:
                                                                        leap = 0
                                                                # print('leap', leap)
                                                                if leap == 1:
                                                                    print('sucess')
                                                                    print(this_round_sampled_action)
                                                                    time.sleep(100)
                                                                    
                                                            # if change_id_leap == 1:
                                                            #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                                            #     print(goal_loss)
                                                            #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                                            #         success_num += 1
                                                            min_action_numpy = min_action[0].cpu().numpy()
                                                            min_action_variance_numpy = min_action_variance[0]
                                                            action_numpy = action_torch.cpu().numpy()

                                                            # for node_pose_iter in range(node_pose_numpy.shape[0]):
                                                            #     self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                                                            # for action_iter in range(1):
                                                            #     self.action_list.append(min_action_numpy[action_iter])
                                                            # for goal_relation_i in range(goal_relation.shape[0]):
                                                            #     for goal_relation_j in range(goal_relation.shape[1]):
                                                            #         self.goal_relation_list.append(goal_relation[goal_relation_i][goal_relation_j])
                                                            self.node_pose_list.append(node_pose_numpy)
                                                            self.action_list.append(min_action_numpy)
                                                            self.action_variance_list.append(min_action_variance_numpy)
                                                            self.goal_relation_list.append(goal_relations)
                                                            self.gt_pose_list.append(self.gt_pose[0])
                                                            self.pc_center_list.append(self.pc_center[0])
                                                            self.gt_orientation_list.append(self.gt_orientation[0])
                                                            self.gt_extents_list.append(self.gt_extents[0])
                                                            self.gt_extents_range_list.append(self.gt_extents_range[0])
                                                            self.predicted_relations.append(pred_relations)
                                                            self.all_index_i_list.append(index_i)
                                                            self.all_index_j_list.append(index_j)
                                                            
                                                            
                                                            
                                                                                                                       
                                                            # simplied version of planned action success rate for pushing task
                                                            success_num = 1
                                                            if min_action_numpy[0][expected_action_list[each_goal][0] - 1] != 1:
                                                                success_num = 0
                                                            if min_action_numpy[0][-2]*expected_action_list[each_goal][1] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                                                success_num = 0
                                                            total_succes_num += success_num
                                                    
                                                    if self.execute_planning:
                                                        print(total_succes_num)
                                                        print(total_succes_num/total_num)               
                                
                            elif self.using_multi_step_latent:
                                if cem_planning:
                                    total_num = 0
                                    total_succes_num = 0
                                    

                                    if not self.using_sub_goal:
                                        sub_goal_list_i.append(1)
                                        sub_goal_list_j.append(1)
                                    # if self.using_multi_step:
                                    #     sub_goal_list_i = []
                                    #     sub_goal_list_j = []

                                    
                                    if True:
                                        for each_index in range(1):
                                            all_goal_list = []
                                            all_index_i_list = []
                                            all_index_j_list = []
                                            
                                            each_goal = 5
                                            index_i = [2,1]
                                            index_j = [3,3]

                                            all_goal_list.append(each_goal)
                                            all_index_i_list.append(index_i)
                                            all_index_j_list.append(index_j)
                                      
                                            each_goal = 1
                                            index_i = [1]
                                            index_j = [2]

                                            all_goal_list.append(each_goal)
                                            all_index_i_list.append(index_i)
                                            all_index_j_list.append(index_j)

                                            #print('index', [index_i, index_j])
                                            if self.manual_relations:
                                                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[each_goal]).to(device)
                                            
                                            total_num = 0
                                            total_succes_num = 0
                                            success_num = 0
                                            total_num += 1
                                            planning_success_num = 0
                                            planning_total_num = 0
                                            planning_threshold = threshold
                                            for test_iter in range(total_num):
                                                print('enter')
                                                plannning_sequence = 2
                                                sample_sequence = 2
                                                num_nodes = self.num_nodes
                                                if self.mlp:
                                                    data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                                                    # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
                                                else:
                                                    data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                                batch = Batch.from_data_list([data_1]).to(device)
                                                outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)



                                                data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
                                                batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                                                outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

                                                min_cost = 1e5
                                                all_action_sequence_list = []
                                                # all_node_pose_list = []
                                                # all_node_pose_list.append(node_pose.cpu().numpy())
                                                #print('init node pose', node_pose)
                                                all_execute_action = []
                                                for test_seq in range(plannning_sequence):
                                                    loss_list = []
                                                    sample_list = []
                                                    for _ in range(500):
                                                        this_sequence = []
                                                        sample_sequence = sample_sequence - test_seq
                                                        this_sequence = []
                                                        for seq in range(sample_sequence):
                                                            obj_mov = np.random.randint(num_nodes)
                                                            action_numpy = np.zeros((num_nodes, 3))
                                                            action_numpy[obj_mov][0] = np.random.uniform(-0.05,0.05)
                                                            action_numpy[obj_mov][1] = np.random.uniform(-0.3,0.3)
                                                            action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                            
                                                            if self.set_max:
                                                                action = np.zeros((num_nodes, self.max_objects + 3))
                                                            else:
                                                                action = np.zeros((num_nodes, num_nodes + 3))
                                                            for i in range(action.shape[0]):
                                                                action[i][obj_mov] = 1
                                                                action[i][-3:] = action_numpy[obj_mov]
                                                            
                                                            sample_action = torch.Tensor(action).to(device)

                                                            
                                                            # obj_mov = np.random.randint(num_nodes)
                                                            # action_numpy = np.zeros((num_nodes, 3))
                                                            # action_numpy[obj_mov][0] = np.random.uniform(-1,1)
                                                            # action_numpy[obj_mov][1] = np.random.uniform(-1,1)
                                                            # # z_choice_list = [0,0.10,0.18]
                                                            # # action_numpy[obj_mov][2] = z_choice_list[np.random.randint(len(z_choice_list))]
                                                            # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                            # for i in range(action_1.shape[0]):
                                                            #     action_1[i][obj_mov] = 1
                                                            #     action_1[i][3:] = action_numpy[obj_mov]
                                                            # sample_action = torch.Tensor(action_1)
                                                            # sample_action = sample_action.to(device)
                                                            # if _ == 0:
                                                            #     sample_action = action_torch
                                                        
                                                            this_sequence.append(sample_action)
                                                        loss_func = nn.MSELoss()
                                                        test_loss = 0
                                                        current_latent = outs['current_embed']
                                                        egde_latent = outs['edge_embed']
                                                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                        for seq in range(len(this_sequence)):
                                                            #print([current_latent, this_sequence[seq]])
                                                            if self.use_graph_dynamics:
                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                            else:
                                                                current_action = self.classif_model.action_emb(this_sequence[seq])

                                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                        for seq in range(len(this_sequence)):
                                                            #print([current_latent, this_sequence[seq]])
                                                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                            edge_num = egde_latent.shape[0]
                                                            edge_action_list = []
                                                            if self.use_graph_dynamics:
                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                            else:
                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                            for _ in range(edge_num):
                                                                edge_action_list.append(current_action[0])
                                                            edge_action = torch.stack(edge_action_list)
                                                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                        #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                        #print(egde_latent.shape)
                                                        # print(current_latent)
                                                        # print(egde_latent)
                                                        # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                
                                                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                        
                                                        
                                                        
                                                
                                                        
                                                        if self.all_gt_sigmoid:
                                                            for each_index in range(len(all_index_i_list)):
                                                                index_i = all_index_i_list[each_index]
                                                                index_j = all_index_j_list[each_index]
                                                                current_goal = all_goal_list[each_index]
                                                                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[current_goal]).to(device)
                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                            #test_loss += self.bce_loss(torch.Tensor(all_relations).to(device), x_tensor_dict_next['batch_all_obj_pair_relation'])
                                        
                                                        

                                                        loss_list.append(test_loss)
                                                        if(test_loss.detach().cpu().numpy() < min_cost):
                                                            min_action = this_sequence
                                                            min_cost = test_loss.detach().cpu().numpy()
                                                    execute_action = min_action[0]
                                                    all_execute_action.append(execute_action)

                                                    #print(loss_list)
                                                    node_pose_numpy = node_pose.detach().cpu().numpy()
                                                    
                                                print(min_action)
                                                print(action_torch)
                                                #print()
                                                print(min_cost)

                                                success_num = 1
                                                # if min_action_numpy[0][expected_action_list[each_goal][0] - 1] != 1:
                                                #     success_num = 0
                                                # if min_action_numpy[0][-2]*expected_action_list[each_goal][1] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                                #     success_num = 0
                                                total_succes_num = success_num
                                                planning_success_num = 0
                                                

                                                #print(node_pose_goal)
                                                #print(edge_feature_2)
                                                #print(generate_edge_embed_list)
                                                #print(all_action_sequence_list)
                                                #print(all_node_pose_list)
                                                #print(loss_func(edge_feature_2, torch.stack(generate_edge_embed_list)))
                                                # goal_loss = loss_func(torch.stack(scene_emb_list_next[0][:12]), torch.stack(generate_edge_embed_list))
                                                # print(goal_loss)
                                                # if(goal_loss.detach().cpu().numpy() < 1e-3):
                                                #     success_num += 1
                                                # else:
                                                #     print(loss_list)
                                                #     print(sample_list)
                                                # print(loss_list)
                                                # print(sample_list)
                                                #print(self.dynamics_loss(node_pose[:,:6], node_pose_goal[:,:6]))
                                            print(success_num)
                                            print(success_num/total_num)

                            elif self.using_multi_step_statistics:
                                if cem_planning:
                                    total_num = 0
                                    total_succes_num = 0
                                    

                                    # 0 (2,-1)
                                    # 1 (2,1)
                                    # 2 (3,1)
                                    # 3 (3,-1)
                                    # 4 (4,1)
                                    # 5 (4,-1)
                                    
                                    

                                    if not self.using_sub_goal:
                                        sub_goal_list_i.append(1)
                                        sub_goal_list_j.append(1)
                                    # if self.using_multi_step:
                                    #     sub_goal_list_i = []
                                    #     sub_goal_list_j = []

                                    
                                    if True:
                                        if self.total_sub_step == 2:
                                            each_goal_all_list = [[2,0],
                                            [3,1],
                                            [4,0],
                                            [5,1],
                                            [4,3],
                                            [5,2]]
                                            each_index_all_list = [[[[1,0], [2, 2]], [[0], [3]]],
                                            [[[1,0], [3,3]], [[0], [2]]],
                                            [[[2,1], [2,2]], [[0], [3]]],
                                            [[[2,1], [3,3]], [[0], [2]]],
                                            [[[2,1], [2,2]], [[1], [3]]],
                                            [[[2,1], [3,3]], [[1], [2]]]
                                            ]
                                        elif self.total_sub_step == 3:
                                            each_goal_all_list = [[4,3,1],
                                            [5,2,0]]
                                            each_index_all_list = [[[[2,1], [2,2]], [[1,0], [3,3]], [[0], [2]]],
                                            [[[2,1], [3,3]], [[1,0], [2,2]], [[0], [3]]]]
                                        this_sample_id = np.random.randint(len(each_goal_all_list))
                                        sampled_goal = each_goal_all_list[this_sample_id]
                                        sampled_index = each_index_all_list[this_sample_id]
                                        for each_index in range(self.total_sub_step):
                                            if True:
                                                each_goal = sampled_goal[each_index]
                                                index_i = sampled_index[each_index][0]
                                                index_j = sampled_index[each_index][1]
                                                # if each_index == 0:
                                                #     # each_goal = 3
                                                #     # index_i = [1,0]
                                                #     # index_j = [3,4]
                                                #     each_goal = 5
                                                #     index_i = [2,1]
                                                #     index_j = [3,3]
                                                # else:
                                                #     # each_goal = 1
                                                #     # index_i = [0]
                                                #     # index_j = [2]
                                                #     each_goal = 1
                                                #     index_i = [1]
                                                #     index_j = [2]
                                            print('index', [index_i, index_j])
                                            if self.manual_relations:
                                                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[each_goal]).to(device)
                                            
                                            if self.all_gt_sigmoid:
                                                success_num = 0
                                                total_num += 1
                                                planning_success_num = 0
                                                planning_total_num = 0
                                                planning_threshold = threshold

                                                for test_iter in range(1):
                                                    print('enter')
                                                    plannning_sequence = 1
                                                    sample_sequence = 1
                                                    num_nodes = self.num_nodes


                                                    if self.mlp:
                                                        data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                                                        # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
                                                    else:
                                                        if self.test_next_step:
                                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, 0, None, action_torch)
                                                        else:
                                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                                    if each_index == 0:
                                                        batch = Batch.from_data_list([data_1]).to(device)
                                                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)

                                                        this_time_step_embed = outs['current_embed']
                                                        this_time_step_edge_embed = outs['edge_embed']



                                                    # data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
                                                    # batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                                                    # outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

                                                    # if self.real_data:
                                                    #     print('current relations', outs_decoder['pred_sigmoid'][:])
                                                    #     print('ground truth relations', x_tensor_dict['batch_all_obj_pair_relation'])
                                                    #     print('difference between current relations and GT relations', self.dynamics_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']))

                                                    

                                                    # data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), action_torch)
                                                    # data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, x_tensor_dict_next['batch_all_obj_pair_relation'], action_torch)
                                                    # batch = Batch.from_data_list([data]).to(device)
                                                    # outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                                    # batch2 = Batch.from_data_list([data_next]).to(device)
                                                    # outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                                    
                                                    
                                                    if self.execute_planning:
                                                        min_cost = 1e5
                                                        loss_list = []
                                                        all_action_list = []
                                                        
                                                        print('actual action', expected_action_list[each_goal])
                                                        
                                                        for obj_mov in range(self.num_nodes):
                                                            if self.seperate_range:
                                                                middle_point = [[1,0.3], [0,0.3]]
                                                            else:
                                                                middle_point = [[0.5,0.6]]
                                                            for current_middle_point in middle_point:
                                                                print('mov obj', obj_mov)
                                                                action_selections = 500
                                                                action_mu = np.zeros((action_selections, 1, 2))
                                                                action_sigma = np.ones((action_selections, 1, 2))
                                                                
                                                                
                                                                for i_iter in range(5):
                                                                    action_noise = np.zeros((action_selections, 1, 2))
                                                                    action_noise[:,:,0] = (np.random.rand(action_selections, 1) - 0.5) * 0.1
                                                                    action_noise[:,:,1] = (np.random.rand(action_selections, 1) - current_middle_point[0]) * current_middle_point[1]
                                                                    #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                                                                    act = action_mu + action_noise*action_sigma
                                                                    costs = []
                                                                    for j in range(action_selections):
                                                                        action_numpy = np.zeros((num_nodes, 3))
                                                                        action_numpy[obj_mov][0] = act[j, 0, 0]
                                                                        action_numpy[obj_mov][1] = act[j, 0, 1]
                                                                        action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                                        
                                                                        if self.set_max:
                                                                            action = np.zeros((num_nodes, self.max_objects + 3))
                                                                        else:
                                                                            action = np.zeros((num_nodes, num_nodes + 3))
                                                                        for i in range(action.shape[0]):
                                                                            action[i][obj_mov] = 1
                                                                            action[i][-3:] = action_numpy[obj_mov]
                                                                        
                                                                        sample_action = torch.Tensor(action).to(device)
                                                                        
                                                                        # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                                        # for i in range(action_1.shape[0]):
                                                                        #     action_1[i][obj_mov] = 1
                                                                        #     action_1[i][3:] = action_numpy[obj_mov]
                                                                        # sample_action = torch.Tensor(action_1)
                                                                        #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                                        # if(_ == 0):
                                                                        #     sample_action = action
                                                                        this_sequence = []
                                                                        this_sequence.append(sample_action)
                                                                        loss_func = nn.MSELoss()
                                                                        test_loss = 0
                                                                        current_latent = this_time_step_embed 
                                                                        egde_latent = this_time_step_edge_embed 
                                                                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])

                                                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                            edge_num = egde_latent.shape[0]
                                                                            edge_action_list = []
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            for _ in range(edge_num):
                                                                                edge_action_list.append(current_action[0])
                                                                            edge_action = torch.stack(edge_action_list)
                                                                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                        #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                        #print(egde_latent.shape)
                                                                        # print(current_latent)
                                                                        # print(egde_latent)
                                                                        # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                
                                                                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                        
                                                                        
                                                                        
                                                                        if self.all_gt_sigmoid:
                                                                            if self.using_sub_goal:
                                                                                #print(x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j].shape)
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                            else:
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                        
                                                                        costs.append(test_loss.detach().cpu().numpy())
                                                                        # if(test_loss.detach().cpu().numpy() < min_cost):
                                                                        #     min_action = this_sequence
                                                                        #     min_cost = test_loss.detach().cpu().numpy()
                                                                
                                                                        #     costs.append(test_loss)

                                                                    index = np.argsort(costs)
                                                                    elite = act[index,:,:]
                                                                    elite = elite[:3, :, :]
                                                                        # print('elite')
                                                                        # print(elite)
                                                                    act_mu = elite.mean(axis = 0)
                                                                    act_sigma = elite.std(axis = 0)
                                                                    print([act_mu, act_sigma])
                                                                    # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                                    #     break
                                                                    #print(act_sigma)
                                                                # print('find_actions')
                                                                # print(act_mu)
                                                                chosen_action = act_mu
                                                                action_numpy = np.zeros((num_nodes, 3))
                                                                action_numpy[obj_mov][0] = chosen_action[0, 0]
                                                                action_numpy[obj_mov][1] = chosen_action[0, 1]
                                                                action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)

                                                                action_numpy_variance = np.zeros((num_nodes, 2))
                                                                action_numpy_variance[obj_mov][0] = act_sigma[0, 0]
                                                                action_numpy_variance[obj_mov][1] = act_sigma[0, 1]
                                                                if self.set_max:
                                                                    action = np.zeros((num_nodes, self.max_objects + 3))
                                                                else:
                                                                    action = np.zeros((num_nodes, num_nodes + 3))
                                                                for i in range(action.shape[0]):
                                                                    action[i][obj_mov] = 1
                                                                    action[i][-3:] = action_numpy[obj_mov]
                                                                        
                                                                sample_action = torch.Tensor(action).to(device)
                                                                # if(_ == 0):
                                                                #     sample_action = action
                                                                this_sequence = []
                                                                this_sequence.append(sample_action)

                                                                this_sequence_variance = []
                                                                this_sequence_variance.append(action_numpy_variance)
                                                                if True:
                                                                    loss_func = nn.MSELoss()
                                                                    test_loss = 0
                                                                    current_latent = outs['current_embed']
                                                                    egde_latent = outs['edge_embed']
                                                                    #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                    for seq in range(len(this_sequence)):
                                                                        #print([current_latent, this_sequence[seq]])
                                                                        if self.use_graph_dynamics:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        else:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                    for seq in range(len(this_sequence)):
                                                                        #print([current_latent, this_sequence[seq]])
                                                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                        edge_num = egde_latent.shape[0]
                                                                        edge_action_list = []
                                                                        if self.use_graph_dynamics:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        else:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        for _ in range(edge_num):
                                                                            edge_action_list.append(current_action[0])
                                                                        edge_action = torch.stack(edge_action_list)
                                                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                    #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                    #print(egde_latent.shape)
                                                                    # print(current_latent)
                                                                    # print(egde_latent)
                                                                    #outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                
                                                                    batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                    outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                    
                                                                    
                                                                    
                                                                    if self.all_gt_sigmoid:
                                                                        if self.using_sub_goal:
                                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                        else:
                                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                            
                                                                    #sample_list.append(outs_edge['pred_edge'])
                                                                    print('test_loss', test_loss)
                                                                    if self.using_sub_goal:
                                                                        print('predicted_relations',outs_decoder_2['pred_sigmoid'][index_i, index_j])
                                                                        print('ground truth relations', x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                    loss_list.append(test_loss)
                                                                    if(test_loss.detach().cpu().numpy() < min_cost):
                                                                        min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                                                                        min_action = this_sequence
                                                                        min_action_variance = this_sequence_variance
                                                                        min_pose = outs_decoder_2['pred'][:, :]
                                                                        min_cost = test_loss.detach().cpu().numpy()
                                                                        this_time_step_embed = current_latent
                                                                        this_time_step_edge_embed = egde_latent

                                                        pred_relations = min_prediction.cpu().detach().numpy()
                                                        goal_relations = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()

                                                        planning_success_num = 1
                                                        for obj_id in range(pred_relations.shape[0]):
                                                            for relation_id in range(pred_relations.shape[1]):
                                                                if goal_relations[obj_id][relation_id] == 1:
                                                                    if pred_relations[obj_id][relation_id] < planning_threshold:
                                                                        planning_success_num = 0
                                                                elif goal_relations[obj_id][relation_id] == 0:
                                                                    if pred_relations[obj_id][relation_id] > 1 - planning_threshold:
                                                                        planning_success_num = 0

                                                        print('pred_relations', pred_relations)
                                                        print('goal_relations shape', goal_relations.shape)
                                                        print('goal_relations', goal_relations)
                                                        print('planning_success_num', planning_success_num)
                                                        node_pose_numpy = node_pose.detach().cpu().numpy()
                                                        change_id_leap = 0
                                                        if True: #for seq in range(len(min_action)):
                                                            this_seq_numpy = min_action[0].cpu().numpy()
                                                            change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                                #print(change_id)
                                                            if change_id == 0:
                                                                change_id_leap = 1
                                                                node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                                            #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                                            #     #if(this_seq_numpy[0])
                                                            # all_node_pose_list.append(node_pose_numpy)
                                                            # node_pose = torch.Tensor(node_pose_numpy)
                                                            # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                                        node_pose_numpy_goal = node_pose_goal.detach().cpu().numpy()
                                                        choose_list = [0,-1]
                                                        # print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                                        
                                                        
                                                        print(min_action)
                                                        print(min_action_variance)
                                                        #min_action_numpy
                                                        print(action_torch)
                                                        #print()
                                                        print(min_cost)
                                                        # if change_id_leap == 1:
                                                        #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                                        #     print(goal_loss)
                                                        #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                                        #         success_num += 1
                                                        min_action_numpy = min_action[0].cpu().numpy()
                                                        min_action_variance_numpy = min_action_variance[0]
                                                        action_numpy = action_torch.cpu().numpy()

                                                        # for node_pose_iter in range(node_pose_numpy.shape[0]):
                                                        #     self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                                                        # for action_iter in range(1):
                                                        #     self.action_list.append(min_action_numpy[action_iter])
                                                        # for goal_relation_i in range(goal_relation.shape[0]):
                                                        #     for goal_relation_j in range(goal_relation.shape[1]):
                                                        #         self.goal_relation_list.append(goal_relation[goal_relation_i][goal_relation_j])
                                                        self.node_pose_list.append(node_pose_numpy)
                                                        self.action_list.append(min_action_numpy)
                                                        self.action_variance_list.append(min_action_variance_numpy)
                                                        self.goal_relation_list.append(goal_relations)
                                                        self.gt_pose_list.append(self.gt_pose[0])
                                                        self.pc_center_list.append(self.pc_center[0])
                                                        self.gt_orientation_list.append(self.gt_orientation[0])
                                                        self.gt_extents_list.append(self.gt_extents[0])
                                                        self.gt_extents_range_list.append(self.gt_extents_range[0])
                                                        self.predicted_relations.append(pred_relations)
                                                        self.all_index_i_list.append(index_i)
                                                        self.all_index_j_list.append(index_j)
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        # simplied version of planned action success rate for pushing task
                                                        success_num = 1
                                                        if min_action_numpy[0][expected_action_list[each_goal][0] - 1] != 1:
                                                            success_num = 0
                                                        if min_action_numpy[0][-2]*expected_action_list[each_goal][1] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                                            success_num = 0
                                                        total_succes_num += success_num
                                                    
                                                if self.execute_planning:
                                                    print(total_succes_num)
                                                    print(total_succes_num/total_num)               
                            
                            elif self.using_multi_step:
                                if cem_planning:
                                    total_num = 0
                                    total_succes_num = 0
                                    

                                    if not self.using_sub_goal:
                                        sub_goal_list_i.append(1)
                                        sub_goal_list_j.append(1)
                                    # if self.using_multi_step:
                                    #     sub_goal_list_i = []
                                    #     sub_goal_list_j = []

                                    
                                    if True:
                                        for each_index in range(2):
                                            if self.test_next_step:
                                                each_goal = 1
                                                index_i = [0]
                                                index_j = [2]
                                                # each_goal = 1
                                                # index_i = [1]
                                                # index_j = [2]
                                            else:
                                                if each_index == 0:
                                                    # each_goal = 3
                                                    # index_i = [1,0]
                                                    # index_j = [3,4]
                                                    each_goal = 5
                                                    index_i = [2,1]
                                                    index_j = [3,3]
                                                else:
                                                    # each_goal = 1
                                                    # index_i = [0]
                                                    # index_j = [2]
                                                    each_goal = 1
                                                    index_i = [1]
                                                    index_j = [2]
                                            print('index', [index_i, index_j])
                                            if self.manual_relations:
                                                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[each_goal]).to(device)
                                            
                                            if self.all_gt_sigmoid:
                                                success_num = 0
                                                total_num += 1
                                                planning_success_num = 0
                                                planning_total_num = 0
                                                planning_threshold = threshold

                                                for test_iter in range(1):
                                                    print('enter')
                                                    plannning_sequence = 1
                                                    sample_sequence = 1
                                                    num_nodes = self.num_nodes


                                                    if self.mlp:
                                                        data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                                                        # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
                                                    else:
                                                        if self.test_next_step:
                                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, 0, None, action_torch)
                                                        else:
                                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                                    if each_index == 0:
                                                        batch = Batch.from_data_list([data_1]).to(device)
                                                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)

                                                        this_time_step_embed = outs['current_embed']
                                                        this_time_step_edge_embed = outs['edge_embed']



                                                    # data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
                                                    # batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                                                    # outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

                                                    # if self.real_data:
                                                    #     print('current relations', outs_decoder['pred_sigmoid'][:])
                                                    #     print('ground truth relations', x_tensor_dict['batch_all_obj_pair_relation'])
                                                    #     print('difference between current relations and GT relations', self.dynamics_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']))

                                                    

                                                    # data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), action_torch)
                                                    # data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, x_tensor_dict_next['batch_all_obj_pair_relation'], action_torch)
                                                    # batch = Batch.from_data_list([data]).to(device)
                                                    # outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                                    # batch2 = Batch.from_data_list([data_next]).to(device)
                                                    # outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                                    
                                                    
                                                    if self.execute_planning:
                                                        min_cost = 1e5
                                                        loss_list = []
                                                        all_action_list = []
                                                        
                                                        print('actual action', expected_action_list[each_goal])
                                                        
                                                        for obj_mov in range(self.num_nodes):
                                                            if self.seperate_range:
                                                                middle_point = [[1,0.3], [0,0.3]]
                                                            else:
                                                                middle_point = [[0.5,0.6]]
                                                            for current_middle_point in middle_point:
                                                                print('mov obj', obj_mov)
                                                                action_selections = 500
                                                                action_mu = np.zeros((action_selections, 1, 2))
                                                                action_sigma = np.ones((action_selections, 1, 2))
                                                                
                                                                
                                                                for i_iter in range(5):
                                                                    action_noise = np.zeros((action_selections, 1, 2))
                                                                    action_noise[:,:,0] = (np.random.rand(action_selections, 1) - 0.5) * 0.1
                                                                    action_noise[:,:,1] = (np.random.rand(action_selections, 1) - current_middle_point[0]) * current_middle_point[1]
                                                                    #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                                                                    act = action_mu + action_noise*action_sigma
                                                                    costs = []
                                                                    for j in range(action_selections):
                                                                        action_numpy = np.zeros((num_nodes, 3))
                                                                        action_numpy[obj_mov][0] = act[j, 0, 0]
                                                                        action_numpy[obj_mov][1] = act[j, 0, 1]
                                                                        action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                                        
                                                                        if self.set_max:
                                                                            action = np.zeros((num_nodes, self.max_objects + 3))
                                                                        else:
                                                                            action = np.zeros((num_nodes, num_nodes + 3))
                                                                        for i in range(action.shape[0]):
                                                                            action[i][obj_mov] = 1
                                                                            action[i][-3:] = action_numpy[obj_mov]
                                                                        
                                                                        sample_action = torch.Tensor(action).to(device)
                                                                        
                                                                        # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                                        # for i in range(action_1.shape[0]):
                                                                        #     action_1[i][obj_mov] = 1
                                                                        #     action_1[i][3:] = action_numpy[obj_mov]
                                                                        # sample_action = torch.Tensor(action_1)
                                                                        #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                                        # if(_ == 0):
                                                                        #     sample_action = action
                                                                        this_sequence = []
                                                                        this_sequence.append(sample_action)
                                                                        loss_func = nn.MSELoss()
                                                                        test_loss = 0
                                                                        current_latent = this_time_step_embed 
                                                                        egde_latent = this_time_step_edge_embed 
                                                                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])

                                                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                            edge_num = egde_latent.shape[0]
                                                                            edge_action_list = []
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            for _ in range(edge_num):
                                                                                edge_action_list.append(current_action[0])
                                                                            edge_action = torch.stack(edge_action_list)
                                                                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                        #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                        #print(egde_latent.shape)
                                                                        # print(current_latent)
                                                                        # print(egde_latent)
                                                                        # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                
                                                                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                        
                                                                        
                                                                        
                                                                        if self.all_gt_sigmoid:
                                                                            if self.using_sub_goal:
                                                                                #print(x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j].shape)
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                            else:
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                        
                                                                        costs.append(test_loss.detach().cpu().numpy())
                                                                        # if(test_loss.detach().cpu().numpy() < min_cost):
                                                                        #     min_action = this_sequence
                                                                        #     min_cost = test_loss.detach().cpu().numpy()
                                                                
                                                                        #     costs.append(test_loss)

                                                                    index = np.argsort(costs)
                                                                    elite = act[index,:,:]
                                                                    elite = elite[:3, :, :]
                                                                        # print('elite')
                                                                        # print(elite)
                                                                    act_mu = elite.mean(axis = 0)
                                                                    act_sigma = elite.std(axis = 0)
                                                                    print([act_mu, act_sigma])
                                                                    # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                                    #     break
                                                                    #print(act_sigma)
                                                                # print('find_actions')
                                                                # print(act_mu)
                                                                chosen_action = act_mu
                                                                action_numpy = np.zeros((num_nodes, 3))
                                                                action_numpy[obj_mov][0] = chosen_action[0, 0]
                                                                action_numpy[obj_mov][1] = chosen_action[0, 1]
                                                                action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)

                                                                action_numpy_variance = np.zeros((num_nodes, 2))
                                                                action_numpy_variance[obj_mov][0] = act_sigma[0, 0]
                                                                action_numpy_variance[obj_mov][1] = act_sigma[0, 1]
                                                                if self.set_max:
                                                                    action = np.zeros((num_nodes, self.max_objects + 3))
                                                                else:
                                                                    action = np.zeros((num_nodes, num_nodes + 3))
                                                                for i in range(action.shape[0]):
                                                                    action[i][obj_mov] = 1
                                                                    action[i][-3:] = action_numpy[obj_mov]
                                                                        
                                                                sample_action = torch.Tensor(action).to(device)
                                                                # if(_ == 0):
                                                                #     sample_action = action
                                                                this_sequence = []
                                                                this_sequence.append(sample_action)

                                                                this_sequence_variance = []
                                                                this_sequence_variance.append(action_numpy_variance)
                                                                if True:
                                                                    loss_func = nn.MSELoss()
                                                                    test_loss = 0
                                                                    current_latent = outs['current_embed']
                                                                    egde_latent = outs['edge_embed']
                                                                    #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                    for seq in range(len(this_sequence)):
                                                                        #print([current_latent, this_sequence[seq]])
                                                                        if self.use_graph_dynamics:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        else:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                    for seq in range(len(this_sequence)):
                                                                        #print([current_latent, this_sequence[seq]])
                                                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                        edge_num = egde_latent.shape[0]
                                                                        edge_action_list = []
                                                                        if self.use_graph_dynamics:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        else:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        for _ in range(edge_num):
                                                                            edge_action_list.append(current_action[0])
                                                                        edge_action = torch.stack(edge_action_list)
                                                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                    #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                    #print(egde_latent.shape)
                                                                    # print(current_latent)
                                                                    # print(egde_latent)
                                                                    #outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                
                                                                    batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                    outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                    
                                                                    
                                                                    
                                                                    if self.all_gt_sigmoid:
                                                                        if self.using_sub_goal:
                                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                        else:
                                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                            
                                                                    #sample_list.append(outs_edge['pred_edge'])
                                                                    print('test_loss', test_loss)
                                                                    if self.using_sub_goal:
                                                                        print('predicted_relations',outs_decoder_2['pred_sigmoid'][index_i, index_j])
                                                                        print('ground truth relations', x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                    loss_list.append(test_loss)
                                                                    if(test_loss.detach().cpu().numpy() < min_cost):
                                                                        min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                                                                        min_action = this_sequence
                                                                        min_action_variance = this_sequence_variance
                                                                        min_pose = outs_decoder_2['pred'][:, :]
                                                                        min_cost = test_loss.detach().cpu().numpy()
                                                                        this_time_step_embed = current_latent
                                                                        this_time_step_edge_embed = egde_latent

                                                        pred_relations = min_prediction.cpu().detach().numpy()
                                                        goal_relations = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()

                                                        planning_success_num = 1
                                                        for obj_id in range(pred_relations.shape[0]):
                                                            for relation_id in range(pred_relations.shape[1]):
                                                                if goal_relations[obj_id][relation_id] == 1:
                                                                    if pred_relations[obj_id][relation_id] < planning_threshold:
                                                                        planning_success_num = 0
                                                                elif goal_relations[obj_id][relation_id] == 0:
                                                                    if pred_relations[obj_id][relation_id] > 1 - planning_threshold:
                                                                        planning_success_num = 0

                                                        print('pred_relations', pred_relations)
                                                        print('goal_relations shape', goal_relations.shape)
                                                        print('goal_relations', goal_relations)
                                                        print('planning_success_num', planning_success_num)
                                                        node_pose_numpy = node_pose.detach().cpu().numpy()
                                                        change_id_leap = 0
                                                        if True: #for seq in range(len(min_action)):
                                                            this_seq_numpy = min_action[0].cpu().numpy()
                                                            change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                                #print(change_id)
                                                            if change_id == 0:
                                                                change_id_leap = 1
                                                                node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                                            #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                                            #     #if(this_seq_numpy[0])
                                                            # all_node_pose_list.append(node_pose_numpy)
                                                            # node_pose = torch.Tensor(node_pose_numpy)
                                                            # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                                        node_pose_numpy_goal = node_pose_goal.detach().cpu().numpy()
                                                        choose_list = [0,-1]
                                                        # print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                                        
                                                        
                                                        print(min_action)
                                                        print(min_action_variance)
                                                        #min_action_numpy
                                                        print(action_torch)
                                                        #print()
                                                        print(min_cost)
                                                        # if change_id_leap == 1:
                                                        #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                                        #     print(goal_loss)
                                                        #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                                        #         success_num += 1
                                                        min_action_numpy = min_action[0].cpu().numpy()
                                                        min_action_variance_numpy = min_action_variance[0]
                                                        action_numpy = action_torch.cpu().numpy()

                                                        # for node_pose_iter in range(node_pose_numpy.shape[0]):
                                                        #     self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                                                        # for action_iter in range(1):
                                                        #     self.action_list.append(min_action_numpy[action_iter])
                                                        # for goal_relation_i in range(goal_relation.shape[0]):
                                                        #     for goal_relation_j in range(goal_relation.shape[1]):
                                                        #         self.goal_relation_list.append(goal_relation[goal_relation_i][goal_relation_j])
                                                        self.node_pose_list.append(node_pose_numpy)
                                                        self.action_list.append(min_action_numpy)
                                                        self.action_variance_list.append(min_action_variance_numpy)
                                                        self.goal_relation_list.append(goal_relations)
                                                        self.gt_pose_list.append(self.gt_pose[0])
                                                        self.pc_center_list.append(self.pc_center[0])
                                                        self.gt_orientation_list.append(self.gt_orientation[0])
                                                        self.gt_extents_list.append(self.gt_extents[0])
                                                        self.gt_extents_range_list.append(self.gt_extents_range[0])
                                                        self.predicted_relations.append(pred_relations)
                                                        self.all_index_i_list.append(index_i)
                                                        self.all_index_j_list.append(index_j)
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        # simplied version of planned action success rate for pushing task
                                                        success_num = 1
                                                        if min_action_numpy[0][expected_action_list[each_goal][0] - 1] != 1:
                                                            success_num = 0
                                                        if min_action_numpy[0][-2]*expected_action_list[each_goal][1] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                                            success_num = 0
                                                        total_succes_num += success_num
                                                    
                                                if self.execute_planning:
                                                    print(total_succes_num)
                                                    print(total_succes_num/total_num)               
                            else:
                                if cem_planning:
                                    total_num = 0
                                    total_succes_num = 0
                                    

                                    if not self.using_sub_goal:
                                        sub_goal_list_i.append(1)
                                        sub_goal_list_j.append(1)
                                    
                                    self.manual_specificy_goal_list = False
                                    for each_goal in range(len(goal_relations_list)):
                                        for each_index in range(len(sub_goal_list_i)):
                                            if self.manual_specificy_goal_list:
                                                each_goal = 1
                                                index_i = [0,1]
                                                index_j = [2,2]
                                            else:    
                                                if self.real_data:
                                                    each_goal = 1 # np.random.randint(len(goal_relations_list))
                                                    each_index = np.random.randint(len(sub_goal_list_i))
                                                    print('total length', [len(goal_relations_list), len(sub_goal_list_i)])
                                                    print('sampled goal, index', [each_goal, each_index])
                                                index_i = sub_goal_list_i[each_index]
                                                index_j = sub_goal_list_j[each_index]
                                                print('index', [index_i, index_j])
                                            if self.manual_relations:
                                                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[each_goal]).to(device)
                                            
                                            if self.all_gt_sigmoid:
                                                success_num = 0
                                                total_num += 1
                                                planning_success_num = 0
                                                planning_total_num = 0
                                                planning_threshold = threshold

                                                for test_iter in range(1):
                                                    print('enter')
                                                    plannning_sequence = 1
                                                    sample_sequence = 1
                                                    num_nodes = self.num_nodes


                                                    if self.mlp:
                                                        data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.node_inp_size*2, img_emb, action_torch)
                                                        # data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, self.node_inp_size*2, img_emb_next, action_torch)
                                                    else:
                                                        data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                                    batch = Batch.from_data_list([data_1]).to(device)
                                                    outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)



                                                    data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
                                                    batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                                                    outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

                                                    if self.real_data:
                                                        print('current relations', outs_decoder['pred_sigmoid'][:])
                                                        print('ground truth relations', x_tensor_dict['batch_all_obj_pair_relation'])
                                                        print('difference between current relations and GT relations', self.dynamics_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']))

                                                    

                                                    # data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), action_torch)
                                                    # data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, x_tensor_dict_next['batch_all_obj_pair_relation'], action_torch)
                                                    # batch = Batch.from_data_list([data]).to(device)
                                                    # outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                                    # batch2 = Batch.from_data_list([data_next]).to(device)
                                                    # outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                                    
                                                    
                                                    if self.execute_planning:
                                                        min_cost = 1e5
                                                        loss_list = []
                                                        all_action_list = []
                                                        
                                                        print('actual action', expected_action_list[each_goal])
                                                        
                                                        for obj_mov in range(self.num_nodes):
                                                            if self.seperate_range:
                                                                middle_point = [[1,0.3], [0,0.3]]
                                                            else:
                                                                middle_point = [[0.5,0.6]]
                                                            for current_middle_point in middle_point:
                                                                print('mov obj', obj_mov)
                                                                action_selections = 500
                                                                action_mu = np.zeros((action_selections, 1, 2))
                                                                action_sigma = np.ones((action_selections, 1, 2))
                                                                
                                                                
                                                                for i_iter in range(5):
                                                                    action_noise = np.zeros((action_selections, 1, 2))
                                                                    action_noise[:,:,0] = (np.random.rand(action_selections, 1) - 0.5) * 0.1
                                                                    action_noise[:,:,1] = (np.random.rand(action_selections, 1) - current_middle_point[0]) * current_middle_point[1]
                                                                    #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                                                                    act = action_mu + action_noise*action_sigma
                                                                    costs = []
                                                                    for j in range(action_selections):
                                                                        action_numpy = np.zeros((num_nodes, 3))
                                                                        action_numpy[obj_mov][0] = act[j, 0, 0]
                                                                        action_numpy[obj_mov][1] = act[j, 0, 1]
                                                                        action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                                        
                                                                        if self.set_max:
                                                                            action = np.zeros((num_nodes, self.max_objects + 3))
                                                                        else:
                                                                            action = np.zeros((num_nodes, num_nodes + 3))
                                                                        for i in range(action.shape[0]):
                                                                            action[i][obj_mov] = 1
                                                                            action[i][-3:] = action_numpy[obj_mov]
                                                                        
                                                                        sample_action = torch.Tensor(action).to(device)
                                                                        
                                                                        # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                                        # for i in range(action_1.shape[0]):
                                                                        #     action_1[i][obj_mov] = 1
                                                                        #     action_1[i][3:] = action_numpy[obj_mov]
                                                                        # sample_action = torch.Tensor(action_1)
                                                                        #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                                        # if(_ == 0):
                                                                        #     sample_action = action
                                                                        this_sequence = []
                                                                        this_sequence.append(sample_action)
                                                                        loss_func = nn.MSELoss()
                                                                        test_loss = 0
                                                                        current_latent = outs['current_embed']
                                                                        egde_latent = outs['edge_embed']
                                                                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])

                                                                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                        for seq in range(len(this_sequence)):
                                                                            #print([current_latent, this_sequence[seq]])
                                                                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                            edge_num = egde_latent.shape[0]
                                                                            edge_action_list = []
                                                                            if self.use_graph_dynamics:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            else:
                                                                                current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                            for _ in range(edge_num):
                                                                                edge_action_list.append(current_action[0])
                                                                            edge_action = torch.stack(edge_action_list)
                                                                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                        #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                        #print(egde_latent.shape)
                                                                        # print(current_latent)
                                                                        # print(egde_latent)
                                                                        # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                
                                                                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                        
                                                                        
                                                                        
                                                                        if self.all_gt_sigmoid:
                                                                            if self.using_sub_goal:
                                                                                #print(x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j].shape)
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                            else:
                                                                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                        
                                                                        costs.append(test_loss.detach().cpu().numpy())
                                                                        # if(test_loss.detach().cpu().numpy() < min_cost):
                                                                        #     min_action = this_sequence
                                                                        #     min_cost = test_loss.detach().cpu().numpy()
                                                                
                                                                        #     costs.append(test_loss)

                                                                    index = np.argsort(costs)
                                                                    elite = act[index,:,:]
                                                                    elite = elite[:3, :, :]
                                                                        # print('elite')
                                                                        # print(elite)
                                                                    act_mu = elite.mean(axis = 0)
                                                                    act_sigma = elite.std(axis = 0)
                                                                    print([act_mu, act_sigma])
                                                                    # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                                    #     break
                                                                    #print(act_sigma)
                                                                # print('find_actions')
                                                                # print(act_mu)
                                                                chosen_action = act_mu
                                                                action_numpy = np.zeros((num_nodes, 3))
                                                                action_numpy[obj_mov][0] = chosen_action[0, 0]
                                                                action_numpy[obj_mov][1] = chosen_action[0, 1]
                                                                action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)

                                                                action_numpy_variance = np.zeros((num_nodes, 2))
                                                                action_numpy_variance[obj_mov][0] = act_sigma[0, 0]
                                                                action_numpy_variance[obj_mov][1] = act_sigma[0, 1]
                                                                if self.set_max:
                                                                    action = np.zeros((num_nodes, self.max_objects + 3))
                                                                else:
                                                                    action = np.zeros((num_nodes, num_nodes + 3))
                                                                for i in range(action.shape[0]):
                                                                    action[i][obj_mov] = 1
                                                                    action[i][-3:] = action_numpy[obj_mov]
                                                                        
                                                                sample_action = torch.Tensor(action).to(device)
                                                                # if(_ == 0):
                                                                #     sample_action = action
                                                                this_sequence = []
                                                                this_sequence.append(sample_action)

                                                                this_sequence_variance = []
                                                                this_sequence_variance.append(action_numpy_variance)
                                                                if True:
                                                                    loss_func = nn.MSELoss()
                                                                    test_loss = 0
                                                                    current_latent = outs['current_embed']
                                                                    egde_latent = outs['edge_embed']
                                                                    #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                                    for seq in range(len(this_sequence)):
                                                                        #print([current_latent, this_sequence[seq]])
                                                                        if self.use_graph_dynamics:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        else:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                                    for seq in range(len(this_sequence)):
                                                                        #print([current_latent, this_sequence[seq]])
                                                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                                        edge_num = egde_latent.shape[0]
                                                                        edge_action_list = []
                                                                        if self.use_graph_dynamics:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        else:
                                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                                        for _ in range(edge_num):
                                                                            edge_action_list.append(current_action[0])
                                                                        edge_action = torch.stack(edge_action_list)
                                                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                                    #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                                    #print(egde_latent.shape)
                                                                    # print(current_latent)
                                                                    # print(egde_latent)
                                                                    #outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                
                                                                    batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                                    outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                    
                                                                    
                                                                    
                                                                    if self.all_gt_sigmoid:
                                                                        if self.using_sub_goal:
                                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                        else:
                                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                            
                                                                    #sample_list.append(outs_edge['pred_edge'])
                                                                    print('test_loss', test_loss)
                                                                    if self.using_sub_goal:
                                                                        print('predicted_relations',outs_decoder_2['pred_sigmoid'][index_i, index_j])
                                                                        print('ground truth relations', x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                                    loss_list.append(test_loss)
                                                                    if(test_loss.detach().cpu().numpy() < min_cost):
                                                                        min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                                                                        min_action = this_sequence
                                                                        min_action_variance = this_sequence_variance
                                                                        min_pose = outs_decoder_2['pred'][:, :]
                                                                        min_cost = test_loss.detach().cpu().numpy()

                                                        pred_relations = min_prediction.cpu().detach().numpy()
                                                        goal_relations = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()

                                                        planning_success_num = 1
                                                        for obj_id in range(pred_relations.shape[0]):
                                                            for relation_id in range(pred_relations.shape[1]):
                                                                if goal_relations[obj_id][relation_id] == 1:
                                                                    if pred_relations[obj_id][relation_id] < planning_threshold:
                                                                        planning_success_num = 0
                                                                elif goal_relations[obj_id][relation_id] == 0:
                                                                    if pred_relations[obj_id][relation_id] > 1 - planning_threshold:
                                                                        planning_success_num = 0

                                                        print('pred_relations', pred_relations)
                                                        print('goal_relations shape', goal_relations.shape)
                                                        print('goal_relations', goal_relations)
                                                        print('planning_success_num', planning_success_num)
                                                        node_pose_numpy = node_pose.detach().cpu().numpy()
                                                        change_id_leap = 0
                                                        if True: #for seq in range(len(min_action)):
                                                            this_seq_numpy = min_action[0].cpu().numpy()
                                                            change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                                #print(change_id)
                                                            if change_id == 0:
                                                                change_id_leap = 1
                                                                node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                                            #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                                            #     #if(this_seq_numpy[0])
                                                            # all_node_pose_list.append(node_pose_numpy)
                                                            # node_pose = torch.Tensor(node_pose_numpy)
                                                            # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                                        node_pose_numpy_goal = node_pose_goal.detach().cpu().numpy()
                                                        choose_list = [0,-1]
                                                        # print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                                        

                                                        
                                                        print(min_action)
                                                        print(min_action_variance)
                                                        #min_action_numpy
                                                        print(action_torch)
                                                        #print()
                                                        print(min_cost)
                                                        # if change_id_leap == 1:
                                                        #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                                        #     print(goal_loss)
                                                        #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                                        #         success_num += 1
                                                        min_action_numpy = min_action[0].cpu().numpy()
                                                        min_action_variance_numpy = min_action_variance[0]
                                                        action_numpy = action_torch.cpu().numpy()

                                                        # for node_pose_iter in range(node_pose_numpy.shape[0]):
                                                        #     self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                                                        # for action_iter in range(1):
                                                        #     self.action_list.append(min_action_numpy[action_iter])
                                                        # for goal_relation_i in range(goal_relation.shape[0]):
                                                        #     for goal_relation_j in range(goal_relation.shape[1]):
                                                        #         self.goal_relation_list.append(goal_relation[goal_relation_i][goal_relation_j])
                                                        self.node_pose_list.append(node_pose_numpy)
                                                        self.action_list.append(min_action_numpy)
                                                        self.action_variance_list.append(min_action_variance_numpy)
                                                        self.goal_relation_list.append(goal_relations)
                                                        self.gt_pose_list.append(self.gt_pose[0])
                                                        self.pc_center_list.append(self.pc_center[0])
                                                        self.gt_orientation_list.append(self.gt_orientation[0])
                                                        self.gt_extents_list.append(self.gt_extents[0])
                                                        self.gt_extents_range_list.append(self.gt_extents_range[0])
                                                        self.predicted_relations.append(pred_relations)
                                                        self.all_index_i_list.append(index_i)
                                                        self.all_index_j_list.append(index_j)
                                                        
                                                        
                                                        
                                                        
                                                        
                                                        # simplied version of planned action success rate for pushing task
                                                        success_num = 1
                                                        if min_action_numpy[0][expected_action_list[each_goal][0] - 1] != 1:
                                                            success_num = 0
                                                        if min_action_numpy[0][-2]*expected_action_list[each_goal][1] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                                            success_num = 0
                                                        total_succes_num += success_num
                                                    
                                                if self.execute_planning:
                                                    print(total_succes_num)
                                                    print(total_succes_num/total_num)   
                                            if self.real_data:
                                                break
                                        if self.real_data:
                                            break                 
            else:
                if graph_latent:
                    if True:
                        if edge_classifier:
                            if cem_planning:
                                if self.all_gt_sigmoid:
                                    success_num = 0
                                    total_num = 1
                                    planning_success_num = 0
                                    planning_total_num = 0
                                    planning_threshold = threshold
                                    print('gt_pose', self.gt_pose)

                                    for test_iter in range(total_num):
                                        print('enter')
                                        plannning_sequence = 1
                                        sample_sequence = 1
                                        num_nodes = self.num_nodes

                                        

                                        if self.test_next_step:
                                            #data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, 0, None, action_torch)
                                        else:
                                            data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                                        current_goal_relations = np.array([[1., 0., 0., 1., 0., 0., 0.],  # 1-2  # (1,0) for z as below as anchor, other
                                                    [1., 0., 0., 1., 0., 0., 0.],   # 1-3
                                                    [0., 0., 1., 0., 0., 0., 0.],   # 2-1
                                                    [0., 0., 0., 1., 0., 0., 0.],   # 2-3  need action as -y?   (0,1) as -y for based on the anchor, other, same for x
                                                    [0., 0., 1., 0., 0., 0., 0.],   # 3-1
                                                    [0., 0., 1., 0., 0., 0., 0.]]) # 3-2
                                        
                                        if self.test_next_step:
                                            x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(current_goal_relations).to(device)
                                            index_i = [1]
                                            index_j = [3]
                                            # index_i = [1]
                                            # index_j = [0]
                                            print('index', [index_i, index_j])
                                        
                                        batch = Batch.from_data_list([data_1]).to(device)
                                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)



                                        data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'],self. edge_emb_size, outs['pred_edge'], action_torch)                    
                                        batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                                        outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)



                                        # data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), action_torch)
                                        # data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, x_tensor_dict_next['batch_all_obj_pair_relation'], action_torch)
                                        # batch = Batch.from_data_list([data]).to(device)
                                        # outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                        # batch2 = Batch.from_data_list([data_next]).to(device)
                                        # outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                        
                                        
                                        min_cost = 1e5
                                        loss_list = []
                                        all_action_list = []
                                        print('actual action', action)
                                        for obj_mov in range(self.num_nodes):
                                            print('mov obj', obj_mov)
                                            action_selections = 500
                                            action_mu = np.zeros((action_selections, 1, 2))
                                            action_sigma = np.ones((action_selections, 1, 2))
                                            for i_iter in range(5):
                                                action_noise = np.zeros((action_selections, 1, 2))
                                                action_noise[:,:,0] = (np.random.rand(action_selections, 1) - 0.5)*0.6
                                                action_noise[:,:,1] = (np.random.rand(action_selections, 1) - 0.5)*2.2
                                                #action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 0.4 # change range to (-0.2, 0.2)
                                                act = action_mu + action_noise*action_sigma
                                                costs = []
                                                for j in range(action_selections):
                                                    action_numpy = np.zeros((num_nodes, 3))
                                                    action_numpy[obj_mov][0] = act[j, 0, 0]
                                                    action_numpy[obj_mov][1] = act[j, 0, 1]
                                                    action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                    
                                                    if self.set_max:
                                                        action = np.zeros((num_nodes, self.max_objects + 3))
                                                    else:
                                                        action = np.zeros((num_nodes, num_nodes + 3))
                                                    for i in range(action.shape[0]):
                                                        action[i][obj_mov] = 1
                                                        action[i][-3:] = action_numpy[obj_mov]
                                                    
                                                    sample_action = torch.Tensor(action).to(device)
                                                    
                                                    # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                    # for i in range(action_1.shape[0]):
                                                    #     action_1[i][obj_mov] = 1
                                                    #     action_1[i][3:] = action_numpy[obj_mov]
                                                    # sample_action = torch.Tensor(action_1)
                                                    #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                    # if(_ == 0):
                                                    #     sample_action = action
                                                    this_sequence = []
                                                    this_sequence.append(sample_action)
                                                    loss_func = nn.MSELoss()
                                                    test_loss = 0
                                                    current_latent = outs['current_embed']
                                                    egde_latent = outs['edge_embed']
                                                    #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                    for seq in range(len(this_sequence)):
                                                        #print([current_latent, this_sequence[seq]])
                                                        if self.use_graph_dynamics:
                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                        else:
                                                            current_action = self.classif_model.action_emb(this_sequence[seq])

                                                        graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                    for seq in range(len(this_sequence)):
                                                        #print([current_latent, this_sequence[seq]])
                                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                        edge_num = egde_latent.shape[0]
                                                        edge_action_list = []
                                                        if self.use_graph_dynamics:
                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                        else:
                                                            current_action = self.classif_model.action_emb(this_sequence[seq])
                                                        for _ in range(edge_num):
                                                            edge_action_list.append(current_action[0])
                                                        edge_action = torch.stack(edge_action_list)
                                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                    #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                    #print(egde_latent.shape)
                                                    # print(current_latent)
                                                    # print(egde_latent)
                                                    # outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                    data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                            
                                                    batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                    outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                    
                                                    
                                                    
                                                    if self.all_gt_sigmoid:
                                                        if self.test_next_step:
                                                            #print(x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j].shape)
                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                        else:
                                                            test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                
                                                    costs.append(test_loss.detach().cpu().numpy())
                                                    # if(test_loss.detach().cpu().numpy() < min_cost):
                                                    #     min_action = this_sequence
                                                    #     min_cost = test_loss.detach().cpu().numpy()
                                            
                                                    #     costs.append(test_loss)

                                                index = np.argsort(costs)
                                                elite = act[index,:,:]
                                                elite = elite[:3, :, :]
                                                    # print('elite')
                                                    # print(elite)
                                                act_mu = elite.mean(axis = 0)
                                                act_sigma = elite.std(axis = 0)
                                                print([act_mu, act_sigma])
                                                # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                #     break
                                                #print(act_sigma)
                                            # print('find_actions')
                                            # print(act_mu)
                                            chosen_action = act_mu
                                            action_numpy = np.zeros((num_nodes, 3))
                                            action_numpy[obj_mov][0] = chosen_action[0, 0]
                                            action_numpy[obj_mov][1] = chosen_action[0, 1]
                                            action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                            if self.set_max:
                                                action = np.zeros((num_nodes, self.max_objects + 3))
                                            else:
                                                action = np.zeros((num_nodes, num_nodes + 3))
                                            for i in range(action.shape[0]):
                                                action[i][obj_mov] = 1
                                                action[i][-3:] = action_numpy[obj_mov]
                                                    
                                            sample_action = torch.Tensor(action).to(device)
                                            # if(_ == 0):
                                            #     sample_action = action
                                            this_sequence = []
                                            this_sequence.append(sample_action)
                                            if True:
                                                loss_func = nn.MSELoss()
                                                test_loss = 0
                                                current_latent = outs['current_embed']
                                                egde_latent = outs['edge_embed']
                                                #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                for seq in range(len(this_sequence)):
                                                    #print([current_latent, this_sequence[seq]])
                                                    if self.use_graph_dynamics:
                                                        current_action = self.classif_model.action_emb(this_sequence[seq])
                                                    else:
                                                        current_action = self.classif_model.action_emb(this_sequence[seq])
                                                    graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                                    current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                for seq in range(len(this_sequence)):
                                                    #print([current_latent, this_sequence[seq]])
                                                    #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                    edge_num = egde_latent.shape[0]
                                                    edge_action_list = []
                                                    if self.use_graph_dynamics:
                                                        current_action = self.classif_model.action_emb(this_sequence[seq])
                                                    else:
                                                        current_action = self.classif_model.action_emb(this_sequence[seq])
                                                    for _ in range(edge_num):
                                                        edge_action_list.append(current_action[0])
                                                    edge_action = torch.stack(edge_action_list)
                                                    graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                    egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                #print(egde_latent.shape)
                                                # print(current_latent)
                                                # print(egde_latent)
                                                #outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)

                                                data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                            
                                                batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                                outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                
                                                
                                                
                                                if self.all_gt_sigmoid:
                                                    if self.test_next_step:
                                                        test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                    else:
                                                        test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                    
                                                #sample_list.append(outs_edge['pred_edge'])
                                                loss_list.append(test_loss)
                                                print(test_loss)
                                                #print(outs_decoder_2['pred_sigmoid'][:, :])
                                                if self.test_next_step:
                                                    print('predicted_relations',outs_decoder_2['pred_sigmoid'][index_i, index_j])
                                                    print('ground truth relations', x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                                                if(test_loss.detach().cpu().numpy() < min_cost):
                                                    min_prediction = outs_decoder_2['pred_sigmoid'][:, :]
                                                    min_action = this_sequence
                                                    min_pose = outs_decoder_2['pred'][:, :]
                                                    min_cost = test_loss.detach().cpu().numpy()

                                                

                                                

                                        
                                        # print('initial_edge_embed', x_tensor_dict['batch_all_obj_pair_relation'][0][:])
                                        # print('min_prediction', min_prediction)
                                        # print('min node pose prediction', min_pose)
                                        pred_relations = min_prediction.cpu().detach().numpy()
                                        goal_relations = x_tensor_dict_next['batch_all_obj_pair_relation'].cpu().detach().numpy()

                                        planning_success_num = 1
                                        for obj_id in range(pred_relations.shape[0]):
                                            for relation_id in range(pred_relations.shape[1]):
                                                if goal_relations[obj_id][relation_id] == 1:
                                                    if pred_relations[obj_id][relation_id] < planning_threshold:
                                                        planning_success_num = 0
                                                elif goal_relations[obj_id][relation_id] == 0:
                                                    if pred_relations[obj_id][relation_id] > 1 - planning_threshold:
                                                        planning_success_num = 0

                                        print('pred_relations', pred_relations)
                                        print('goal_relations shape', goal_relations.shape)
                                        print('goal_relations', goal_relations)
                                        print('planning_success_num', planning_success_num)
                                        node_pose_numpy = node_pose.detach().cpu().numpy()
                                        change_id_leap = 0
                                        if True: #for seq in range(len(min_action)):
                                            this_seq_numpy = min_action[0].cpu().numpy()
                                            change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                #print(change_id)
                                            if change_id == 0:
                                                change_id_leap = 1
                                                node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                            #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                            #     #if(this_seq_numpy[0])
                                            # all_node_pose_list.append(node_pose_numpy)
                                            # node_pose = torch.Tensor(node_pose_numpy)
                                            # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                        node_pose_numpy_goal = node_pose_goal.detach().cpu().numpy()
                                        choose_list = [0,-1]
                                        #print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                        

                                        
                                        print(min_action)
                                        #min_action_numpy
                                        print(action_torch)
                                        #print()
                                        print(min_cost)
                                        # if change_id_leap == 1:
                                        #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                        #     print(goal_loss)
                                        #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                        #         success_num += 1
                                        min_action_numpy = min_action[0].cpu().numpy()
                                        action_numpy = action_torch.cpu().numpy()

                                        # for node_pose_iter in range(node_pose_numpy.shape[0]):
                                        #     self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                                        # for action_iter in range(1):
                                        #     self.action_list.append(min_action_numpy[action_iter])
                                        # for goal_relation_i in range(goal_relation.shape[0]):
                                        #     for goal_relation_j in range(goal_relation.shape[1]):
                                        #         self.goal_relation_list.append(goal_relation[goal_relation_i][goal_relation_j])
                                        self.node_pose_list.append(node_pose_numpy)
                                        self.action_list.append(min_action_numpy)
                                        self.goal_relation_list.append(goal_relations)
                                        self.gt_pose_list.append(self.gt_pose[0])
                                        self.gt_extents_range_list.append(self.gt_extents_range[0])
                                        
                                        
                                        
                                        

                                        
                                        # simplied version of planned action success rate for pushing task
                                        success_num = 1
                                        for action_i in range(min_action_numpy.shape[1] - 3):
                                            if(min_action_numpy[0][action_i] != action_numpy[0][action_i]):
                                                success_num = 0
                                        if min_action_numpy[0][-2]*action_numpy[0][-2] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                            success_num = 0

                                        total_succes_num = 0
                                        planning_success_num = 0
                                        #print(node_pose)

                                        
                                    print(success_num)
                                    print(success_num/total_num)                    
                                elif self.all_classifier:
                                    success_num = 0
                                    total_num = 1
                                    for test_iter in range(total_num):
                                        print('enter')
                                        plannning_sequence = 1
                                        sample_sequence = 1
                                        num_nodes = self.num_nodes
                                        data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, x_tensor_dict['batch_all_obj_pair_relation'], action_torch)
                                        data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, x_tensor_dict_next['batch_all_obj_pair_relation'], action_torch)
                                        batch = Batch.from_data_list([data]).to(device)
                                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                        batch2 = Batch.from_data_list([data_next]).to(device)
                                        outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                        
                                        
                                        min_cost = 1e5
                                        loss_list = []
                                        all_action_list = []
                                        print('actual action', action)
                                        for obj_mov in range(self.num_nodes):
                                            print('mov obj', obj_mov)
                                            action_selections = 500
                                            action_mu = np.zeros((action_selections, 1, 2))
                                            action_sigma = np.ones((action_selections, 1, 2))
                                            for i_iter in range(5):
                                                action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 2 # change range to (-0.2, 0.2)
                                                act = action_mu + action_noise*action_sigma
                                                costs = []
                                                for j in range(action_selections):
                                                    action_numpy = np.zeros((num_nodes, 3))
                                                    action_numpy[obj_mov][0] = act[j, 0, 0]
                                                    action_numpy[obj_mov][1] = act[j, 0, 1]
                                                    action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                    action = np.zeros((num_nodes, num_nodes + 3))
                                                    for i in range(action.shape[0]):
                                                        action[i][obj_mov] = 1
                                                        action[i][-3:] = action_numpy[obj_mov]
                                                    
                                                    sample_action = torch.Tensor(action).to(device)
                                                    
                                                    # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                    # for i in range(action_1.shape[0]):
                                                    #     action_1[i][obj_mov] = 1
                                                    #     action_1[i][3:] = action_numpy[obj_mov]
                                                    # sample_action = torch.Tensor(action_1)
                                                    #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                    # if(_ == 0):
                                                    #     sample_action = action
                                                    this_sequence = []
                                                    this_sequence.append(sample_action)
                                                    loss_func = nn.MSELoss()
                                                    test_loss = 0
                                                    current_latent = outs['current_embed']
                                                    egde_latent = outs['edge_embed']
                                                    #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                    for seq in range(len(this_sequence)):
                                                        #print([current_latent, this_sequence[seq]])
                                                        graph_node_action = torch.cat((current_latent, this_sequence[seq]), axis = 1)
                                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                    for seq in range(len(this_sequence)):
                                                        #print([current_latent, this_sequence[seq]])
                                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                        edge_num = egde_latent.shape[0]
                                                        edge_action_list = []
                                                        for _ in range(edge_num):
                                                            edge_action_list.append(this_sequence[seq][0])
                                                        edge_action = torch.stack(edge_action_list)
                                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                    #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                    #print(egde_latent.shape)
                                                    # print(current_latent)
                                                    # print(egde_latent)
                                                    outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)
                                                    
                                                    
                                                    
                                                    if self.all_gt:
                                                        test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                    elif self.all_classifier:
                                                        
                                                        horizon_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,1]
                                                        right_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,2]
                                                        left_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,3]
                                                        vertical_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,4]
                                                        front_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,5]
                                                        behind_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,6]
                                                        if self.stacking:
                                                            stack_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,7]
                                                            top_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,8]
                                                            below_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,9]

                                                        
                                                        
                                                        
                                                        
                                                        test_loss += self.CE_loss(outs_edge['pred_edge_classifier'][0], right_tensor_dict) + \
                                                                self.CE_loss(outs_edge['pred_edge_classifier'][1], left_tensor_dict) + \
                                                                self.CE_loss(outs_edge['pred_edge_classifier'][2], front_tensor_dict) + \
                                                                self.CE_loss(outs_edge['pred_edge_classifier'][3], behind_tensor_dict) + \
                                                                self.CE_loss(outs_edge['pred_edge_classifier'][4], stack_tensor_dict) + \
                                                                self.CE_loss(outs_edge['pred_edge_classifier'][5], top_tensor_dict) + \
                                                                self.CE_loss(outs_edge['pred_edge_classifier'][6], below_tensor_dict)
                                                    else:
                                                        test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                    #test_loss += loss_func(goal_relation, outs_edge['pred_edge'])
                                                    #test_loss += loss_func(egde_latent, outs_2['edge_embed'])
                                                    #print(outs_edge['pred_edge'].shape)
                                                    # print(sample_action)
                                                    # print(test_loss)
                                                    #sample_list.append(outs_edge['pred_edge'])
                                                    #loss_list.append(test_loss)
                                                    costs.append(test_loss.detach().cpu().numpy())
                                                    # if(test_loss.detach().cpu().numpy() < min_cost):
                                                    #     min_action = this_sequence
                                                    #     min_cost = test_loss.detach().cpu().numpy()
                                            
                                                    #     costs.append(test_loss)

                                                index = np.argsort(costs)
                                                elite = act[index,:,:]
                                                elite = elite[:3, :, :]
                                                    # print('elite')
                                                    # print(elite)
                                                act_mu = elite.mean(axis = 0)
                                                act_sigma = elite.std(axis = 0)
                                                print([act_mu, act_sigma])
                                                # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                #     break
                                                #print(act_sigma)
                                            # print('find_actions')
                                            # print(act_mu)
                                            chosen_action = act_mu
                                            action_numpy = np.zeros((num_nodes, 3))
                                            action_numpy[obj_mov][0] = chosen_action[0, 0]
                                            action_numpy[obj_mov][1] = chosen_action[0, 1]
                                            action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                            action = np.zeros((num_nodes, num_nodes + 3))
                                            for i in range(action.shape[0]):
                                                action[i][obj_mov] = 1
                                                action[i][-3:] = action_numpy[obj_mov]
                                                    
                                            sample_action = torch.Tensor(action).to(device)
                                            # if(_ == 0):
                                            #     sample_action = action
                                            this_sequence = []
                                            this_sequence.append(sample_action)
                                            if True:
                                                loss_func = nn.MSELoss()
                                                test_loss = 0
                                                current_latent = outs['current_embed']
                                                egde_latent = outs['edge_embed']
                                                #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                for seq in range(len(this_sequence)):
                                                    #print([current_latent, this_sequence[seq]])
                                                    graph_node_action = torch.cat((current_latent, this_sequence[seq]), axis = 1)
                                                    current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                for seq in range(len(this_sequence)):
                                                    #print([current_latent, this_sequence[seq]])
                                                    #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                    edge_num = egde_latent.shape[0]
                                                    edge_action_list = []
                                                    for _ in range(edge_num):
                                                        edge_action_list.append(this_sequence[seq][0])
                                                    edge_action = torch.stack(edge_action_list)
                                                    graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                    egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                #print(egde_latent.shape)
                                                # print(current_latent)
                                                # print(egde_latent)
                                                outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)
                                                
                                                
                                                
                                                if self.all_gt:
                                                    test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                elif self.all_classifier:
                                                    horizon_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,1]
                                                    right_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,2]
                                                    left_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,3]
                                                    vertical_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,4]
                                                    front_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,5]
                                                    behind_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,6]
                                                    if self.stacking:
                                                        stack_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,7]
                                                        top_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,8]
                                                        below_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,9]

                                                        
                                                        
                                                        
                                                    print(outs_edge['pred_edge_classifier'])
                    
                                                    test_loss += self.CE_loss(outs_edge['pred_edge_classifier'][0], right_tensor_dict) + \
                                                            self.CE_loss(outs_edge['pred_edge_classifier'][1], left_tensor_dict) + \
                                                            self.CE_loss(outs_edge['pred_edge_classifier'][2], front_tensor_dict) + \
                                                            self.CE_loss(outs_edge['pred_edge_classifier'][3], behind_tensor_dict) + \
                                                            self.CE_loss(outs_edge['pred_edge_classifier'][4], stack_tensor_dict) + \
                                                            self.CE_loss(outs_edge['pred_edge_classifier'][5], top_tensor_dict) + \
                                                            self.CE_loss(outs_edge['pred_edge_classifier'][6], below_tensor_dict)
                                                else:
                                                    test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                #test_loss += loss_func(torch.stack(scene_emb_list_next[0][:]), outs_edge['pred_edge'][:, :])
                                                #test_loss += loss_func(goal_relation, outs_edge['pred_edge'])
                                                #test_loss += loss_func(egde_latent, outs_2['edge_embed'])
                                                #print(outs_edge['pred_edge'].shape)
                                                # print(sample_action)
                                                # print(test_loss)
                                                #sample_list.append(outs_edge['pred_edge'])
                                                loss_list.append(test_loss)
                                                if(test_loss.detach().cpu().numpy() < min_cost):
                                                    min_prediction = outs_edge['pred_edge'][:, :]
                                                    min_action = this_sequence
                                                    min_pose = outs_edge['pred'][:, :]
                                                    min_cost = test_loss.detach().cpu().numpy()

                                        
                                        print('initial_edge_embed', x_tensor_dict['batch_all_obj_pair_relation'][0][:])
                                        print('min_prediction', min_prediction)
                                        print('min node pose prediction', min_pose)
                                        node_pose_numpy = node_pose.cpu().numpy()
                                        change_id_leap = 0
                                        if True: #for seq in range(len(min_action)):
                                            this_seq_numpy = min_action[0].cpu().numpy()
                                            change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                #print(change_id)
                                            if change_id == 0:
                                                change_id_leap = 1
                                                node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                            #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                            #     #if(this_seq_numpy[0])
                                            # all_node_pose_list.append(node_pose_numpy)
                                            # node_pose = torch.Tensor(node_pose_numpy)
                                            # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                        node_pose_numpy_goal = node_pose_goal.cpu().numpy()
                                        choose_list = [0,-1]
                                        print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                        

                                        
                                        print(min_action)
                                        print(action_torch)
                                        #print()
                                        print(min_cost)
                                        # if change_id_leap == 1:
                                        #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                        #     print(goal_loss)
                                        #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                        #         success_num += 1
                                        min_action_numpy = min_action[0].cpu().numpy()
                                        action_numpy = action_torch.cpu().numpy()
                                        # success_num = 1
                                        # for action_i in range(min_action_numpy.shape[1] - 3):
                                        #     if(min_action_numpy[0][action_i] != action_numpy[0][action_i]):
                                        #         success_num = 0
                                        # if np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                        #     success_num = 0
                                        #print(node_pose)

                                        #print(node_pose_goal)
                                        #print(edge_feature_2)
                                        #print(generate_edge_embed_list)
                                        #print(all_action_sequence_list)
                                        #print(all_node_pose_list)
                                        #print(loss_func(edge_feature_2, torch.stack(generate_edge_embed_list)))
                                        # goal_loss = loss_func(torch.stack(scene_emb_list_next[0][:12]), torch.stack(generate_edge_embed_list))
                                        # print(goal_loss)
                                        # if(goal_loss.detach().cpu().numpy() < 1e-3):
                                        #     success_num += 1
                                        # else:
                                        #     print(loss_list)
                                        #     print(sample_list)
                                        # print(loss_list)
                                        # print(sample_list)
                                        #print(self.dynamics_loss(node_pose[:,:6], node_pose_goal[:,:6]))
                                    print(success_num)
                                    print(success_num/total_num)                    
                                elif self.all_gt:
                                    success_num = 0
                                    total_num = 1
                                    for test_iter in range(total_num):
                                        print('enter')
                                        plannning_sequence = 1
                                        sample_sequence = 1
                                        num_nodes = self.num_nodes
                                        data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, x_tensor_dict['batch_all_obj_pair_relation'], action_torch)
                                        data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, x_tensor_dict_next['batch_all_obj_pair_relation'], action_torch)
                                        batch = Batch.from_data_list([data]).to(device)
                                        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                        batch2 = Batch.from_data_list([data_next]).to(device)
                                        outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                        
                                        
                                        min_cost = 1e5
                                        loss_list = []
                                        all_action_list = []
                                        print('actual action', action)
                                        for obj_mov in range(self.num_nodes):
                                            print('mov obj', obj_mov)
                                            action_selections = 500
                                            action_mu = np.zeros((action_selections, 1, 2))
                                            action_sigma = np.ones((action_selections, 1, 2))
                                            for i_iter in range(5):
                                                action_noise = (np.random.rand(action_selections, 1, 2) - 0.5) * 2 # change range to (-0.2, 0.2)
                                                act = action_mu + action_noise*action_sigma
                                                costs = []
                                                for j in range(action_selections):
                                                    action_numpy = np.zeros((num_nodes, 3))
                                                    action_numpy[obj_mov][0] = act[j, 0, 0]
                                                    action_numpy[obj_mov][1] = act[j, 0, 1]
                                                    action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                                    action = np.zeros((num_nodes, num_nodes + 3))
                                                    for i in range(action.shape[0]):
                                                        action[i][obj_mov] = 1
                                                        action[i][-3:] = action_numpy[obj_mov]
                                                    
                                                    sample_action = torch.Tensor(action).to(device)
                                                    
                                                    # action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                    # for i in range(action_1.shape[0]):
                                                    #     action_1[i][obj_mov] = 1
                                                    #     action_1[i][3:] = action_numpy[obj_mov]
                                                    # sample_action = torch.Tensor(action_1)
                                                    #sample_action = (torch.rand((num_nodes, node_inp_size)) - 0.5)*20
                                                    # if(_ == 0):
                                                    #     sample_action = action
                                                    this_sequence = []
                                                    this_sequence.append(sample_action)
                                                    loss_func = nn.MSELoss()
                                                    test_loss = 0
                                                    current_latent = outs['current_embed']
                                                    egde_latent = outs['edge_embed']
                                                    #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                    for seq in range(len(this_sequence)):
                                                        #print([current_latent, this_sequence[seq]])
                                                        graph_node_action = torch.cat((current_latent, this_sequence[seq]), axis = 1)
                                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                    for seq in range(len(this_sequence)):
                                                        #print([current_latent, this_sequence[seq]])
                                                        #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                        edge_num = egde_latent.shape[0]
                                                        edge_action_list = []
                                                        for _ in range(edge_num):
                                                            edge_action_list.append(this_sequence[seq][0])
                                                        edge_action = torch.stack(edge_action_list)
                                                        graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                        egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                    #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                    #print(egde_latent.shape)
                                                    # print(current_latent)
                                                    # print(egde_latent)
                                                    outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)
                                                    
                                                    
                                                    
                                                    if self.all_gt_sigmoid:
                                                        test_loss += self.bce_loss(outs_edge['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                    elif self.all_gt:
                                                        test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                    elif self.all_classifier:
                                                        horizon_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,1]
                                                        right_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,2]
                                                        left_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,3]
                                                        vertical_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,4]
                                                        front_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,5]
                                                        behind_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,6]
                                                        if self.stacking:
                                                            stack_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,7]
                                                            top_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,8]
                                                            below_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,9]

                                                        
                                                        
                                                        
                                                        test_loss += self.CE_loss(outs['pred_edge_classifier'][0], right_tensor_dict) + \
                                                                self.CE_loss(outs['pred_edge_classifier'][1], left_tensor_dict) + \
                                                                self.CE_loss(outs['pred_edge_classifier'][2], front_tensor_dict) + \
                                                                self.CE_loss(outs['pred_edge_classifier'][3], behind_tensor_dict) + \
                                                                self.CE_loss(outs['pred_edge_classifier'][4], stack_tensor_dict) + \
                                                                self.CE_loss(outs['pred_edge_classifier'][5], top_tensor_dict) + \
                                                                self.CE_loss(outs['pred_edge_classifier'][6], below_tensor_dict)
                                                    else:
                                                        test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                    #test_loss += loss_func(goal_relation, outs_edge['pred_edge'])
                                                    #test_loss += loss_func(egde_latent, outs_2['edge_embed'])
                                                    #print(outs_edge['pred_edge'].shape)
                                                    # print(sample_action)
                                                    # print(test_loss)
                                                    #sample_list.append(outs_edge['pred_edge'])
                                                    #loss_list.append(test_loss)
                                                    costs.append(test_loss.detach().cpu().numpy())
                                                    # if(test_loss.detach().cpu().numpy() < min_cost):
                                                    #     min_action = this_sequence
                                                    #     min_cost = test_loss.detach().cpu().numpy()
                                            
                                                    #     costs.append(test_loss)

                                                index = np.argsort(costs)
                                                elite = act[index,:,:]
                                                elite = elite[:3, :, :]
                                                    # print('elite')
                                                    # print(elite)
                                                act_mu = elite.mean(axis = 0)
                                                act_sigma = elite.std(axis = 0)
                                                print([act_mu, act_sigma])
                                                # if(act_sigma[0][0] < 0.1 and act_sigma[0][1] < 0.1):
                                                #     break
                                                #print(act_sigma)
                                            # print('find_actions')
                                            # print(act_mu)
                                            chosen_action = act_mu
                                            action_numpy = np.zeros((num_nodes, 3))
                                            action_numpy[obj_mov][0] = chosen_action[0, 0]
                                            action_numpy[obj_mov][1] = chosen_action[0, 1]
                                            action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)
                                            action = np.zeros((num_nodes, num_nodes + 3))
                                            for i in range(action.shape[0]):
                                                action[i][obj_mov] = 1
                                                action[i][-3:] = action_numpy[obj_mov]
                                                    
                                            sample_action = torch.Tensor(action).to(device)
                                            # if(_ == 0):
                                            #     sample_action = action
                                            this_sequence = []
                                            this_sequence.append(sample_action)
                                            if True:
                                                loss_func = nn.MSELoss()
                                                test_loss = 0
                                                current_latent = outs['current_embed']
                                                egde_latent = outs['edge_embed']
                                                #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                                for seq in range(len(this_sequence)):
                                                    #print([current_latent, this_sequence[seq]])
                                                    graph_node_action = torch.cat((current_latent, this_sequence[seq]), axis = 1)
                                                    current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                                for seq in range(len(this_sequence)):
                                                    #print([current_latent, this_sequence[seq]])
                                                    #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                    edge_num = egde_latent.shape[0]
                                                    edge_action_list = []
                                                    for _ in range(edge_num):
                                                        edge_action_list.append(this_sequence[seq][0])
                                                    edge_action = torch.stack(edge_action_list)
                                                    graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                    egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                                #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                                #print(egde_latent.shape)
                                                # print(current_latent)
                                                # print(egde_latent)
                                                outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)
                                                
                                                
                                                
                                                if self.all_gt_sigmoid:
                                                    test_loss += self.bce_loss(outs_edge['pred_sigmoid'][:, :], x_tensor_dict_next['batch_all_obj_pair_relation'])
                                                elif self.all_gt:
                                                    test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                elif self.all_classifier:
                                                    horizon_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,1]
                                                    right_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,2]
                                                    left_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,3]
                                                    vertical_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,4]
                                                    front_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,5]
                                                    behind_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,6]
                                                    if self.stacking:
                                                        stack_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,7]
                                                        top_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,8]
                                                        below_tensor_dict = x_tensor_dict_next['batch_all_obj_pair_relation'][:,9]

                                                        
                                                        
                                                        
                                                    test_loss += self.CE_loss(outs['pred_edge_classifier'][0], right_tensor_dict) + \
                                                            self.CE_loss(outs['pred_edge_classifier'][1], left_tensor_dict) + \
                                                            self.CE_loss(outs['pred_edge_classifier'][2], front_tensor_dict) + \
                                                            self.CE_loss(outs['pred_edge_classifier'][3], behind_tensor_dict) + \
                                                            self.CE_loss(outs['pred_edge_classifier'][4], stack_tensor_dict) + \
                                                            self.CE_loss(outs['pred_edge_classifier'][5], top_tensor_dict) + \
                                                            self.CE_loss(outs['pred_edge_classifier'][6], below_tensor_dict)
                                                else:
                                                    test_loss += loss_func(x_tensor_dict_next['batch_all_obj_pair_relation'], outs_edge['pred_edge'][:, :])
                                                #test_loss += loss_func(torch.stack(scene_emb_list_next[0][:]), outs_edge['pred_edge'][:, :])
                                                #test_loss += loss_func(goal_relation, outs_edge['pred_edge'])
                                                #test_loss += loss_func(egde_latent, outs_2['edge_embed'])
                                                #print(outs_edge['pred_edge'].shape)
                                                # print(sample_action)
                                                # print(test_loss)
                                                #sample_list.append(outs_edge['pred_edge'])
                                                loss_list.append(test_loss)
                                                if(test_loss.detach().cpu().numpy() < min_cost):
                                                    min_prediction = outs_edge['pred_edge'][:, :]
                                                    min_action = this_sequence
                                                    min_pose = outs_edge['pred'][:, :]
                                                    min_cost = test_loss.detach().cpu().numpy()

                                        
                                        print('initial_edge_embed', x_tensor_dict['batch_all_obj_pair_relation'][0][:])
                                        print('min_prediction', min_prediction)
                                        print('min node pose prediction', min_pose)
                                        node_pose_numpy = node_pose.cpu().numpy()
                                        change_id_leap = 0
                                        if True: #for seq in range(len(min_action)):
                                            this_seq_numpy = min_action[0].cpu().numpy()
                                            change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                                                #print(change_id)
                                            if change_id == 0:
                                                change_id_leap = 1
                                                node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]
                                            #node_pose_numpy[0][3:6] += this_seq_numpy[0][-3:]
                                            #     #if(this_seq_numpy[0])
                                            # all_node_pose_list.append(node_pose_numpy)
                                            # node_pose = torch.Tensor(node_pose_numpy)
                                            # generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                        node_pose_numpy_goal = node_pose_goal.cpu().numpy()
                                        choose_list = [0,-1]
                                        print('node_pose', [node_pose_numpy[[0,-1]], node_pose_goal[[0,-1]]])
                                        

                                        
                                        print(min_action)
                                        #min_action_numpy
                                        print(action_torch)
                                        #print()
                                        print(min_cost)
                                        # if change_id_leap == 1:
                                        #     goal_loss = loss_func(torch.stack(current_relations[:]), torch.stack(goal_relations[:]))
                                        #     print(goal_loss)
                                        #     if(goal_loss.detach().cpu().numpy() < 1e-3):
                                        #         success_num += 1
                                        min_action_numpy = min_action[0].cpu().numpy()
                                        action_numpy = action_torch.cpu().numpy()

                                        for node_pose_iter in range(node_pose_numpy.shape[0]):
                                            self.node_pose_list.append(node_pose_numpy[node_pose_iter])
                                        for action_iter in range(1):
                                            self.action_list.append(min_action_numpy[action_iter])
                                        # success_num = 1
                                        # for action_i in range(min_action_numpy.shape[1] - 3):
                                        #     if(min_action_numpy[0][action_i] != action_numpy[0][action_i]):
                                        #         success_num = 0
                                        # if np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                                        #     success_num = 0
                                        #print(node_pose)

                                        #print(node_pose_goal)
                                        #print(edge_feature_2)
                                        #print(generate_edge_embed_list)
                                        #print(all_action_sequence_list)
                                        #print(all_node_pose_list)
                                        #print(loss_func(edge_feature_2, torch.stack(generate_edge_embed_list)))
                                        # goal_loss = loss_func(torch.stack(scene_emb_list_next[0][:12]), torch.stack(generate_edge_embed_list))
                                        # print(goal_loss)
                                        # if(goal_loss.detach().cpu().numpy() < 1e-3):
                                        #     success_num += 1
                                        # else:
                                        #     print(loss_list)
                                        #     print(sample_list)
                                        # print(loss_list)
                                        # print(sample_list)
                                        #print(self.dynamics_loss(node_pose[:,:6], node_pose_goal[:,:6]))
                                        
                                    print(success_num)
                                    print(success_num/total_num)                    
                            else:
                                success_num = 0
                                total_num = 1
                                for test_iter in range(total_num):
                                    print('enter')
                                    plannning_sequence = 1
                                    sample_sequence = 1
                                    num_nodes = self.num_nodes
                                    data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), action_torch)
                                    data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, torch.stack(scene_emb_list_next[0]), action_torch)
                                    batch = Batch.from_data_list([data]).to(device)
                                    outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                    batch2 = Batch.from_data_list([data_next]).to(device)
                                    outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                                    
                                    min_cost = 1e5
                                    all_action_sequence_list = []
                                    all_node_pose_list = []
                                    all_node_pose_list.append(node_pose.cpu().numpy())
                                    #print('init node pose', node_pose)
                                    for test_seq in range(plannning_sequence):
                                        loss_list = []
                                        sample_list = []
                                        for _ in range(200):
                                            this_sequence = []
                                            for seq in range(sample_sequence):
                                                obj_mov = np.random.randint(num_nodes)
                                                action_numpy = np.zeros((num_nodes, 3))
                                                action_numpy[obj_mov][0] = np.random.uniform(-1,1)
                                                action_numpy[obj_mov][1] = np.random.uniform(-1,1)
                                                # z_choice_list = [0,0.10,0.18]
                                                # action_numpy[obj_mov][2] = z_choice_list[np.random.randint(len(z_choice_list))]
                                                action_1 = np.zeros((num_nodes, 3 + num_nodes))
                                                for i in range(action_1.shape[0]):
                                                    action_1[i][obj_mov] = 1
                                                    action_1[i][3:] = action_numpy[obj_mov]
                                                sample_action = torch.Tensor(action_1)
                                                sample_action = sample_action.to(device)
                                                if _ == 0:
                                                    sample_action = action_torch
                                                this_sequence.append(sample_action)
                                            loss_func = nn.MSELoss()
                                            test_loss = 0
                                            current_latent = outs['current_embed']
                                            egde_latent = outs['edge_embed']
                                            #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                            for seq in range(len(this_sequence)):
                                                #print([current_latent, this_sequence[seq]])
                                                graph_node_action = torch.cat((current_latent, this_sequence[seq]), axis = 1)
                                                current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                            for seq in range(len(this_sequence)):
                                                #print([current_latent, this_sequence[seq]])
                                                edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                                graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                                egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)
                                            #test_loss += loss_func(current_latent, outs_2['current_embed'])
                                            #print(egde_latent.shape)
                                            # print(current_latent)
                                            # print(egde_latent)
                                            outs_edge = self.classif_model.forward_decoder(current_latent, batch.edge_index, egde_latent, batch.batch, sample_action)
                                            
                                            test_loss = loss_func(torch.stack(scene_emb_list_next[0][:]), outs_edge['pred_edge'][:])
                                            #test_loss += loss_func(egde_latent, outs_2['edge_embed'])
                                            #print(outs_edge['pred_edge'].shape)
                                            # print(sample_action)
                                            # print(test_loss)
                                            sample_list.append(outs_edge['pred_edge'])
                                            loss_list.append(test_loss)
                                            if(test_loss.detach().cpu().numpy() < min_cost):
                                                min_action = this_sequence
                                                min_cost = test_loss.detach().cpu().numpy()
                                        #print(loss_list)
                                        node_pose_numpy = node_pose.cpu().numpy()
                                        if True: #for seq in range(len(min_action)):
                                            this_seq_numpy = min_action[0].cpu().numpy()
                                            all_action_sequence_list.append(this_seq_numpy[0])
                                            #print(this_seq_numpy[0][:3])
                                            change_id = np.argmax(this_seq_numpy[0][:3])
                                            #print(change_id)
                                            node_pose_numpy[change_id][3:6] += this_seq_numpy[0][3:]
                                            #if(this_seq_numpy[0])
                                            all_node_pose_list.append(node_pose_numpy)
                                            node_pose = torch.Tensor(node_pose_numpy)
                                            generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                            #print(generate_edge_embed_list)
                                            # data = self.create_graph(num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(generate_edge_embed_list), min_action[0])
                                            # batch = Batch.from_data_list([data]).to(device)
                                            # outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                                    print(min_action)
                                    print(action)
                                    #print()
                                    print(min_cost)
                                    min_action_numpy = min_action[0].cpu().numpy()
                                    action_numpy = action.cpu().numpy()
                                    success_num = 1
                                    for action_i in range(min_action_numpy.shape[1] - 3):
                                        if(min_action_numpy[0][action_i] != action_numpy[0][0][action_i]):
                                            success_num = 0
                                    if np.abs(min_action_numpy[0][3] - action_numpy[0][0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][0][4]) >= 0.06:
                                        success_num = 0
                                    #print(node_pose)

                                    #print(node_pose_goal)
                                    #print(edge_feature_2)
                                    #print(generate_edge_embed_list)
                                    #print(all_action_sequence_list)
                                    #print(all_node_pose_list)
                                    #print(loss_func(edge_feature_2, torch.stack(generate_edge_embed_list)))
                                    # goal_loss = loss_func(torch.stack(scene_emb_list_next[0][:12]), torch.stack(generate_edge_embed_list))
                                    # print(goal_loss)
                                    # if(goal_loss.detach().cpu().numpy() < 1e-3):
                                    #     success_num += 1
                                    # else:
                                    #     print(loss_list)
                                    #     print(sample_list)
                                    # print(loss_list)
                                    # print(sample_list)
                                    #print(self.dynamics_loss(node_pose[:,:6], node_pose_goal[:,:6]))
                                print(success_num)
                                print(success_num/total_num)

                        else:
                            plannning_sequence = 2
                            sample_sequence = 1
                            print('init_node_pose', node_pose)
                            data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), action_torch)
                            data_next = self.create_graph(self.num_nodes, self.node_inp_size, node_pose_goal, self.edge_inp_size, torch.stack(scene_emb_list_next[0]), action_torch)
                            batch = Batch.from_data_list([data]).to(device)
                            outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                            batch2 = Batch.from_data_list([data_next]).to(device)
                            outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
                            min_cost = 1e5
                            all_action_sequence_list = []
                            all_node_pose_list = []
                            all_node_pose_list.append(node_pose.cpu().numpy())
                            for test_seq in range(plannning_sequence):
                                loss_list = []
                                for _ in range(1000):
                                    this_sequence = []
                                    for seq in range(sample_sequence):
                                        obj_mov = np.random.randint(self.num_nodes)
                                        action_numpy = np.zeros((self.num_nodes, 3))
                                        action_numpy[obj_mov][0] = np.random.uniform(-0.3,0.3)
                                        action_numpy[obj_mov][1] = np.random.uniform(-0.6,0.6)
                                        z_choice_list = [0,0.10,0.18]
                                        action_numpy[obj_mov][2] = z_choice_list[np.random.randint(len(z_choice_list))]
                                        action_1 = np.zeros((self.num_nodes, 3 + self.num_nodes))
                                        for i in range(action_1.shape[0]):
                                            action_1[i][obj_mov] = 1
                                            action_1[i][3:] = action_numpy[obj_mov]
                                        sample_action = torch.Tensor(action_1).to(device)
                                        this_sequence.append(sample_action)
                                    loss_func = nn.MSELoss()
                                    test_loss = 0
                                    current_latent = outs['current_embed']
                                    for seq in range(len(this_sequence)):
                                        #print([current_latent, this_sequence[seq]])
                                        graph_node_action = torch.cat((current_latent, this_sequence[seq]), axis = 1)
                                        current_latent = self.classif_model.graph_dynamics(graph_node_action)
                                    test_loss += loss_func(current_latent, outs_2['current_embed'])
                                    # print(sample_action)
                                    # print(test_loss)
                                    loss_list.append(test_loss)
                                    if(test_loss.detach().cpu().numpy() < min_cost):
                                        min_action = this_sequence
                                        min_cost = test_loss.detach().cpu().numpy()
                                #print(loss_list)
                                node_pose_numpy = node_pose.cpu().numpy()
                                if True: #for seq in range(len(min_action)):
                                    this_seq_numpy = min_action[0].cpu().numpy()
                                    all_action_sequence_list.append(this_seq_numpy[0])
                                    #print(this_seq_numpy[0][:3])
                                    change_id = np.argmax(this_seq_numpy[0][:3])
                                    #print(change_id)
                                    node_pose_numpy[change_id][3:6] += this_seq_numpy[0][3:]
                                    node_pose_numpy[change_id][5] -= 0.02
                                    #if(this_seq_numpy[0])
                                    all_node_pose_list.append(node_pose_numpy)
                                    node_pose = torch.Tensor(node_pose_numpy).to(device)
                                    generate_edge_embed_list = self.generate_edge_embed(node_pose)
                                    #print(generate_edge_embed_list)
                                    data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, self.edge_inp_size, torch.stack(scene_emb_list[0]), min_action[0])
                                    batch = Batch.from_data_list([data]).to(device)
                                    outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
                            # print(min_action)
                            # print(min_cost)
                            print(node_pose)

                            #print(node_pose_numpy)
                            print(node_pose_goal)
                            print(all_action_sequence_list)
                            #print(all_node_pose_list)
                            print(self.dynamics_loss(node_pose[:,:6], node_pose_goal[:,:6]))
                                #print(action)

        # #print(batch_result_dict)
        planning_leap = 0
        if not train and self.execute_planning and not self.evaluate_end_relations:
            leap  = total_succes_num
            planning_leap = planning_success_num
        else:
            leap = 0
            planning_leap = 0
        return batch_result_dict, leap, planning_leap 
          
    def get_next_data_from_dataloader(self, dataloader, train):
        args = self.config.args
        data = None
        if args.train_type == 'all_object_pairs':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij_cut_food':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij_attn':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_g_f_ij_box_stacking_node_label':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'all_object_pairs_gnn':
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == ALL_OBJ_PAIRS_GNN_NEW:
            if self.data_sequence:
                data, data_next = dataloader.get_next_all_object_pairs_for_scene_sequence(train) # get the sequence results
            else:
                data, data_next = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == ALL_OBJ_PAIRS_GNN_RAW_INFO:
            data = dataloader.get_next_all_object_pairs_for_scene(train)
        elif args.train_type == 'unfactored_scene':
            data = dataloader.get_next_voxel_for_scene(train=train)
        elif args.train_type == 'unfactored_scene_resnet18':
            data = dataloader.get_next_voxel_for_scene(train=train)
        else:
            raise ValueError(f"Invalid train type: {args.train_type}")
        return data, data_next
   
    def train_next(self, train=True, viz_images=False, save_embedding=True, log_prefix='', threshold = 0.8):
        #Entrypoint to train/test

        args = self.config.args
        log_freq_iters = args.log_freq_iters if train else 10
        dataloader = self.dataloader
        device = self.config.get_device()

        train_data_size = dataloader.number_of_scene_data(train)

        # Reset log counter 
        train_step_count, test_step_count = 0, 0

        #planning_success_list = []


        self.set_model_device(device)

        result_dict = {
            'data_info': {
                'path': [],
                'info': [],
            },
            # 'emb' Key is saved in hdf5 files, hence add keys here that will
            #  be numpy arrays.
            'emb': {
                'train_img_emb': [],
                'test_img_emb': [],
                'train_gt': [],
                'train_pred': [],
                'test_gt': [],
                'test_pred': [],
            },
            'output': {
                'gt': [],
                'pred': [],
                'test_gt': [],
                'test_pred': [],
                'best_test_gt': [],
                'best_test_pred': [],
                'test_f1_score': [],
                'test_wt_f1_score': [],
                'test_conf': [],
                'scene_path': [],
                'scene_all_object_pair_path': [],
                'test_scene_path': [],
                'test_scene_all_object_pair_path': [],
                'best_test_scene_path': [],
                'best_test_scene_all_object_pair_path': [],
            },
            'conf': {
                'train': [],
                'test': [],
            }
        }
        
        if self.evaluate_end_relations:
            if self.using_sub_goal:
                total_num_relations = 5
                #self.fail_mp_num, self.total_num = self.dataloader.get_fail_motion_planner_num()
                self.fail_exe_num = np.zeros((total_num_relations, 1))
                self.success_exe_num = np.zeros((total_num_relations, 1))
                self.fail_pred_exe_num = np.zeros((total_num_relations, 1))
                self.success_pred_exe_num = np.zeros((total_num_relations, 1))
                self.fail_pred_num = np.zeros((total_num_relations, 1))
                self.success_pred_num = np.zeros((total_num_relations, 1))
                self.success_planning_num = np.zeros((total_num_relations, 1))
                self.fail_planning_num = np.zeros((total_num_relations, 1))
                self.planning_pr = np.zeros((total_num_relations , 1,4))
                self.total_num = np.zeros((total_num_relations, 1))
                failed_reasoning_num = dataloader.get_fail_reasoning_num()
                if self.using_multi_step_statistics:
                    self.total_num[0,0] = failed_reasoning_num
                self.detection_pr = np.zeros((7,4))
            else:
                self.fail_mp_num, self.total_num = self.dataloader.get_fail_motion_planner_num()
                self.fail_exe_num = 0
                self.success_exe_num = 0
                self.fail_pred_exe_num = 0
                self.success_pred_exe_num = 0
                self.fail_pred_num = 0
                self.success_pred_num = 0
                self.success_planning_num = 0
                self.fail_planning_num = 0
                self.planning_pr = np.zeros((1,4))
                self.detection_pr = np.zeros((7,4))
                
        
        num_epochs = args.num_epochs if train else 1
        if args.save_embedding_only:
            num_epochs = 1

        total_leap = 0
        total_true_leap = 0

        total_sudo_success_num = 0
        total_sudo_total_num = 0

        total_planning_num = 0
        total_success_planning_num = 0

        if self.pointconv_baselines:
            self.planning_pr = np.zeros((1,4))
            self.detection_pr = np.zeros((7,4))
            self.pointconv_run_label = 0
        
        if train_data_size == 0:
            num_batches = 1000
            if not train:
                num_epochs = 1
                num_batches = 100
            for e in range(num_epochs):
                for batch_idx in range(num_batches):
                    if True:
                            batch_result_dict, leap = self.run_model_on_batch_ground_truth(
                            train=train,
                            save_preds=True,
                            save_emb=True)
                    run_batch_end_time = time.time()
                    if leap != -1:
                        total_leap += 1
                        total_true_leap += leap
                    train_step_count += 1

                    if train and train_step_count % args.save_freq_iters == 0:
                        self.save_checkpoint(train_step_count)
        else:
            for e in range(num_epochs):
                dataloader.reset_scene_batch_sampler(train=train, shuffle=train)

                batch_size = args.batch_size #if train else 32
                num_batches = train_data_size // batch_size
                if train_data_size % batch_size != 0:
                    num_batches += 1

                # print(train_data_size)
                # print(num_batches)
                data_idx = 0

                n_classes = args.classif_num_classes
                result_dict['conf']['train'].append(
                    np.zeros((n_classes, n_classes), dtype=np.int32))
                for k in ['gt', 'pred', 'scene_path', 'scene_all_object_pair_path', 
                        'test_scene_path', 'test_scene_all_object_pair_path']:
                    result_dict['output'][k] = []
                for k in ['train_img_emb', 'train_gt', 'train_pred']:
                    result_dict['emb'][k] = []

        
                for batch_idx in range(num_batches):

                    batch_data = []
                    batch_data_next = []

                    while len(batch_data) < batch_size and data_idx < train_data_size:  # in current version, we totally ignore batch size
                        data, data_next = self.get_next_data_from_dataloader(dataloader, train)
                        batch_data.append(data)
                        batch_data_next.append(data_next)
                        data_idx = data_idx + 1

                    proc_data_start_time = time.time()

                    
                    x_dict = self.process_raw_batch_data_point_cloud(batch_data)
                    # Now collate the batch data together
                    x_tensor_dict = self.collate_batch_data_to_tensors_point_cloud(x_dict)

                    x_dict_next = self.process_raw_batch_data_point_cloud(batch_data_next)
                    # Now collate the batch data together
                    x_tensor_dict_next = self.collate_batch_data_to_tensors_point_cloud(x_dict_next)

                    if self.use_point_cloud_embedding:
                        if self.e2e:
                            if self.manual_relations: 
                                batch_result_dict, leap, planning_leap = self.run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e_manual_relations(
                                    x_tensor_dict,
                                    x_tensor_dict_next,
                                    batch_size,
                                    train=train,
                                    save_preds=True,
                                    save_emb=True, 
                                    threshold = threshold)
                            else:
                                batch_result_dict, leap, planning_leap = self.run_model_on_batch_torch_geometry_pick_primitive_new_relational_classifier_point_cloud_e2e(
                                    x_tensor_dict,
                                    x_tensor_dict_next,
                                    batch_size,
                                    train=train,
                                    save_preds=True,
                                    save_emb=True, 
                                    threshold = threshold)

                    if leap != -1:
                        if self.manual_relations:
                            total_leap += 1
                        else:
                            total_leap += 1
                        total_true_leap += leap
                        if planning_leap == 1:
                            total_sudo_total_num += 1
                            total_sudo_success_num += leap
                    if planning_leap != -1:
                        total_planning_num += 1
                        total_success_planning_num += planning_leap
                    train_step_count += 1
                    if not train:
                        if total_leap != 0:
                            print('sudo execution success num',total_true_leap/total_leap)
                        if total_sudo_total_num != 0:
                            print('sudo execution success num based on planning threshold',total_sudo_success_num/total_sudo_total_num)

                        if total_planning_num != 0:
                            print('planning success num', total_success_planning_num/total_planning_num)

                        
                    

                    if train and train_step_count % args.save_freq_iters == 0:
                        self.save_checkpoint(train_step_count)
        
        if total_leap != 0:
            print(total_true_leap/total_leap)
        if total_sudo_total_num != 0:
            sudo_success_rate = total_sudo_success_num/total_sudo_total_num
        else:
            sudo_success_rate = -1
        return batch_result_dict, total_success_planning_num/total_planning_num, sudo_success_rate

def main(args):
    
    dtype = torch.FloatTensor
    if args.cuda:
        dtype = torch.cuda.FloatTensor

    config = BaseVAEConfig(args, dtype=dtype) #This just contains the dtype, args, and separates out result_dir.
    create_log_dirs(config) #Creates result dir structure if it doesn't exist.

    trainer = MultiObjectVoxelPrecondTrainerE2E(config)

    get_planning_success_leap = config.args.get_planning_success_leap #whether to use a sequence of planning success threshold #Not sure what this means.
    test_end_relations_leap = config.args.test_end_relations #whether to use the test_end_relations #Also pretty vague
    if get_planning_success_leap:
        threshold_list = []
        for i in range(11):
            threshold_list.append(0+ 0.1*i)

    planning_success_list = []
    sudo_execution_success_list = []
    result_dict_list = []
    success_planning_rate_list = []
    success_exe_rate_list= []
    success_pred_rate_list= []
    success_pred_exe_rate_list= []
    pure_success_planning_rate_list = []
    pure_success_exe_rate_list= []
    pure_success_pred_rate_list= []
    pure_success_pred_exe_rate_list= []
    fall_mp_list = []
    if config.args.using_sub_goal:
        planning_prf = np.zeros((5,1,3))
    else:
        planning_prf = np.zeros((1,3))
    detection_prf = np.zeros((7,3))


    #####################################################################################
    # Entry points:
    #####################################################################################

    ##Test
    if len(args.checkpoint_path) > 0:
        # Checkpoint path provided, so presumably for testing. #Note: Why is testing sent to the train_next with a parameter of 'train' = false. This seems silly.
        trainer.load_checkpoint(args.checkpoint_path)
                
        if get_planning_success_leap:
            for i in range(len(threshold_list)):
                
                # Test_end_relations_leap FALSE ; get_planning_success_leap TRUE
                ##TRAIN##
                result_dict, planning_success_rate, execution_success_list = trainer.train_next(train=False,
                                            viz_images=False,
                                            save_embedding=True, 
                                            threshold = threshold_list[i])


                planning_success_list.append(planning_success_rate)
                sudo_execution_success_list.append(execution_success_list)
                print('all planning success list', planning_success_list)
                print('all sudo execution success list', sudo_execution_success_list)
        else:

            # Test_end_relations_leap FALSE ; get_planning_success_leap FALSE
            ##TRAIN##
            result_dict, planning_success_rate, execution_success_list = trainer.train_next(train=False,
                                                viz_images=False,
                                                save_embedding=True, 
                                                threshold = 0)

            # PRINT AND COMPUTE SOME RESULTS                              
            if config.args.evaluate_end_relations:
                if config.args.using_sub_goal:
                    for relation_i in range(result_dict['planning_pr'].shape[0]):
                        
                        planning_prf[relation_i,0,0] = result_dict['planning_pr'][relation_i][0][0] / (result_dict['planning_pr'][relation_i][0][0] + result_dict['planning_pr'][relation_i][0][1])
                        planning_prf[relation_i,0,1] = result_dict['planning_pr'][relation_i][0][0] / (result_dict['planning_pr'][relation_i][0][0] + result_dict['planning_pr'][relation_i][0][3])
                        planning_prf[relation_i,0,2] = (2*planning_prf[relation_i,0,0]*planning_prf[relation_i,0,1])/(planning_prf[relation_i,0,0] + planning_prf[relation_i,0,1])

                        
                        for detection_i in range(detection_prf.shape[0]):
                            detection_prf[detection_i][0] = result_dict['detection_pr'][detection_i][0] / (result_dict['detection_pr'][detection_i][0] + result_dict['detection_pr'][detection_i][1])
                            detection_prf[detection_i][1] = result_dict['detection_pr'][detection_i][0] / (result_dict['detection_pr'][detection_i][0] + result_dict['detection_pr'][detection_i][3])
                            detection_prf[detection_i][2] = (2*detection_prf[detection_i][0]*detection_prf[detection_i][1])/(detection_prf[detection_i][0] + detection_prf[detection_i][1])
                        
                        
                        success_planning_rate_list.append(result_dict['success_planning_num'][relation_i]/result_dict['total_num'])
                        success_exe_rate_list.append(result_dict['success_exe_num'][relation_i][0]/result_dict['total_num'][relation_i][0])
                        success_pred_exe_rate_list.append(result_dict['success_pred_exe_num'][relation_i][0]/result_dict['total_num'][relation_i][0])
                        success_pred_rate_list.append(result_dict['success_pred_num'][relation_i][0]/result_dict['total_num'][relation_i][0])
                        pure_success_planning_rate_list.append(result_dict['success_planning_num'][relation_i][0]/(result_dict['total_num'][relation_i][0] - result_dict['fail_mp_num']))
                        pure_success_exe_rate_list.append(result_dict['success_exe_num'][relation_i][0]/((result_dict['total_num'][relation_i][0]) - result_dict['fail_mp_num'] - result_dict['point_cloud_not_complete']))
                        pure_success_pred_exe_rate_list.append(result_dict['success_pred_exe_num'][relation_i][0]/((result_dict['total_num'][relation_i][0]) - result_dict['fail_mp_num'] - result_dict['point_cloud_not_complete']))
                        pure_success_pred_rate_list.append(result_dict['success_pred_num'][relation_i][0]/((result_dict['total_num'][relation_i][0]) - result_dict['fail_mp_num']))
                    fall_mp_list.append(result_dict['fail_mp_num']/result_dict['total_num'][relation_i][0])
                    result_dict_list.append(result_dict)
                else:
                    print('result_dict', result_dict)
                    print(result_dict['planning_pr'])
                    print(result_dict['detection_pr'])
                    planning_prf[0,0] = result_dict['planning_pr'][0][0] / (result_dict['planning_pr'][0][0] + result_dict['planning_pr'][0][1])
                    planning_prf[0,1] = result_dict['planning_pr'][0][0] / (result_dict['planning_pr'][0][0] + result_dict['planning_pr'][0][3])
                    planning_prf[0,2] = (2*planning_prf[0,0]*planning_prf[0,1])/(planning_prf[0,0] + planning_prf[0,1])

                    print('planning prf', planning_prf)
                    for detection_i in range(detection_prf.shape[0]):
                        detection_prf[detection_i][0] = result_dict['detection_pr'][detection_i][0] / (result_dict['detection_pr'][detection_i][0] + result_dict['detection_pr'][detection_i][1])
                        detection_prf[detection_i][1] = result_dict['detection_pr'][detection_i][0] / (result_dict['detection_pr'][detection_i][0] + result_dict['detection_pr'][detection_i][3])
                        detection_prf[detection_i][2] = (2*detection_prf[detection_i][0]*detection_prf[detection_i][1])/(detection_prf[detection_i][0] + detection_prf[detection_i][1])
                    print('detection prf', detection_prf)
                    result_dict_list.append(result_dict)
                    success_planning_rate_list.append(result_dict['success_planning_num']/result_dict['total_num'])
                    success_exe_rate_list.append(result_dict['success_exe_num']/result_dict['total_num'])
                    success_pred_exe_rate_list.append(result_dict['success_pred_exe_num']/result_dict['total_num'])
                    success_pred_rate_list.append(result_dict['success_pred_num']/result_dict['total_num'])
                    pure_success_planning_rate_list.append(result_dict['success_planning_num']/(result_dict['total_num'] - result_dict['fail_mp_num']))
                    pure_success_exe_rate_list.append(result_dict['success_exe_num']/((result_dict['total_num']) - result_dict['fail_mp_num'] - result_dict['point_cloud_not_complete']))
                    pure_success_pred_exe_rate_list.append(result_dict['success_pred_exe_num']/((result_dict['total_num']) - result_dict['fail_mp_num'] - result_dict['point_cloud_not_complete']))
                    pure_success_pred_rate_list.append(result_dict['success_pred_num']/((result_dict['total_num']) - result_dict['fail_mp_num']))
                    fall_mp_list.append(result_dict['fail_mp_num']/result_dict['total_num'])
        
        #save and print more results
        test_result_dir = os.path.join(
            os.path.dirname(args.checkpoint_path), '{}_result_{}'.format(
                args.cp_prefix, os.path.basename(args.checkpoint_path)[:-4]))
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)

        pkl_path = os.path.join(test_result_dir, 'result_info.pkl')
        with open(pkl_path, 'wb') as pkl_f:
            result_pkl_dict = result_dict 
            # = {
            #     'data_info': result_dict['data_info'],
            #     'output': result_dict['output'],
            #     'conf': result_dict['conf']}
            pickle.dump(result_pkl_dict, pkl_f, protocol=2)
            print("Did save test info: {}".format(pkl_path))
        #print(result_dict_list)
        
        if 'planning_pr' in result_dict:
            print(result_dict['planning_pr'])
            print(result_dict['detection_pr'])
            print(planning_prf.shape)
            print(detection_prf.shape)
            print('planning prf', planning_prf)
            print('detection prf', detection_prf)

            print('planning prf', planning_prf[:,0, -1].tolist())
            print('detection prf', detection_prf[:, -1].tolist())
            print('success_exe_rate_list', success_exe_rate_list)
            print('success_pred_exe_rate_list', success_pred_exe_rate_list)
            print('success_pred_rate_list', success_pred_rate_list)
        # print('pure_success_exe_rate_list', pure_success_exe_rate_list)
        # print('pure_success_pred_exe_rate_list', pure_success_pred_exe_rate_list)
        # print('pure_success_pred_rate_list', pure_success_pred_rate_list)
        # print('fall_mp_list', fall_mp_list)
    
    #Train
    else:
        #Checkpoint path not provided

        config_pkl_path = os.path.join(args.result_dir, 'config.pkl')
        config_json_path = os.path.join(args.result_dir, 'config.json')
        with open(config_pkl_path, 'wb') as config_f:
            pickle.dump((args), config_f, protocol=2)
            print(bcolors.c_red("Did save config: {}".format(config_pkl_path)))
        with open(config_json_path, 'w') as config_json_f:
            config_json_f.write(json.dumps(args.__dict__))

        result_dict = trainer.train_next(viz_images=True)
        
        if args.save_embedding_only:
            if not os.path.exists(args.emb_save_path):
                # Create all intermediate dirs if required
                os.makedirs(args.emb_save_path)
            save_emb_data_to_h5(args.emb_save_path, result_dict)
        print(result_dict_list)
        print(success_exe_rate_list)
        print(success_pred_exe_rate_list)
        print(success_pred_rate_list)
        print(fall_mp_list)


if __name__ == '__main__':
    #Review needed args
    parser = argparse.ArgumentParser(
        description='Train for precond classification directly from images.')
    add_common_args_to_parser(parser,
                              cuda=True,
                              result_dir=True,
                              checkpoint_path=True,
                              num_epochs=True,
                              batch_size=True,
                              lr=True,
                              save_freq_iters=True,
                              log_freq_iters=True,
                              print_freq_iters=True,
                              test_freq_iters=True)

    parser.add_argument('--train_dir', required=True, action='append',
                        help='Path to hdf5 file.')
    parser.add_argument('--test_dir', required=True, action='append',
                        help='Path to hdf5 file.')
    
    parser.add_argument('--train_type', type=str, default='all_pairs',
                        choices=[
                            'all_object_pairs', 
                            'unfactored_scene',
                            'unfactored_scene_resnet18',
                            'all_object_pairs_g_f_ij',
                            'all_object_pairs_g_f_ij_cut_food',
                            'all_object_pairs_g_f_ij_attn',
                            'all_object_pairs_g_f_ij_box_stacking_node_label',
                            'all_object_pairs_gnn',
                            'all_object_pairs_gnn_raw_obj_info',
                            'all_object_pairs_gnn_new',
                            'pointcloud'
                            ],
                        help='Training type to follow.')
    parser.add_argument('--emb_lr', required=True, type=float, default=0.0,
                        help='Learning rate to use for embeddings.')

    parser.add_argument('--cp_prefix', type=str, default='',
                        help='Prefix to be used to save embeddings.')
    parser.add_argument('--max_train_data_size', type=int, default=10000,
                        help='Max train data size.')
    parser.add_argument('--max_test_data_size', type=int, default=10000,
                        help='Max test data size.')

    parser.add_argument('--z_dim', type=int, default=128,
                        help='Embedding size to extract from image.')

    # Loss weights
    parser.add_argument('--loss_type', type=str, default='classif',
                        choices=['classif'], help='Loss type to use')
    parser.add_argument('--weight_precond_loss', type=float, default=1.0,
                        help='Weight for precond pred loss.')
    parser.add_argument('--classif_num_classes', type=int, default=2,
                        help='Number of classes for classification.')
    parser.add_argument('--use_dynamic_bce_loss', type=str2bool, default=False,
                        help='Use dynamic BCE loss.')

    parser.add_argument('--add_xy_channels', type=int, default=0,
                        choices=[0, 1],
                        help='0: no xy append, 1: xy append '
                             '2: xy centered on bb')
    parser.add_argument('--use_bb_in_input', type=int, default=1, choices=[0,1],
                        help='Use bb in input')
    # 0: sparse voxels that is the scene size is fixed and we have the voxels in there.
    # 1: dense voxels, such that the given scene is rescaled to fit the max size.
    parser.add_argument('--voxel_datatype', type=int, default=0,
                         choices=[0, 1],
                         help='Voxel datatype to use.')
    parser.add_argument('--use_spatial_softmax', type=str2bool, default=False,
                         help='Use spatial softmax.')
    parser.add_argument('--save_full_3d', type=str2bool, default=False,
                        help='Save 3d voxel representation in memory.')
    parser.add_argument('--expand_voxel_points', type=str2bool, default=False,
                        help='Expand voxel points to internal points of obj.')

    # Get Embeddings for data
    parser.add_argument('--save_embedding_only', type=str2bool, default=False,
                        help='Do not train precond model, just save the embedding for train data.')
    parser.add_argument('--emb_checkpoint_path', type=str, default='', 
                        help='Checkpoint path for embedding model.')
    parser.add_argument('--emb_save_path', type=str, default='', 
                        help='Path to save embeddings.')
    parser.add_argument('--save_data_path', type=str, default='', 
                        help='Path to savetxt file to get goal relations.')
                    
    parser.add_argument('--use_contact_edges_only', type=str2bool, default=False,
                        help='Use contact edges only')
    parser.add_argument('--use_backbone_for_emb', type=str2bool, default=True,
                        help='Use backbone for learned emb.')
    parser.add_argument('--test_end_relations', type=str2bool, default=False,
                        help='whether to use the test_end_relations')
    parser.add_argument('--evaluate_end_relations', type=str2bool, default=False,
                        help='whether to use the evaluate the end relations especially for the real-world case')
    parser.add_argument('--set_max', type=str2bool, default=True,
                        help='whether to use set_max method')
    parser.add_argument('--max_objects', type=int, default=5,
                        help='max_objects in this experiments')
    parser.add_argument('--total_sub_step', type=int, default=2,
                        help='total sub steps for multi-step test')
    parser.add_argument('--save_sampling_points', type=str2bool, default=False,
                        help='whether to use save and sampling points')
    parser.add_argument('--sampling_points', type=str2bool, default=False,
                        help='whether to use sampling points') 
    parser.add_argument('--recursive_saving', type=str2bool, default=False,
                        help='whether to use recursive saving') 
    parser.add_argument('--get_planning_success_leap', type=str2bool, default=False,
                        help='whether to use a sequence of planning success threshold')
    parser.add_argument('--save_all_planning_info', type=str2bool, default=False,
                        help='whether to save all planning info')
    parser.add_argument('--real_data', type=str2bool, default=False,
                        help='whether to use real data')
    parser.add_argument('--pointconv_baselines', type=str2bool, default=False,
                        help='whether to use pointconv baselines as a comparison')
    parser.add_argument('--bounding_box_baselines', type=str2bool, default=False,
                        help='whether to use bounding_box baselines as a comparison')
    parser.add_argument('--mlp', type=str2bool, default=False,
                        help='whether to use mlp baselines as a comparison')
    parser.add_argument('--use_multiple_train_dataset', type=str2bool, default=False,
                        help='whether to use use_multiple_train_dataset')
    parser.add_argument('--use_multiple_test_dataset', type=str2bool, default=False,
                        help='whether to use use_multiple_test_dataset')
    parser.add_argument('--manual_relations', type=str2bool, default=False,
                        help='whether to use manual_relations')
    parser.add_argument('--execute_planning', type=str2bool, default=True,
                        help='whether to use execute_planning in the test')
    parser.add_argument('--consider_current_relations', type=str2bool, default=False,
                        help='whether consider current relations')
    parser.add_argument('--consider_end_relations', type=str2bool, default=True,
                        help='whether consider end relations in the evaluate_end_relations')
    parser.add_argument('--updated_behavior_params', type=str2bool, default=True,
                        help='whether to use updated_behavior_params')
    parser.add_argument('--using_sub_goal', type=str2bool, default=False,
                        help='whether to use using_sub_goal')
    parser.add_argument('--start_id', type=int, default=5,
                        help='start_id in hthe training')
    parser.add_argument('--max_size', type=int, default=10,
                        help='max_size if the training dataset')
    parser.add_argument('--start_test_id', type=int, default=10,
                        help='start_test_id of the test dataset')
    parser.add_argument('--test_max_size', type=int, default=10,
                        help='test_max_size of the test dataset') #execute_planning
    parser.add_argument('--set_random_seed', type=str2bool, default=False,
                        help='whether to set random seed')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for numpy and torch') #execute_planning
    parser.add_argument('--seperate_range', type=str2bool, default=False,
                        help='whether to seperate_range')
    parser.add_argument('--random_sampling_relations', type=str2bool, default=True,
                        help='whether to random_sampling_relations')
    parser.add_argument('--using_delta_training', type=str2bool, default=False,
                        help='whether to use using_delta_training')
    parser.add_argument('--cem_planning', type=str2bool, default=True,
                        help='whether to use cem planning')
    parser.add_argument('--pick_place', type=str2bool, default=False,
                        help='whether to use pick place skill')        
    parser.add_argument('--pushing', type=str2bool, default=True,
                        help='whether to use pushing skill')  
    parser.add_argument('--rcpe', type=str2bool, default=False,
                        help='whether to use relational classifier and pose estimation baselines')    
    parser.add_argument('--using_multi_step', type=str2bool, default=False,
                        help='whether to use multi step planning as a task and motion planning style') 
    parser.add_argument('--graph_search', type=str2bool, default=False,
                        help='whether to use graph search in the multi-step planning')
    parser.add_argument('--using_multi_step_latent', type=str2bool, default=False,
                        help='whether to use using_multi_step_latent which means some model-based RL sampling methods') 
    parser.add_argument('--test_next_step', type=str2bool, default=False,
                        help='whether to use test_next_step which means plans actions based on first steps. also whether to use test_next_step for pickplace') 
    parser.add_argument('--using_latent_regularization', type=str2bool, default=True,
                        help='whether to use use regularization loss in the latent space.') 
    parser.add_argument('--save_many_data', type=str2bool, default=False,
                        help='whether to use save many data together') 
    parser.add_argument('--using_multi_step_statistics', type=str2bool, default=False,
                        help='whether to use using_multi_step_statistics to get statistics for the multi-step test.')
    parser.add_argument('--sampling_once', type=str2bool, default=False,
                        help='whether to sampling_once for random sample goal.')
    
    
    # parser.add_argument('--test_next_step', type=str2bool, default=False,
    #                     help='whether to use test_next_step for pickplace') 

    
    
    
    
                                      
                        


    args = parser.parse_args()
    pprint.pprint(args.__dict__)
    np.set_printoptions(precision=4, linewidth=120)

    if args.set_random_seed:
        seed = args.seed  # previous version is all 0
        np.random.seed(seed)
        torch.manual_seed(seed)

    main(args)