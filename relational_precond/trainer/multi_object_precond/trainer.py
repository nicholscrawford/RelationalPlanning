import argparse
import copy
import json
import os
import pickle
import pprint
import sys
import time

from itertools import permutations

#sys.path.append(os.getcwd())

import h5py
import numpy as np
from relational_precond.trainer.base_train import BaseVAETrainer
import torch
import torch.nn as nn
import torch.optim as optim

from relational_precond.model.GNN_pytorch_geometry import GNNTrainer, GNNModel, MLPModel, GNNModelOptionalEdge, MLPModelOptionalEdge

from relational_precond.dataloader.real_robot_dataloader import AllPairVoxelDataloaderPointCloud3stack
from relational_precond.model.contact_model import PointConv


from torch_geometric.data import Batch, Data, DataLoader


class Trainer(BaseVAETrainer):
    def __init__(self, max_objects: int = 8):
        self.max_objects = max_objects

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.cuda.FloatTensor
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.FloatTensor
            
        self.emb_model = PointConv(normal_channel=False)
        self.classif_model = GNNModelOptionalEdge(
                    128, #self.node_inp_size
                    128, #input arg z_dim self.edge_inp_size,
                    relation_output_size = 7, #input arg z_dim
                    node_output_size = 128, 
                    predict_edge_output = True,
                    edge_output_size = 128,
                    graph_output_emb_size=16, 
                    node_emb_size=128, 
                    edge_emb_size=128,
                    message_output_hidden_layer_size=128,  
                    message_output_size=128, 
                    node_output_hidden_layer_size=64,
                    all_classifier = False,
                    predict_obj_masks=False,
                    predict_graph_output=False,
                    use_edge_embedding = False,
                    use_edge_input = False, 
                    max_objects = 8 #args max objects
                            )
        self.classif_model_decoder = GNNModelOptionalEdge(
                    128, #self.node_emb_size, 
                    128, #self.edge_emb_size,
                    relation_output_size = 7, #args.z_dim, 
                    node_output_size = 128, #self.node_inp_size, 
                    predict_edge_output = True,
                    edge_output_size = 7,  #edge_inp_size,
                    graph_output_emb_size=16, 
                    node_emb_size=128, 
                    edge_emb_size=128,
                    message_output_hidden_layer_size=128,  
                    message_output_size=128, 
                    node_output_hidden_layer_size=64,
                    all_classifier = False,
                    predict_obj_masks=False,
                    predict_graph_output=False,
                    use_edge_embedding = False,
                    use_edge_input = True, 
                    max_objects = 8
                )
        
        self.bce_loss = nn.BCELoss()
        self.dynamics_loss = nn.MSELoss()
        
        self.opt_emb = optim.Adam(self.emb_model.parameters(), lr=0.0001)
        self.opt_classif = optim.Adam(self.classif_model.parameters(), lr=1e-4) 
        self.opt_classif_decoder = optim.Adam(self.classif_model_decoder.parameters(), lr=1e-4) 

    # TODO: REVIEW AND EDIT
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

        #args = self.config.args
        x_dict = proc_batch_dict

        for b, data in enumerate(batch_data):
            if True:
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

    # TODO: REVIEW AND EDIT
    def collate_batch_data_to_tensors_point_cloud(self, proc_batch_dict):
        '''Collate processed batch into tensors.'''
        # Now collate the batch data together
        x_tensor_dict = {}
        x_dict = proc_batch_dict
        device = self.device

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
   
    # TODO: Remove unused params.
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

    def set_model_device(self, device=torch.device("cpu")):
        model_list = self.get_model_list()
        for m in model_list:
            m.to(device)

    # Potentially uneccessary
    def get_model_list(self):
        return [self.emb_model, self.classif_model,self.classif_model_decoder]

    def run_model(self,
                    x_tensor_dict,
                    x_tensor_dict_next,
                    batch_size: int,
                    learning_rate: float=0.0001
                    ):
        
        device = self.device

        """
            TODO: Figure this function out more.

            Re-implimenting batches may be useful for training speed/accuracy?
        """

        voxel_data_single = x_tensor_dict['batch_voxel_single']
        voxel_data_next_single = x_tensor_dict_next['batch_voxel_single']
        select_obj_num_range = x_tensor_dict['batch_select_obj_num_range']
        select_obj_num_range_next = x_tensor_dict_next['batch_select_obj_num_range']
        action = x_tensor_dict_next['batch_action']
        self.num_nodes = x_tensor_dict['batch_num_objects'].cpu().numpy().astype(int)[0]


        img_emb_single = self.emb_model(voxel_data_single)
        img_emb_next_single = self.emb_model(voxel_data_next_single)

        #Can check for image embedding zeroing out.
        #print(f"VOXEL_DATA_SINGLE: {voxel_data_single[0][0][0]}\tIMG_EMB_SINGLE: {img_emb_single[0][0]}")                
        
        one_hot_encoding = torch.eye(self.num_nodes).float().to(device)

        action_list = []
        for _ in range(self.num_nodes):
            action_list.append(action[0][0][:])
        action_torch = torch.stack(action_list)

        x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'][0]
        x_tensor_dict_next['batch_all_obj_pair_relation'] = x_tensor_dict_next['batch_all_obj_pair_relation'][0]
        select_obj_num_range = select_obj_num_range.cpu().numpy()[0]

        one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))

                    
        for one_hot_i in range(len(select_obj_num_range)):
            one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
        one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
        latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
  
        node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)


        select_obj_num_range_next = select_obj_num_range_next.cpu().numpy()[0]


        one_hot_encoding_next = np.zeros((self.num_nodes, self.max_objects))


        for one_hot_i in range(len(select_obj_num_range_next)):
            one_hot_encoding_next[one_hot_i][(int)(select_obj_num_range_next[one_hot_i])] = 1
        one_hot_encoding_next_tensor = torch.Tensor(one_hot_encoding_next).to(device)
        latent_one_hot_encoding_next = self.classif_model.one_hot_encoding_embed(one_hot_encoding_next_tensor)
        node_pose_goal = torch.cat([img_emb_next_single, latent_one_hot_encoding_next], dim = 1)
  
        data = self.create_graph(self.num_nodes, -1 , node_pose, 0, None, action_torch)
        data_next = self.create_graph(self.num_nodes,  -1 , node_pose_goal, 0, None, action_torch)


        batch = Batch.from_data_list([data]).to(device)

        outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
        data_1_decoder = self.create_graph(self.num_nodes, -1, outs['pred'], -1, outs['pred_edge'], action_torch)
        batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
        outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)
                        
        batch2 = Batch.from_data_list([data_next]).to(device)

        outs_2 = self.classif_model(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch, batch2.action)
        data_2_decoder = self.create_graph(self.num_nodes, -1, outs_2['pred'], -1, outs_2['pred_edge'], action_torch)
        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)
        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                        
        data_2_decoder_edge = self.create_graph(self.num_nodes, -1, outs['pred_embedding'], -1, outs['pred_edge_embed'], action_torch)
        batch_decoder_2_edge = Batch.from_data_list([data_2_decoder_edge]).to(device)
        outs_decoder_2_edge = self.classif_model_decoder(batch_decoder_2_edge.x, batch_decoder_2_edge.edge_index, batch_decoder_2_edge.edge_attr, batch_decoder_2_edge.batch, batch_decoder_2_edge.action)
                        
        total_loss = 0


        total_loss += self.bce_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation'][:, :])
        total_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])                        
        total_loss += self.dynamics_loss(outs['pred_embedding'], outs_2['current_embed'])
        total_loss += self.dynamics_loss(outs['pred_edge_embed'], outs_2['edge_embed'])
        total_loss += self.bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
        print(f"Total loss:\t{total_loss}")
        
        self.opt_emb.zero_grad()
        self.opt_classif.zero_grad()
        self.opt_classif_decoder.zero_grad()

        total_loss.backward()
        if learning_rate >= 1e-5:
            self.opt_emb.step()

        self.opt_classif.step()
        self.opt_classif_decoder.step()

    def train(
        self,
        num_epochs :int,
        batch_size :int,
        dataloader :AllPairVoxelDataloaderPointCloud3stack
        ):

        self.set_model_device(self.device)

        num_batches = dataloader.number_of_scene_data() // batch_size
        if dataloader.number_of_scene_data() % batch_size != 0:
            num_batches += 1

        for e_idx in range(num_epochs):
            dataloader.reset_scene_batch_sampler()
            data_idx = 0

            for _ in range(num_batches):
                data_batch = []
                data_next_batch = []

                while len(data_batch) < batch_size and data_idx < dataloader.number_of_scene_data():  # in current version, we totally ignore batch size
                    data, data_next = dataloader.get_next_all_object_pairs_for_scene()
                    data_batch.append(data)
                    data_next_batch.append(data_next)
                    data_idx = data_idx + 1
                
                x_dict = self.process_raw_batch_data_point_cloud(data_batch)
                    # Now collate the batch data together
                x_tensor_dict = self.collate_batch_data_to_tensors_point_cloud(x_dict)

                x_dict_next = self.process_raw_batch_data_point_cloud(data_next_batch)
                # Now collate the batch data together
                x_tensor_dict_next = self.collate_batch_data_to_tensors_point_cloud(x_dict_next)

                self.run_model(
                    x_tensor_dict,
                    x_tensor_dict_next,
                    batch_size
                    )
                print(f"Epoch:\t{e_idx}/{num_epochs}")

    def train_run_model(self,
        x_tensor_dict,
        x_tensor_dict_next):

        device = self.device

        voxel_data_single = x_tensor_dict['batch_voxel_single']
        voxel_data_next_single = x_tensor_dict_next['batch_voxel_single']
        this_one_hot_encoding_numpy = x_tensor_dict_next['batch_this_one_hot_encoding'].detach().cpu().numpy()[0,0]

        #Is this necessary?
        """
        for check_i in range(this_one_hot_encoding_numpy.shape[0]):
            if this_one_hot_encoding_numpy[check_i] == 0:
                for check_i_again in range(check_i,this_one_hot_encoding_numpy.shape[0]):
                    if this_one_hot_encoding_numpy[check_i_again] == 1:
                        print(this_one_hot_encoding_numpy)
                        raise ValueError("invalide this one hot encoding")
        """

        select_obj_num_range = x_tensor_dict['batch_select_obj_num_range']
        select_obj_num_range_next = x_tensor_dict_next['batch_select_obj_num_range']
        self.pc_center = x_tensor_dict['batch_pc_center'].cpu().detach().numpy()

        action = x_tensor_dict_next['batch_action']
        self.num_nodes = x_tensor_dict['batch_num_objects'].cpu().numpy().astype(int)[0]

        img_emb_single = self.emb_model(voxel_data_single)
        img_emb_next_single = self.emb_model(voxel_data_next_single)

        one_hot_encoding = torch.eye(self.num_nodes).float().to(device)

        node_pose = torch.cat((one_hot_encoding, x_tensor_dict['batch_all_obj_pair_pos'][0]), 1)
        node_pose_goal = torch.cat((one_hot_encoding, x_tensor_dict_next['batch_all_obj_pair_pos'][0]), 1)

        x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'][0]
        x_tensor_dict_next['batch_all_obj_pair_relation'] = x_tensor_dict_next['batch_all_obj_pair_relation'][0]

        select_obj_num_range = select_obj_num_range.cpu().numpy()[0]
        
        one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))

        for one_hot_i in range(len(select_obj_num_range)):
            one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
        one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
        latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
        print('latent_one_hot_encoding, img_emb_single', [latent_one_hot_encoding.shape, img_emb_single.shape])
        node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)

        select_obj_num_range_next = select_obj_num_range_next.cpu().numpy()[0]
        
        one_hot_encoding_next = np.zeros((self.num_nodes, self.max_objects))
        for one_hot_i in range(len(select_obj_num_range_next)):
            one_hot_encoding_next[one_hot_i][(int)(select_obj_num_range_next[one_hot_i])] = 1
        one_hot_encoding_next_tensor = torch.Tensor(one_hot_encoding_next).to(device)
        latent_one_hot_encoding_next = self.classif_model.one_hot_encoding_embed(one_hot_encoding_next_tensor)
        node_pose_goal = torch.cat([img_emb_next_single, latent_one_hot_encoding_next], dim = 1)

        action_list = []
        for _ in range(self.num_nodes):
            action_list.append(action[0][0][:])
        action_torch = torch.stack(action_list)

        data = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
        data_next = self.create_graph(self.num_nodes, self.node_emb_size, node_pose_goal, 0, None, action_torch)
        
        batch = Batch.from_data_list([data]).to(device)

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
        
        print(f"Total loss:\t{total_loss}")

        goal_relations_list = []   
        expected_action_list = []

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


        sub_goal_list_i = []
        sub_goal_list_j = []
        consider_end_range = 5

        self.sampling_relations_number_range = [1,5,9,13,17]

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

        total_num = 0
        total_succes_num = 0

        for each_goal in range(len(goal_relations_list)):
            for each_index in range(len(sub_goal_list_i)):
                index_i = sub_goal_list_i[each_index]
                index_j = sub_goal_list_j[each_index]
                x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(goal_relations_list[each_goal]).to(device)

                success_num = 0
                total_num += 1
                planning_success_num = 0
                planning_total_num = 0
                planning_threshold = threshold

                plannning_sequence = 1
                sample_sequence = 1
                num_nodes = self.num_nodes

                data_1 = self.create_graph(self.num_nodes, self.node_inp_size, node_pose, 0, None, action_torch)
                batch = Batch.from_data_list([data_1]).to(device)
                outs = self.classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)


                data_1_decoder = self.create_graph(self.num_nodes, self.node_emb_size, outs['pred'], self.edge_emb_size, outs['pred_edge'], action_torch)                    
                batch_decoder = Batch.from_data_list([data_1_decoder]).to(device)
                outs_decoder = self.classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

                min_cost = 1e5
                loss_list = []

                for obj_mov in range(self.num_nodes):
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
                        
                                action = np.zeros((num_nodes, self.max_objects + 3))

                                for i in range(action.shape[0]):
                                    action[i][obj_mov] = 1
                                    action[i][-3:] = action_numpy[obj_mov]
                                
                                sample_action = torch.Tensor(action).to(device)

                                this_sequence = []
                                this_sequence.append(sample_action)
                                loss_func = nn.MSELoss()
                                test_loss = 0
                                current_latent = outs['current_embed']
                                egde_latent = outs['edge_embed']
                                #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                                for seq in range(len(this_sequence)):
                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                    graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                                    current_latent = self.classif_model.graph_dynamics(graph_node_action)

                                for seq in range(len(this_sequence)):
                                    #print([current_latent, this_sequence[seq]])
                                    #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                                    edge_num = egde_latent.shape[0]
                                    edge_action_list = []
                                    current_action = self.classif_model.action_emb(this_sequence[seq])
                                    for _ in range(edge_num):
                                        edge_action_list.append(current_action[0])
                                    edge_action = torch.stack(edge_action_list)
                                    graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                                    egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)

                                data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                    
                                batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                                outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)

                                test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])

                                costs.append(test_loss.detach().cpu().numpy())

                            index = np.argsort(costs)
                            elite = act[index,:,:]
                            elite = elite[:3, :, :]
                                # print('elite')
                                # print(elite)
                            act_mu = elite.mean(axis = 0)
                            act_sigma = elite.std(axis = 0)
                            print([act_mu, act_sigma])

                        chosen_action = act_mu
                        action_numpy = np.zeros((num_nodes, 3))
                        action_numpy[obj_mov][0] = chosen_action[0, 0]
                        action_numpy[obj_mov][1] = chosen_action[0, 1]
                        action_numpy[obj_mov][2] = 0 #np.random.uniform(-0.2,0.2)

                        action_numpy_variance = np.zeros((num_nodes, 2))
                        action_numpy_variance[obj_mov][0] = act_sigma[0, 0]
                        action_numpy_variance[obj_mov][1] = act_sigma[0, 1]

                        action = np.zeros((num_nodes, self.max_objects + 3))
                        
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
    
                        loss_func = nn.MSELoss()
                        test_loss = 0
                        current_latent = outs['current_embed']
                        egde_latent = outs['edge_embed']
                        #print('prev_embed, edge_embed, action', [current_latent, egde_latent, sample_action])
                        for seq in range(len(this_sequence)):
                            current_action = self.classif_model.action_emb(this_sequence[seq])
                            graph_node_action = torch.cat((current_latent, current_action), axis = 1)
                            current_latent = self.classif_model.graph_dynamics(graph_node_action)
                        for seq in range(len(this_sequence)):
                            #print([current_latent, this_sequence[seq]])
                            #edge_action = torch.stack([this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0], this_sequence[seq][0]])
                            edge_num = egde_latent.shape[0]
                            edge_action_list = []
                            current_action = self.classif_model.action_emb(this_sequence[seq])
                            for _ in range(edge_num):
                                edge_action_list.append(current_action[0])
                            edge_action = torch.stack(edge_action_list)
                            graph_edge_action = torch.cat((egde_latent, edge_action), axis = 1)
                            egde_latent = self.classif_model.graph_edge_dynamics(graph_edge_action)

                        data_2_decoder = self.create_graph(self.num_nodes, self.node_emb_size, current_latent, self.edge_emb_size, egde_latent, sample_action)
                                                    
                        batch_decoder_2 = Batch.from_data_list([data_2_decoder]).to(device)

                        outs_decoder_2 = self.classif_model_decoder(batch_decoder_2.x, batch_decoder_2.edge_index, batch_decoder_2.edge_attr, batch_decoder_2.batch, batch_decoder_2.action)
                                                                        
                        test_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][index_i, index_j], x_tensor_dict_next['batch_all_obj_pair_relation'][index_i, index_j])
                        print('test_loss', test_loss)
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
                
                this_seq_numpy = min_action[0].cpu().numpy()
                change_id = np.argmax(this_seq_numpy[0][:self.num_nodes])
                    #print(change_id)
                if change_id == 0:
                    change_id_leap = 1
                    node_pose_numpy[0][-3:] += this_seq_numpy[0][-3:]

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
                

                success_num = 1
                if min_action_numpy[0][expected_action_list[each_goal][0] - 1] != 1:
                    success_num = 0
                if min_action_numpy[0][-2]*expected_action_list[each_goal][1] < 0: #np.abs(min_action_numpy[0][3] - action_numpy[0][3]) >= 0.06 or np.abs(min_action_numpy[0][4] - action_numpy[0][4]) >= 0.06:
                    success_num = 0
                total_succes_num += success_num
                

                print(total_succes_num)
                print(total_succes_num/total_num) 

        return

        
    def test(self, dataloader):
        
        train_data_size = dataloader.number_of_scene_data(False)
        train_step_count = 0
        device = self.device

        self.set_model_device(device)

        dataloader.reset_scene_batch_sampler(False, False)
        
        batch_size = 1
        num_batches = train_data_size // batch_size

        if train_data_size % batch_size != 0:
            num_batches += 1

        data_idx = 0

        for _ in range(num_batches):

            batch_data=[]
            batch_data_next=[]

            while len(batch_data) < batch_size and data_idx < train_data_size:  # in current version, we totally ignore batch size
                data, data_next = self.get_next_data_from_dataloader(dataloader, train)
                batch_data.append(data)
                batch_data_next.append(data_next)
                data_idx = data_idx + 1
                
                x_dict = self.process_raw_batch_data_point_cloud(batch_data)
                # Now collate the batch data together
                x_tensor_dict = self.collate_batch_data_to_tensors_point_cloud(x_dict)

                x_dict_next = self.process_raw_batch_data_point_cloud(batch_data_next)
                # Now collate the batch data together
                x_tensor_dict_next = self.collate_batch_data_to_tensors_point_cloud(x_dict_next)

                self.train_run_model(x_tensor_dict, x_tensor_dict_next)