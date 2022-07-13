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

from relational_precond.dataloader.real_robot_dataloader import AllPairVoxelDataloaderPointCloud3stack
from zmq import device

class Trainer:
    def __init__():
        pass
    
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

    # TODO: REVIEW AND EDIT
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

    def run_model(self,
                    x_tensor_dict,
                    x_tensor_dict_next,
                    batch_size: int,
                    learning_rate: float=0.0001
                    ):
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        
        img_emb_anchor = self.emb_model(x_tensor_dict_next['batch_anchor_voxel'])
        img_emb_other = self.emb_model(x_tensor_dict_next['batch_other_voxel'])
        img_emb_single = self.emb_model(x_tensor_dict['batch_voxel_single'])
        img_emb_next_single = self.emb_model(x_tensor_dict_next['batch_voxel_single'])


        node_pose = torch.cat((one_hot_encoding, x_tensor_dict['batch_all_obj_pair_pos'][0]), 1)
        node_pose_goal = torch.cat((one_hot_encoding, x_tensor_dict_next['batch_all_obj_pair_pos'][0]), 1)

        #WTF????
        x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'][0]
        x_tensor_dict_next['batch_all_obj_pair_relation'] = x_tensor_dict_next['batch_all_obj_pair_relation'][0]#print('edge shape', outs['edge_embed'].shape)

        one_hot_encoding = np.zeros((self.num_nodes, self.max_objects))

        for one_hot_i in range(len(select_obj_num_range)):
            one_hot_encoding[one_hot_i][(int)(select_obj_num_range[one_hot_i])] = 1
        one_hot_encoding_tensor = torch.Tensor(one_hot_encoding).to(device)
        latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)

        node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)

        select_obj_num_range_next = select_obj_num_range_next.cpu().numpy()[0]

        if self.set_max:
            one_hot_encoding_next = np.zeros((self.num_nodes, self.max_objects))
        else:
            one_hot_encoding_next = np.zeros((self.num_nodes, self.num_nodes))
        
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

        total_loss += self.bce_loss(outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation'][:, :])
        total_loss += self.bce_loss(outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])
        
        # print('current', [outs_decoder['pred_sigmoid'][:], x_tensor_dict['batch_all_obj_pair_relation']])
        # print('pred', [outs_decoder_2['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation']])
        
        total_loss += self.dynamics_loss(outs['pred_embedding'], outs_2['current_embed'])
        total_loss += self.dynamics_loss(outs['pred_edge_embed'], outs_2['edge_embed'])
        total_loss += self.bce_loss(outs_decoder_2_edge['pred_sigmoid'][:], x_tensor_dict_next['batch_all_obj_pair_relation'][:, :])

        print(total_loss)   

        self.opt_emb.zero_grad()
        self.opt_classif.zero_grad()
        self.opt_classif_decoder.zero_grad()
      
        total_loss.backward()
        if args.emb_lr >= 1e-5:
            self.opt_emb.step()
        self.opt_classif.step()
        self.opt_classif_decoder.step()

        

    def train(
        self,
        num_epochs :int,
        batch_size :int,
        dataloader :AllPairVoxelDataloaderPointCloud3stack
        ):

        num_batches = dataloader.number_of_scene_data() // batch_size
        if dataloader.number_of_scene_data() % batch_size != 0:
            num_batches += 1

        for e_idx in range(num_epochs):
            dataloader.reset_scene_batch_sampler()

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


def main(args):
    
    dtype = torch.FloatTensor
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor

    trainer = Trainer()

    

if __name__ == '__main__':
    #Limiting arguments. Will add config file if needed.
    parser = argparse.ArgumentParser(
        description='Train relational predictor model.'
        )

    parser.add_argument('--train_dir', required=True, action='append',
                        help='Path to training data.')

    args = parser.parse_args()
    main(args)
