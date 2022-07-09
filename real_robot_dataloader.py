"""
This file is different from dataloader/real_robot_dataloader.py but I'm not sure exaclty how. 
"""
import numpy as np

import csv
import h5py
import os
import ipdb
import pickle
import glob
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import time
import json
import copy

from itertools import permutations, combinations

from relational_precond.utils import data_utils
from relational_precond.utils import image_utils
from relational_precond.utils.colors import bcolors
#from relational_precond.dataloader.robot_octree_data import RobotVoxels, SceneVoxels, BoxStackingSceneVoxels
#from relational_precond.dataloader.robot_octree_data import RobotAllPairVoxels
from relational_precond.dataloader.farthest_point_sampling import farthest_point_sampling
from relational_precond.utils import math_util

import torch
from torchvision.transforms import functional as F

from typing import List, Dict

def _convert_string_key_to_int(d: dict) -> dict:
    dict_with_int_key = {}
    for k in d.keys():
        dict_with_int_key[int(k)] = d[k]
    assert len(d) == len(dict_with_int_key), "Emb data size changed."
    return dict_with_int_key


def plot_voxel_plot(voxel_3d):
    # ==== Visualize ====
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = voxel_3d[0, ...].nonzero()
    # ax.scatter(x - 25, y - 25, z - 25)
    ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


def load_emb_data_from_h5_path(h5_path, pkl_path, max_data_size=None):
    h5f = h5py.File(h5_path, 'r')
    emb_data = data_utils.recursively_get_dict_from_group(h5f)
    # emb_data keys are stored as str in hdf5 file. Convert them to integers
    # for direct comparison with the file data.
    emb_data = _convert_string_key_to_int(emb_data)

    with open(pkl_path, 'rb') as pkl_f:
        emb_data_pkl = pickle.load(pkl_f)
    emb_data_pkl = _convert_string_key_to_int(emb_data_pkl)
    
    assert sorted(emb_data.keys()) == sorted(emb_data_pkl.keys())

    data_keys = sorted(emb_data.keys())
    if max_data_size is None:
        max_data_size = len(data_keys)
    else:
        max_data_size = min(max_data_size, len(data_keys))
    
    data_dict = {}
    for i in range(max_data_size):
        key = data_keys[i]
        val = emb_data[key]
        pkl_val = emb_data_pkl[key]
        assert pkl_val['precond_label'] == int(val['precond_label'])
        assert len(pkl_val['all_object_pair_path']) == val['emb'].shape[0]
        data_dict[i] = {
            'path': pkl_val['path'],
            'precond_label': int(val['precond_label']),
            'all_object_pair_path': pkl_val['all_object_pair_path'],
            'emb': val['emb'],
        }
    
    return data_dict


def get_all_object_pcd_path_for_data_in_line(scene_path):
    files = os.listdir(scene_path)
    return [os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]

def get_all_object_voxel_path_for_data_in_line(scene_path):
    files = sorted(os.listdir(scene_path))
    return [os.path.join(scene_path, p) for p in files if 'voxel_data' in p]

def get_all_object_image_path_for_data_in_line(scene_path):
    files = sorted(os.listdir(scene_path))
    return [os.path.join(scene_path, p) for p in files if 'img_data.pkl' in p]


def get_all_object_pcd_path_for_cut_food(scene_path):
    files = os.listdir(scene_path)
    obj_path = [os.path.join(scene_path, p) for p in sorted(files) 
                if 'cloud_cluster' in p]
    knife_path = os.path.join(scene_path, 'knife_object.pcd')
    assert os.path.basename(obj_path[0]) == 'cloud_cluster_0.pcd'
    return [knife_path] + obj_path


def get_all_object_pcd_path_for_box_stacking(scene_path, remove_obj_id):
    files = os.listdir(os.path.join(scene_path, 'object_cloud'))
    remove_cloud_fname = f'cloud_cluster_{remove_obj_id}.pcd'
    if remove_obj_id != -1:
        assert remove_cloud_fname in files, "To remove obj id not found"
    path_list = [os.path.join(scene_path, 'object_cloud', p) for p in sorted(files) 
                 if 'cloud_cluster' in p and p != remove_cloud_fname]
    return path_list


def get_valid_objects_to_remove_for_box_stacking(scene_path, scene_info):
    # For objects that only have one object at the lowest level, we cannot remove
    # that object since if we remove those voxels the scene has no way of knowing
    # where the ground is and the system would be trivially unstable (knowing the ground)
    pos_list, obj_id_list = [], []
    for k in sorted(scene_info.keys()):
        if k == 'precond_obj_id' or k == 'test_removal' or k == 'contact_edges':
            continue
        pos_list.append(scene_info[k]['pos']) 
        obj_id_list.append(int(k))
    pos_arr = np.array(pos_list)
    pos_arr_on_ground = pos_arr[:, 2] > 0.82
    ground_obj_idx = np.where(pos_arr[:, 2] > 0.82)[0]
    assert len(ground_obj_idx) >= 1, "Atleast one object should be on ground"

    obj_pcd = os.listdir(os.path.join(scene_path, 'object_cloud'))
    # assert len(obj_pcd) == len(obj_id_list)
    obj_id_list = obj_id_list[:len(obj_pcd)]

    if len(ground_obj_idx) > 1:
        return sorted(obj_id_list)
    else:
        assert np.sum(pos_arr_on_ground) == 1
        print(f"Only 1 obj on ground {obj_id_list[ground_obj_idx[0]]}, {scene_path}. Will remove")
        valid_obj_id_list = [obj_id for i, obj_id in enumerate(obj_id_list) 
                             if not pos_arr_on_ground[i]]
        return valid_obj_id_list
  
class AllPairVoxelDataloader(object):
    def __init__(self, 
                 config,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False):

        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        # print('train dir')
        # print(self.train_dir_list)
        self.train_idx_to_data_dict = {}
        self.max_train_data_size = 100000
        for train_dir in self.train_dir_list:
            # print('enter')
            # print(train_dir)
            idx_to_data_dict = self.load_voxel_data(
                train_dir, 
                max_data_size=self.max_train_data_size, 
                demo_idx=len(self.train_idx_to_data_dict),
                max_data_from_dir=None)
            if len(self.train_idx_to_data_dict) > 0:
                assert (max(list(self.train_idx_to_data_dict.keys())) < 
                        max(list(idx_to_data_dict.keys())))

            self.train_idx_to_data_dict.update(idx_to_data_dict)
            if len(self.train_idx_to_data_dict) >= self.max_train_data_size:
                break

        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))

        self.test_idx_to_data_dict = {}
        self.max_test_data_size = 50000
        for test_dir in self.test_dir_list:
            idx_to_data_dict = self.load_voxel_data(
                test_dir, 
                max_data_size=self.max_test_data_size, 
                demo_idx=len(self.test_idx_to_data_dict),
                max_data_from_dir=None)
            if len(self.test_idx_to_data_dict) > 0:
                assert (max(list(self.test_idx_to_data_dict.keys())) < 
                        max(list(idx_to_data_dict.keys())))

            self.test_idx_to_data_dict.update(idx_to_data_dict)
            if len(self.test_idx_to_data_dict) >= self.max_test_data_size:
                break

        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        # The following dicts contain two keys ('idx' and 'order')
        self.train_all_pair_sample_order = {}
        self.test_all_pair_sample_order = {}
        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}


    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()
            # print(root)
            # print(dirs)
            # print(files)

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line':
                if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                    continue
                # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                #     continue #yixuan test

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                elif self.scene_type == 'box_stacking_node_label':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type, load_voxels_of_removed_obj=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                elif self.scene_type == "box_stacking_node_label":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                             use_node_label=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        if shuffle:
            np.random.shuffle(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        #print(path)
        precond_label = 1 if 'true' in path.split('/') else 0
        # if precond_label == 0:
        #     assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        all_obj_pair_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos = \
            scene_voxel_obj.get_all_object_pair_voxels()
        
        # import pdb; pdb.set_trace()
        # for l in range(len(all_obj_pair_voxels)):
        #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
        # import pdb; pdb.set_trace()

        #print(all_obj_pair_pos)
        data = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos
        }
        # obj_center_list = scene_voxel_obj.get_object_center_list()
        # data['obj_center_list'] = obj_center_list # yixuan test
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            obj_center_list = scene_voxel_obj.get_object_center_list()
            data['obj_center_list'] = obj_center_list

            contact_edges = scene_voxel_obj.contact_edges
            stable_obj_label = scene_voxel_obj.get_stable_object_label_tensor()
            data['precond_stable_obj_ids'] = stable_obj_label
            if contact_edges is not None:
                data['contact_edges'] = contact_edges
            data['box_stacking_remove_obj_id'] = scene_voxel_obj.get_object_id_to_remove()

        return data
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        #print(sample_order_dict)
        sample_idx = sample_order_dict['idx']
        length = self.number_of_scene_data(train) - 3
        print(length)
        sample_idx = np.random.randint(length)
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        #sample_order_dict['idx'] += 1

        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        sample_idx = np.random.randint(length)
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        #sample_order_dict['idx'] += 1 # yixuan test
        return data, data_next

    def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        if return_obj_voxels:
            obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
            data.update(obj_voxel_dict)

        return data

    def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                               return_obj_voxels=return_obj_voxels)
        sample_order_dict['idx'] += 1
        return data


class AllPairVoxelDataloaderPointCloud(object):
    def __init__(self, 
                 config,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False):

        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}
        max_train_data_size = 120000
        curr_dir_max_data = None
        print('train')
        print(self.train_dir_list)
        print(self.test_dir_list)
        files = sorted(os.listdir(self.train_dir_list[0]))
        
        self.train_pcd_path = [
            os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
        max_size = 500
        data_size = 500
        for train_dir in self.train_pcd_path[:max_size]:
            # print('enter')
            # print(train_dir)
            # idx_to_data_dict = self.load_voxel_data(
            #     train_dir, 
            #     max_data_size=max_train_data_size, 
            #     max_action_idx=8, 
            #     demo_idx=demo_idx,
            #     curr_dir_max_data=None)

            
            
            
            with open(train_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
                continue
            if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
                continue
            #print(data)

            # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
            # for _ in range(3):
            #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
            #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
            idx_to_data_dict[demo_idx] = {}
            # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            idx_to_data_dict[demo_idx]['objects'] = data['objects']
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']
            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            all_pair_scene_object =  RobotAllPairSceneObjectPointCloud(train_dir, self.scene_type) # yixuan test
            idx_to_data_dict[demo_idx]['path'] = train_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1           
        self.train_idx_to_data_dict.update(idx_to_data_dict)
            

        
        #time.sleep(10)
        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))
        # for _ in range(len(self.train_idx_to_data_dict)):
        #     print(self.train_idx_to_data_dict[_])

        self.test_idx_to_data_dict = {}
        idx_to_data_dict = {}
        
        files = sorted(os.listdir(self.test_dir_list[0]))
        
        self.test_pcd_path = [
            os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
        demo_idx = 0
        for test_dir in self.test_pcd_path[:max_size]:
            #data_size = 1500
            
            with open(test_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            
            if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
                continue
            if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
                continue
            #print(data['objects'])

            # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
            # for _ in range(3):
            #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
            #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
            idx_to_data_dict[demo_idx] = {}
            # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            idx_to_data_dict[demo_idx]['objects'] = data['objects']
            
            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']
            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            all_pair_scene_object =  RobotAllPairSceneObjectPointCloud(test_dir, self.scene_type) # yixuan test
            idx_to_data_dict[demo_idx]['path'] = train_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1
        self.test_idx_to_data_dict.update(idx_to_data_dict)
        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        # The following dicts contain two keys ('idx' and 'order')
        self.train_all_pair_sample_order = {}
        self.test_all_pair_sample_order = {}
        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}


    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()
            # print(root)
            # print(dirs)
            # print(files)

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line':
                if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                    continue
                # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                #     continue #yixuan test

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                elif self.scene_type == 'box_stacking_node_label':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type, load_voxels_of_removed_obj=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                elif self.scene_type == "box_stacking_node_label":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                             use_node_label=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        #print(order)
        if shuffle:
            np.random.shuffle(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        #print(path)
        precond_label = 1 if 'true' in path.split('/') else 0
        # if precond_label == 0:
        #     assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos = \
            scene_voxel_obj.get_all_object_pair_voxels()
        
        #print(len(all_obj_pair_voxels))
        # import pdb; pdb.set_trace()
        # for l in range(len(all_obj_pair_voxels)):
        #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
        # import pdb; pdb.set_trace()

        #print(all_obj_pair_pos)
        data = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos
        }
        # obj_center_list = scene_voxel_obj.get_object_center_list()
        # data['obj_center_list'] = obj_center_list # yixuan test
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            obj_center_list = scene_voxel_obj.get_object_center_list()
            data['obj_center_list'] = obj_center_list

            contact_edges = scene_voxel_obj.contact_edges
            stable_obj_label = scene_voxel_obj.get_stable_object_label_tensor()
            data['precond_stable_obj_ids'] = stable_obj_label
            if contact_edges is not None:
                data['contact_edges'] = contact_edges
            data['box_stacking_remove_obj_id'] = scene_voxel_obj.get_object_id_to_remove()

        return data
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        #print(sample_order_dict)
        sample_idx = sample_order_dict['idx']
        length = self.number_of_scene_data(train) - 3
        #print(length)
        sample_idx = np.random.randint(length)
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        # print(scene_idx)
        # scene_idx = 0
        # print(train)
        data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        #sample_order_dict['idx'] += 1
        #print(sample_idx)

        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        sample_idx = np.random.randint(length)
        #print(sample_idx)
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        # print(scene_idx)
        # scene_idx = 0
        data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        #sample_order_dict['idx'] += 1 # yixuan test
        print([data['all_obj_pair_pos'], data_next['all_obj_pair_pos']])
        return data, data_next

    def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        if return_obj_voxels:
            obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
            data.update(obj_voxel_dict)

        return data

    def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                               return_obj_voxels=return_obj_voxels)
        sample_order_dict['idx'] += 1
        return data


class AllPairVoxelDataloaderPointCloudSavePoints(object):
    def __init__(self, 
                 config,
                 four_data = False,
                 classify_data = False,
                 stacking = False, 
                 pushing = False,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False, 
                 test_end_relations = False):
        stacking = stacking
        self.classify_data = classify_data
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels
        self.test_end_relations = test_end_relations

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}
        max_train_data_size = 120000
        curr_dir_max_data = None
        print('train')
        print(self.train_dir_list)
        print(self.test_dir_list)
        files = sorted(os.listdir(self.train_dir_list[0]))
        
        self.train_pcd_path = [
            os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
        max_size = 500
        data_size = 200
        total_objects = 0
        for train_dir in self.train_pcd_path[:]:
            
            print(train_dir)
            # idx_to_data_dict = self.load_voxel_data(
            #     train_dir, 
            #     max_data_size=max_train_data_size, 
            #     max_action_idx=8, 
            #     demo_idx=demo_idx,
            #     curr_dir_max_data=None)

            
            
            
            with open(train_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            if total_objects == 0:
                for k, v in data['objects'].items():
                    if 'block' in k:
                        total_objects += 1

            if self.test_end_relations:
                if data['depth'].shape[0] < 4:
                    data['fail_mp'] = 1
                else:
                    data['fail_mp'] = 0
            #print()
            if four_data:
                sample_time_step = [0, 10, 18, -1]
            elif self.classify_data:
                sample_time_step = [0]
            elif pushing:
                sample_time_step = [0, -1]
            else:
                sample_time_step = [0, 11, -1]
            save_step = 0
            print(total_objects)
            if 'point_cloud_' not in data: #(data['point_cloud_1'].shape[0] == 0): #True: #(data['point_cloud_1'].shape[0] == 0):
                # if(type(data['point_cloud_1']) != list):
                #     data['point_cloud_1'] = []
                #     data['point_cloud_2'] = []
                #     data['point_cloud_3'] = []
                print('enter')
                point_string = 'point_cloud_'
                
                for i in range(total_objects):
                    data[point_string + str(i+1)] = []
                    # data['point_cloud_1'] = data['point_cloud_1'].tolist()
                    # data['point_cloud_2'] = data['point_cloud_2'].tolist()
                    # data['point_cloud_3'] = data['point_cloud_3'].tolist()
                for step in sample_time_step:
                    total_point_cloud = self._get_point_cloud_3(data['depth'][step], data['segmentation'][step], data['projection_matrix'][step], data['view_matrix'][step], total_objects)
                    
                    # print(total_point_cloud[0].shape)
                    # print(total_point_cloud[1].shape)
                    # print(total_point_cloud[2].shape)
                    for i in range(total_objects):
                        #data[point_string + str(i+1)] = []
                        data[point_string + str(i+1)].append(
                            total_point_cloud[i])
                        
                        # data['point_cloud_2'].append(
                        #     total_point_cloud[1])
                        
                        # data['point_cloud_3'].append(
                        #     total_point_cloud[2])
                    save_step += 1
                
                for i in range(total_objects):
                    #data[point_string + str(i+1)] = []
                    data[point_string + str(i+1)] = np.array(data[point_string + str(i+1)], dtype = object)
                # data['point_cloud_2'] = np.array(data['point_cloud_2'], dtype = object)
                # data['point_cloud_3'] = np.array(data['point_cloud_3'], dtype = object)
                #print(data['point_cloud_1'][0].shape)
                # A = np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2
                # A_1 = [A[1], A[2], A[0]]
                #print(self.get_point_cloud_center())
                # print(np.mean(data['point_cloud_1'][0], axis = 0))
                # print(np.mean(data['point_cloud_2'][0], axis = 0))
                # print(np.mean(data['point_cloud_3'][0], axis = 0))
                # print(data['point_cloud_1'].shape)
                # print(data['point_cloud_2'].shape)
                # print(train_dir)
                with open(train_dir, 'wb') as f:
                    pickle.dump((data, attrs), f)
            #print('finish')
            
            
        # for _ in range(len(self.train_idx_to_data_dict)):
        #     print(self.train_idx_to_data_dict[_])

    def _get_point_cloud_3(self, depth, segmentation, projection_matrix, view_matrix, total_objects):
        points = []
        points1 = []
        points2 = []
        points3 = []
        color = []
        for i_o in range(total_objects):
            points.append([])
        cam_width = 512
        cam_height = 512
        #pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(cam_width, cam_height, fx, fy, cam_width/2.0 , -cam_height/2.0), view_matrix)
        # pcd = o3d.geometry.create_point_cloud_from_depth_image(depth_o3d, o3d.camera.PinholeCameraIntrinsic(512, 512, fx, fy, 256.0, 256.0))
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

        # Retrieve depth and segmentation buffer
        # print(depth)
        # print(segmentation)
        depth_buffer = depth
        seg_buffer = segmentation

        # Get camera view matrix and invert it to transform points from camera to world space
        #print(view_matrix)
        vinv = np.linalg.inv(np.matrix(view_matrix))

        # Get camera projection matrix and necessary scaling coefficients for deprojection
        proj = projection_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        depth_buffer[seg_buffer == 0] = -10001

        centerU = cam_width/2
        centerV = cam_height/2
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_buffer[j, i] < -10000:
                    continue
                # This will take all segmentation IDs. Can look at specific objects by
                # setting equal to a specific segmentation ID, e.g. seg_buffer[j, i] == 2
                
                
                for i_o in range(total_objects):
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        #print(p2)
                        #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
                # if seg_buffer[j, i] == 1:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points1.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
                # if seg_buffer[j, i] == 2:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points2.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
                # if seg_buffer[j, i] == 3:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points3.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
        for i_o in range(total_objects):
            points[i_o] = np.array(points[i_o])
        return points #np.array(points1), np.array(points2), np.array(points3)

    def _get_point_cloud_3_noise(self, depth, segmentation, projection_matrix, view_matrix, total_objects):
        points = []
        points1 = []
        points2 = []
        points3 = []
        color = []

        depth = self.add_noise_to_depth(depth, 2, 2)
        depth = self.dropout_random_ellipses(depth, 0.5, 2, 2)

        for i_o in range(total_objects):
            points.append([])
        cam_width = 512
        cam_height = 512
        #pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(cam_width, cam_height, fx, fy, cam_width/2.0 , -cam_height/2.0), view_matrix)
        # pcd = o3d.geometry.create_point_cloud_from_depth_image(depth_o3d, o3d.camera.PinholeCameraIntrinsic(512, 512, fx, fy, 256.0, 256.0))
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

        # Retrieve depth and segmentation buffer
        # print(depth)
        # print(segmentation)
        depth_buffer = depth
        seg_buffer = segmentation

        # Get camera view matrix and invert it to transform points from camera to world space
        #print(view_matrix)
        vinv = np.linalg.inv(np.matrix(view_matrix))

        # Get camera projection matrix and necessary scaling coefficients for deprojection
        proj = projection_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        depth_buffer[seg_buffer == 0] = -10001

        centerU = cam_width/2
        centerV = cam_height/2
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_buffer[j, i] < -10000:
                    continue
                # This will take all segmentation IDs. Can look at specific objects by
                # setting equal to a specific segmentation ID, e.g. seg_buffer[j, i] == 2
                
                
                for i_o in range(total_objects):
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        #print(p2)
                        #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
                # if seg_buffer[j, i] == 1:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points1.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
                # if seg_buffer[j, i] == 2:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points2.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
                # if seg_buffer[j, i] == 3:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points3.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
        for i_o in range(total_objects):
            points[i_o] = np.array(points[i_o])
        return points #np.array(points1), np.array(points2), np.array(points3)


    def add_noise_to_depth(self, depth_img, gamma_shape, gamma_scale):
        """ Distort depth image with multiplicative gamma noise.
            This is adapted from the DexNet 2.0 codebase.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Multiplicative noise: Gamma random variable
        multiplicative_noise = np.random.gamma(gamma_shape, gamma_scale)
        depth_img = multiplicative_noise * depth_img

        return depth_img

    def add_noise_to_xyz(self, xyz_img, depth_img, noise_params):
        """ Add (approximate) Gaussian Process noise to ordered point cloud.
            This is adapted from the DexNet 2.0 codebase.
            @param xyz_img: a [H x W x 3] ordered point cloud
        """
        xyz_img = xyz_img.copy()

        H, W, C = xyz_img.shape

        # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
        #                 which is rescaled with bicubic interpolation.
        small_H, small_W = (np.array([H, W]) / noise_params['gp_rescale_factor']).astype(int)
        additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(small_H, small_W, C))
        additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
        xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

        return xyz_img

    def dropout_random_ellipses(self, depth_img, ellipse_dropout_mean, ellipse_gamma_shape, ellipse_gamma_scale):
        """ Randomly drop a few ellipses in the image for robustness.
            This is adapted from the DexNet 2.0 codebase.
            Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py
            @param depth_img: a [H x W] set of depth z values
        """
        depth_img = depth_img.copy()

        # Sample number of ellipses to dropout
        num_ellipses_to_dropout = np.random.poisson(ellipse_dropout_mean)

        # Sample ellipse centers
        nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T # Shape: [#nonzero_pixels x 2]
        dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
        dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

        # Sample ellipse radii and angles
        x_radii = np.random.gamma(ellipse_gamma_shape, ellipse_gamma_scale, size=num_ellipses_to_dropout)
        y_radii = np.random.gamma(ellipse_gamma_shape, ellipse_gamma_scale, size=num_ellipses_to_dropout)
        angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

        # Dropout ellipses
        for i in range(num_ellipses_to_dropout):
            center = dropout_centers[i, :]
            x_radius = np.round(x_radii[i]).astype(int)
            y_radius = np.round(y_radii[i]).astype(int)
            angle = angles[i]

            # dropout the ellipse
            mask = np.zeros_like(depth_img)
            mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
            depth_img[mask == 1] = 0

        return depth_img

        
        def get_point_cloud_center(self, v):
            A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
            A_1 = [A[1], A[2], A[0]]
            return A_1
        
        def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
            '''Load emb data from h5 files. '''
            self.train_h5_data_dict = load_emb_data_from_h5_path(
                train_h5_path, 
                train_pkl_path,
                max_data_size=self.max_train_data_size)
            self.test_h5_data_dict = load_emb_data_from_h5_path(
                test_h5_path, 
                test_pkl_path,
                max_data_size=self.max_test_data_size)
            
            for k in sorted(self.train_idx_to_data_dict.keys()):
                train_idx_data = self.train_idx_to_data_dict[k]
                h5_data = self.train_h5_data_dict[k]
                scene_voxel_obj = train_idx_data['scene_voxel_obj']
                assert h5_data['path'] == train_idx_data['path']
                assert h5_data['all_object_pair_path'] == \
                    scene_voxel_obj.get_all_object_pair_path()
                assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

            for k in sorted(self.test_idx_to_data_dict.keys()):
                test_idx_data = self.test_idx_to_data_dict[k]
                h5_data = self.test_h5_data_dict[k]
                scene_voxel_obj = test_idx_data['scene_voxel_obj']
                assert h5_data['path'] == test_idx_data['path']
                assert h5_data['all_object_pair_path'] == \
                    scene_voxel_obj.get_all_object_pair_path()
                assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                            max_data_from_dir=None):
            '''Load images for all demonstrations.

            demo_dir: str. Path to directory containing the demonstration data.
            '''
            demo_idx_to_path_dict = {}
            args = self._config.args

            if not os.path.exists(demo_dir):
                print("Dir does not exist: {}".format(demo_dir))
                return demo_idx_to_path_dict

            data_count_curr_dir = 0
            for root, dirs, files in os.walk(demo_dir, followlinks=False):
                if max_data_size is not None and demo_idx > max_data_size:
                    break
                if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                    break

                # Sort the data order so that we do not have randomness associated
                # with it.
                dirs.sort()
                files.sort()
                # print(root)
                # print(dirs)
                # print(files)

                # ==== Used for data_in_line scee ====
                if self.scene_type == 'data_in_line':
                    if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                        continue
                    # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                    #     continue #yixuan test

                # ==== Used for cut food ====
                if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                    continue
                    
                # ==== Used for box stacking ====
                if self.scene_type == 'box_stacking' and 'info.json' not in files:
                    continue

                # ==== Used for box stacking ====
                if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                    continue

                # TODO: Add size_channels flag
                if self.load_all_object_pair_voxels:
                    if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                        all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                    elif self.scene_type == 'box_stacking':
                        multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                            root, self.scene_type)
                    elif self.scene_type == 'box_stacking_node_label':
                        multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                            root, self.scene_type, load_voxels_of_removed_obj=True)
                    else:
                        raise ValueError(f"Invalid scene type {self.scene_type}")

                    # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                    #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                    #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
                else:     
                    all_pair_scene_object = None
            
                if self.load_scene_voxels:
                    if self.scene_type == "data_in_line":
                        single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                    elif self.scene_type == "cut_food":
                        single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                    elif self.scene_type == "box_stacking":
                        multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                    elif self.scene_type == "box_stacking_node_label":
                        multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                                use_node_label=True)
                    else:
                        raise ValueError(f"Invalid scene type {self.scene_type}")
                else:
                    single_scene_voxel_obj = None
                
                if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                    if self.load_scene_voxels:
                        all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                    if self.load_all_object_pair_voxels:
                        assert not self.load_scene_voxels
                        all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                        if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                            self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                        
                    for scene_idx, scene in enumerate(all_scene_list):
                        demo_idx_to_path_dict[demo_idx] = {}
                        demo_idx_to_path_dict[demo_idx]['path'] = root

                        if self.load_all_object_pair_voxels:
                            demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                            demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                                multi_all_pair_scene_object 
                        else:
                            demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                        if self.load_scene_voxels:
                            demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                            demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                                multi_scene_voxel_obj
                        else:
                            demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                        demo_idx = demo_idx + 1 
                        data_count_curr_dir = data_count_curr_dir + 1

                else:
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root
                    demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                    demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

                if demo_idx % 10 == 0:
                    print("Did process: {}".format(demo_idx))
                        
            print(f"Did load {data_count_curr_dir} from {demo_dir}")
            return demo_idx_to_path_dict            

        def get_demo_data_dict(self, train=True):
            data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
            return data_dict
        
        def total_pairs_in_all_scenes(self, train=True):
            data_dict = self.get_demo_data_dict(train)
            data_size = 0
            for scene_idx, scene in data_dict.items():
                data_size += scene['scene_voxel_obj'].number_of_object_pairs
            return data_size
        
        def number_of_scenes(self, train=True):
            data_dict = self.get_demo_data_dict(train)
            return len(data_dict)

        def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
            if train:
                sampling_dict = self.train_scene_sample_order
            elif not train:
                sampling_dict = self.test_scene_sample_order
            else:
                raise ValueError("Invalid value")

            data_dict = self.get_demo_data_dict(train)
            order = sorted(data_dict.keys())
            #print(order)
            if shuffle:
                np.random.shuffle(order)

            sampling_dict['order'] = order
            sampling_dict['idx'] = 0
            
        def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
            if train:
                sampler_dict = self.train_all_pair_sample_order
            else:
                sampler_dict = self.test_all_pair_sample_order

            data_dict = self.get_demo_data_dict(train)

            # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
            order = []
            for scene_idx, scene_dict in data_dict.items():
                for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                    order.append((scene_idx, i))

            if shuffle:
                np.random.shuffle(order)

            sampler_dict['order'] = order
            sampler_dict['idx'] = 0

        def number_of_scene_data(self, train=True):
            return self.number_of_scenes(train)

        def number_of_pairs_data(self, train=True):
            return self.total_pairs_in_all_scenes(train)

        def get_scene_voxel_obj_at_idx(self, idx, train=True):
            data_dict = self.get_demo_data_dict(train)
            return data_dict[idx]['scene_voxel_obj']
        
        def get_some_object_pair_train_data_at_idx(self, idx, train=True):
            # Get the actual data idx for this idx. Since we shuffle the data
            # internally these are not same values
            sample_order_dict = self.train_all_pair_sample_order if train else \
                self.test_all_pair_sample_order
            (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

            data_dict = self.get_demo_data_dict(train)[scene_idx]

            path = data_dict['path']
            scene_voxel_obj = data_dict['scene_voxel_obj']
            precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

            data = {
                'scene_path': path,
                'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
                'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
                'precond_label': precond_label,
            }
            return data
        
        def get_precond_label_for_demo_data_dict(self, demo_data_dict):
            if self.scene_type == 'data_in_line':
                return self.get_precond_label_for_path(demo_data_dict['path'])
            elif self.scene_type == 'cut_food':
                return self.get_precond_label_for_path(demo_data_dict['path'])
            elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_all_object_pair_voxels:
                    return demo_data_dict['scene_voxel_obj'].precond_label
                elif self.load_scene_voxels:
                    return demo_data_dict['single_scene_voxel_obj'].precond_label
                else:
                    raise ValueError("Invalid label")

        def get_precond_label_for_path(self, path):
            #print(path)
            precond_label = 1 if 'true' in path.split('/') else 0
            # if precond_label == 0:
            #     assert 'false' in path.split('/')
            return precond_label
        
        def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
            data_dict = self.get_demo_data_dict(train)[scene_idx]

            #print(data_dict)
            path = data_dict['path']
            scene_voxel_obj = data_dict['scene_voxel_obj']
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
            precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action = \
                scene_voxel_obj.get_all_object_pair_voxels()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last = \
                scene_voxel_obj.get_all_object_pair_voxels_last()
            
            #print(len(all_obj_pair_voxels))
            # import pdb; pdb.set_trace()
            # for l in range(len(all_obj_pair_voxels)):
            #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
            # import pdb; pdb.set_trace()

            #print(all_obj_pair_pos)
            data = {
                'scene_path': path,
                'num_objects': scene_voxel_obj.number_of_objects,
                'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
                'all_object_pair_voxels': all_obj_pair_voxels,
                'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
                'all_object_pair_other_voxels': all_obj_pair_other_voxels,
                'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
                'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
                'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
                'precond_label': precond_label,
                'all_obj_pair_pos': all_obj_pair_pos,
                'all_obj_pair_orient': all_obj_pair_orient,
                'action': action
            }

            data_last = {
                'scene_path': path,
                'num_objects': scene_voxel_obj.number_of_objects,
                'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
                'all_object_pair_voxels': all_obj_pair_voxels_last,
                'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
                'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
                'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
                'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
                'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
                'precond_label': precond_label,
                'all_obj_pair_pos': all_obj_pair_pos_last,
                'all_obj_pair_orient': all_obj_pair_orient_last,
                'action': action_last
            }
            return data, data_last
        
        def get_next_all_object_pairs_for_scene(self, train=True):
            # First find the next scene index based on the current index
            sample_order_dict = self.train_scene_sample_order if train else \
                self.test_scene_sample_order
            # Get the current sample pointer
            #print(sample_order_dict)
            sample_idx = sample_order_dict['idx']
            length = self.number_of_scene_data(train)
            #print(length)
            sample_idx = np.random.randint(length)
            # Get the actual scene index.
            scene_idx = sample_order_dict['order'][sample_idx]
            # print(scene_idx)
            # scene_idx = 0
            # print(train)
            data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            #sample_order_dict['idx'] += 1
            #print(sample_idx)
            if train:
                return data, data_next
            else:
                sample_idx = np.random.randint(length)
                # Get the actual scene index.
                scene_idx = sample_order_dict['order'][sample_idx]
                test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
                return data, data_next# , test_data#

        def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
            data_dict = self.get_demo_data_dict(train)[scene_idx]

            path = data_dict['path']
            single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
            precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
            voxels = single_scene_voxel_obj.get_scene_voxels()

            # if '3_obj/type_9/28' in path:
            #     plot_voxel_plot(voxels)
            #     import pdb; pdb.set_trace()

            data = {
                'scene_path': path,
                'num_objects': single_scene_voxel_obj.number_of_objects,
                'scene_voxels': torch.FloatTensor(voxels),
                'precond_label': precond_label,
            }

            if return_obj_voxels:
                obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
                data.update(obj_voxel_dict)

            return data

        def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
            # First find the next scene index based on the current index
            sample_order_dict = self.train_scene_sample_order if train else \
                self.test_scene_sample_order
            # Get the current sample pointer
            sample_idx = sample_order_dict['idx']
            # Get the actual scene index.
            scene_idx = sample_order_dict['order'][sample_idx]
            data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                                return_obj_voxels=return_obj_voxels)
            sample_order_dict['idx'] += 1
            return data


class AllPairVoxelDataloaderPointCloudFarthesetSampling(object):
    def __init__(self, 
                 config,
                 classify_data = False,
                 stacking = False, 
                 pushing = False,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False, 
                 real_data = False):
        stacking = stacking
        self.real_data = real_data
        self.classify_data = classify_data
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}
        max_train_data_size = 120000
        curr_dir_max_data = None
        print('train')
        print(self.train_dir_list)
        print(self.test_dir_list)
        files = sorted(os.listdir(self.train_dir_list[0]))
        
        self.train_pcd_path = [
            os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
        max_size = 500
        data_size = 128
        total_objects = 0
        for train_dir in self.train_pcd_path[:]:
            
            print(train_dir)
            total_objects = 0
            # idx_to_data_dict = self.load_voxel_data(
            #     train_dir, 
            #     max_data_size=max_train_data_size, 
            #     max_action_idx=8, 
            #     demo_idx=demo_idx,
            #     curr_dir_max_data=None)

            
            
            
            if self.real_data:
                with open(train_dir, 'rb') as f:
                    data = pickle.load(f)
                #print(data)
                if total_objects == 0:
                    for k, v in data.items():
                        print(k)
                        if 'point_cloud_' in k and 'sampling' not in k:
                            total_objects += 1
                point_string = 'point_cloud_'
                #total_objects = 4
                print(total_objects)
                for i in range(total_objects):
                    data[point_string + str(i+1) + "sampling"] = []
                if True: #
                    for i in range(total_objects):
                        for step in range(data[point_string + str(i+1)].shape[0]):
                            pc = data[point_string + str(i+1)][step]
                            print(pc.shape)
                            if pc.shape[0] < data_size:   # sampling points for any size of the points
                                continue
                            print('enter')
                            sampling_number = 128
                            #print(pc.shape)
                            farthest_indices,_ = farthest_point_sampling(pc, sampling_number)
                            pc_resampled = pc[farthest_indices.squeeze()]
                            #print(pc_resampled.shape)
                            data[point_string + str(i+1) + "sampling"].append(pc_resampled)
                        
                        data[point_string + str(i+1) + "sampling"] = np.array(data[point_string + str(i+1) + "sampling"], dtype = object)
                    with open(train_dir, 'wb') as f:
                        pickle.dump(data, f)
                    print('finish')    
            else:
                with open(train_dir, 'rb') as f:
                    data, attrs = pickle.load(f)
                    leap = 1
                    for k, v in data.items():
                        if 'sampling' in k:
                            leap = 0
                    # if leap == 0:
                    #     print('already sampling')
                    #     continue
                    if total_objects == 0:
                        for k, v in data['objects'].items():
                            if 'block' in k:
                                total_objects += 1

                    #print()
                    if self.classify_data:
                        sample_time_step = [0]
                    elif pushing:
                        sample_time_step = [0, -1]
                    else:
                        sample_time_step = [0, 11, -1]
                    save_step = 0
                    point_string = 'point_cloud_'
                    #print(total_objects)
                    for i in range(total_objects):
                        data[point_string + str(i+1) + "sampling"] = []
                    if True: #
                        for i in range(total_objects):
                            for step in range(data[point_string + str(i+1)].shape[0]):
                                pc = data[point_string + str(i+1)][step]
                                #print(pc.shape)
                                if pc.shape[0] < data_size:   # sampling points for any size of the points
                                    continue
                                print('enter')
                                sampling_number = 128
                                #print(pc.shape)
                                farthest_indices,_ = farthest_point_sampling(pc, sampling_number)
                                pc_resampled = pc[farthest_indices.squeeze()]
                                #print(pc_resampled.shape)
                                data[point_string + str(i+1) + "sampling"].append(pc_resampled)
                            
                            data[point_string + str(i+1) + "sampling"] = np.array(data[point_string + str(i+1) + "sampling"], dtype = object)
                        with open(train_dir, 'wb') as f:
                            pickle.dump((data, attrs), f)
                        print('finish')    


    def _get_point_cloud_3(self, depth, segmentation, projection_matrix, view_matrix, total_objects):
        points = []
        points1 = []
        points2 = []
        points3 = []
        color = []
        for i_o in range(total_objects):
            points.append([])
        cam_width = 512
        cam_height = 512
        #pcd = o3d.geometry.create_point_cloud_from_rgbd_image(rgbd, o3d.camera.PinholeCameraIntrinsic(cam_width, cam_height, fx, fy, cam_width/2.0 , -cam_height/2.0), view_matrix)
        # pcd = o3d.geometry.create_point_cloud_from_depth_image(depth_o3d, o3d.camera.PinholeCameraIntrinsic(512, 512, fx, fy, 256.0, 256.0))
        color_map = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1], [1, 0, 1]])

        # Retrieve depth and segmentation buffer
        # print(depth)
        # print(segmentation)
        depth_buffer = depth
        seg_buffer = segmentation

        # Get camera view matrix and invert it to transform points from camera to world space
        #print(view_matrix)
        vinv = np.linalg.inv(np.matrix(view_matrix))

        # Get camera projection matrix and necessary scaling coefficients for deprojection
        proj = projection_matrix
        fu = 2/proj[0, 0]
        fv = 2/proj[1, 1]

        # Ignore any points which originate from ground plane or empty space
        depth_buffer[seg_buffer == 0] = -10001

        centerU = cam_width/2
        centerV = cam_height/2
        for i in range(cam_width):
            for j in range(cam_height):
                if depth_buffer[j, i] < -10000:
                    continue
                # This will take all segmentation IDs. Can look at specific objects by
                # setting equal to a specific segmentation ID, e.g. seg_buffer[j, i] == 2
                
                
                for i_o in range(total_objects):
                    if seg_buffer[j, i] == i_o + 1:
                        u = -(i-centerU)/(cam_width)  # image-space coordinate
                        v = (j-centerV)/(cam_height)  # image-space coordinate
                        d = depth_buffer[j, i]  # depth buffer value
                        X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                        p2 = X2*vinv  # Inverse camera view to get world coordinates
                        #print(p2)
                        #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        points[i_o].append([p2[0, 2], p2[0, 0], p2[0, 1]])
                        color.append(0)
                # if seg_buffer[j, i] == 1:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points1.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
                # if seg_buffer[j, i] == 2:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points2.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
                # if seg_buffer[j, i] == 3:
                #     u = -(i-centerU)/(cam_width)  # image-space coordinate
                #     v = (j-centerV)/(cam_height)  # image-space coordinate
                #     d = depth_buffer[j, i]  # depth buffer value
                #     X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                #     p2 = X2*vinv  # Inverse camera view to get world coordinates
                #     #print(p2)
                #     #points.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     points3.append([p2[0, 2], p2[0, 0], p2[0, 1]])
                #     color.append(0)
        for i_o in range(total_objects):
            points[i_o] = np.array(points[i_o])
        return points #np.array(points1), np.array(points2), np.array(points3)



    def get_point_cloud_center(self, v):
        A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return A_1
    
    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()
            # print(root)
            # print(dirs)
            # print(files)

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line':
                if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                    continue
                # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                #     continue #yixuan test

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                elif self.scene_type == 'box_stacking_node_label':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type, load_voxels_of_removed_obj=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                elif self.scene_type == "box_stacking_node_label":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                             use_node_label=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        #print(order)
        if shuffle:
            np.random.shuffle(order)
        #print(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        #print(path)
        precond_label = 1 if 'true' in path.split('/') else 0
        # if precond_label == 0:
        #     assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action = \
            scene_voxel_obj.get_all_object_pair_voxels()

        all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last = \
            scene_voxel_obj.get_all_object_pair_voxels_last()
        
        #print(len(all_obj_pair_voxels))
        # import pdb; pdb.set_trace()
        # for l in range(len(all_obj_pair_voxels)):
        #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
        # import pdb; pdb.set_trace()

        #print(all_obj_pair_pos)
        data = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos,
            'all_obj_pair_orient': all_obj_pair_orient,
            'action': action
        }

        data_last = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels_last,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos_last,
            'all_obj_pair_orient': all_obj_pair_orient_last,
            'action': action_last
        }
        return data, data_last
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        #print(sample_order_dict)
        sample_idx = sample_order_dict['idx']
        length = self.number_of_scene_data(train)
        #print(length)
        sample_idx = np.random.randint(length)
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        # print(scene_idx)
        # scene_idx = 0
        # print(train)
        data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        #sample_order_dict['idx'] += 1
        #print(sample_idx)
        if train:
            return data, data_next
        else:
            sample_idx = np.random.randint(length)
            # Get the actual scene index.
            scene_idx = sample_order_dict['order'][sample_idx]
            test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            return data, data_next# , test_data#

    def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        if return_obj_voxels:
            obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
            data.update(obj_voxel_dict)

        return data

    def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                               return_obj_voxels=return_obj_voxels)
        sample_order_dict['idx'] += 1
        return data


class AllPairVoxelDataloaderPointCloudPrimitive(object):
    def __init__(self, 
                 config,
                 stacking = False, 
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False):
        stacking = stacking
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}
        max_train_data_size = 120000
        curr_dir_max_data = None
        print('train')
        print(self.train_dir_list)
        print(self.test_dir_list)
        files = sorted(os.listdir(self.train_dir_list[0]))
        
        self.train_pcd_path = [
            os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
        max_size = 100
        data_size = 256
        for train_dir in self.train_pcd_path[:max_size]:
            
            #print(train_dir)
            # idx_to_data_dict = self.load_voxel_data(
            #     train_dir, 
            #     max_data_size=max_train_data_size, 
            #     max_action_idx=8, 
            #     demo_idx=demo_idx,
            #     curr_dir_max_data=None)

            
            
            
            with open(train_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
            #     continue
            # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
            #     continue
            leap = 1
            
            for k, v in data.items():
                if 'point_cloud' in k:
                    #print([k, v.shape])
                    if len(v.shape) !=3:
                        leap = 0
                        break
                    if v.shape[1] < data_size:
                        leap = 0
                        break
                #print(k)
            eps = 1e-3
            #print('enter', leap)
            
            pos_diff_list_0 = []
            for k, v in data['objects'].items():    
                if 'block' in k:
                    #print(v['position'])
                    if(np.abs(v['position'][0][0] - v['position'][-1][0]) > 1e-5):
                        pos_diff_list_0.append(np.abs(v['position'][0][0] - v['position'][-1][0]))
                        #print(pos_diff_list_0)
                    # self.all_pos_list.append(v['position'][0])
                    # self.all_pos_list_last.append(v['position'][-1])
            
            
            if not stacking:
                for i in range(len(pos_diff_list_0)):
                    for j in range(len(pos_diff_list_0)):
                        if(pos_diff_list_0[i] - pos_diff_list_0[j] > eps):
                            leap = 0
                            #print(leap)
                            break

            pos_diff_list_1 = []
            for k, v in data['objects'].items():    
                if 'block' in k:
                    #print(v['position'])
                    if(np.abs(v['position'][0][0] - v['position'][-1][0]) > 1e-5):
                        pos_diff_list_1.append(np.abs(v['position'][0][0] - v['position'][-1][0]))
                        
            
            
            if not stacking:
                for i in range(len(pos_diff_list_1)):
                    for j in range(len(pos_diff_list_1)):
                        if(pos_diff_list_1[i] - pos_diff_list_1[j] > eps):
                            leap = 0
                            #print(leap)
                            break
                
            
            if leap == 0:
                continue
            #print(data)

            # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
            # for _ in range(3):
            #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
            #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
            idx_to_data_dict[demo_idx] = {}
            # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            idx_to_data_dict[demo_idx]['objects'] = data['objects']
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']
            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            all_pair_scene_object =  RobotAllPairSceneObjectPointCloud(train_dir, self.scene_type) # yixuan test
            idx_to_data_dict[demo_idx]['path'] = train_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1           
        self.train_idx_to_data_dict.update(idx_to_data_dict)
            

        
        #time.sleep(10)
        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))
        # for _ in range(len(self.train_idx_to_data_dict)):
        #     print(self.train_idx_to_data_dict[_])

        self.test_idx_to_data_dict = {}
        idx_to_data_dict = {}
        
        files = sorted(os.listdir(self.test_dir_list[0]))
        
        self.test_pcd_path = [
            os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
        demo_idx = 0
        start_test_id = 0
        for test_dir in self.test_pcd_path[start_test_id: start_test_id + max_size]:
            #print('enter')
            with open(test_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
            #     continue
            # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
            #     continue
            leap = 1
            for k, v in data.items():
                if 'point_cloud' in k:
                    if len(v.shape) !=3:
                        leap = 0
                        break
                    if v.shape[1] < data_size:
                        leap = 0
                        break
            
            pos_diff_list_0 = []
            for k, v in data['objects'].items():    
                if 'block' in k:
                    #print(v['position'])
                    if(np.abs(v['position'][0][0] - v['position'][-1][0]) > 1e-5):
                        pos_diff_list_0.append(np.abs(v['position'][0][0] - v['position'][-1][0]))
                        #print(pos_diff_list_0)
                    # self.all_pos_list.append(v['position'][0])
                    # self.all_pos_list_last.append(v['position'][-1])
            
            
            if not stacking:
                for i in range(len(pos_diff_list_0)):
                    for j in range(len(pos_diff_list_0)):
                        if(pos_diff_list_0[i] - pos_diff_list_0[j] > eps):
                            leap = 0
                            #print(leap)
                            break

            pos_diff_list_1 = []
            for k, v in data['objects'].items():    
                if 'block' in k:
                    #print(v['position'])
                    if(np.abs(v['position'][0][0] - v['position'][-1][0]) > 1e-5):
                        pos_diff_list_1.append(np.abs(v['position'][0][0] - v['position'][-1][0]))
                        
            
            
            if not stacking:
                for i in range(len(pos_diff_list_1)):
                    for j in range(len(pos_diff_list_1)):
                        if(pos_diff_list_1[i] - pos_diff_list_1[j] > eps):
                            leap = 0
                            #print(leap)
                            break
            
            if leap == 0:
                continue
            #print(data)

            # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
            # for _ in range(3):
            #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
            #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
            idx_to_data_dict[demo_idx] = {}
            # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            idx_to_data_dict[demo_idx]['objects'] = data['objects']
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']
            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            all_pair_scene_object =  RobotAllPairSceneObjectPointCloud(test_dir, self.scene_type) # yixuan test
            idx_to_data_dict[demo_idx]['path'] = test_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1           
        self.test_idx_to_data_dict.update(idx_to_data_dict)

        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        # The following dicts contain two keys ('idx' and 'order')
        self.train_all_pair_sample_order = {}
        self.test_all_pair_sample_order = {}
        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}


    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()
            # print(root)
            # print(dirs)
            # print(files)

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line':
                if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                    continue
                # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                #     continue #yixuan test

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                elif self.scene_type == 'box_stacking_node_label':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type, load_voxels_of_removed_obj=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                elif self.scene_type == "box_stacking_node_label":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                             use_node_label=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        #print(order)
        if shuffle:
            np.random.shuffle(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        #print(path)
        precond_label = 1 if 'true' in path.split('/') else 0
        # if precond_label == 0:
        #     assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action = \
            scene_voxel_obj.get_all_object_pair_voxels()

        all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last = \
            scene_voxel_obj.get_all_object_pair_voxels_last()
        
        #print(len(all_obj_pair_voxels))
        # import pdb; pdb.set_trace()
        # for l in range(len(all_obj_pair_voxels)):
        #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
        # import pdb; pdb.set_trace()

        #print(all_obj_pair_pos)
        data = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos,
            'all_obj_pair_orient': all_obj_pair_orient,
            'action': action
        }

        data_last = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels_last,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos_last,
            'all_obj_pair_orient': all_obj_pair_orient_last,
            'action': action_last
        }
        return data, data_last # yixuan test
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        # sample_order_dict = self.train_scene_sample_order if train else \
        #     self.test_scene_sample_order
        
        # sample_idx = sample_order_dict['idx']
        # sample_idx = np.random.randint(length)
        # # Get the actual scene index.
        # scene_idx = sample_order_dict['order'][sample_idx]
        # print(scene_idx)

        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        #data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        

        data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        #sample_order_dict['idx'] += 1
        #print(sample_idx)
        if train:
            return data, data_next
        else:
            # sample_idx = np.random.randint(length)
            # # Get the actual scene index.
            # scene_idx = sample_order_dict['order'][sample_idx]
            # test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            return data, data_next# , test_data#

    def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        if return_obj_voxels:
            obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
            data.update(obj_voxel_dict)

        return data

    def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                               return_obj_voxels=return_obj_voxels)
        sample_order_dict['idx'] += 1
        return data


class AllPairVoxelDataloaderPointCloud3stack(object):
    def __init__(self, 
                 config,
                 max_objects = 5, 
                 use_multiple_train_dataset = False,
                 use_multiple_test_dataset = False,
                 four_data = False,
                 pick_place = False,
                 pushing = False,
                 stacking = False, 
                 set_max = False,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False,
                 test_end_relations = False,
                 real_data = False, 
                 start_id = 0, 
                 max_size = 0,   # begin on max_size = 8000 for all disturbance data
                 start_test_id = 0, 
                 test_max_size = 2,
                 updated_behavior_params = False,
                 pointconv_baselines = False, 
                 save_data_path = None, 
                 evaluate_end_relations = False,
                 using_multi_step_statistics = False,
                 total_multi_steps = 0
                 ):
        #self.train = train
        self.total_multi_steps = total_multi_steps
        self.using_multi_step_statistics = using_multi_step_statistics
        self.evaluate_end_relations = evaluate_end_relations
        self.updated_behavior_params = updated_behavior_params
        self.real_data = real_data
        stacking = stacking
        self.set_max = set_max
        self.four_data = four_data
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.pick_place = pick_place
        self.pushing = pushing
        self.stacking = stacking
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels
        self.test_end_relations = test_end_relations

        self.fail_reasoning_num = 0

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}
        max_train_data_size = 120000
        curr_dir_max_data = None
        
        
        # start_id = 0
        # max_size = 0  # begin on max_size = 8000 for all disturbance data
        # start_test_id = 0
        # test_max_size = 2
        
        data_size = 128
        self.max_size = max_size
        self.test_max_size = test_max_size
        # if self.pushing:
        #     data_size = 128
        #print(len(self.train_pcd_path))
        #if 
        
        if self.four_data:
            total_steps = 4
        if self.pick_place:
            total_steps = 2
        elif self.pushing:
            total_steps = 2
        elif self.stacking:
            total_steps = 3
        self.motion_planner_fail_num = 0
        self.train_id = 0
        self.test_id = 0    


        # save_path = "/home/yhuang/Desktop/mohit_code/savetxt/2022-03-06-20-54-22.pickle" #string_1 + str(10) + ".pickle"

        # with open(save_path, 'rb') as f:
        #     data = pickle.load(f)
        # indent = ''
        
        



        self.use_multiple_train_dataset = use_multiple_train_dataset
        if not self.use_multiple_train_dataset:
            self.train_dir_list = train_dir_list \
                if train_dir_list is not None else config.args.train_dir
        


        if save_data_path != '':
            print(save_data_path)
            save_data_path = os.path.join("/home/yhuang/Desktop/mohit_code/savetxt/", save_data_path + "/0.pickle")
            print(save_data_path)
            
            with open(save_data_path, 'rb') as f:
                data = pickle.load(f)
            self.all_goal_relations = data['goal_relations']
            self.all_predicted_relations = data['predicted_relations']
            self.all_index_i_list = data['all_index_i_list']
            self.all_index_j_list = data['all_index_j_list']
            # print(data['gt_pose'][0])
            # print(data['gt_orientation'][0])
            self.all_data_planned_pose = data['gt_pose']
            self.all_data_planned_orientation = data['gt_orientation']
            self.all_actions = data['action']

            #print(data['gt_extents'][0])
        else:
            self.all_goal_relations = np.ones((50000,5,1))
            self.all_predicted_relations = np.ones((50000,5,1))
            self.all_index_i_list = np.ones((50000,5,1))
            self.all_index_j_list = np.ones((50000,5,1))
        
        if self.use_multiple_train_dataset:
            #self.train_dir_list_total = ['/home/yhuang/Desktop/mohit_code/saved_dataset/data/iiwa_push_2stack_4','/home/yhuang/Desktop/mohit_code/saved_dataset/data/4push_high_success_rate_2']
            self.train_dir_list_total = ['/home/yhuang/Desktop/mohit_code/saved_dataset/data/4push_high_success_rate_4_disturbance_return', 
            '/home/yhuang/Desktop/mohit_code/saved_dataset/data/iiwa_push_2stack_4_disturbance_return']
            
            #self.train_dir_list_total = ['/home/yhuang/Desktop/mohit_code/saved_dataset/data/real_data']
            print('train')
            print(self.train_dir_list_total)
            for train_dir_list in self.train_dir_list_total:
                
                self.train_dir_list = [train_dir_list]
                print(self.train_dir_list)
                
                files = sorted(os.listdir(self.train_dir_list[0]))
                
                self.train_pcd_path = [
                    os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
                # if train_dir_list == '/home/yhuang/Desktop/mohit_code/saved_dataset/data/push_stack_contact_4':
                #     max_size = max_size*2
                for train_dir in self.train_pcd_path[start_id:start_id+max_size]:
                    self.current_goal_relations = self.all_goal_relations[0] # simplify for without test end_relations
                    self.current_predicted_relations = self.all_goal_relations[0] # simplify for without test end_relations
                    self.train_id += 1
                    print(train_dir)            
                    with open(train_dir, 'rb') as f:
                        data, attrs = pickle.load(f)
                    # print((data['point_cloud_1'].shape))
                    # print((data['point_cloud_2'].shape))
                    # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
                    #     continue
                    # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
                    #     continue
                    leap = 1

                    # if self.test_end_relations and data['fail_mp'] == 1:
                    #     leap = 0
                    #     self.motion_planner_fail_num += 1

                    if self.test_end_relations:
                        if 'push' not in attrs['behavior_params']['']['behaviors'] or 'target_pose' not in attrs['behavior_params']['']['behaviors']['push']:
                            leap = 0
                            self.motion_planner_fail_num += 1
                    
                    for k, v in data.items():
                        if 'point_cloud' in k and 'last' not in k:
                            # print(k)
                            # print(v.shape)
                            if(v.shape[0] == 0):
                                leap = 0
                                break
                            if(v.shape[0] != total_steps): # 2 will be total steps of the task
                                leap = 0
                                break
                            for i in range((v.shape[0])):   ## filter out some examples that have too fewer points, for example, object fall over the table. 
                                #print([k, v[i].shape])
                                if(v[i].shape[0] < data_size):
                                    leap = 0
                                    break
                            # if len(v.shape) !=3:
                            #     leap = 0
                            #     break
                            # if v.shape[1] < data_size:
                            #     leap = 0
                            #     break
                        #print(k)
                    # print(data['objects']['block_1']['position'])
                    # print(data['objects']['block_2']['position'])
                    # print(data['objects']['block_3']['position'])
                    if stacking and not pick_place and not pushing:
                        if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58):
                            leap = 0
                        #break
                    eps = 1e-3
                    print(leap)
                    if leap == 0:
                        continue
                    #print(data)

                    # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
                    # for _ in range(3):
                    #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
                    #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
                    idx_to_data_dict[demo_idx] = {}
                    # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                    # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                    idx_to_data_dict[demo_idx]['objects'] = data['objects']
                    # print((data['point_cloud_1'].shape))
                    # print((data['point_cloud_2'].shape))
                    idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

                    idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
                    idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations

                    this_one_hot_encoding = np.zeros((1, 3)) ## quick test
                    idx_to_data_dict[demo_idx]['this_one_hot_encoding'] = this_one_hot_encoding
                    idx_to_data_dict[demo_idx]['index_i'] = [1]
                    idx_to_data_dict[demo_idx]['index_j'] = [1]
                    
                    
                    total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
                    idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
                    idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
                    idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
                    idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
                    self.train = True
                    if self.four_data:
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                    elif not pushing:
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                        #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(train_dir, self.scene_type) # yixuan test
                    else: 
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                    idx_to_data_dict[demo_idx]['path'] = train_dir
                    idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                    demo_idx += 1           
                self.train_idx_to_data_dict.update(idx_to_data_dict)
        else:
            files = sorted(os.listdir(self.train_dir_list[0]))       
            self.train_pcd_path = [
                os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
            for train_dir in self.train_pcd_path[start_id:start_id+max_size]:
                self.current_goal_relations = self.all_goal_relations[self.train_id] # simplify for without test end_relations
                self.current_predicted_relations = self.all_predicted_relations[self.train_id]
                self.current_index_i = self.all_index_i_list[self.train_id]  ## test_id?  I change it to train_id
                self.current_index_j = self.all_index_j_list[self.train_id]
                self.train_id += 1
                
                print(train_dir)            
                with open(train_dir, 'rb') as f:
                    data, attrs = pickle.load(f)
                total_objects = 0
                for k, v in data.items():
                    if 'point_cloud' in k and 'sampling' in k and 'last' not in k:
                        total_objects += 1
                this_one_hot_encoding = np.zeros((1, total_objects))
                # print((data['point_cloud_1'].shape))
                # print((data['point_cloud_2'].shape))
                # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
                #     continue
                # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
                #     continue
                leap = 1

                # if self.test_end_relations and data['fail_mp'] == 1:
                #     leap = 0
                #     self.motion_planner_fail_num += 1

                if self.test_end_relations:
                    if 'push' not in attrs['behavior_params']['']['behaviors'] or 'target_pose' not in attrs['behavior_params']['']['behaviors']['push']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                
                if self.evaluate_end_relations:
                    obj_id = -1
                    for k, v in data.items():
                        if 'point_cloud' in k and 'last' not in k and 'sampling' not in k:
                            this_leap = 1
                            obj_id += 1
                            if(v.shape[0] == 0):
                                this_leap = 0
                                this_one_hot_encoding[0][obj_id] = this_leap
                                break
                            if(v.shape[0] != total_steps): # 2 will be total steps of the task
                                this_leap = 0
                                this_one_hot_encoding[0][obj_id] = this_leap
                                break
                            for i in range((v.shape[0])):
                                #print([k, v[i].shape])
                                if(v[i].shape[0] < data_size):
                                    this_leap = 0
                                    this_one_hot_encoding[0][obj_id] = this_leap
                                    break
                            this_one_hot_encoding[0][obj_id] = this_leap
                else:
                    for k, v in data.items():
                        if 'point_cloud' in k and 'last' not in k:
                            # print(k)
                            # print(v.shape[0])
                            if(v.shape[0] == 0):
                                leap = 0
                                break
                            if(v.shape[0] != total_steps): # 2 will be total steps of the task
                                leap = 0
                                break
                            for i in range((v.shape[0])):
                                #print([k, v[i].shape])
                                if(v[i].shape[0] < data_size):
                                    leap = 0
                                    break
                        
                # print(data['objects']['block_3']['position'])
                if stacking and not pick_place and not pushing:
                    if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58):
                        leap = 0
                    #break
                eps = 1e-3
                print(leap)
                if leap == 0 and not self.evaluate_end_relations:
                    continue
                #print(data)

                # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
                # for _ in range(3):
                #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
                #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
                idx_to_data_dict[demo_idx] = {}
                # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                idx_to_data_dict[demo_idx]['objects'] = data['objects']
                # print((data['point_cloud_1'].shape))
                # print((data['point_cloud_2'].shape))
                idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

                idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
                idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations
                idx_to_data_dict[demo_idx]['index_i'] = self.current_index_i
                idx_to_data_dict[demo_idx]['index_j'] = self.current_index_j
                idx_to_data_dict[demo_idx]['this_one_hot_encoding'] = this_one_hot_encoding

                total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
                idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
                idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
                idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
                idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
                self.train = True
                if self.four_data:
                    all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                elif not pushing:
                    all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                    #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(train_dir, self.scene_type) # yixuan test
                else: 
                    all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                idx_to_data_dict[demo_idx]['path'] = train_dir
                idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx += 1           
            self.train_idx_to_data_dict.update(idx_to_data_dict)
                    

        
        self.use_multiple_test_dataset = use_multiple_test_dataset
        #time.sleep(10)
        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))
        # for _ in range(len(self.train_idx_to_data_dict)):
        #     print(self.train_idx_to_data_dict[_])

        if not self.use_multiple_test_dataset:
            self.test_dir_list = test_dir_list \
                if test_dir_list is not None else config.args.test_dir
        self.test_idx_to_data_dict = {}
        idx_to_data_dict = {}
        if self.use_multiple_test_dataset:
            self.train_dir_list_total = ['/home/yhuang/Desktop/mohit_code/saved_dataset/data/iiwa_push_2stack_4', 
            '/home/yhuang/Desktop/mohit_code/saved_dataset/data/4push_high_success_rate_2', 
            '/home/yhuang/Desktop/mohit_code/saved_dataset/data/iiwa_push_2stack_4_disturbance', 
            '/home/yhuang/Desktop/mohit_code/saved_dataset/data/4push_high_success_rate_2_disturbance']
            self.test_dir_list_total = ['/home/yhuang/Desktop/mohit_code/saved_dataset/data/5push_high_success_3',
            '/home/yhuang/Desktop/mohit_code/saved_dataset/data/iiwa_push_2stack_4_disturbance']
            
            #self.train_dir_list_total = ['/home/yhuang/Desktop/mohit_code/saved_dataset/data/real_data']
            print('test')
            print(self.test_dir_list_total)
            demo_idx = 0
            for test_dir_list in self.test_dir_list_total:
                print(test_dir_list)
                self.test_dir_list = [test_dir_list]
                if test_dir_list in self.train_dir_list_total:
                    start_test_id = 8010
                else:
                    start_test_id = 0
                files = sorted(os.listdir(self.test_dir_list[0]))
                
                if self.real_data:
                    self.test_pcd_path = [
                        os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
                    files = sorted(os.listdir(self.train_dir_list[0]))
                    self.test_pcd_path_1 = [
                        os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
                    #print(self.test_pcd_path_1)
                    #time.sleep(10)
                    
                else:
                    self.test_pcd_path = [
                        os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
                
                
                for test_dir in self.test_pcd_path[start_test_id: start_test_id + test_max_size]:
                    self.current_goal_relations = self.all_goal_relations[0] # simplify for without test end_relations
                    
                    
                    print(test_dir)
                    if self.real_data:
                        #test_dir = self.test_pcd_path[10]
                        print(self.test_pcd_path_1)
                        #time.sleep(1)
                        test_dir_1 = self.test_pcd_path_1[-1]
                        print(test_dir_1)
                        with open(test_dir_1, 'rb') as f:
                            real_data = pickle.load(f)
                        #print(real_data)
                    with open(test_dir, 'rb') as f:
                        data, attrs = pickle.load(f)
                    # print((data['point_cloud_1'].shape))
                    # print((data['point_cloud_2'].shape))
                    # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
                    #     continue
                    # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
                    #     continue
                    leap = 1
                    
                    for k, v in data.items():
                        if 'point_cloud' in k and 'last' not in k:
                            if(v.shape[0] == 0):
                                leap = 0
                                break
                            # if(v.shape[0] != total_steps): # 2 will be total steps of the task
                            #     leap = 0
                            #     break
                            for i in range((v.shape[0])):
                                #print([k, v[i].shape])
                                if(v[i].shape[0] < data_size):
                                    leap = 0
                                    break
                            # if len(v.shape) !=3:
                            #     leap = 0
                            #     break
                            # if v.shape[1] < data_size:
                            #     leap = 0
                            #     break
                        #print(k)
                    
                    #print(leap)
                    # print(data['objects']['block_1']['position'])
                    # print(data['objects']['block_2']['position'])
                    # print(data['objects']['block_3']['position'])
                    if stacking and not pick_place and not pushing:
                        if 'block_4' in data['objects']:
                            if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58 or data['objects']['block_4']['position'][-1][2] < 0.66):
                                leap = 0
                        else:
                            if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58):
                                leap = 0
                        #break
                    eps = 1e-3
                    #print(leap)
                    if leap == 0:
                        continue
                    #print(data)

                    # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
                    # for _ in range(3):
                    #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
                    #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
                    idx_to_data_dict[demo_idx] = {}
                    # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                    # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                    idx_to_data_dict[demo_idx]['objects'] = data['objects']
                    # print((data['point_cloud_1'].shape))
                    # print((data['point_cloud_2'].shape))
                    idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations # keep all the data as the same because we do not need it. 
                    idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']
                    total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
                    idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T    
                    idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
                    idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
                    idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
                    self.train = False
                    if self.real_data:
                        if self.four_data:
                            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects ,train = self.train, real_data = self.real_data, test_dir_1 = test_dir_1, updated_behavior_params = self.updated_behavior_params)
                        elif not pushing:
                            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, real_data = self.real_data, test_dir_1 = test_dir_1, updated_behavior_params = self.updated_behavior_params)
                            #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(test_dir, self.scene_type) # yixuan test
                        else: 
                            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, real_data = self.real_data, test_dir_1 = test_dir_1, updated_behavior_params = self.updated_behavior_params)
                    else:
                        if self.four_data:
                            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                        elif not pushing:
                            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                            #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(test_dir, self.scene_type) # yixuan test
                        else: 
                            all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params)
                    idx_to_data_dict[demo_idx]['path'] = test_dir
                    idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                    demo_idx += 1           
            self.test_idx_to_data_dict.update(idx_to_data_dict)
        else:
            
            files = sorted(os.listdir(self.test_dir_list[0]))
            
            if self.real_data:
                self.test_pcd_path = [
                    os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
                files = sorted(os.listdir(self.train_dir_list[0]))
                self.test_pcd_path_1 = [
                    os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
                #print(self.test_pcd_path_1)
                #time.sleep(10)
                
            else:
                self.test_pcd_path = [
                    os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
            demo_idx = 0
            
            real_idx = 0
            for test_dir in self.test_pcd_path[start_test_id: start_test_id + test_max_size]:
                #self.current_goal_relations = self.all_goal_relations[0] # simplify for without test end_relations
                
                fmp_leap = 1
                if self.using_multi_step_statistics and self.evaluate_end_relations:
                    if self.total_multi_steps == 2:
                        self.current_index_i = []
                        self.current_index_j = []
                        self.current_goal_relations = np.zeros((self.all_goal_relations[self.test_id*self.total_multi_steps].shape[0], self.all_goal_relations[self.test_id*self.total_multi_steps].shape[1]))
                        self.current_index_i.append(self.all_index_i_list[self.test_id*self.total_multi_steps][0])
                        self.current_index_i.append(self.all_index_i_list[self.test_id*self.total_multi_steps + 1][0])
                        self.current_index_j.append(self.all_index_j_list[self.test_id*self.total_multi_steps][0])
                        self.current_index_j.append(self.all_index_j_list[self.test_id*self.total_multi_steps + 1][0])
                        for array_i in range(self.current_goal_relations.shape[0]):
                            for array_j in range(self.current_goal_relations.shape[1]):
                                if self.all_goal_relations[self.test_id*self.total_multi_steps][array_i][array_j] == 1 or self.all_goal_relations[self.test_id*self.total_multi_steps + 1][array_i][array_j] == 1:
                                    self.current_goal_relations[array_i][array_j] = 1
                        
                        self.current_predicted_relations = self.all_predicted_relations[self.test_id*self.total_multi_steps + 1]
                        
                        push_id_1 = np.argmax(self.all_actions[self.test_id*self.total_multi_steps][0][:max_objects])
                        
                        push_id_2 = np.argmax(self.all_actions[self.test_id*self.total_multi_steps + 1][0][:max_objects])
                        #print([max_objects, push_id_1, push_id_2])
                        if push_id_2 >= push_id_1:
                            self.fail_reasoning_num += 1
                            #fmp_leap = 0
                        # print(self.all_actions[self.test_id*2][0])
                        # print(self.all_actions[self.test_id*2 + 1][0])
                    elif self.total_multi_steps == 3:
                        self.current_index_i = []
                        self.current_index_j = []
                        self.current_goal_relations = np.zeros((self.all_goal_relations[self.test_id*self.total_multi_steps].shape[0], self.all_goal_relations[self.test_id*self.total_multi_steps].shape[1]))
                        self.current_index_i.append(self.all_index_i_list[self.test_id*self.total_multi_steps][0])
                        self.current_index_i.append(self.all_index_i_list[self.test_id*self.total_multi_steps + 1][0])
                        self.current_index_i.append(self.all_index_i_list[self.test_id*self.total_multi_steps + 2][0])
                        self.current_index_j.append(self.all_index_j_list[self.test_id*self.total_multi_steps][0])
                        self.current_index_j.append(self.all_index_j_list[self.test_id*self.total_multi_steps + 1][0])
                        self.current_index_j.append(self.all_index_j_list[self.test_id*self.total_multi_steps + 2][0])
                        for array_i in range(self.current_goal_relations.shape[0]):
                            for array_j in range(self.current_goal_relations.shape[1]):
                                if self.all_goal_relations[self.test_id*self.total_multi_steps][array_i][array_j] == 1 or self.all_goal_relations[self.test_id*self.total_multi_steps + 1][array_i][array_j] == 1:
                                    self.current_goal_relations[array_i][array_j] = 1
                        
                        self.current_predicted_relations = self.all_predicted_relations[self.test_id*self.total_multi_steps + 1]
                        
                        push_id_1 = np.argmax(self.all_actions[self.test_id*self.total_multi_steps][0][:max_objects])
                        
                        push_id_2 = np.argmax(self.all_actions[self.test_id*self.total_multi_steps + 1][0][:max_objects])

                        push_id_3 = np.argmax(self.all_actions[self.test_id*self.total_multi_steps + 2][0][:max_objects])
                        
                        print('fmp', fmp_leap)
                        if push_id_2 >= push_id_1:
                            self.fail_reasoning_num += 1
                            fmp_leap = 0
                        elif push_id_3 >= push_id_2:
                            self.fail_reasoning_num += 1
                            fmp_leap = 0
                        print('fmp', fmp_leap)
                        print([push_id_1, push_id_2, push_id_3])
                    # print(self.all_goal_relations[self.test_id])
                    # print(self.all_index_i_list[self.test_id])
                    # print(self.all_index_j_list[self.test_id])
                    #time.sleep(10)
                else:
                    self.current_goal_relations = self.all_goal_relations[self.test_id] # simplify for without test end_relations
                    self.current_predicted_relations = self.all_predicted_relations[self.test_id]
                    self.current_index_i = self.all_index_i_list[self.test_id]
                    self.current_index_j = self.all_index_j_list[self.test_id]
                if self.evaluate_end_relations:
                    self.current_planned_pose = self.all_data_planned_pose[self.test_id]
                    self.current_planned_orientation = self.all_data_planned_orientation[self.test_id]
                self.test_id += 1

                
                
                if self.real_data:
                    #test_dir = self.test_pcd_path[10]
                    print(self.test_pcd_path_1)
                    #time.sleep(1)
                    test_dir_1 = self.test_pcd_path_1[real_idx]
                    print(test_dir_1)
                    with open(test_dir_1, 'rb') as f:
                        real_data = pickle.load(f)
                    #print(real_data)
                real_idx += 1
                
                
                
                with open(test_dir, 'rb') as f:
                    data, attrs = pickle.load(f)

                if self.real_data:
                    total_objects = 0
                    for k, v in real_data.items():
                        if 'point_cloud' in k and 'sampling' in k and 'last' not in k:
                            total_objects += 1
                    this_one_hot_encoding = np.zeros((1, total_objects))
                else:
                    total_objects = 0
                    for k, v in data.items():
                        #print(k)
                        if 'point_cloud' in k and 'sampling' in k and 'last' not in k:
                            total_objects += 1
                    print(total_objects)
                    this_one_hot_encoding = np.zeros((1, total_objects))
                # print((data['point_cloud_1'].shape))
                # print((data['point_cloud_2'].shape))
                # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
                #     continue
                # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
                #     continue
                leap = 1
                
                # if not self.real_data:
                #     for k, v in data.items():
                #         if 'point_cloud' in k and 'last' not in k:
                #             if(v.shape[0] == 0):
                #                 leap = 0
                #                 break
                #             # if(v.shape[0] != total_steps): # 2 will be total steps of the task
                #             #     leap = 0
                #             #     break
                #             print(leap)
                #             for i in range((v.shape[0])):
                #                 print([k, v[i].shape])
                #                 if(v[i].shape[0] < data_size):
                #                     leap = 0
                #                     break
                            # if len(v.shape) !=3:
                            #     leap = 0
                            #     break
                            # if v.shape[1] < data_size:
                            #     leap = 0
                            #     break
                        #print(k)
                if self.evaluate_end_relations:
                    if self.real_data:
                        obj_id = -1
                        for k, v in real_data.items():
                            if 'point_cloud' in k and 'last' not in k and 'sampling' not in k:
                                this_leap = 1
                                obj_id += 1
                                if(v.shape[0] == 0):
                                    this_leap = 0
                                    this_one_hot_encoding[0][obj_id] = this_leap
                                    break
                                if(v.shape[0] != total_steps): # 2 will be total steps of the task
                                    this_leap = 0
                                    this_one_hot_encoding[0][obj_id] = this_leap
                                    break
                                for i in range((v.shape[0])):
                                    #print([k, v[i].shape])
                                    if(v[i].shape[0] < data_size):
                                        this_leap = 0
                                        this_one_hot_encoding[0][obj_id] = this_leap
                                        break
                                this_one_hot_encoding[0][obj_id] = this_leap
                    else:
                        obj_id = -1
                        for k, v in data.items():
                            if 'point_cloud' in k and 'last' not in k and 'sampling' not in k:
                                this_leap = 1
                                obj_id += 1
                                if(v.shape[0] == 0):
                                    this_leap = 0
                                    this_one_hot_encoding[0][obj_id] = this_leap
                                    break
                                if(v.shape[0] != total_steps): # 2 will be total steps of the task
                                    this_leap = 0
                                    this_one_hot_encoding[0][obj_id] = this_leap
                                    break
                                for i in range((v.shape[0])):
                                    #print([k, v[i].shape])
                                    if(v[i].shape[0] < data_size):
                                        this_leap = 0
                                        this_one_hot_encoding[0][obj_id] = this_leap
                                        break
                                this_one_hot_encoding[0][obj_id] = this_leap
                else:
                    if not self.real_data:
                        for k, v in data.items():
                            if 'point_cloud' in k and 'last' not in k:
                                if(v.shape[0] == 0):
                                    leap = 0
                                    break
                                if(v.shape[0] != total_steps): # 2 will be total steps of the task
                                    leap = 0
                                    break
                                for i in range((v.shape[0])):
                                    #print([k, v[i].shape])
                                    if(v[i].shape[0] < data_size):
                                        leap = 0
                                        break
                 
                
                print(this_one_hot_encoding)
                print(leap)
                #print(data['objects'])
                self.off_policy_evaluation = False
                if self.off_policy_evaluation:
                    # print(data['objects'])
                    
                    generated_pose = np.zeros((self.current_planned_pose[0].shape[0], self.current_planned_pose[0].shape[1]))
                    generated_orientation = np.zeros((self.current_planned_orientation[0].shape[0], self.current_planned_orientation[0].shape[1]))
                    for this_obj_i in range(generated_pose.shape[0]):
                        this_block_string = 'block_' + str(this_obj_i + 1)
                        generated_pose[this_obj_i] = data['objects'][this_block_string]['position'][0]
                        generated_orientation[this_obj_i] = data['objects'][this_block_string]['orientation'][0]
                    
                    print('max error pose', np.max(generated_pose - self.current_planned_pose[0]))
                    print('max error orientation', np.max(generated_orientation - self.current_planned_orientation[0]))
                    if np.abs(np.max(generated_pose - self.current_planned_pose[0])) > 0.02: # maximum around 2cm
                        print(generated_pose)
                        print(self.current_planned_pose[0])
                        print(generated_orientation)
                        print(self.current_planned_orientation[0])
                    if np.abs(np.max(generated_orientation - self.current_planned_orientation[0])) > 0.02: # maximum around 1
                        print(generated_pose)
                        print(self.current_planned_pose[0])
                        print(generated_orientation)
                        print(self.current_planned_orientation[0])
                    #time.sleep(10)
                # print(data['objects']['block_1']['position'])
                # print(data['objects']['block_2']['position'])
                # print(data['objects']['block_3']['position'])
                if stacking and not pick_place and not pushing:
                    if 'block_4' in data['objects']:
                        if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58 or data['objects']['block_4']['position'][-1][2] < 0.66):
                            leap = 0
                    else:
                        if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58):
                            leap = 0
                    #break
                
                
                if pointconv_baselines:
                    if not self.updated_behavior_params:
                        #print(attrs['behavior_params']['']['behaviors'])
                        if 'push' not in attrs['behavior_params']['']['behaviors']:
                            fmp_leap = 0
                            self.motion_planner_fail_num += 1
                        elif 'target_pose' not in attrs['behavior_params']['']['behaviors']['push']:
                            fmp_leap = 0
                            self.motion_planner_fail_num += 1
                    else:
                        if 'target_object_pose' not in attrs['behavior_params']['']:
                            fmp_leap = 0
                            self.motion_planner_fail_num += 1
                
                
                if self.evaluate_end_relations:
                    if not self.updated_behavior_params:
                        #print(attrs['behavior_params']['']['behaviors'])
                        if 'push' not in attrs['behavior_params']['']['behaviors']:
                            fmp_leap = 0
                            self.motion_planner_fail_num += 1
                        elif 'target_pose' not in attrs['behavior_params']['']['behaviors']['push']:
                            fmp_leap = 0
                            self.motion_planner_fail_num += 1
                    else:
                        print('fmp_leap', fmp_leap)
                        print(attrs['behavior_params'])
                        if self.using_multi_step_statistics :
                            if self.total_multi_steps == 2:
                                if 'target_object_pose' not in attrs['behavior_params']['push_step_1'] or 'target_object_pose' not in attrs['behavior_params']['push_step_2']:
                                    fmp_leap = 0
                                    self.motion_planner_fail_num += 1
                            elif self.total_multi_steps == 3:
                                if 'target_object_pose' not in attrs['behavior_params']['push_step_1'] or 'target_object_pose' not in attrs['behavior_params']['push_step_2'] or 'target_object_pose' not in attrs['behavior_params']['push_step_3']:
                                    print('enter')
                                    fmp_leap = 0
                                    self.motion_planner_fail_num += 1
                        else:
                            if 'target_object_pose' not in attrs['behavior_params']['']:
                                fmp_leap = 0
                                self.motion_planner_fail_num += 1
                
                this_one_hot_encoding_numpy = copy.deepcopy(this_one_hot_encoding)
                this_one_hot_encoding_numpy = this_one_hot_encoding_numpy.T
                print(this_one_hot_encoding_numpy.shape)
                print(this_one_hot_encoding_numpy)
                print(fmp_leap)
                one_hot_encoding_leap = 1
                if not self.real_data:
                    for check_i in range(this_one_hot_encoding_numpy.shape[0]):
                        if this_one_hot_encoding_numpy[check_i] == 0:
                            for check_i_again in range(check_i,this_one_hot_encoding_numpy.shape[0]):
                                if this_one_hot_encoding_numpy[check_i_again] == 1:
                                    one_hot_encoding_leap = 0
                                #print(this_one_hot_encoding_numpy)
                        
                eps = 1e-3
                #print(leap)
                # if one_hot_encoding_leap == 0:
                #     time.sleep(10)
                if fmp_leap == 0 or one_hot_encoding_leap == 0:
                    continue
                if leap == 0 and not self.evaluate_end_relations:
                    continue
                #print(data)

                # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
                # for _ in range(3):
                #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
                #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
                idx_to_data_dict[demo_idx] = {}
                # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
                idx_to_data_dict[demo_idx]['objects'] = data['objects']
                # print((data['point_cloud_1'].shape))
                # print((data['point_cloud_2'].shape))
                idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
                idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations
                idx_to_data_dict[demo_idx]['index_i'] = self.current_index_i
                idx_to_data_dict[demo_idx]['index_j'] = self.current_index_j
                idx_to_data_dict[demo_idx]['this_one_hot_encoding'] = this_one_hot_encoding
                idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']
                total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
                idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T    
                idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
                idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
                idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
                self.train = False
                if self.real_data:
                    if self.four_data:
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects ,train = self.train, real_data = self.real_data, test_dir_1 = test_dir_1, updated_behavior_params = self.updated_behavior_params, evaluate_end_relations = self.evaluate_end_relations,this_one_hot_encoding = this_one_hot_encoding)
                    elif not pushing:
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, real_data = self.real_data, test_dir_1 = test_dir_1, updated_behavior_params = self.updated_behavior_params, evaluate_end_relations = self.evaluate_end_relations, this_one_hot_encoding = this_one_hot_encoding)
                        #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(test_dir, self.scene_type) # yixuan test
                    else: 
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, real_data = self.real_data, test_dir_1 = test_dir_1, updated_behavior_params = self.updated_behavior_params, evaluate_end_relations = self.evaluate_end_relations, this_one_hot_encoding = this_one_hot_encoding)
                else:
                    if self.four_data:
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params, evaluate_end_relations = self.evaluate_end_relations, this_one_hot_encoding = this_one_hot_encoding)
                    elif not pushing:
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params, evaluate_end_relations = self.evaluate_end_relations, this_one_hot_encoding = this_one_hot_encoding)
                        #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(test_dir, self.scene_type) # yixuan test
                    else: 
                        all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params, evaluate_end_relations = self.evaluate_end_relations, this_one_hot_encoding = this_one_hot_encoding)
                idx_to_data_dict[demo_idx]['path'] = test_dir
                idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx += 1           
            self.test_idx_to_data_dict.update(idx_to_data_dict)

        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        # The following dicts contain two keys ('idx' and 'order')
        self.train_all_pair_sample_order = {}
        self.test_all_pair_sample_order = {}
        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}

    def get_fail_motion_planner_num(self):
        return self.motion_planner_fail_num, self.test_max_size

    def get_fail_reasoning_num(self):
        return self.fail_reasoning_num
    

    
    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()
            # print(root)
            # print(dirs)
            # print(files)

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line':
                if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                    continue
                # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                #     continue #yixuan test

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                elif self.scene_type == 'box_stacking_node_label':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type, load_voxels_of_removed_obj=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                elif self.scene_type == "box_stacking_node_label":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                             use_node_label=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        #print(order)
        if shuffle:
            np.random.shuffle(order)
        #print(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        #print(path)
        precond_label = 1 if 'true' in path.split('/') else 0
        # if precond_label == 0:
        #     assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]
 
        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        # we need to make sure all the following return arguments have the same value and structure. 
        if self.pushing or self.pick_place:  
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_pc_center,all_obj_pair_orient, action, relation_list, gt_pose_list, gt_orientation_list, gt_extents_list, gt_extents_range_list, voxel_single_list, select_obj_num_range, bounding_box, rotated_bounding_box = \
                scene_voxel_obj.get_all_object_pair_voxels_0()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_pc_center_last, all_obj_pair_orient_last, action_last, relation_list_last, gt_pose_list_last, gt_orientation_list_last, gt_extents_list_last, gt_extents_range_list_last, voxel_single_list_last, select_obj_num_range_last, bounding_box_last, rotated_bounding_box_last = \
                scene_voxel_obj.get_all_object_pair_voxels_1()
        elif self.four_data:
            if(np.random.rand() < 0.33):
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_0()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()
            elif(np.random.rand() > 0.33 and np.random.rand() < 0.66):
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()
            else:
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_3()
        else:
            print(np.random.rand())
            if (np.random.rand() < 0.5):
                print('enter 1')
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_pc_center,all_obj_pair_orient, action, relation_list, gt_pose_list, gt_orientation_list, gt_extents_list, gt_extents_range_list, voxel_single_list, select_obj_num_range, bounding_box, rotated_bounding_box = \
                    scene_voxel_obj.get_all_object_pair_voxels_0()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_pc_center_last, all_obj_pair_orient_last, action_last, relation_list_last, gt_pose_list_last, gt_orientation_list_last, gt_extents_list_last, gt_extents_range_list_last, voxel_single_list_last, select_obj_num_range_last, bounding_box_last, rotated_bounding_box_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()
            else:
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_pc_center,all_obj_pair_orient, action, relation_list, gt_pose_list, gt_orientation_list, gt_extents_list, gt_extents_range_list, voxel_single_list, select_obj_num_range, bounding_box, rotated_bounding_box = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_pc_center_last, all_obj_pair_orient_last, action_last, relation_list_last, gt_pose_list_last, gt_orientation_list_last, gt_extents_list_last, gt_extents_range_list_last, voxel_single_list_last, select_obj_num_range_last, bounding_box_last, rotated_bounding_box_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()

        obj_num = scene_voxel_obj.get_obj_num()
        data = {
            'scene_path': path,
            'num_objects': obj_num,
            'select_obj_num_range': select_obj_num_range,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos,
            'pc_center': all_pc_center,
            'all_obj_pair_orient': all_obj_pair_orient,
            'action': action,
            'relation': relation_list, 
            'gt_pose_list': gt_pose_list,
            'gt_orientation_list': gt_orientation_list,
            'gt_extents_list': gt_extents_list,
            'gt_extents_range_list': gt_extents_range_list,
            'goal_relations': data_dict['goal_relations'],
            'predicted_relations': data_dict['predicted_relations'],
            'this_one_hot_encoding': data_dict['this_one_hot_encoding'],
            'index_i': data_dict['index_i'],
            'index_j': data_dict['index_j'],
            'all_object_pair_voxels_single': voxel_single_list,
            'bounding_box': bounding_box,
            'rotated_bounding_box' : rotated_bounding_box
        }

        data_last = {
            'scene_path': path,
            'num_objects': obj_num,
            'select_obj_num_range': select_obj_num_range_last,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels_last,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos_last,
            'pc_center': all_pc_center_last,
            'all_obj_pair_orient': all_obj_pair_orient_last,
            'action': action_last,
            'relation': relation_list_last, 
            'gt_pose_list': gt_pose_list_last,
            'gt_orientation_list': gt_orientation_list_last,
            'gt_extents_list': gt_extents_list_last,
            'gt_extents_range_list': gt_extents_range_list_last,
            'goal_relations': data_dict['goal_relations'],
            'predicted_relations': data_dict['predicted_relations'],
            'this_one_hot_encoding': data_dict['this_one_hot_encoding'],
            'index_i': data_dict['index_i'],
            'index_j': data_dict['index_j'],
            'all_object_pair_voxels_single': voxel_single_list_last,
            'bounding_box': bounding_box_last, 
            'rotated_bounding_box' : rotated_bounding_box_last
        }
        #print('return here!!!')
        return data, data_last
    
    def get_all_object_pairs_for_scene_index_sequence(self, scene_idx, sequence, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]
 
        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        if sequence == 0:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_0()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()
        elif sequence == 1:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()
        elif sequence == 2:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_3()

        #print(all_obj_pair_pos)
        data = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos,
            'all_obj_pair_orient': all_obj_pair_orient,
            'action': action,
            'relation': relation_list
        }

        data_last = {
            'scene_path': path,
            'num_objects': scene_voxel_obj.number_of_objects,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels_last,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos_last,
            'all_obj_pair_orient': all_obj_pair_orient_last,
            'action': action_last,
            'relation': relation_list_last
        }
        return data, data_last
    
    
    
    def get_next_all_object_pairs_for_scene_sequence(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        #data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        

        data = [] 
        data_next = []

        if self.pushing:
            total_sequence_num = 2
        elif self.four_data:
            total_sequence_num = 4
        else:
            total_sequence_num = 3
        
        #data, data_next = 
        for i in range(total_sequence_num - 1):
            current_data, current_data_next = self.get_all_object_pairs_for_scene_index_sequence(scene_idx, i, train=train)
            data.append(current_data)
            data_next.append(current_data_next)

        sample_order_dict['idx'] += 1
        
        #sample_order_dict['idx'] += 1
        #print(sample_idx)
        if train:
            return data, data_next
        else:
            # length = self.number_of_scene_data(train)
            # sample_idx = np.random.randint(length)
            # # Get the actual scene index.
            # scene_idx = sample_order_dict['order'][sample_idx]
            # test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            return data, data_next# , test_data#
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        #data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        

        data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        
        #sample_order_dict['idx'] += 1
        print(sample_idx)
        if train:
            return data, data_next
        else:
            # length = self.number_of_scene_data(train)
            # sample_idx = np.random.randint(length)
            # # Get the actual scene index.
            # scene_idx = sample_order_dict['order'][sample_idx]
            # test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            return data, data_next# , test_data#

    def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        if return_obj_voxels:
            obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
            data.update(obj_voxel_dict)

        return data

    def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                               return_obj_voxels=return_obj_voxels)
        sample_order_dict['idx'] += 1
        return data


class AllPairVoxelDataloaderPointCloud3stackEndRelations(object):
    def __init__(self, 
                 config,
                 max_objects = 5, 
                 four_data = False,
                 pick_place = False,
                 pushing = False,
                 stacking = False, 
                 set_max = False,
                 save_data_path = None,
                 train_dir_list=None,
                 test_dir_list=None,
                 voxel_datatype_to_use=1,
                 load_contact_data=False,
                 load_all_object_pair_voxels=True,
                 load_scene_voxels=False,
                 test_end_relations = False,
                 data_index = 0, 
                 start_id = 0, 
                 max_size = 0, 
                 start_test_id = 0, 
                 test_max_size = 0,
                 updated_behavior_params = False):
        #self.train = train
        self.data_index = data_index
        stacking = stacking
        self.set_max = set_max
        self.four_data = four_data
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.pick_place = pick_place
        self.pushing = pushing
        self.stacking = stacking
        self.voxel_datatype_to_use = voxel_datatype_to_use
        self.load_contact_data = load_contact_data
        self.load_all_object_pair_voxels = load_all_object_pair_voxels
        self.load_scene_voxels = load_scene_voxels
        self.test_end_relations = test_end_relations

        self.valid_scene_types = ("data_in_line", "cut_food", "box_stacking", "box_stacking_node_label", "test")
        #self.scene_type = "box_stacking"
        self.scene_type = "data_in_line"

        self.train_dir_list = train_dir_list \
            if train_dir_list is not None else config.args.train_dir
        self.test_dir_list = test_dir_list \
            if test_dir_list is not None else config.args.test_dir

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}
        max_train_data_size = 120000
        curr_dir_max_data = None
        print('train')
        print(self.train_dir_list)
        print(self.test_dir_list)
        
        
        
        
        original_pcd_path = self.train_dir_list[0]
        print(original_pcd_path)
        sorted_orginal_pcd_path = os.listdir(original_pcd_path)
        print(sorted_orginal_pcd_path)
        if len(sorted_orginal_pcd_path) >= 11:
            sorted_orginal_pcd_path.sort() 
            for sort_i in range(2,10):
                sorted_orginal_pcd_path[sort_i] = sorted_orginal_pcd_path[sort_i + 1]
            sorted_orginal_pcd_path[10] = '10'  # temporay fix about "solve the sort problem about the comparison of 2 and 10"
        else:
            sorted_orginal_pcd_path.sort() 
        
        print(sorted_orginal_pcd_path)
        this_path = os.path.join(self.train_dir_list[0], sorted_orginal_pcd_path[self.data_index])
        #this_path = os.path.join(self.train_dir_list[0], sorted_orginal_pcd_path[3])

        files = sorted(os.listdir(this_path))
        #files = (os.listdir(this_path))
        
        #print(this_path)
        #time.sleep(10)
        self.train_pcd_path = [
            os.path.join(this_path, p) for p in files if 'demo' in p]

        self.test_pcd_path = [
            os.path.join(this_path, p) for p in files if 'demo' in p]

        
        # max_size = 30
        # start_test_id = 0
        # test_max_size = 30
        data_size = 128
        self.max_size = max_size
        self.test_max_size = test_max_size
        self.updated_behavior_params = updated_behavior_params
        # if self.pushing:
        #     data_size = 128
        #print(len(self.train_pcd_path))
        #if 
        
        if self.four_data:
            total_steps = 4
        if self.pick_place or self.pushing:
            total_steps = 2
        elif self.stacking:
            total_steps = 3
        self.motion_planner_fail_num = 0
        self.train_id = 0

        self.test_id = 0


        #save_path = "/home/yhuang/Desktop/mohit_code/savetxt/2022-03-02-19-55-14.pickle" #string_1 + str(10) + ".pickle"

        #save_data_path = "/home/yhuang/Desktop/mohit_code/savetxt/2022-03-07-11-33-06"
        #save_data_path = "/home/yhuang/Desktop/mohit_code/savetxt/2022-04-18-00-17-34"
        # save_data_path = "/home/yhuang/Desktop/mohit_code/savetxt/2022-05-01-18-59-37planning" # train on 4 objects and test on 4 objects
        #save_data_path = save_data_path[0]
        save_data_path = os.path.join("/home/yhuang/Desktop/mohit_code/savetxt/", save_data_path)
        print(save_data_path)
        #save_data_path = save_data_path
        # save_data_path = "/home/yhuang/Desktop/mohit_code/savetxt/2022-04-20-03-48-24" # train on 4 objects and test on 5 objects
        sorted_save_data_path_dir = os.listdir(save_data_path)
        sorted_save_data_path_dir.sort()
        
        save_path = os.path.join(save_data_path, sorted_save_data_path_dir[self.data_index])
        
        #print('save_path', save_path)
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        indent = ''
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(indent + k, v.shape)
            elif isinstance(v, dict):
                print(indent + k)
                print_data(v, indent + '    ')
            else:
                raise ValueError(f"Unknown datatype: {type(v)}")

        self.all_goal_relations = data['goal_relations']
        self.all_predicted_relations = data['predicted_relations']
        # print('self.all_goal_relations shape', self.all_goal_relations.shape)
        # time.sleep(10)

        
        



        #print('train_pcd_path', self.train_pcd_path)
        for train_dir in self.train_pcd_path[:max_size]: # train_pcd_path contains 11 sub directories. 
            self.current_goal_relations = self.all_goal_relations[self.train_id]
            self.current_predicted_relations = self.all_predicted_relations[self.train_id]
            self.train_id += 1
            #print(train_dir)            
            #time.sleep(10)
            with open(train_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            indent = ''
            #print(data)
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
            #     continue
            # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
            #     continue
            leap = 1

            # if self.test_end_relations and data['fail_mp'] == 1:
            #     leap = 0
            #     self.motion_planner_fail_num += 1

            # print(attrs)

            # print('self.test_end_relations', self.test_end_relations)
            
            if self.test_end_relations:
                if not self.updated_behavior_params:
                    #print(attrs['behavior_params']['']['behaviors'])
                    if 'push' not in attrs['behavior_params']['']['behaviors']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                    elif 'target_pose' not in attrs['behavior_params']['']['behaviors']['push']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                else:
                    if 'target_object_pose' not in attrs['behavior_params']['']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                    #print(attrs)


            
            print('leap', leap)
            for k, v in data.items():   # yixuan comment out 04-18 for the test purpose
                if 'point_cloud' in k and 'last' not in k:
                    if(v.shape[0] == 0):
                        leap = 0
                        break
                    if(v.shape[0] != total_steps): # 2 will be total steps of the task
                        leap = 0
                        break
                    for i in range((v.shape[0])):
                        #print([k, v[i].shape])
                        if(v[i].shape[0] < data_size):
                            leap = 0
                            break
                    # if len(v.shape) !=3:
                    #     leap = 0
                    #     break
                    # if v.shape[1] < data_size:
                    #     leap = 0
                    #     break
                #print(k)
            # print(data['objects']['block_1']['position'])
            # print(data['objects']['block_2']['position'])
            # print(data['objects']['block_3']['position'])
            if stacking and not pick_place and not pushing:
                if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58):
                    leap = 0
                #break
            eps = 1e-3
            print('leap', leap)
            if leap == 0:  ## in the test_end_relations case, when leap == 0, it's either falied motion planner or some object fall the table. We need some automated mechanisms to deal with it. But it may take some time, so I can ignore it a bit. 
                continue
            #print(data)

            # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
            # for _ in range(3):
            #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
            #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
            idx_to_data_dict[demo_idx] = {}
            # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            idx_to_data_dict[demo_idx]['objects'] = data['objects']
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

            idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
            idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations

            #print(data)
            #time.sleep(10)
            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            self.train = True
            if self.four_data:
                all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects, train = self.train, updated_behavior_params = self.updated_behavior_params)
            elif not pushing:
                all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects, train = self.train, updated_behavior_params = self.updated_behavior_params)
                #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(train_dir, self.scene_type) # yixuan test
            else: 
                all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(train_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects, train = self.train, updated_behavior_params = self.updated_behavior_params)
            idx_to_data_dict[demo_idx]['path'] = train_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1           
        self.train_idx_to_data_dict.update(idx_to_data_dict)
            

        
        #time.sleep(10)
        print("Did create index for train data: {}".format(
            len(self.train_idx_to_data_dict)))
        # for _ in range(len(self.train_idx_to_data_dict)):
        #     print(self.train_idx_to_data_dict[_])

        self.test_idx_to_data_dict = {}
        idx_to_data_dict = {}
        
        files = sorted(os.listdir(self.test_dir_list[0]))
        
        # self.test_pcd_path = [
        #     os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]
        demo_idx = 0
        
        
        if test_max_size > 0:
            self.motion_planner_fail_num = 0
        print(start_test_id)
        print(test_max_size)
        #time.sleep(10)
        for test_dir in self.test_pcd_path[start_test_id : start_test_id + test_max_size]: # test_pcd_path contains 11 sub directories. 
            self.current_goal_relations = self.all_goal_relations[self.test_id]
            self.current_predicted_relations = self.all_predicted_relations[self.test_id]
            self.test_id += 1
            #print(test_dir)            
            with open(test_dir, 'rb') as f:
                data, attrs = pickle.load(f)
            indent = ''
            #print(data)
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            # if len(data['point_cloud_1'].shape) !=3 or len(data['point_cloud_2'].shape)!=3 or len(data['point_cloud_3'].shape)!=3:
            #     continue
            # if data['point_cloud_1'].shape[1] < data_size or data['point_cloud_2'].shape[1] < data_size or data['point_cloud_3'].shape[1] < data_size:
            #     continue
            leap = 1

            # if self.test_end_relations and data['fail_mp'] == 1:
            #     leap = 0
            #     self.motion_planner_fail_num += 1

            # print(attrs)

            # print('self.test_end_relations', self.test_end_relations)
            if self.test_end_relations:
                if not self.updated_behavior_params:
                    #print(attrs['behavior_params']['']['behaviors'])
                    if 'push' not in attrs['behavior_params']['']['behaviors']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                    elif 'target_pose' not in attrs['behavior_params']['']['behaviors']['push']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                else:
                    if 'target_object_pose' not in attrs['behavior_params']['']:
                        leap = 0
                        self.motion_planner_fail_num += 1
                    #print(attrs)
            
            print('leap', leap)
            for k, v in data.items():   # yixuan comment out 04-18 for the test purpose
                if 'point_cloud' in k and 'last' not in k:
                    if(v.shape[0] == 0):
                        leap = 0
                        break
                    if(v.shape[0] != total_steps): # 2 will be total steps of the task
                        leap = 0
                        break
                    for i in range((v.shape[0])):
                        #print([k, v[i].shape])
                        if(v[i].shape[0] < data_size):
                            leap = 0
                            break
                    # if len(v.shape) !=3:
                    #     leap = 0
                    #     break
                    # if v.shape[1] < data_size:
                    #     leap = 0
                    #     break
                #print(k)
            # print(data['objects']['block_1']['position'])
            # print(data['objects']['block_2']['position'])
            # print(data['objects']['block_3']['position'])
            if stacking and not pick_place and not pushing:
                if(data['objects']['block_2']['position'][-1][2] < 0.5 or data['objects']['block_3']['position'][-1][2] < 0.58):
                    leap = 0
                #break
            eps = 1e-3
            print('leap', leap)
            if leap == 0:
                continue
            #print(data)

            # print([data['point_cloud_1'].shape, data['point_cloud_2'].shape])
            # for _ in range(3):
            #     print([np.max(data['point_cloud_1'][0,:,_]), np.max(data['point_cloud_2'][0,:,_])])
            #     print([np.min(data['point_cloud_1'][0,:,_]), np.min(data['point_cloud_2'][0,:,_])])
            idx_to_data_dict[demo_idx] = {}
            # idx_to_data_dict[demo_idx]['voxel_1'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            # idx_to_data_dict[demo_idx]['voxel_2'] = OctreePickleLoader_pointcloud(data['point_cloud_1'], data['point_cloud_2'])
            idx_to_data_dict[demo_idx]['objects'] = data['objects']
            # print((data['point_cloud_1'].shape))
            # print((data['point_cloud_2'].shape))
            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

            idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations

            idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations

            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
            idx_to_data_dict[demo_idx]['point_cloud_1'] = data['point_cloud_1'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_2'] = data['point_cloud_2'][0][:data_size][:].T
            idx_to_data_dict[demo_idx]['point_cloud_3'] = data['point_cloud_3'][0][:data_size][:].T
            self.train = False
            if self.four_data:
                all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, four_data = self.four_data, set_max = self.set_max, max_objects = max_objects, train = self.train, updated_behavior_params = self.updated_behavior_params)
            elif not pushing:
                all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects, train = self.train, updated_behavior_params = self.updated_behavior_params)
                #all_pair_scene_object =  RobotAllPairSceneObjectPointCloud3stack(test_dir, self.scene_type) # yixuan test
            else: 
                all_pair_scene_object =  RobotAllPairSceneObjectPointCloudVariablestack(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects, train = self.train, updated_behavior_params = self.updated_behavior_params)
            idx_to_data_dict[demo_idx]['path'] = test_dir
            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
            demo_idx += 1           
        
        self.test_idx_to_data_dict.update(idx_to_data_dict)

        print("Did create index for test data: {}".format(
            len(self.test_idx_to_data_dict)))

        # The following dicts contain two keys ('idx' and 'order')
        self.train_all_pair_sample_order = {}
        self.test_all_pair_sample_order = {}
        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}

    def get_fail_motion_planner_num(self):
        return self.motion_planner_fail_num, len(self.train_pcd_path)

    
    def load_emb_data(self, train_h5_path, train_pkl_path, test_h5_path, test_pkl_path):
        '''Load emb data from h5 files. '''
        self.train_h5_data_dict = load_emb_data_from_h5_path(
            train_h5_path, 
            train_pkl_path,
            max_data_size=self.max_train_data_size)
        self.test_h5_data_dict = load_emb_data_from_h5_path(
            test_h5_path, 
            test_pkl_path,
            max_data_size=self.max_test_data_size)
        
        for k in sorted(self.train_idx_to_data_dict.keys()):
            train_idx_data = self.train_idx_to_data_dict[k]
            h5_data = self.train_h5_data_dict[k]
            scene_voxel_obj = train_idx_data['scene_voxel_obj']
            assert h5_data['path'] == train_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

        for k in sorted(self.test_idx_to_data_dict.keys()):
            test_idx_data = self.test_idx_to_data_dict[k]
            h5_data = self.test_h5_data_dict[k]
            scene_voxel_obj = test_idx_data['scene_voxel_obj']
            assert h5_data['path'] == test_idx_data['path']
            assert h5_data['all_object_pair_path'] == \
                   scene_voxel_obj.get_all_object_pair_path()
            assert h5_data['emb'].shape[0] == scene_voxel_obj.number_of_object_pairs

    def load_voxel_data(self, demo_dir, max_data_size=None, demo_idx=0, 
                        max_data_from_dir=None):
        '''Load images for all demonstrations.

        demo_dir: str. Path to directory containing the demonstration data.
        '''
        demo_idx_to_path_dict = {}
        args = self._config.args

        if not os.path.exists(demo_dir):
            print("Dir does not exist: {}".format(demo_dir))
            return demo_idx_to_path_dict

        data_count_curr_dir = 0
        for root, dirs, files in os.walk(demo_dir, followlinks=False):
            if max_data_size is not None and demo_idx > max_data_size:
                break
            if max_data_from_dir is not None and data_count_curr_dir >= max_data_from_dir:
                break

            # Sort the data order so that we do not have randomness associated
            # with it.
            dirs.sort()
            files.sort()
            # print(root)
            # print(dirs)
            # print(files)

            # ==== Used for data_in_line scee ====
            if self.scene_type == 'data_in_line':
                if '0_voxel_data.pkl' not in files and '1_voxel_data.pkl' not in files:
                    continue
                # if 'projected_cloud.pcd' not in files and 'cloud_cluster_0.pcd' not in files:
                #     continue #yixuan test

            # ==== Used for cut food ====
            if self.scene_type == 'cut_food' and 'knife_object.pcd' not in files:
                continue
                
            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking' and 'info.json' not in files:
                continue

            # ==== Used for box stacking ====
            if self.scene_type == 'box_stacking_node_label' and 'info.json' not in files:
                continue

            # TODO: Add size_channels flag
            if self.load_all_object_pair_voxels:
                if self.scene_type == 'data_in_line' or self.scene_type == 'cut_food':
                    all_pair_scene_object =  RobotAllPairSceneObjectSpatial(root, self.scene_type) # yixuan test
                elif self.scene_type == 'box_stacking':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type)
                elif self.scene_type == 'box_stacking_node_label':
                    multi_all_pair_scene_object = RobotBoxStackingAllPairSceneObject(
                        root, self.scene_type, load_voxels_of_removed_obj=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")

                # if self.pos_grid is None and self.voxel_datatype_to_use == 0 and \
                #    self.scene_type != 'box_stacking' and self.scene_type != 'box_stacking_node_label':
                #     self.pos_grid = torch.Tensor(all_pair_scene_object.create_position_grid()) # yixuan test
            else:     
                all_pair_scene_object = None
        
            if self.load_scene_voxels:
                if self.scene_type == "data_in_line":
                    single_scene_voxel_obj = RobotSceneObject(root, self.scene_type)
                elif self.scene_type == "cut_food":
                    single_scene_voxel_obj = RobotSceneCutFoodObject(root, self.scene_type)
                elif self.scene_type == "box_stacking":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type)
                elif self.scene_type == "box_stacking_node_label":
                    multi_scene_voxel_obj = RobotSceneMultiBoxStackingObject(root, self.scene_type, 
                                                                             use_node_label=True)
                else:
                    raise ValueError(f"Invalid scene type {self.scene_type}")
            else:
                single_scene_voxel_obj = None
            
            if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
                if self.load_scene_voxels:
                    all_scene_list = multi_scene_voxel_obj.create_scenes_by_removing_voxels()
                if self.load_all_object_pair_voxels:
                    assert not self.load_scene_voxels
                    all_scene_list = multi_all_pair_scene_object.create_scenes_by_removing_voxels()

                    if self.pos_grid is None and self.voxel_datatype_to_use == 0:
                        self.pos_grid = torch.Tensor(all_scene_list[0].create_position_grid())
                    
                for scene_idx, scene in enumerate(all_scene_list):
                    demo_idx_to_path_dict[demo_idx] = {}
                    demo_idx_to_path_dict[demo_idx]['path'] = root

                    if self.load_all_object_pair_voxels:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_all_pair_scene_obj'] = \
                            multi_all_pair_scene_object 
                    else:
                        demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = None

                    if self.load_scene_voxels:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = scene
                        demo_idx_to_path_dict[demo_idx]['multi_scene_voxel_obj'] = \
                            multi_scene_voxel_obj
                    else:
                        demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = None

                    demo_idx = demo_idx + 1 
                    data_count_curr_dir = data_count_curr_dir + 1

            else:
                demo_idx_to_path_dict[demo_idx] = {}
                demo_idx_to_path_dict[demo_idx]['path'] = root
                demo_idx_to_path_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                demo_idx_to_path_dict[demo_idx]['single_scene_voxel_obj'] = single_scene_voxel_obj 
                demo_idx = demo_idx + 1 
                data_count_curr_dir = data_count_curr_dir + 1

            if demo_idx % 10 == 0:
                print("Did process: {}".format(demo_idx))
                    
        print(f"Did load {data_count_curr_dir} from {demo_dir}")
        return demo_idx_to_path_dict            

    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    def total_pairs_in_all_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        data_size = 0
        for scene_idx, scene in data_dict.items():
            data_size += scene['scene_voxel_obj'].number_of_object_pairs
        return data_size
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        #print(order)
        if shuffle:
            np.random.shuffle(order)

        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        
    def reset_all_pair_batch_sampler(self, train=True, shuffle=True):
        if train:
            sampler_dict = self.train_all_pair_sample_order
        else:
            sampler_dict = self.test_all_pair_sample_order

        data_dict = self.get_demo_data_dict(train)

        # Get all keys. Each key is a tuple of the (scene_idx, pair_idx)
        order = []
        for scene_idx, scene_dict in data_dict.items():
            for i in scene_dict['scene_voxel_obj'].number_of_object_pairs:
                order.append((scene_idx, i))

        if shuffle:
            np.random.shuffle(order)

        sampler_dict['order'] = order
        sampler_dict['idx'] = 0

    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    def number_of_pairs_data(self, train=True):
        return self.total_pairs_in_all_scenes(train)

    def get_scene_voxel_obj_at_idx(self, idx, train=True):
        data_dict = self.get_demo_data_dict(train)
        return data_dict[idx]['scene_voxel_obj']
    
    def get_some_object_pair_train_data_at_idx(self, idx, train=True):
        # Get the actual data idx for this idx. Since we shuffle the data
        # internally these are not same values
        sample_order_dict = self.train_all_pair_sample_order if train else \
            self.test_all_pair_sample_order
        (scene_idx, scene_obj_pair_idx) = sample_order_dict['order'][idx]

        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)

        data = {
            'scene_path': path,
            'voxels': scene_voxel_obj.get_object_pair_voxels_at_index(scene_obj_pair_idx),
            'object_pair_path': scene_voxel_obj.get_object_pair_path_at_index(scene_obj_pair_idx),
            'precond_label': precond_label,
        }
        return data
    
    def get_precond_label_for_demo_data_dict(self, demo_data_dict):
        if self.scene_type == 'data_in_line':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'cut_food':
            return self.get_precond_label_for_path(demo_data_dict['path'])
        elif self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            if self.load_all_object_pair_voxels:
                return demo_data_dict['scene_voxel_obj'].precond_label
            elif self.load_scene_voxels:
                return demo_data_dict['single_scene_voxel_obj'].precond_label
            else:
                raise ValueError("Invalid label")

    def get_precond_label_for_path(self, path):
        #print(path)
        precond_label = 1 if 'true' in path.split('/') else 0
        # if precond_label == 0:
        #     assert 'false' in path.split('/')
        return precond_label
    
    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]
 
        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        if self.pushing:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_pc_center,all_obj_pair_orient, action, relation_list, gt_pose_list, gt_orientation_list,  gt_extents_range_list, voxel_single_list, select_obj_num_range= \
                scene_voxel_obj.get_all_object_pair_voxels_0()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_pc_center_last, all_obj_pair_orient_last, action_last, relation_list_last, gt_pose_list_last, gt_orientation_list_last, gt_extents_range_list_last, voxel_single_list_last, select_obj_num_range_last = \
                scene_voxel_obj.get_all_object_pair_voxels_1()
        elif self.four_data:
            if(np.random.rand() < 0.33):
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_0()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()
            elif(np.random.rand() > 0.33 and np.random.rand() < 0.66):
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()
            else:
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_3()
        else:
            if (np.random.rand() < 0.5):
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_0()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()
            else:
                all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()

                all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()

        # if not train:
        #     all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action = \
        #         scene_voxel_obj.get_all_object_pair_voxels_0()

        #     all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last = \
        #         scene_voxel_obj.get_all_object_pair_voxels_2()
        
        #print(len(all_obj_pair_voxels))
        # import pdb; pdb.set_trace()
        # for l in range(len(all_obj_pair_voxels)):
        #     plot_voxel_plot(all_obj_pair_voxels[l].numpy())
        # import pdb; pdb.set_trace()

        #print(all_obj_pair_pos)
        # print('gt_pose_list', gt_pose_list)
        # print('gt_extents_range_list', gt_extents_range_list)
        obj_num = scene_voxel_obj.get_obj_num()
        data = {
            'scene_path': path,
            'num_objects': obj_num,
            'select_obj_num_range': select_obj_num_range,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos,
            'pc_center': all_pc_center,
            'all_obj_pair_orient': all_obj_pair_orient,
            'action': action,
            'relation': relation_list, 
            'gt_pose_list': gt_pose_list,
            'gt_orientation_list': gt_orientation_list,
            'gt_extents_range_list': gt_extents_range_list,
            'goal_relations': data_dict['goal_relations'],
            'predicted_relations': data_dict['predicted_relations'],
            'all_object_pair_voxels_single': voxel_single_list
        }

        data_last = {
            'scene_path': path,
            'num_objects': obj_num,
            'select_obj_num_range': select_obj_num_range_last,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels_last,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos_last,
            'pc_center': all_pc_center_last,
            'all_obj_pair_orient': all_obj_pair_orient_last,
            'action': action_last,
            'relation': relation_list_last, 
            'gt_pose_list': gt_pose_list_last,
            'gt_orientation_list': gt_orientation_list_last,
            'gt_extents_range_list': gt_extents_range_list_last,
            'goal_relations': data_dict['goal_relations'],
            'predicted_relations': data_dict['predicted_relations'],
            'all_object_pair_voxels_single': voxel_single_list_last
        }
        print('return here !!!')
        return data, data_last
    
    def get_all_object_pairs_for_scene_index_sequence(self, scene_idx, sequence, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]
 
        #print(data_dict)
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{scene_voxel_obj.box_stacking_remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        if sequence == 0:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_0()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()
        elif sequence == 1:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_1()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()
        elif sequence == 2:
            all_obj_pair_voxels, all_obj_pair_anchor_voxels, all_obj_pair_other_voxels, all_obj_pair_far_apart_status, all_obj_pair_pos, all_obj_pair_orient, action, relation_list = \
                    scene_voxel_obj.get_all_object_pair_voxels_2()

            all_obj_pair_voxels_last, all_obj_pair_anchor_voxels_last, all_obj_pair_other_voxels_last, all_obj_pair_far_apart_status_last, all_obj_pair_pos_last, all_obj_pair_orient_last, action_last, relation_list_last = \
                    scene_voxel_obj.get_all_object_pair_voxels_3()

        #print(all_obj_pair_pos)
        data = {
            'scene_path': path,
            'num_objects': obj_num,
            'select_obj_num_range': select_obj_num_range,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos,
            'all_obj_pair_orient': all_obj_pair_orient,
            'action': action,
            'relation': relation_list, 
            'gt_pose_list': gt_pose_list,
            'gt_extents_range_list': gt_extents_range_list,
            'goal_relations': data_dict['goal_relations'],
            'all_object_pair_voxels_single': voxel_single_list
        }

        data_last = {
            'scene_path': path,
            'num_objects': obj_num,
            'select_obj_num_range': select_obj_num_range_last,
            'num_object_pairs': scene_voxel_obj.number_of_object_pairs,
            'all_object_pair_voxels': all_obj_pair_voxels_last,
            'all_object_pair_anchor_voxels': all_obj_pair_anchor_voxels_last,
            'all_object_pair_other_voxels': all_obj_pair_other_voxels_last,
            'all_object_pair_far_apart_status': all_obj_pair_far_apart_status_last,
            'all_object_pair_path': scene_voxel_obj.get_all_object_pair_path(),
            'object_pcd_path_list': scene_voxel_obj.get_object_pcd_paths(),
            'precond_label': precond_label,
            'all_obj_pair_pos': all_obj_pair_pos_last,
            'all_obj_pair_orient': all_obj_pair_orient_last,
            'action': action_last,
            'relation': relation_list_last, 
            'gt_pose_list': gt_pose_list_last,
            'gt_extents_range_list': gt_extents_range_list_last,
            'goal_relations': data_dict['goal_relations'],
            'all_object_pair_voxels_single': voxel_single_list_last
        }
        print('return here!!')
        return data, data_last
    
    
    
    def get_next_all_object_pairs_for_scene_sequence(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        #data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        

        data = [] 
        data_next = []

        if self.pushing:
            total_sequence_num = 2
        elif self.four_data:
            total_sequence_num = 4
        else:
            total_sequence_num = 3
        
        #data, data_next = 
        for i in range(total_sequence_num - 1):
            current_data, current_data_next = self.get_all_object_pairs_for_scene_index_sequence(scene_idx, i, train=train)
            data.append(current_data)
            data_next.append(current_data_next)

        sample_order_dict['idx'] += 1
        
        #sample_order_dict['idx'] += 1
        #print(sample_idx)
        if train:
            return data, data_next
        else:
            # length = self.number_of_scene_data(train)
            # sample_idx = np.random.randint(length)
            # # Get the actual scene index.
            # scene_idx = sample_order_dict['order'][sample_idx]
            # test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            return data, data_next# , test_data#
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        #data = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        

        data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        
        #sample_order_dict['idx'] += 1
        #print(sample_idx)
        if train:
            return data, data_next
        else:
            # length = self.number_of_scene_data(train)
            # sample_idx = np.random.randint(length)
            # # Get the actual scene index.
            # scene_idx = sample_order_dict['order'][sample_idx]
            # test_data, test_data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
            return data, data_next# , test_data#

    def get_voxels_for_scene_index(self, scene_idx, train=True, return_obj_voxels=False):
        data_dict = self.get_demo_data_dict(train)[scene_idx]

        path = data_dict['path']
        single_scene_voxel_obj = data_dict['single_scene_voxel_obj']
        if self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label':
            path = path + f'/remove_obj_id_{single_scene_voxel_obj.remove_obj_id}'
        precond_label = self.get_precond_label_for_demo_data_dict(data_dict)
        voxels = single_scene_voxel_obj.get_scene_voxels()

        # if '3_obj/type_9/28' in path:
        #     plot_voxel_plot(voxels)
        #     import pdb; pdb.set_trace()

        data = {
            'scene_path': path,
            'num_objects': single_scene_voxel_obj.number_of_objects,
            'scene_voxels': torch.FloatTensor(voxels),
            'precond_label': precond_label,
        }

        if return_obj_voxels:
            obj_voxel_dict = single_scene_voxel_obj.get_scene_voxels_for_each_object()
            data.update(obj_voxel_dict)

        return data

    def get_next_voxel_for_scene(self, train=True, return_obj_voxels=False):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        data = self.get_voxels_for_scene_index(scene_idx, train=train, 
                                               return_obj_voxels=return_obj_voxels)
        sample_order_dict['idx'] += 1
        return data
