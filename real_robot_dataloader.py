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
from relational_precond.dataloader.robot_octree_data import RobotVoxels, SceneVoxels, BoxStackingSceneVoxels
from relational_precond.dataloader.robot_octree_data import RobotAllPairVoxels
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

class RobotSceneObject(object):
    def __init__(self, scene_path, scene_type):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = sorted(os.listdir(scene_path))
        assert 'projected_cloud.pcd' in files or 'cloud_cluster_0.pcd'
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.
        self.cloud_cluster_pcd_path = [
            os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]
        
        self.scene_voxels = SceneVoxels(self.cloud_cluster_pcd_path, scene_type, use_object_idx=False)
        self.scene_voxels.init_voxel_index()

    @classmethod
    def is_dir_with_files_valid(path_to_dir, files):
        return 'projected_cloud.pcd' in files
    
    def get_scene_voxels(self, return_tensor=False):
        _, voxels = self.scene_voxels.parse()
        if return_tensor:
            return torch.FloatTensor(voxels)
        else:
            return voxels
    
    def get_scene_voxels_for_each_object(self):
        voxel_index_list = self.scene_voxels.get_scene_voxels_for_each_object()
        min_voxel_idx = np.array([1000, 1000, 1000])
        for obj_voxels in voxel_index_list:
            min_voxel_idx = np.minimum(min_voxel_idx, obj_voxels.min(axis=0))

        edge_voxel_index_list = []
        for obj_voxel in voxel_index_list:
            obj_voxel = obj_voxel - min_voxel_idx
            edge_obj_voxel = self.scene_voxels.get_edges_for_object_voxels(obj_voxel)
            edge_voxel_index_list.append(edge_obj_voxel)

        # Some objects occupy similar voxels remove them. Find object voxels that do not
        # interesect with other voxels.
        non_intersect_obj_edge_voxel = self.scene_voxels.remove_similar_voxels_from_different_objects(
            edge_voxel_index_list
        )
        return {
            'obj_voxel_index_list': voxel_index_list,
            'obj_edge_voxel_index_list': edge_voxel_index_list,
            'non_intersect_obj_edge_voxel': non_intersect_obj_edge_voxel,
        }

    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path)


class RobotSceneCutFoodObject(object):
    def __init__(self, scene_path, scene_type):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = os.listdir(scene_path)
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.
        self.cloud_cluster_pcd_path = [
            os.path.join(scene_path, p) for p in sorted(files) if 'cloud_cluster' in p]
        self.knife_pcd_path = os.path.join(scene_path, 'knife_object.pcd')
        
        self.scene_voxels = SceneVoxels(
            self.cloud_cluster_pcd_path + [self.knife_pcd_path],
            scene_type)
        self.scene_voxels.init_voxel_index()
    
    @classmethod
    def is_dir_with_files_valid(path_to_dir, files):
        return os.path.basename(path_to_dir) == 'object_cloud' and \
            'knife_object.pcd' in files

    def get_scene_voxels(self, return_tensor=False):
        _, voxels = self.scene_voxels.parse()
        if return_tensor:
            return torch.FloatTensor(voxels)
        else:
            return voxels

    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path) + 1

class RobotSceneBoxStackingObject(object):
    def __init__(self, scene_path, scene_type, precond_label, remove_obj_id, remove_obj_from_voxels=True):
        self.scene_path = scene_path
        self.scene_type = scene_type
        self.precond_label = precond_label
        self.remove_obj_id = remove_obj_id
        self.remove_obj_from_voxels = remove_obj_from_voxels

        # Now we have multiple objects each with their own cloud cluster
        # in this scene.
        remove_obj_fname = f'cloud_cluster_{remove_obj_id}.pcd'
        object_cloud_dir = os.path.join(scene_path, 'object_cloud')

        files = os.listdir(object_cloud_dir)
        if remove_obj_id != -1:
            assert remove_obj_fname in files, "To remove obj-id not found"
        
        if not remove_obj_from_voxels:
            self.cloud_cluster_pcd_path = [os.path.join(object_cloud_dir, p) 
                                           for p in sorted(files) if 'cloud_cluster' in p]
            self.scene_voxels = BoxStackingSceneVoxels(self.cloud_cluster_pcd_path, scene_type,
                                                       use_object_idx=False,
                                                       to_remove_obj_idx = remove_obj_id)
        else:
            self.cloud_cluster_pcd_path = [os.path.join(object_cloud_dir, p) 
                for p in sorted(files) if 'cloud_cluster' in p and p != remove_obj_fname]
            self.scene_voxels = SceneVoxels(self.cloud_cluster_pcd_path, scene_type)

        self.scene_voxels.init_voxel_index()

    @classmethod
    def is_dir_with_files_valid(path_to_dir, files):
        return os.path.basename(path_to_dir) == 'object_cloud' and \
            'cloud_cluster_0.pcd' in files

    def get_scene_voxels(self, return_tensor=False):
        _, voxels = self.scene_voxels.parse()
        if return_tensor:
            return torch.FloatTensor(voxels)
        else:
            return voxels

    def get_scene_voxels_for_each_object(self):
        voxel_index_list = self.scene_voxels.get_scene_voxels_for_each_object()
        min_voxel_idx = np.array([1000, 1000, 1000])
        for obj_voxels in voxel_index_list:
            min_voxel_idx = np.minimum(min_voxel_idx, obj_voxels.min(axis=0))

        edge_voxel_index_list = []
        for obj_voxel in voxel_index_list:
            obj_voxel = obj_voxel - min_voxel_idx
            edge_obj_voxel = self.scene_voxels.get_edges_for_object_voxels(obj_voxel)
            edge_voxel_index_list.append(edge_obj_voxel)

        # Some objects occupy similar voxels remove them. Find object voxels that do not
        # interesect with other voxels.
        non_intersect_obj_edge_voxel = self.scene_voxels.remove_similar_voxels_from_different_objects(
            edge_voxel_index_list
        )
        return {
            'obj_voxel_index_list': voxel_index_list,
            'obj_edge_voxel_index_list': edge_voxel_index_list,
            'non_intersect_obj_edge_voxel': non_intersect_obj_edge_voxel,
        }

    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path)
    
    @property
    def scene_precond_label(self):
        return self.precond_label
    

class RobotSceneMultiBoxStackingObject(object):
    def __init__(self, scene_path, scene_type, use_node_label=False):
        self.scene_path = scene_path
        self.scene_type = scene_type
        # If this flag is true we do not remove the main node but instead label it differently
        self.use_node_label = use_node_label

        # Read the json file
        info_json_path = os.path.join(scene_path, 'info.json')
        assert os.path.exists(info_json_path), f"JSON file does not exist: {info_json_path}"
        with open(info_json_path, 'r') as json_fp:
            self.scene_info = json.load(json_fp)

        assert self.scene_info.get('precond_obj_id') is not None

    def get_total_number_of_objects(self):
         obj_cloud_path = os.path.join(self.scene_path, 'object_cloud')
         num_objects = len(
             [p for p in os.listdir(obj_cloud_path) if 'cloud_cluster' in p])
         return num_objects

    def create_scenes_by_removing_voxels(self):
        precond_obj_ids = self.scene_info['precond_obj_id']
        scene_obj_list = []
        num_objects = self.get_total_number_of_objects()
        valid_objs = get_valid_objects_to_remove_for_box_stacking(
            self.scene_path, 
            self.scene_info)

        for i, obj_id in enumerate(valid_objs):
            label = 0 if obj_id in precond_obj_ids else 1
            remove_obj_from_voxels = (not self.use_node_label)
            obj = RobotSceneBoxStackingObject(self.scene_path, 
                                              self.scene_type, 
                                              label,
                                              obj_id,
                                              remove_obj_from_voxels=remove_obj_from_voxels)
            
            scene_obj_list.append(obj)

        if not self.use_node_label:
            scene_obj_list.append(self.create_scene_with_all_voxels())
        return scene_obj_list

    def create_scene_with_all_voxels(self):
        # The entire scene is always stable
        assert not self.use_node_label, "Cannot create scene with all voxels with none to remove"
        scene_obj = RobotSceneBoxStackingObject(self.scene_path, 
                                                self.scene_type, 
                                                1,
                                                -1,
                                                remove_obj_from_voxels=True)
        return scene_obj
    

class RobotAllPairSceneObject(object):
    def __init__(self, 
                 scene_path, 
                 scene_type, 
                 precond_label=None, 
                 box_stacking_remove_obj_id=None,
                 scene_pos_info=None,
                 load_voxels_of_removed_obj=False):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = sorted(os.listdir(scene_path))
        if scene_type == 'data_in_line':
            #assert 'projected_cloud.pcd' in files or 'cloud_cluster_0.pcd' in files # yixuan test
            assert '0_voxel_data.pkl' in files
        elif scene_type == 'cut_food':
            assert 'knife_object.pcd' in files
        elif scene_type == 'box_stacking' or scene_type == 'box_stacking_node_label':
            assert precond_label is not None
            assert box_stacking_remove_obj_id is not None 
            assert 'info.json' in files
            assert scene_pos_info is not None
            self.precond_label = precond_label
            self.box_stacking_remove_obj_id = box_stacking_remove_obj_id
        else:
            raise ValueError(f"Invalid scene type: {scene_type}")
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.

        # self.cloud_cluster_pcd_path = [
        #     os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]
        if scene_type == 'data_in_line':
            self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_data_in_line(
                scene_path)
        elif scene_type == 'cut_food':
            self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_cut_food(
                scene_path)
        elif scene_type == 'box_stacking' or scene_type == 'box_stacking_node_label':
            if load_voxels_of_removed_obj:
                self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_box_stacking(
                    scene_path, -1)
            else:
                self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_box_stacking(
                    scene_path, box_stacking_remove_obj_id)
            remove_obj_id_str = str(box_stacking_remove_obj_id)
            self.scene_pos_info = {} 
            self.contact_edges = None
            self.load_voxels_of_removed_obj = load_voxels_of_removed_obj

            # Create a map from the original obj_ids to new_obj_ids
            self.old_obj_id_to_new_obj_id_dict = dict()
            for k, v in scene_pos_info.items():
                if k == 'precond_obj_id' or k == 'test_removal' or k == 'contact_edges':
                    continue
                k_int = int(k)
                if box_stacking_remove_obj_id == -1 or load_voxels_of_removed_obj:
                    new_k_int = k_int
                elif box_stacking_remove_obj_id >= 0:
                    if k_int < box_stacking_remove_obj_id:
                        new_k_int = k_int
                    elif k_int > box_stacking_remove_obj_id:
                        new_k_int = k_int -1 
                    else:
                        new_k_int = -1
                self.old_obj_id_to_new_obj_id_dict[k_int] = new_k_int
                if new_k_int >= 0:
                    self.scene_pos_info[new_k_int] = copy.deepcopy(scene_pos_info[k])
                    
            # Get the precond obj ids for this scene
            self.precond_obj_id_list = []
            for obj_id in scene_pos_info['precond_obj_id']:
                if self.old_obj_id_to_new_obj_id_dict[obj_id] != -1:
                    self.precond_obj_id_list.append(self.old_obj_id_to_new_obj_id_dict[obj_id])
            
            # Create new set of edges based on the old_obj to new_obj mapping.
            if scene_pos_info.get('contact_edges') is not None:
                self.contact_edges = []
                for e in scene_pos_info['contact_edges']:
                    new_e = (
                        self.old_obj_id_to_new_obj_id_dict[e[0]],
                        self.old_obj_id_to_new_obj_id_dict[e[1]],
                    )
                    if new_e[0] == -1 or new_e[1] == -1:
                        continue
                    self.contact_edges.append(new_e)
                # import ipdb; ipdb.set_trace()
            else:
                # import ipdb; ipdb.set_trace()
                pass
                
            assert len(self.scene_pos_info) == len(self.cloud_cluster_pcd_path) #yixuan test

        else:
            raise ValueError(f"Invalid scene type {scene_type}")
        
        # relation embeddings are not commutative, hence we find N^2 relations
        self.obj_voxels_by_obj_pair_dict = dict()
        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()

        self.obj_pair_list = list(permutations(range(len(self.cloud_cluster_pcd_path)), 2))

        self.robot_all_pair_voxels = RobotAllPairVoxels(self.cloud_cluster_pcd_path)

        # Load all the obj pair embeddings for now. this is because we can have
        # a case where there are two objects with no edge between them and this
        # will result in no pairwise embedding. However, for the downstream task
        # we might end up using the embeddings between objects which have contacts
        # only.
        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair
            status, robot_voxels = self.robot_all_pair_voxels.init_voxels_for_pcd_pair(
                anchor_idx, other_idx
            )

            self.obj_voxels_by_obj_pair_dict[obj_pair] = robot_voxels
            self.obj_voxels_status_by_obj_pair_dict[obj_pair] = status
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                self.cloud_cluster_pcd_path[anchor_idx],
                self.cloud_cluster_pcd_path[other_idx]
            )

    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def number_of_objects(self):
        return len(self.cloud_cluster_pcd_path)
    
    @property
    def number_of_object_pairs(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def stable_object_ids(self):
        assert self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label', \
            f"Invalid func for scene {self.scene_type}"
        return self.precond_obj_id_list
    
    def get_object_center_list_2(self):
        # return self.robot_all_pair_voxels.get_object_center_list()
        center_list = []
        for k in sorted(self.scene_pos_info.keys()):
            assert self.scene_pos_info[k]['orient'] in (0, 1)
            orient_list = [1, 0] if self.scene_pos_info[k]['orient'] == 0 else [0, 1]
            center_list.append(self.scene_pos_info[k]['pos'] + orient_list)
        return center_list

    def get_object_center_list(self):
        obj_info_list = []
        for k in sorted(self.scene_pos_info.keys()):
            orient = self.scene_pos_info[k]['orient']
            assert orient in (0, 1)
            pos = self.scene_pos_info[k]['pos']
            if orient == 0:
                size_by_axes = [0.048, 0.15, 0.03]
            elif orient == 1:
                size_by_axes = [0.15, 0.048, 0.03]
            bound_1 = [pos[0] - size_by_axes[0]/2.0,
                       pos[1] - size_by_axes[1]/2.0,
                       pos[2]]
            bound_2 = [pos[0] + size_by_axes[0]/2.0,
                       pos[1] + size_by_axes[1]/2.0,
                       pos[2] + size_by_axes[2]]
            
            orient_info = [1, 0] if orient == 0 else [0, 1]
            obj_info_list.append(
                pos + orient_info + bound_1 + bound_2
            )
        return obj_info_list
    
    def get_object_id_to_remove(self):
        return self.box_stacking_remove_obj_id

    def get_stable_object_label_tensor(self):
        label = torch.zeros((self.number_of_objects)).long()
        for i in self.stable_object_ids:
            label[i] = 1
        return label
    
    def create_position_grid(self):
        for v in self.obj_voxels_by_obj_pair_dict.values():
            return v.create_position_grid()
    
    def get_pair_status_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_voxels_status_by_obj_pair_dict[obj_pair_key]
    
    # ==== Functions that return object point cloud paths ====
    
    def get_object_pair_path_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
    
    def get_all_object_pair_path(self):
        path_list = []
        for obj_pair_key in self.obj_pair_list:
            path = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            path_list.append(path)
        return path_list

    # ==== Functions that return voxel objects ====
    
    def get_object_pair_voxels_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj.parse()[1]
    
    def get_all_object_pair_voxels(self):
        voxel_list, are_obj_far_apart_list = [], []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            _, voxels = voxel_obj.parse()
            if voxels is not None:
                voxel_list.append(torch.FloatTensor(voxels))
                are_obj_far_apart_list.append(0)
            else:
                assert voxel_obj.objects_are_far_apart
                voxels = voxel_obj.get_all_zero_voxels()
                voxel_list.append(torch.FloatTensor(voxels))
                obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
                # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
                are_obj_far_apart_list.append(0)

        return voxel_list, are_obj_far_apart_list
    
    # ==== Functions that return voxel objects ====

    def get_object_pair_voxel_object_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj
    
    def get_all_object_pair_voxel_object(self):
        voxel_obj_list = []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            voxel_obj_list.append(voxel_obj)
        return voxel_obj_list
    
    def get_object_pcd_paths(self):
        return self.cloud_cluster_pcd_path


class RobotAllPairSceneObjectSpatial(object):
    def __init__(self, 
                 scene_path, 
                 scene_type, 
                 precond_label=None, 
                 box_stacking_remove_obj_id=None,
                 scene_pos_info=None,
                 load_voxels_of_removed_obj=False):
        self.scene_path = scene_path
        self.scene_type = scene_type

        files = sorted(os.listdir(scene_path))
        if scene_type == 'data_in_line':
            #assert 'projected_cloud.pcd' in files or 'cloud_cluster_0.pcd' in files # yixuan test
            assert '0_voxel_data.pkl' in files
        elif scene_type == 'cut_food':
            assert 'knife_object.pcd' in files
        elif scene_type == 'box_stacking' or scene_type == 'box_stacking_node_label':
            assert precond_label is not None
            assert box_stacking_remove_obj_id is not None 
            assert 'info.json' in files
            assert scene_pos_info is not None
            self.precond_label = precond_label
            self.box_stacking_remove_obj_id = box_stacking_remove_obj_id
        else:
            raise ValueError(f"Invalid scene type: {scene_type}")
            
        # Now we have multiple objects each with their own cloud cluster
        # in this scene.

        # self.cloud_cluster_pcd_path = [
        #     os.path.join(scene_path, p) for p in files if 'cloud_cluster' in p]
        if scene_type == 'data_in_line':
            # self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_data_in_line(
            #     scene_path) # yixuan test
            self.voxel_cluster_path = get_all_object_voxel_path_for_data_in_line(scene_path)
            self.image_cluster_path = get_all_object_image_path_for_data_in_line(scene_path)
        elif scene_type == 'cut_food':
            self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_cut_food(
                scene_path)
        elif scene_type == 'box_stacking' or scene_type == 'box_stacking_node_label':
            if load_voxels_of_removed_obj:
                self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_box_stacking(
                    scene_path, -1)
            else:
                self.cloud_cluster_pcd_path = get_all_object_pcd_path_for_box_stacking(
                    scene_path, box_stacking_remove_obj_id)
            remove_obj_id_str = str(box_stacking_remove_obj_id)
            self.scene_pos_info = {} 
            self.contact_edges = None
            self.load_voxels_of_removed_obj = load_voxels_of_removed_obj

            # Create a map from the original obj_ids to new_obj_ids
            self.old_obj_id_to_new_obj_id_dict = dict()
            for k, v in scene_pos_info.items():
                if k == 'precond_obj_id' or k == 'test_removal' or k == 'contact_edges':
                    continue
                k_int = int(k)
                if box_stacking_remove_obj_id == -1 or load_voxels_of_removed_obj:
                    new_k_int = k_int
                elif box_stacking_remove_obj_id >= 0:
                    if k_int < box_stacking_remove_obj_id:
                        new_k_int = k_int
                    elif k_int > box_stacking_remove_obj_id:
                        new_k_int = k_int -1 
                    else:
                        new_k_int = -1
                self.old_obj_id_to_new_obj_id_dict[k_int] = new_k_int
                if new_k_int >= 0:
                    self.scene_pos_info[new_k_int] = copy.deepcopy(scene_pos_info[k])
                    
            # Get the precond obj ids for this scene
            self.precond_obj_id_list = []
            for obj_id in scene_pos_info['precond_obj_id']:
                if self.old_obj_id_to_new_obj_id_dict[obj_id] != -1:
                    self.precond_obj_id_list.append(self.old_obj_id_to_new_obj_id_dict[obj_id])
            
            # Create new set of edges based on the old_obj to new_obj mapping.
            if scene_pos_info.get('contact_edges') is not None:
                self.contact_edges = []
                for e in scene_pos_info['contact_edges']:
                    new_e = (
                        self.old_obj_id_to_new_obj_id_dict[e[0]],
                        self.old_obj_id_to_new_obj_id_dict[e[1]],
                    )
                    if new_e[0] == -1 or new_e[1] == -1:
                        continue
                    self.contact_edges.append(new_e)
                # import ipdb; ipdb.set_trace()
            else:
                # import ipdb; ipdb.set_trace()
                pass
                
            assert len(self.scene_pos_info) == len(self.cloud_cluster_pcd_path) #yixuan test

        else:
            raise ValueError(f"Invalid scene type {scene_type}")
        
        # relation embeddings are not commutative, hence we find N^2 relations
        max_length = 4
        perturbations = np.random.permutation(8)
        #self.voxel_cluster_path = self.voxel_cluster_path[perturbations]
        self.voxel_cluster_path_1 = []
        for i in range(perturbations.shape[0]):
            self.voxel_cluster_path_1.append(self.voxel_cluster_path[perturbations[i]])
        self.voxel_cluster_path = self.voxel_cluster_path_1[:max_length]
        #self.image_cluster_path = self.image_cluster_path[perturbations]
        self.image_cluster_path_1 = []
        for i in range(perturbations.shape[0]):
            self.image_cluster_path_1.append(self.image_cluster_path[perturbations[i]])
        self.image_cluster_path = self.image_cluster_path_1[:max_length]
        #print(self.voxel_cluster_path)

        self.obj_voxels_by_obj_pair_dict = dict()
        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()

        self.obj_pair_list = list(permutations(range(len(self.voxel_cluster_path)), 2))

        self.robot_all_pair_voxels = RobotAllPairVoxels(self.voxel_cluster_path, self.image_cluster_path)

        # Load all the obj pair embeddings for now. this is because we can have
        # a case where there are two objects with no edge between them and this
        # will result in no pairwise embedding. However, for the downstream task
        # we might end up using the embeddings between objects which have contacts
        # only.
        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair
            status, robot_voxels = self.robot_all_pair_voxels.init_voxels_for_pcd_pair(
                anchor_idx, other_idx
            )

            self.obj_voxels_by_obj_pair_dict[obj_pair] = robot_voxels
            self.obj_voxels_status_by_obj_pair_dict[obj_pair] = status
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                self.voxel_cluster_path[anchor_idx],
                self.voxel_cluster_path[other_idx]
            )

    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def number_of_objects(self):
        return len(self.voxel_cluster_path)
    
    @property
    def number_of_object_pairs(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def stable_object_ids(self):
        assert self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label', \
            f"Invalid func for scene {self.scene_type}"
        return self.precond_obj_id_list
    
    def get_object_center_list_2(self):
        # return self.robot_all_pair_voxels.get_object_center_list()
        center_list = []
        for k in sorted(self.scene_pos_info.keys()):
            assert self.scene_pos_info[k]['orient'] in (0, 1)
            orient_list = [1, 0] if self.scene_pos_info[k]['orient'] == 0 else [0, 1]
            center_list.append(self.scene_pos_info[k]['pos'] + orient_list)
        return center_list

    def get_object_center_list(self):
        obj_info_list = []
        for k in sorted(self.scene_pos_info.keys()):
            orient = self.scene_pos_info[k]['orient']
            assert orient in (0, 1)
            pos = self.scene_pos_info[k]['pos']
            if orient == 0:
                size_by_axes = [0.048, 0.15, 0.03]
            elif orient == 1:
                size_by_axes = [0.15, 0.048, 0.03]
            bound_1 = [pos[0] - size_by_axes[0]/2.0,
                       pos[1] - size_by_axes[1]/2.0,
                       pos[2]]
            bound_2 = [pos[0] + size_by_axes[0]/2.0,
                       pos[1] + size_by_axes[1]/2.0,
                       pos[2] + size_by_axes[2]]
            
            orient_info = [1, 0] if orient == 0 else [0, 1]
            obj_info_list.append(
                pos + orient_info + bound_1 + bound_2
            )
        return obj_info_list
    
    def get_object_id_to_remove(self):
        return self.box_stacking_remove_obj_id

    def get_stable_object_label_tensor(self):
        label = torch.zeros((self.number_of_objects)).long()
        for i in self.stable_object_ids:
            label[i] = 1
        return label
    
    def create_position_grid(self):
        for v in self.obj_voxels_by_obj_pair_dict.values():
            return v.create_position_grid()
    
    def get_pair_status_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_voxels_status_by_obj_pair_dict[obj_pair_key]
    
    # ==== Functions that return object point cloud paths ====
    
    def get_object_pair_path_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
    
    def get_all_object_pair_path(self):
        path_list = []
        for obj_pair_key in self.obj_pair_list:
            path = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            path_list.append(path)
        return path_list

    # ==== Functions that return voxel objects ====
    
    def get_object_pair_voxels_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj.parse()[1]
    
    def get_all_object_pair_voxels(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            _, voxels, anchor_pos, other_pos = voxel_obj.parse()
            if anchor_pos not in pos_list:
                pos_list.append(anchor_pos)
            if voxels is not None:
                voxel_list.append(torch.FloatTensor(voxels))
                are_obj_far_apart_list.append(0)
            else:
                assert voxel_obj.objects_are_far_apart
                voxels = voxel_obj.get_all_zero_voxels()
                voxel_list.append(torch.FloatTensor(voxels))
                obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
                # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
                are_obj_far_apart_list.append(0)

        return voxel_list, are_obj_far_apart_list, pos_list
    
    # ==== Functions that return voxel objects ====

    def get_object_pair_voxel_object_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj
    
    def get_all_object_pair_voxel_object(self):
        voxel_obj_list = []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            voxel_obj_list.append(voxel_obj)
        return voxel_obj_list
    
    def get_object_pcd_paths(self):
        return self.voxel_cluster_path


class RobotAllPairSceneObjectPointCloud(object):
    def __init__(self,  # put the specifc file list to here and seperate several point clouds to get a better representation
                 scene_path, 
                 scene_type, 
                 precond_label=None, 
                 box_stacking_remove_obj_id=None,
                 scene_pos_info=None,
                 load_voxels_of_removed_obj=False):
        self.scene_path = scene_path
        self.scene_type = scene_type

        with open(self.scene_path, 'rb') as f:
            data, attrs = pickle.load(f)
        self.all_point_cloud = []
        self.all_point_cloud_last = []
        data_size = 256
        self.scale = 1
        self.all_pos_list = []
        self.all_pos_list_last = []
        self.all_orient_list = []
        self.all_orient_list_last = []
        for k, v in data.items():
            if 'point_cloud' in k and 'sampling' in k:
                #print([k, v.shape])
                self.all_point_cloud.append(v[0,:, :]) # yixuan test
                self.all_point_cloud_last.append(v[1,:, :])
                self.all_pos_list.append(self.get_point_cloud_center(v[0,:, :]))
                self.all_pos_list_last.append(self.get_point_cloud_center(v[1,:, :]))
                
                #print(v.shape)
                # if 'last' in k:
                #     #print(v.shape)
                #     self.all_point_cloud_last.append(v[0,:data_size, :])
                # else:
                #     self.all_point_cloud.append(v[0,:data_size, :])
                #     #print(v[0,:, :].shape)
                    
                #     # print(np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2) # here is the function to get the center of pointcloud, sequence(2,0,1)
                #     A = np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2
                #     A_1 = [A[1], A[2], A[0]]
                    #print(A_1) # A_1 here is the center of the point cloud
        # for k, v in data['objects'].items():
        #     if 'block' in k:
        #         #print([k, v['position'].shape])
        #         self.all_pos_list.append(v['position'][0])
        #         self.all_pos_list_last.append(v['position'][-1])
        #         # self.all_orient_list.append(v['orientation'][0]) # how to get orientation from point cloud directly is another problem.
        #         # self.all_orient_list_last.append(v['orientation'][-1])
        #         #print(v['position'][0])

        #print(self.all_point_cloud[0].shape)

        
        
        self.action = data['action_primitives']*self.scale
        #print(self.action)
        # relation embeddings are not commutative, hence we find N^2 relations
        # max_length = 4
        # perturbations = np.random.permutation(8)
        # #self.voxel_cluster_path = self.voxel_cluster_path[perturbations]
        # self.voxel_cluster_path_1 = []
        # for i in range(perturbations.shape[0]):
        #     self.voxel_cluster_path_1.append(self.voxel_cluster_path[perturbations[i]])
        # self.voxel_cluster_path = self.voxel_cluster_path_1[:max_length]
        # #self.image_cluster_path = self.image_cluster_path[perturbations]
        # self.image_cluster_path_1 = []
        # for i in range(perturbations.shape[0]):
        #     self.image_cluster_path_1.append(self.image_cluster_path[perturbations[i]])
        # self.image_cluster_path = self.image_cluster_path_1[:max_length]
        #print(self.voxel_cluster_path)

        self.obj_voxels_by_obj_pair_dict = dict()
        self.obj_voxels_by_obj_pair_dict_anchor = dict()
        self.obj_voxels_by_obj_pair_dict_other = dict()


        self.obj_voxels_by_obj_pair_dict_last = dict()
        self.obj_voxels_by_obj_pair_dict_anchor_last = dict()
        self.obj_voxels_by_obj_pair_dict_other_last = dict()

        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()

        self.obj_pair_list = list(permutations(range(len(self.all_point_cloud)), 2))

        #self.robot_all_pair_voxels = RobotAllPairVoxels(self.voxel_cluster_path, self.image_cluster_path)

        # Load all the obj pair embeddings for now. this is because we can have
        # a case where there are two objects with no edge between them and this
        # will result in no pairwise embedding. However, for the downstream task
        # we might end up using the embeddings between objects which have contacts
        # only.
        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair
            #print(obj_pair)
            # status, robot_voxels = self.robot_all_pair_voxels.init_voxels_for_pcd_pair(
            #     anchor_idx, other_idx
            # )

            #print(self.all_point_cloud[anchor_idx].shape)
            total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            #print(total_point_cloud.shape)
            self.obj_voxels_by_obj_pair_dict[obj_pair] = total_point_cloud.T
            self.obj_voxels_by_obj_pair_dict_anchor[obj_pair] = self.all_point_cloud[anchor_idx].T
            self.obj_voxels_by_obj_pair_dict_other[obj_pair] = self.all_point_cloud[other_idx].T

            total_point_cloud_last = np.concatenate((self.all_point_cloud_last[anchor_idx], self.all_point_cloud_last[other_idx]), axis = 0)
            #print(total_point_cloud.shape)
            self.obj_voxels_by_obj_pair_dict_last[obj_pair] = total_point_cloud_last.T
            self.obj_voxels_by_obj_pair_dict_anchor_last[obj_pair] = self.all_point_cloud_last[anchor_idx].T
            self.obj_voxels_by_obj_pair_dict_other_last[obj_pair] = self.all_point_cloud_last[other_idx].T
            #self.obj_voxels_status_by_obj_pair_dict[obj_pair] = status
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                scene_path, 
                scene_path # yixuan test
            )
        #print('finish')

    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def number_of_objects(self):
        return len(self.all_point_cloud)
    
    @property
    def number_of_object_pairs(self):
        return len(self.obj_pair_list)
    
    @property
    def stable_object_ids(self):
        assert self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label', \
            f"Invalid func for scene {self.scene_type}"
        return self.precond_obj_id_list
    
    def get_point_cloud_center(self, v):
        A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return np.array(A_1)
    
    def get_object_center_list_2(self):
        # return self.robot_all_pair_voxels.get_object_center_list()
        center_list = []
        for k in sorted(self.scene_pos_info.keys()):
            assert self.scene_pos_info[k]['orient'] in (0, 1)
            orient_list = [1, 0] if self.scene_pos_info[k]['orient'] == 0 else [0, 1]
            center_list.append(self.scene_pos_info[k]['pos'] + orient_list)
        return center_list

    def get_object_center_list(self):
        obj_info_list = []
        for k in sorted(self.scene_pos_info.keys()):
            orient = self.scene_pos_info[k]['orient']
            assert orient in (0, 1)
            pos = self.scene_pos_info[k]['pos']
            if orient == 0:
                size_by_axes = [0.048, 0.15, 0.03]
            elif orient == 1:
                size_by_axes = [0.15, 0.048, 0.03]
            bound_1 = [pos[0] - size_by_axes[0]/2.0,
                       pos[1] - size_by_axes[1]/2.0,
                       pos[2]]
            bound_2 = [pos[0] + size_by_axes[0]/2.0,
                       pos[1] + size_by_axes[1]/2.0,
                       pos[2] + size_by_axes[2]]
            
            orient_info = [1, 0] if orient == 0 else [0, 1]
            obj_info_list.append(
                pos + orient_info + bound_1 + bound_2
            )
        return obj_info_list
    
    def get_object_id_to_remove(self):
        return self.box_stacking_remove_obj_id

    def get_stable_object_label_tensor(self):
        label = torch.zeros((self.number_of_objects)).long()
        for i in self.stable_object_ids:
            label[i] = 1
        return label
    
    def create_position_grid(self):
        for v in self.obj_voxels_by_obj_pair_dict.values():
            return v.create_position_grid()
    
    def get_pair_status_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_voxels_status_by_obj_pair_dict[obj_pair_key]
    
    # ==== Functions that return object point cloud paths ====
    
    def get_object_pair_path_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
    
    def get_all_object_pair_path(self):
        path_list = []
        for obj_pair_key in self.obj_pair_list:
            path = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            path_list.append(path)
        return path_list

    # ==== Functions that return voxel objects ====
    
    def get_object_pair_voxels_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj.parse()[1]
    
    def get_all_object_pair_voxels(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, orient_list, self.action

    def get_all_object_pair_voxels_last(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels = self.obj_voxels_by_obj_pair_dict_last[obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor_last[obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            
            voxels_other = self.obj_voxels_by_obj_pair_dict_other_last[obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list_last
            orient_list = self.all_orient_list_last
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, orient_list, self.action


    # ==== Functions that return voxel objects ====

    def get_object_pair_voxel_object_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj
    
    def get_all_object_pair_voxel_object(self):
        voxel_obj_list = []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            voxel_obj_list.append(voxel_obj)
        return voxel_obj_list
    
    def get_object_pcd_paths(self):
        return []#self.voxel_cluster_path

class RobotAllPairSceneObjectPointCloud3stack(object):
    def __init__(self,  # put the specifc file list to here and seperate several point clouds to get a better representation
                 scene_path, 
                 scene_type, 
                 precond_label=None, 
                 box_stacking_remove_obj_id=None,
                 scene_pos_info=None,
                 load_voxels_of_removed_obj=False):
        self.scene_path = scene_path
        self.scene_type = scene_type

        with open(self.scene_path, 'rb') as f:
            data, attrs = pickle.load(f)
        #print('enter')
        self.all_point_cloud_1 = []
        self.all_point_cloud_2 = []
        self.all_point_cloud_3 = []
            
        self.all_point_cloud_last = []
        data_size = 256
        self.scale = 1
        self.all_pos_list_1 = []
        self.all_pos_list_2 = []
        self.all_pos_list_3 = []
        self.all_pos_list_last = []
        self.all_orient_list = []
        self.all_orient_list_last = []
        self.all_point_cloud = []
        self.all_pos_list = []
        self.all_pos_list_p = []
        total_object = 3
        for i in range(total_object):
            self.all_point_cloud.append([])
            self.all_pos_list.append([])
            self.all_pos_list_p.append([])
        # for k, v in data.items():
        #     if 'point_cloud' in k:
        #         #print(v.shape)
        #         if 'last' in k:
        #             #print(v.shape)
        #             self.all_point_cloud_last.append(v[0,:data_size, :])
        #         else:
        #             self.all_point_cloud.append(v[0,:data_size, :])
        #             self.all_point_cloud_1.append(v[11,:data_size, :])
        #             self.all_point_cloud_2.append(v[-1,:data_size, :])
        #             #print(v[0,:, :].shape)
                    
        #             # print(np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2) # here is the function to get the center of pointcloud, sequence(2,0,1)
        #             A = np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2
        #             A_1 = [A[1], A[2], A[0]]
        #             #print(A_1) # A_1 here is the center of the point cloud
        sample_list = [0,11, -1]

        for i in range((data['point_cloud_1'].shape[0])):
            #self.all_point_cloud[i] = [data['point_cloud_1'][i][:data_size, :], data['point_cloud_2'][i][:data_size, :], data['point_cloud_3'][i][:data_size, :]]
            self.all_point_cloud[i] = [data['point_cloud_1sampling'][i][:, :], data['point_cloud_2sampling'][i][:, :], data['point_cloud_3sampling'][i][:, :]]
            #self.all_pos_list[i] = [data['objects']['block_1']['position'][sample_list[i]], data['objects']['block_2']['position'][sample_list[i]], data['objects']['block_3']['position'][sample_list[i]]]
            self.all_pos_list[i] = [self.get_point_cloud_center(data['point_cloud_1'][i]), self.get_point_cloud_center(data['point_cloud_2'][i]), self.get_point_cloud_center(data['point_cloud_3'][i])] 
            # print(self.all_pos_list[i])
            # print(self.all_pos_list_p[i])
            # self.all_point_cloud[0].append(data['point_cloud_1'][i][:data_size, :])
            # self.all_point_cloud[1].append(data['point_cloud_2'][i][:data_size, :])
            # self.all_point_cloud[2].append(data['point_cloud_3'][i][:data_size, :])
            # self.all_pos_list[0].append(data['objects']['block_1']['position'][sample_list[i]])
            # self.all_pos_list[1].append(data['objects']['block_2']['position'][sample_list[i]])
            # self.all_pos_list[2].append(data['objects']['block_3']['position'][sample_list[i]])
        
        # for k, v in data['objects'].items():
        #     if 'block' in k:
        #         #print(k)
        #         self.all_pos_list.append(v['position'][0]*self.scale)
        #         self.all_pos_list_last.append(v['position'][-1]*self.scale)
        #         self.all_orient_list.append(v['orientation'][0]) # how to get orientation from point cloud directly is another problem.
        #         self.all_orient_list_last.append(v['orientation'][-1])
        #         #print(v['position'][0])

        #print(self.all_point_cloud[0].shape)

        
        
        #print(attrs['behavior_params'])
        
        #print()
        self.action_1 = [0,1,0]
        for i in range(3):
            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
            self.action_1.append(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
        self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
        self.action_1 = [self.action_1]
        self.action_2 = [0,0,1]
        #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
        for i in range(3):
            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
            # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
            self.action_2.append(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['pick']['init_object_pose'][i])
        self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
        self.action_2 = [self.action_2]
        #print([self.action_1, self.action_2])
        
        self.action = data['action_primitives']*self.scale

        self.obj_voxels_by_obj_pair_dict = []
        self.obj_voxels_by_obj_pair_dict_anchor = []
        self.obj_voxels_by_obj_pair_dict_other = []
        for i in range(3):
            self.obj_voxels_by_obj_pair_dict.append(dict())
            self.obj_voxels_by_obj_pair_dict_anchor.append(dict())
            self.obj_voxels_by_obj_pair_dict_other.append(dict())


        self.obj_voxels_by_obj_pair_dict_last = dict()
        self.obj_voxels_by_obj_pair_dict_anchor_last = dict()
        self.obj_voxels_by_obj_pair_dict_other_last = dict()

        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()

        self.obj_pair_list = list(permutations(range(len(self.all_point_cloud)), 2))

        #self.robot_all_pair_voxels = RobotAllPairVoxels(self.voxel_cluster_path, self.image_cluster_path)

        # Load all the obj pair embeddings for now. this is because we can have
        # a case where there are two objects with no edge between them and this
        # will result in no pairwise embedding. However, for the downstream task
        # we might end up using the embeddings between objects which have contacts
        # only.
        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair
            #print(obj_pair)
            # status, robot_voxels = self.robot_all_pair_voxels.init_voxels_for_pcd_pair(
            #     anchor_idx, other_idx
            # )

            #print(self.all_point_cloud[anchor_idx].shape)
            for i in range(3):
                total_point_cloud = np.concatenate((self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]), axis = 0)
                #print(total_point_cloud.shape)
                self.obj_voxels_by_obj_pair_dict[i][obj_pair] = total_point_cloud.T
                self.obj_voxels_by_obj_pair_dict_anchor[i][obj_pair] = self.all_point_cloud[i][anchor_idx].T
                self.obj_voxels_by_obj_pair_dict_other[i][obj_pair] = self.all_point_cloud[i][other_idx].T   # yixuan test. Important bugs, reverse the idx of obj_pair and sample_idx, this affect the performance of block stacking task

            # total_point_cloud_last = np.concatenate((self.all_point_cloud_last[anchor_idx], self.all_point_cloud_last[other_idx]), axis = 0)
            # #print(total_point_cloud.shape)
            # self.obj_voxels_by_obj_pair_dict_last[obj_pair] = total_point_cloud_last.T
            # self.obj_voxels_by_obj_pair_dict_anchor_last[obj_pair] = self.all_point_cloud_last[anchor_idx].T
            # self.obj_voxels_by_obj_pair_dict_other_last[obj_pair] = self.all_point_cloud_last[other_idx].T
            #self.obj_voxels_status_by_obj_pair_dict[obj_pair] = status
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                scene_path, 
                scene_path # yixuan test
            )
        #print('finish')

    def get_point_cloud_center(self, v):
        A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return np.array(A_1)
    
    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def number_of_objects(self):
        return len(self.all_point_cloud)
    
    @property
    def number_of_object_pairs(self):
        return len(self.obj_pair_list)
    
    @property
    def stable_object_ids(self):
        assert self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label', \
            f"Invalid func for scene {self.scene_type}"
        return self.precond_obj_id_list
    
    def get_object_center_list_2(self):
        # return self.robot_all_pair_voxels.get_object_center_list()
        center_list = []
        for k in sorted(self.scene_pos_info.keys()):
            assert self.scene_pos_info[k]['orient'] in (0, 1)
            orient_list = [1, 0] if self.scene_pos_info[k]['orient'] == 0 else [0, 1]
            center_list.append(self.scene_pos_info[k]['pos'] + orient_list)
        return center_list

    def get_object_center_list(self):
        obj_info_list = []
        for k in sorted(self.scene_pos_info.keys()):
            orient = self.scene_pos_info[k]['orient']
            assert orient in (0, 1)
            pos = self.scene_pos_info[k]['pos']
            if orient == 0:
                size_by_axes = [0.048, 0.15, 0.03]
            elif orient == 1:
                size_by_axes = [0.15, 0.048, 0.03]
            bound_1 = [pos[0] - size_by_axes[0]/2.0,
                       pos[1] - size_by_axes[1]/2.0,
                       pos[2]]
            bound_2 = [pos[0] + size_by_axes[0]/2.0,
                       pos[1] + size_by_axes[1]/2.0,
                       pos[2] + size_by_axes[2]]
            
            orient_info = [1, 0] if orient == 0 else [0, 1]
            obj_info_list.append(
                pos + orient_info + bound_1 + bound_2
            )
        return obj_info_list
    
    def get_object_id_to_remove(self):
        return self.box_stacking_remove_obj_id

    def get_stable_object_label_tensor(self):
        label = torch.zeros((self.number_of_objects)).long()
        for i in self.stable_object_ids:
            label[i] = 1
        return label
    
    def create_position_grid(self):
        for v in self.obj_voxels_by_obj_pair_dict.values():
            return v.create_position_grid()
    
    def get_pair_status_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_voxels_status_by_obj_pair_dict[obj_pair_key]
    
    # ==== Functions that return object point cloud paths ====
    
    def get_object_pair_path_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
    
    def get_all_object_pair_path(self):
        path_list = []
        for obj_pair_key in self.obj_pair_list:
            path = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            path_list.append(path)
        return path_list

    # ==== Functions that return voxel objects ====
    
    def get_object_pair_voxels_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj.parse()[1]
    
    def get_all_object_pair_voxels_0(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels = self.obj_voxels_by_obj_pair_dict[0][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[0][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[0][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[0]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, orient_list, self.action_1

    def get_all_object_pair_voxels_1(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels = self.obj_voxels_by_obj_pair_dict[1][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[1][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[1][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[1]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, orient_list, self.action_1

    def get_all_object_pair_voxels_2(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            # voxels = self.obj_voxels_by_obj_pair_dict[2][obj_pair_key]
            # voxel_list.append(torch.FloatTensor(voxels))
            # voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[2][obj_pair_key]
            # voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            # voxels_other = self.obj_voxels_by_obj_pair_dict_other[2][obj_pair_key]
            # voxel_list_other.append(torch.FloatTensor(voxels_other))
            voxels = self.obj_voxels_by_obj_pair_dict[2][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[2][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[2][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[2]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, orient_list, self.action_2


    # ==== Functions that return voxel objects ====

    def get_object_pair_voxel_object_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj
    
    def get_all_object_pair_voxel_object(self):
        voxel_obj_list = []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            voxel_obj_list.append(voxel_obj)
        return voxel_obj_list
    
    def get_object_pcd_paths(self):
        return []#self.voxel_cluster_path

class RobotAllPairSceneObjectPointCloudVariablestack(object):
    def __init__(self,  # put the specifc file list to here and seperate several point clouds to get a better representation
                 scene_path, 
                 scene_type, 
                 pick_place = False, 
                 push = False,
                 four_data = False, 
                 set_max = False,
                 train = True,
                 precond_label=None, 
                 box_stacking_remove_obj_id=None,
                 scene_pos_info=None,
                 load_voxels_of_removed_obj=False,
                 max_objects = 5,
                 real_data = False,
                 test_dir_1 = None,
                 updated_behavior_params = False, 
                 evaluate_end_relations = False,
                 this_one_hot_encoding = None):
        self.sudo_push = True
        if self.sudo_push:
            push = True
            pick_place = False
        
        self.evaluate_end_relations = evaluate_end_relations
        self.updated_behavior_params = updated_behavior_params
        self.real_data = real_data
        self.train = train
        self.scene_path = scene_path
        self.scene_type = scene_type
        self.set_max = set_max
        self.pushing = push

       

        print('max_objects', max_objects)
        print('set_max', set_max)
        #time.sleep(10)

        self.pick_place = pick_place
        self.four_data = four_data
        if not self.pushing:
            table_increase = 0.345
        #print(self.scene_path)
        if self.real_data:
            print(test_dir_1)
            with open(test_dir_1, 'rb') as f:
                data = pickle.load(f)
                real_data = data
        with open(self.scene_path, 'rb') as f:
            data, attrs = pickle.load(f)

        
        
        #print('enter')
        self.all_point_cloud_1 = []
        self.all_point_cloud_2 = []
        self.all_point_cloud_3 = []
            
        self.all_point_cloud_last = []
        data_size = 128
        self.scale = 1
        self.all_pos_list_1 = []
        self.all_pos_list_2 = []
        self.all_pos_list_3 = []
        self.all_pos_list_last = []
        self.all_orient_list = []
        self.all_orient_list_last = []
        self.all_point_cloud = []
        self.all_pos_list = []
        self.all_pos_list_p = []
        self.all_gt_pose_list = []
        self.all_gt_orientation_list = []
        self.all_gt_max_pose_list = []
        self.all_gt_min_pose_list = []
        self.all_gt_extents_range_list = []
        self.all_gt_extents_list = []
        self.all_relation_list = []
        self.all_initial_bounding_box = []
        self.all_bounding_box = []
        self.all_rotated_bounding_box = []
        
        
        total_objects = 0
        if total_objects == 0:
            for k, v in data['objects'].items():
                if 'block' in k:
                    total_objects += 1

        self.total_objects = total_objects

        

        self.obj_pair_list = list(permutations(range(total_objects), 2))
        #print(total_objects)
        # if True: #(data['point_cloud_1'].shape[0] == 0):
        #     point_string = 'point_cloud_'
        #     for i in range(total_objects):
        #         data[point_string + str(i+1)] = []
            
        #total_object = 3
        for i in range(data['point_cloud_1'].shape[0]):
            self.all_point_cloud.append([])
            self.all_pos_list.append([])
            self.all_gt_pose_list.append([])
            self.all_gt_orientation_list.append([])
            self.all_gt_max_pose_list.append([])
            self.all_gt_min_pose_list.append([])
            self.all_pos_list_p.append([])
            self.all_relation_list.append([])
            #self.all_initial_bounding_box.append([])
            self.all_bounding_box.append([])
            self.all_rotated_bounding_box.append([])
        # for k, v in data.items():
        #     if 'point_cloud' in k:
        #         #print(v.shape)
        #         if 'last' in k:
        #             #print(v.shape)
        #             self.all_point_cloud_last.append(v[0,:data_size, :])
        #         else:
        #             self.all_point_cloud.append(v[0,:data_size, :])
        #             self.all_point_cloud_1.append(v[11,:data_size, :])
        #             self.all_point_cloud_2.append(v[-1,:data_size, :])
        #             #print(v[0,:, :].shape)

        #             # print(np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2) # here is the function to get the center of pointcloud, sequence(2,0,1)
        #             A = np.min(v[0,:, :], axis = 0) + (np.max(v[0,:, :], axis = 0) - np.min(v[0,:, :], axis = 0))/2
        #             A_1 = [A[1], A[2], A[0]]
        #             #print(A_1) # A_1 here is the center of the point cloud
        sample_list = [0,11, -1]

        #print(data)
        if self.set_max:
            self.max_objects = max_objects
            A = np.arange(self.max_objects)
            #print('max objects',self.max_objects)
            if train:
                np.random.shuffle(A)
                            #print(A)
            select_obj_num_range = A[:total_objects]
            self.select_obj_num_range = select_obj_num_range
            one_hot_encoding = np.zeros((total_objects, self.max_objects))
            for i in range(len(select_obj_num_range)):
                one_hot_encoding[i][select_obj_num_range[i]] = 1
        else:
            A = np.arange(total_objects)
            np.random.shuffle(A)
            select_obj_num_range = A[:total_objects]
            self.select_obj_num_range = select_obj_num_range
        block_string = 'block_'
        for j in range(total_objects):
            #print(attrs['objects'][block_string + str(j+1)])
            #self.all_gt_pose_list.append(attrs['objects'][block_string + str(j+1)]['position'])
            if 'extents' in attrs['objects'][block_string + str(j+1)]: #self.pushing and not self.pick_place:
                self.all_gt_extents_range_list.append(attrs['objects'][block_string + str(j+1)]['extents_ranges'])
                self.all_gt_extents_list.append(attrs['objects'][block_string + str(j+1)]['extents'])
            else:
                self.all_gt_extents_list.append([attrs['objects'][block_string + str(j+1)]['x_extent'], attrs['objects'][block_string + str(j+1)]['y_extent'], attrs['objects'][block_string + str(j+1)]['z_extent']])

        # print(data['objects'])
        print('size', data['point_cloud_1'].shape[0])
        for i in range(data['point_cloud_1'].shape[0]):
            point_string = 'point_cloud_'
            block_string = 'block_'

            

            if i == data['point_cloud_1'].shape[0] - 1:
                i = -1
                if 'contact' in data:
                    contact_array = np.zeros((total_objects, total_objects))

                    #print('contact length', len(data['contact']))
                    time_step = i
                    #print('contact [0] length', len(data['contact'][time_step]))
                    for contact_i in range(len(data['contact'][time_step])):
                        if data['contact'][time_step][contact_i]['body0'] > -1 and data['contact'][time_step][contact_i]['body0'] < total_objects and data['contact'][time_step][contact_i]['body1'] > -1 and data['contact'][time_step][contact_i]['body1'] < total_objects:
                            contact_array[data['contact'][time_step][contact_i]['body0'], data['contact'][time_step][contact_i]['body1']] = 1
                            contact_array[data['contact'][time_step][contact_i]['body1'], data['contact'][time_step][contact_i]['body0']] = 1

                    #print(contact_array)
                else:
                    contact_array = - np.ones((total_objects, total_objects))
                for j in range(total_objects):
                    each_obj = j
                    current_block = "block_" + str(each_obj + 1)
                    initial_bounding_box = []
                    TF_matrix = []

                    if not self.pushing:
                        data[point_string + str(j+1)][i][:,0] += table_increase
                        data[point_string + str(j+1) + 'sampling'][i][:,0] += table_increase

                    
                    for inner_i in range(2):
                        for inner_j in range(2):
                            for inner_k in range(2):
                                if True:
                                    step = 0
                                    if 'extents' in attrs['objects'][current_block]:# self.pushing:
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['extents'][0]/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['extents'][1]/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['extents'][2]/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    else:
                                        #self.all_gt_extents_list.append([attrs['objects'][block_string + str(j+1)]['x_extent'], attrs['objects'][block_string + str(j+1)]['y_extent'], attrs['objects'][block_string + str(j+1)]['z_extent']])
                                        
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['x_extent']/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['y_extent']/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['z_extent']/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                    # print(current_block)
                    # print(data['objects'][current_block]['position'][i])
                    # print(attrs['objects'][current_block]['extents'])
                    # #print(initial_bounding_box)
                    initial_bounding_box = np.array(initial_bounding_box)
                    #print(initial_bounding_box.shape)

                    rotated_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    TF_rotated_bounding_matrix = []
                    Rotated_bounding_box_array = np.zeros((initial_bounding_box.shape[0], 3))
                    
                    for inner_i in range(initial_bounding_box.shape[0]):
                        rotated_bounding_box[inner_i, :, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])@TF_matrix[inner_i]
                        TF_rotated_bounding_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])), rotated_bounding_box[inner_i, :, :]))
                        Rotated_bounding_box_array[inner_i, :] = math_util.homogeneous_to_position(rotated_bounding_box[inner_i, :, :])
                        #print(math_util.homogeneous_to_position(rotated_bounding_box[inner_i, :, :]))
                    # print(Rotated_bounding_box_array)
                    # max_current_pose = np.max(Rotated_bounding_box_array, axis = 0)[:3]
                    # min_current_pose = np.min(Rotated_bounding_box_array, axis = 0)[:3]
                    # max_error = max_current_pose - self.get_point_cloud_max(data[point_string + str(j+1) + 'sampling'][0][:data_size, :])
                    # min_error = min_current_pose - self.get_point_cloud_min(data[point_string + str(j+1) + 'sampling'][0][:data_size, :])
                    # print(max_error)
                    # print(min_error)
                    
                    self.all_rotated_bounding_box[i].append(np.array(TF_rotated_bounding_matrix))
                    #print('final bounding box')
                    final_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    final_array = np.zeros((initial_bounding_box.shape[0], 3))
                    for inner_i in range(rotated_bounding_box.shape[0]):
                        final_bounding_box[inner_i,:, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][i], data['objects'][current_block]['orientation'][i])@TF_rotated_bounding_matrix[inner_i]
                        final_array[inner_i, :] = math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :])
                        #print(math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :]))
                    # print(final_array)
                    # time.sleep(1)
                    #print([i, final_array])
                    self.all_bounding_box[i].append(final_array)

                    max_current_pose = np.max(final_array, axis = 0)[:3]
                    min_current_pose = np.min(final_array, axis = 0)[:3]

                    # print(final_bounding_box)
                    # print(max_current_pose)
                    # print(min_current_pose)
                    # print('point cloud')
                    # print(self.get_point_cloud_max(data[point_string + str(j+1) + 'sampling'][i][:data_size, :]))
                    # print(self.get_point_cloud_min(data[point_string + str(j+1) + 'sampling'][i][:data_size, :]))

                    max_error = max_current_pose - self.get_point_cloud_max(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                    min_error = min_current_pose - self.get_point_cloud_min(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                    # print(sum(max_error))
                    # print(sum(min_error))



                    # print(T_12)

                    # print(data['objects']['table']['orientation'])

                

                    # print([i,j, data[point_string + str(j+1) + 'sampling'].shape])
                    # print([i,j, data[point_string + str(j+1) + 'sampling'][i].shape])
                    self.all_gt_max_pose_list[i].append(max_current_pose)
                    self.all_gt_min_pose_list[i].append(min_current_pose)



                    if self.real_data:  
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            # print(real_data[point_string + str(j+1) + 'sampling'][i].shape)  
                            # time.sleep(10)
                            self.all_point_cloud[i].append(np.zeros((128,3)))
                        else:
                            self.all_point_cloud[i].append(real_data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            self.all_pos_list[i].append([-1,-1,-1])
                        else:
                            self.all_pos_list[i].append(self.get_point_cloud_center(real_data[point_string + str(j+1)][i]))
                    else:
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            # print([j, i])
                            # print(data[point_string + str(j+1) + 'sampling'].shape)
                            # print(data[point_string + str(j+1) + 'sampling'][i].shape)  
                            # print(data[point_string + str(j+1)][i].shape)  
                            # time.sleep(10)
                            self.all_point_cloud[i].append(np.zeros((128,3)))
                        else:
                            self.all_point_cloud[i].append(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            self.all_pos_list[i].append([-1,-1,-1])
                        else:
                            self.all_pos_list[i].append(self.get_point_cloud_center(data[point_string + str(j+1)][i]))
                    if self.set_max and self.train:
                        # obj_input = one_hot_encoding[j].tolist()
                        # obj_input.extend(self.get_point_cloud_center(data[point_string + str(j+1)][i]))
                        # self.all_pos_list[i].append(obj_input)
                        self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][i])
                        self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][i])
                        ## yixuan double check about this part, whether it is important, not important for point cloud embedding for important for centroid estimation
                    else:
                        self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][i])
                        self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][i])
                        
                #print('gt pose lise', self.all_gt_pose_list)
                if self.real_data:
                    for obj_pair in self.obj_pair_list:
                        (anchor_idx, other_idx) = obj_pair
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_ground_truth_relations(self.all_gt_pose_list[i][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_gt_pose_list[i][other_idx], self.all_gt_extents_list[other_idx]))
                        anchor_pose_pc = self.get_point_cloud_center(self.all_point_cloud[i][anchor_idx])
                        anchor_pose_pc_max = self.get_point_cloud_max(self.all_point_cloud[i][anchor_idx])
                        anchor_pose_pc_min = self.get_point_cloud_min(self.all_point_cloud[i][anchor_idx])

                        other_pose_pc = self.get_point_cloud_center(self.all_point_cloud[i][other_idx])
                        other_pose_pc_max = self.get_point_cloud_max(self.all_point_cloud[i][other_idx])
                        other_pose_pc_min = self.get_point_cloud_min(self.all_point_cloud[i][other_idx])
                        self.all_relation_list[i].append(self.get_ground_truth_tf_relations_contact_pc(anchor_pose_pc, anchor_pose_pc_max, anchor_pose_pc_min, other_pose_pc, other_pose_pc_max, other_pose_pc_min))
                else:
                    for obj_pair in self.obj_pair_list:
                        (anchor_idx, other_idx) = obj_pair
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_ground_truth_relations(self.all_gt_pose_list[i][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_gt_pose_list[i][other_idx], self.all_gt_extents_list[other_idx]))
                        if self.pushing:
                            self.all_relation_list[i].append(self.get_ground_truth_tf_relations_contact(contact_array, anchor_idx, other_idx,self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_pose_list[i][other_idx],self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx])[:])
                        else:
                            self.all_relation_list[i].append(self.get_ground_truth_tf_relations_contact(contact_array, anchor_idx, other_idx,self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_pose_list[i][other_idx],self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx])[:])
                            #self.all_relation_list[i].append(self.get_ground_truth_tf_relations(self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_pose_list[i][other_idx],self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx])[:])
                        
                break
            else:
                if 'contact' in data:
                    contact_array = np.zeros((total_objects, total_objects))

                    #print('contact length', len(data['contact']))
                    time_step = i
                    #print('contact [0] length', len(data['contact'][time_step]))
                    for contact_i in range(len(data['contact'][time_step])):
                        if data['contact'][time_step][contact_i]['body0'] > -1 and data['contact'][time_step][contact_i]['body0'] < total_objects and data['contact'][time_step][contact_i]['body1'] > -1 and data['contact'][time_step][contact_i]['body1'] < total_objects:
                            contact_array[data['contact'][time_step][contact_i]['body0'], data['contact'][time_step][contact_i]['body1']] = 1
                            contact_array[data['contact'][time_step][contact_i]['body1'], data['contact'][time_step][contact_i]['body0']] = 1

                    #print(contact_array)
                else:
                    contact_array = - np.ones((total_objects, total_objects))
                for j in range(total_objects):
                    #print(data[point_string + str(j+1)][i].shape)
                    if not self.pushing:
                        data[point_string + str(j+1)][i][:,0] += table_increase
                        data[point_string + str(j+1) + 'sampling'][i][:,0] += table_increase
                    #time.sleep(1)
                    each_obj = j
                    current_block = "block_" + str(each_obj + 1)
                    initial_bounding_box = []
                    TF_matrix = []
                    for inner_i in range(2):
                        for inner_j in range(2):
                            for inner_k in range(2):
                                if True:
                                    step = 0
                                    if 'extents' in attrs['objects'][current_block]: #self.pushing and not self.pick_place: self.pushing:
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['extents'][0]/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['extents'][1]/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['extents'][2]/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    else:
                                        #self.all_gt_extents_list.append([attrs['objects'][block_string + str(j+1)]['x_extent'], attrs['objects'][block_string + str(j+1)]['y_extent'], attrs['objects'][block_string + str(j+1)]['z_extent']])
                                        
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['x_extent']/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['y_extent']/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['z_extent']/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    
                    # print(current_block)
                    # print(data['objects'][current_block]['position'][i])
                    # print(attrs['objects'][current_block]['extents'])
                    # #print(initial_bounding_box)
                    initial_bounding_box = np.array(initial_bounding_box)
                    #print(initial_bounding_box.shape)

                    rotated_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    TF_rotated_bounding_matrix = []
                    
                    for inner_i in range(initial_bounding_box.shape[0]):
                        rotated_bounding_box[inner_i, :, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])@TF_matrix[inner_i]
                        TF_rotated_bounding_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])), rotated_bounding_box[inner_i, :, :]))
                        #print(math_util.homogeneous_to_position(rotated_bounding_box[inner_i, :, :]))

                    
                    #print('final bounding box')
                    self.all_rotated_bounding_box[i].append(np.array(TF_rotated_bounding_matrix))

                    final_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    final_array = np.zeros((initial_bounding_box.shape[0], 3))
                    if self.pick_place or self.pushing or i == 0:
                        for inner_i in range(rotated_bounding_box.shape[0]):
                            final_bounding_box[inner_i,:, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][i], data['objects'][current_block]['orientation'][i])@TF_rotated_bounding_matrix[inner_i]
                            final_array[inner_i, :] = math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :])
                    else:
                        for inner_i in range(rotated_bounding_box.shape[0]): ## special case for the relations at timestep 1 
                            final_bounding_box[inner_i,:, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][11], data['objects'][current_block]['orientation'][11])@TF_rotated_bounding_matrix[inner_i]
                            final_array[inner_i, :] = math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :])
                        #print(math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :]))
                    # print([i, final_array])

                    self.all_bounding_box[i].append(final_array)
                    
                    max_current_pose = np.max(final_array, axis = 0)[:3]
                    min_current_pose = np.min(final_array, axis = 0)[:3]

                    # print('i', i)
                    # #print(final_bounding_box)
                    # print(max_current_pose)
                    # print(min_current_pose)
                    # print('point cloud')
                    # print(self.get_point_cloud_max(data[point_string + str(j+1) + 'sampling'][i][:data_size, :]))
                    # print(self.get_point_cloud_min(data[point_string + str(j+1) + 'sampling'][i][:data_size, :]))
                    max_error = max_current_pose - self.get_point_cloud_max(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                    min_error = min_current_pose - self.get_point_cloud_min(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                    # print(sum(max_error))
                    # print(sum(min_error))




                    # print(T_12)

                    # print(data['objects']['table']['orientation'])

                

                    # print([i,j, data[point_string + str(j+1) + 'sampling'].shape])
                    # print([i,j, data[point_string + str(j+1) + 'sampling'][i].shape])
                    self.all_gt_max_pose_list[i].append(max_current_pose)
                    self.all_gt_min_pose_list[i].append(min_current_pose)



                    # if self.real_data:
                    #     self.all_point_cloud[i].append(real_data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                    #     #print('enter')
                    #     self.all_pos_list[i].append(self.get_point_cloud_center(real_data[point_string + str(j+1)][i]))
                    # else:
                    #     self.all_point_cloud[i].append(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                    #     self.all_pos_list[i].append(self.get_point_cloud_center(data[point_string + str(j+1)][i]))

                    if self.real_data:  
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            # print(real_data[point_string + str(j+1) + 'sampling'][i].shape)  
                            # time.sleep(10)
                            self.all_point_cloud[i].append(np.zeros((128,3)))
                        else:
                            self.all_point_cloud[i].append(real_data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            self.all_pos_list[i].append([-1,-1,-1])
                        else:
                            self.all_pos_list[i].append(self.get_point_cloud_center(real_data[point_string + str(j+1)][i]))
                    else:
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            # print([j, i])
                            # print(data[point_string + str(j+1) + 'sampling'].shape)
                            # print(data[point_string + str(j+1) + 'sampling'][i].shape)  
                            # print(data[point_string + str(j+1)][i].shape)  
                            # time.sleep(10)
                            self.all_point_cloud[i].append(np.zeros((128,3)))
                        else:
                            # print('pc_center', self.get_point_cloud_center(data[point_string + str(j+1)][i]))
                            # time.sleep(10)
                            self.all_point_cloud[i].append(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
                        if self.evaluate_end_relations and this_one_hot_encoding[0][j] == 0:
                            self.all_pos_list[i].append([-1,-1,-1])
                        else:
                            self.all_pos_list[i].append(self.get_point_cloud_center(data[point_string + str(j+1)][i]))
                            

                    if self.set_max and self.train:
                        # obj_input = one_hot_encoding[j].tolist()
                        # obj_input.extend(self.get_point_cloud_center(data[point_string + str(j+1)][i]))
                        # self.all_pos_list[i].append(obj_input)
                        if self.pick_place or self.pushing or i!=1:
                            self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][i]) #data['objects'][block_string + str(j+1)]['orientation'][i]])
                            self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][i])
                        else:
                            self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][11]) #data['objects'][block_string + str(j+1)]['orientation'][i]])
                            self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][11])
                        ## yixuan double check about this part, whether it is important, not important for point cloud embedding for important for centroid estimation
                    else:
                        if self.pick_place or self.pushing or i!=1:
                            self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][i]) #data['objects'][block_string + str(j+1)]['orientation'][i]])
                            self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][i])
                        else:
                            self.all_gt_pose_list[i].append(data['objects'][block_string + str(j+1)]['position'][11]) #data['objects'][block_string + str(j+1)]['orientation'][i]])
                            self.all_gt_orientation_list[i].append(data['objects'][block_string + str(j+1)]['orientation'][11])

                #print('gt pose lise', self.all_gt_pose_list)
                if self.real_data:
                    for obj_pair in self.obj_pair_list:
                        (anchor_idx, other_idx) = obj_pair
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_ground_truth_relations(self.all_gt_pose_list[i][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_gt_pose_list[i][other_idx], self.all_gt_extents_list[other_idx]))
                        anchor_pose_pc = self.get_point_cloud_center(self.all_point_cloud[i][anchor_idx])
                        anchor_pose_pc_max = self.get_point_cloud_max(self.all_point_cloud[i][anchor_idx])
                        anchor_pose_pc_min = self.get_point_cloud_min(self.all_point_cloud[i][anchor_idx])

                        other_pose_pc = self.get_point_cloud_center(self.all_point_cloud[i][other_idx])
                        other_pose_pc_max = self.get_point_cloud_max(self.all_point_cloud[i][other_idx])
                        other_pose_pc_min = self.get_point_cloud_min(self.all_point_cloud[i][other_idx])
                        self.all_relation_list[i].append(self.get_ground_truth_tf_relations_contact_pc(anchor_pose_pc, anchor_pose_pc_max, anchor_pose_pc_min, other_pose_pc, other_pose_pc_max, other_pose_pc_min))
                else:
                    for obj_pair in self.obj_pair_list:
                        (anchor_idx, other_idx) = obj_pair
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_relations_sigmoid_6(self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]))
                        #self.all_relation_list[i].append(self.get_ground_truth_relations(self.all_gt_pose_list[i][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_gt_pose_list[i][other_idx], self.all_gt_extents_list[other_idx]))
                        if self.pushing:
                            self.all_relation_list[i].append(self.get_ground_truth_tf_relations_contact(contact_array, anchor_idx, other_idx ,self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_pose_list[i][other_idx],self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx])[:])
                        else:
                            self.all_relation_list[i].append(self.get_ground_truth_tf_relations_contact(contact_array, anchor_idx, other_idx,self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_pose_list[i][other_idx],self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx])[:])
                            #self.all_relation_list[i].append(self.get_ground_truth_tf_relations(self.all_gt_pose_list[i][anchor_idx], self.all_gt_max_pose_list[i][anchor_idx], self.all_gt_min_pose_list[i][anchor_idx], self.all_gt_pose_list[i][other_idx],self.all_gt_max_pose_list[i][other_idx],self.all_gt_min_pose_list[i][other_idx])[:])
                        
            
            # self.all_point_cloud[i] = [data['point_cloud_1'][i][:data_size, :], data['point_cloud_2'][i][:data_size, :], data['point_cloud_3'][i][:data_size, :]]
            # #self.all_pos_list[i] = [data['objects']['block_1']['position'][sample_list[i]], data['objects']['block_2']['position'][sample_list[i]], data['objects']['block_3']['position'][sample_list[i]]]
            # self.all_pos_list[i] = [self.get_point_cloud_center(data['point_cloud_1'][i]), self.get_point_cloud_center(data['point_cloud_2'][i]), self.get_point_cloud_center(data['point_cloud_3'][i])] 

            
            # self.all_point_cloud[i] = [data['point_cloud_1'][i][:data_size, :], data['point_cloud_2'][i][:data_size, :], data['point_cloud_3'][i][:data_size, :]]
            # #self.all_pos_list[i] = [data['objects']['block_1']['position'][sample_list[i]], data['objects']['block_2']['position'][sample_list[i]], data['objects']['block_3']['position'][sample_list[i]]]
            # self.all_pos_list[i] = [self.get_point_cloud_center(data['point_cloud_1'][i]), self.get_point_cloud_center(data['point_cloud_2'][i]), self.get_point_cloud_center(data['point_cloud_3'][i])] 
            # print(self.all_pos_list[i])
            # print(self.all_pos_list_p[i])
            # self.all_point_cloud[0].append(data['point_cloud_1'][i][:data_size, :])
            # self.all_point_cloud[1].append(data['point_cloud_2'][i][:data_size, :])
            # self.all_point_cloud[2].append(data['point_cloud_3'][i][:data_size, :])
            # self.all_pos_list[0].append(data['objects']['block_1']['position'][sample_list[i]])
            # self.all_pos_list[1].append(data['objects']['block_2']['position'][sample_list[i]])
            # self.all_pos_list[2].append(data['objects']['block_3']['position'][sample_list[i]])
        
        # for k, v in data['objects'].items():
        #     if 'block' in k:
        #         #print(k)
        #         self.all_pos_list.append(v['position'][0]*self.scale)
        #         self.all_pos_list_last.append(v['position'][-1]*self.scale)
        #         self.all_orient_list.append(v['orientation'][0]) # how to get orientation from point cloud directly is another problem.
        #         self.all_orient_list_last.append(v['orientation'][-1])
        #         #print(v['position'][0])

        #print(self.all_point_cloud[0].shape)

        
        
        #print(self.all_relation_list)
        # print(attrs['behavior_params'])
        
        #print()
        #print(attrs['behavior_params'][''])
        #push = True
        #print(attrs['behavior_params']['stack_objects']['behaviors'])
        self.random_push = False
        if push and not self.pick_place:
            self.random_push = True
        
        
        
        if pick_place:
            index_list = []
            move_obj = -1
            for i in range(total_objects):
                if str(i+1) in attrs['behavior_params']['stack_objects']['behaviors']['pick']['target_object']:
                    move_obj = i
                index_list.append(str(i+1))
            if self.set_max:
                self.action_1 = []
                for i in range(self.max_objects):
                    self.action_1.append(0)
            else:
                self.action_1 = []
                for i in range(total_objects):
                    self.action_1.append(0)
            if self.set_max and self.train:
                self.action_1[select_obj_num_range[move_obj]] = 1
            else:
                self.action_1[move_obj] = 1
            for i in range(3):
                # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                self.action_1.append(attrs['behavior_params']['stack_objects']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick']['behaviors']['approach']['target_pose'][i])
            self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
            self.action_1 = [self.action_1]
            self.action_2 = []
        elif self.random_push:
            if not self.set_max:
                #print(attrs['behavior_params'])
                move_obj = -1
                for i in range(total_objects):
                    if str(i+1) in attrs['behavior_params']['']['target_object']:
                        move_obj = i
                self.action_1 = []
                for i in range(total_objects):
                    self.action_1.append(0)
                self.action_1[move_obj] = 1
                
                #print(self.action_1)
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_1.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                self.action_1 = [self.action_1]
                self.action_2 = [] #[1,0,0,0]
                for i in range(total_objects):
                    self.action_2.append(0)
                self.action_2[move_obj] = 1
                #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_2.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                self.action_2 = [self.action_2]
            else:
                #print('enter random push')
                if 'push_step_1' in attrs['behavior_params']:
                    move_obj = -1
                    #print(attrs['behavior_params'])
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params']['push_step_1']['target_object']:
                            move_obj = i
                    self.action_1 = []
                    for i in range(self.max_objects):
                        self.action_1.append(0)
                    if self.train:
                        self.action_1[select_obj_num_range[move_obj]] = 1
                    else:
                        self.action_1[move_obj] = 1
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.action_1.append(attrs['behavior_params']['push_step_1']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['push_step_1']['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.action_1.append(attrs['behavior_params']['push_step_1']['target_object_pose'][i] - attrs['behavior_params']['push_step_1']['init_object_pose'][i])
                    self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                    self.action_1 = [self.action_1]
                        
                    self.action_2 = []
                    for i in range(self.max_objects):
                        self.action_2.append(0)
                    if self.train:
                        self.action_2[select_obj_num_range[move_obj]] = 1
                    else:
                        self.action_2[move_obj] = 1
                        #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.action_2.append(attrs['behavior_params']['push_step_1']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['push_step_1']['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.action_2.append(attrs['behavior_params']['push_step_1']['target_object_pose'][i] - attrs['behavior_params']['push_step_1']['init_object_pose'][i])
                    self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                    self.action_2 = [self.action_2]
                    print(self.action_1)
                    #time.sleep(10)
                else:
                    move_obj = -1
                    #print(attrs['behavior_params'])
                    for i in range(total_objects):
                        if str(i+1) in attrs['behavior_params']['']['target_object']:
                            move_obj = i
                    self.action_1 = []
                    for i in range(self.max_objects):
                        self.action_1.append(0)
                    if self.train:
                        self.action_1[select_obj_num_range[move_obj]] = 1
                    else:
                        self.action_1[move_obj] = 1
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.action_1.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.action_1.append(attrs['behavior_params']['']['target_object_pose'][i] - attrs['behavior_params']['']['init_object_pose'][i])
                    self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                    self.action_1 = [self.action_1]
                        
                    self.action_2 = []
                    for i in range(self.max_objects):
                        self.action_2.append(0)
                    if self.train:
                        self.action_2[select_obj_num_range[move_obj]] = 1
                    else:
                        self.action_2[move_obj] = 1
                        #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                    if not self.updated_behavior_params:
                        for i in range(3):
                            self.action_2.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                    else:
                        for i in range(3):
                            self.action_2.append(attrs['behavior_params']['']['target_object_pose'][i] - attrs['behavior_params']['']['init_object_pose'][i])
                    self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                    self.action_2 = [self.action_2]
                    print(self.action_1)
                    #time.sleep(10)
        elif push:
            if not self.set_max:
                self.action_1 = []
                for i in range(total_objects):
                    self.action_1.append(0)
                self.action_1[0] = 1
                #print(self.action_1)
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_1.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                self.action_1 = [self.action_1]
                self.action_2 = [] #[1,0,0,0]
                for i in range(total_objects):
                    self.action_2.append(0)
                self.action_2[0] = 1
                #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_2.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                self.action_2 = [self.action_2]
            else:
                self.action_1 = []
                for i in range(self.max_objects):
                    self.action_1.append(0)
                if self.train:
                    self.action_1[select_obj_num_range[0]] = 1
                else:
                    self.action_1[0] = 1
                for i in range(3):
                    self.action_1.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                self.action_1 = [self.action_1]
                    
                self.action_2 = []
                for i in range(self.max_objects):
                    self.action_2.append(0)
                if self.train:
                    self.action_2[select_obj_num_range[0]] = 1
                else:
                    self.action_2[0] = 1
                    #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                for i in range(3):
                    self.action_2.append(attrs['behavior_params']['']['behaviors']['push']['target_pose'][i] - attrs['behavior_params']['']['behaviors']['approach']['target_pose'][i])
                self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                self.action_2 = [self.action_2]
        else:
            randomness = 0
            if self.four_data:
                self.action_1 = [0,1,0,0]
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_1.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                self.action_1 = [self.action_1]
                self.action_2 = [0,0,1,0]
                #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_2.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['pick']['init_object_pose'][i])
                self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                self.action_2 = [self.action_2]
                
                self.action_3 = [0,0,0,1]
                    #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                for i in range(3):
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                    # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_3.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_4']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_4']['behaviors']['pick']['init_object_pose'][i])
                self.action_3[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                self.action_3 = [self.action_3]
            else:
                if not self.set_max:
                    self.action_1 = [0,1,0]
                    for i in range(3):
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                        self.action_1.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                    self.action_1 = [self.action_1]
                    self.action_2 = [0,0,1]
                    #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                    for i in range(3):
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                        self.action_2.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['pick']['init_object_pose'][i])
                    self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                    self.action_2 = [self.action_2]
                elif self.set_max:
                    #self.action_1 = [0,1,0]
                    self.action_1 = []
                    for i in range(self.max_objects):
                        self.action_1.append(0)
                    if self.train:
                        self.action_1[select_obj_num_range[1]] = 1
                    else:
                        self.action_1[1] = 1
                    for i in range(3):
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                        self.action_1.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                    self.action_1[-1] = 0 #self.action_1[-1] - 0.02 # 0 with yixuan test
                    self.action_1 = [self.action_1]
                    
                    self.action_2 = []
                    for i in range(self.max_objects):
                        self.action_2.append(0)
                    if self.train:
                        self.action_2[select_obj_num_range[2]] = 1
                    else:
                        self.action_2[2] = 1
                    #self.action_2.extend(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][:3])
                    for i in range(3):
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['place']['target_object_pose'][i])
                        # print(attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_2']['behaviors']['pick']['init_object_pose'][i])
                        self.action_2.append(np.random.uniform(-randomness, randomness) + attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['place']['target_object_pose'][i] - attrs['behavior_params']['stack_objects']['behaviors']['pick_place_block_3']['behaviors']['pick']['init_object_pose'][i])
                    self.action_2[-1] = 0 #self.action_2[-1] - 0.02 # 0 with yixuan test
                    self.action_2 = [self.action_2]
        #print([self.action_1, self.action_2])
        
        # self.action = data['action_primitives']*self.scale # we do not need this any more I guess

        
        self.obj_voxels_single = []
        self.obj_voxels_by_obj_pair_dict = []
        self.obj_voxels_by_obj_pair_dict_anchor = []
        self.obj_voxels_by_obj_pair_dict_other = []
        for i in range(data['point_cloud_1'].shape[0]):
            self.obj_voxels_single.append(dict())
            self.obj_voxels_by_obj_pair_dict.append(dict())
            self.obj_voxels_by_obj_pair_dict_anchor.append(dict())
            self.obj_voxels_by_obj_pair_dict_other.append(dict())


        self.obj_voxels_by_obj_pair_dict_last = dict()
        self.obj_voxels_by_obj_pair_dict_anchor_last = dict()
        self.obj_voxels_by_obj_pair_dict_other_last = dict()

        self.obj_voxels_status_by_obj_pair_dict = dict()
        self.obj_pcd_path_by_obj_pair_dict = dict()

        

        #self.robot_all_pair_voxels = RobotAllPairVoxels(self.voxel_cluster_path, self.image_cluster_path)

        # Load all the obj pair embeddings for now. this is because we can have
        # a case where there are two objects with no edge between them and this
        # will result in no pairwise embedding. However, for the downstream task
        # we might end up using the embeddings between objects which have contacts
        # only.

        for obj_id in range(self.total_objects):
            for i in range(data['point_cloud_1'].shape[0]):
                total_point_cloud = self.all_point_cloud[i][obj_id]
                #print(total_point_cloud.shape)
                self.obj_voxels_single[i][obj_id] = total_point_cloud.T
            
        
        for obj_pair in self.obj_pair_list:
            (anchor_idx, other_idx) = obj_pair
            #print(obj_pair)
            # status, robot_voxels = self.robot_all_pair_voxels.init_voxels_for_pcd_pair(
            #     anchor_idx, other_idx
            # )

            #print(self.all_point_cloud[anchor_idx].shape)
            for i in range(data['point_cloud_1'].shape[0]):
                total_point_cloud = np.concatenate((self.all_point_cloud[i][anchor_idx], self.all_point_cloud[i][other_idx]), axis = 0)
                #print(total_point_cloud.shape)
                self.obj_voxels_by_obj_pair_dict[i][obj_pair] = total_point_cloud.T
                self.obj_voxels_by_obj_pair_dict_anchor[i][obj_pair] = self.all_point_cloud[i][anchor_idx].T
                self.obj_voxels_by_obj_pair_dict_other[i][obj_pair] = self.all_point_cloud[i][other_idx].T

            # total_point_cloud_last = np.concatenate((self.all_point_cloud_last[anchor_idx], self.all_point_cloud_last[other_idx]), axis = 0)
            # #print(total_point_cloud.shape)
            # self.obj_voxels_by_obj_pair_dict_last[obj_pair] = total_point_cloud_last.T
            # self.obj_voxels_by_obj_pair_dict_anchor_last[obj_pair] = self.all_point_cloud_last[anchor_idx].T
            # self.obj_voxels_by_obj_pair_dict_other_last[obj_pair] = self.all_point_cloud_last[other_idx].T
            #self.obj_voxels_status_by_obj_pair_dict[obj_pair] = status
            self.obj_pcd_path_by_obj_pair_dict[obj_pair] = (
                scene_path, 
                scene_path # yixuan test
            )
        #print('finish')

    def get_relations(self, anchor_point_cloud, other_point_cloud):
        action = [0]
        self.point_cloud_max = True
        if self.point_cloud_max:
            # anchor_pose_max = data_dict['anchor_pos_max']#self.get_point_cloud_max(data_dict['anchor_point_cloud'])
            # anchor_pose_min = data_dict['anchor_pos_min']#self.get_point_cloud_min(data_dict['anchor_point_cloud'])

            # other_pose_max = data_dict['other_pos_max']#self.get_point_cloud_max(data_dict['other_point_cloud'])
            # other_pose_min = data_dict['other_pos_min']#self.get_point_cloud_min(data_dict['other_point_cloud'])
            
            # print([anchor_pose_max, anchor_pose_min])
            # print([other_pose_max, other_pose_min])
            # print('anchor_pos', data_dict['anchor_pos'])
            # print('other_pos', data_dict['other_pos'])
            anchor_pose = self.get_point_cloud_center(anchor_point_cloud)
            anchor_pose_max = self.get_point_cloud_max(anchor_point_cloud)
            anchor_pose_min = self.get_point_cloud_min(anchor_point_cloud)

            other_pose = self.get_point_cloud_center(other_point_cloud)
            other_pose_max = self.get_point_cloud_max(other_point_cloud)
            other_pose_min = self.get_point_cloud_min(other_point_cloud)

            self.only_left = False

            if self.pick_place:
                if(anchor_pose[0] < other_pose[0]):
                    right_pred = 1
                    left_pred = 0
                else:
                    right_pred = 0
                    left_pred = 1

                if(anchor_pose[1] < other_pose[1]):
                    front_pred = 1
                    behind_pred = 0
                else:
                    front_pred = 0
                    behind_pred = 1
                #action.append(horizon_pred)
                action.append(right_pred)
                action.append(left_pred)
                #action.append(vertical_pred)
                action.append(front_pred)
                action.append(behind_pred)
            elif self.only_left:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                    
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    action.append(1)
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                    
                else:
                    action.append(0)
                    action.append(0)
                    action.append(0)
                # if(np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < (data_dict['att_objects']['block1']['x_extent'] - data_dict['att_objects']['block2']['x_extent'])/2) and (np.abs(data_dict['objects']['block2']['position'][0][1] - data_dict['objects']['block1']['position'][0][1]) < (data_dict['att_objects']['block1']['y_extent'] - data_dict['att_objects']['block2']['y_extent'])/2):
                #     action.append(1) # form the stack with support relationships yixuan notice need to deal with that carefully when leveraging the stacking relationships
                # else:
                #     action.append(0) # this is actually complex contact
            # yixuan notice, here we assume all the objects with the same size
            # print(data_dict['other_pos'])
            # print(data_dict['anchor_pos'])
            # print(data_dict['att_objects']['block1'])
            if((other_pose[2] - anchor_pose[2]) > 0):
                current_extents = anchor_pose_max - anchor_pose_min
            else:
                current_extents = other_pose_max - other_pose_min
            if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
                if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                    action.append(1)
                    if((other_pose[2] - anchor_pose[2]) > 0): # above
                        action.append(1)
                        action.append(0)
                    else:  # below
                        action.append(0)
                        action.append(1)
                else:
                    action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                action.append(0)
                action.append(0)
                action.append(0)
        return action
    
    def get_relations_1(self, anchor_point_cloud, other_point_cloud):
        action = []
        self.point_cloud_max = True
        if self.point_cloud_max:
            # anchor_pose_max = data_dict['anchor_pos_max']#self.get_point_cloud_max(data_dict['anchor_point_cloud'])
            # anchor_pose_min = data_dict['anchor_pos_min']#self.get_point_cloud_min(data_dict['anchor_point_cloud'])

            # other_pose_max = data_dict['other_pos_max']#self.get_point_cloud_max(data_dict['other_point_cloud'])
            # other_pose_min = data_dict['other_pos_min']#self.get_point_cloud_min(data_dict['other_point_cloud'])
            
            # print([anchor_pose_max, anchor_pose_min])
            # print([other_pose_max, other_pose_min])
            # print('anchor_pos', data_dict['anchor_pos'])
            # print('other_pos', data_dict['other_pos'])
            anchor_pose = self.get_point_cloud_center(anchor_point_cloud)
            anchor_pose_max = self.get_point_cloud_max(anchor_point_cloud)
            anchor_pose_min = self.get_point_cloud_min(anchor_point_cloud)

            other_pose = self.get_point_cloud_center(other_point_cloud)
            other_pose_max = self.get_point_cloud_max(other_point_cloud)
            other_pose_min = self.get_point_cloud_min(other_point_cloud)

            self.only_left = False

            if self.pick_place:
                if(anchor_pose[0] < other_pose[0]):
                    right_pred = 1
                    left_pred = 0
                else:
                    right_pred = 0
                    left_pred = 1

                if(anchor_pose[1] < other_pose[1]):
                    front_pred = 1
                    behind_pred = 0
                else:
                    front_pred = 0
                    behind_pred = 1
                #action.append(horizon_pred)
                action.append(right_pred)
                action.append(1 - right_pred)
                action.append(left_pred)
                action.append(1 - left_pred)
                #action.append(vertical_pred)
                action.append(front_pred)
                action.append(1 - front_pred)
                action.append(behind_pred)
                action.append(1 - behind_pred)
            elif self.pushing:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                        action.append(0)
                        action.append(1)
                    else:
                        action.append(0)
                        action.append(1)
                        action.append(1)
                        action.append(0)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                        action.append(0)
                        action.append(1)
                    else:
                        action.append(0)
                        action.append(1)
                        action.append(1)
                        action.append(0)                    
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
            else:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    action.append(1)
                    action.append(0)
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                        action.append(0)
                        action.append(1)
                    else:
                        action.append(0)
                        action.append(1)
                        action.append(1)
                        action.append(0)
                else:
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    action.append(1)
                    action.append(0)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                        action.append(0)
                        action.append(1)
                    else:
                        action.append(0)
                        action.append(1)
                        action.append(1)
                        action.append(0)                    
                else:
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
                # if(np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < (data_dict['att_objects']['block1']['x_extent'] - data_dict['att_objects']['block2']['x_extent'])/2) and (np.abs(data_dict['objects']['block2']['position'][0][1] - data_dict['objects']['block1']['position'][0][1]) < (data_dict['att_objects']['block1']['y_extent'] - data_dict['att_objects']['block2']['y_extent'])/2):
                #     action.append(1) # form the stack with support relationships yixuan notice need to deal with that carefully when leveraging the stacking relationships
                # else:
                #     action.append(0) # this is actually complex contact
            # yixuan notice, here we assume all the objects with the same size
            # print(data_dict['other_pos'])
            # print(data_dict['anchor_pos'])
            # print(data_dict['att_objects']['block1'])
            if((other_pose[2] - anchor_pose[2]) > 0):
                current_extents = anchor_pose_max - anchor_pose_min
            else:
                current_extents = other_pose_max - other_pose_min
            if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
                if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                    action.append(1)
                    action.append(0)
                    if((other_pose[2] - anchor_pose[2]) > 0): # above
                        action.append(1)
                        action.append(0)
                        action.append(0)
                        action.append(1)
                    else:  # below
                        action.append(0)
                        action.append(1)
                        action.append(1)
                        action.append(0)
                else:
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
                    action.append(0)
                    action.append(1)
            else:
                action.append(0)
                action.append(1)
                action.append(0)
                action.append(1)
                action.append(0)
                action.append(1)
        return action
    
    def get_relations_sigmoid(self, anchor_point_cloud, other_point_cloud):
        action = []
        self.point_cloud_max = True
        if self.point_cloud_max:
            # anchor_pose_max = data_dict['anchor_pos_max']#self.get_point_cloud_max(data_dict['anchor_point_cloud'])
            # anchor_pose_min = data_dict['anchor_pos_min']#self.get_point_cloud_min(data_dict['anchor_point_cloud'])

            # other_pose_max = data_dict['other_pos_max']#self.get_point_cloud_max(data_dict['other_point_cloud'])
            # other_pose_min = data_dict['other_pos_min']#self.get_point_cloud_min(data_dict['other_point_cloud'])
            
            # print([anchor_pose_max, anchor_pose_min])
            # print([other_pose_max, other_pose_min])
            # print('anchor_pos', data_dict['anchor_pos'])
            # print('other_pos', data_dict['other_pos'])
            anchor_pose = self.get_point_cloud_center(anchor_point_cloud)
            anchor_pose_max = self.get_point_cloud_max(anchor_point_cloud)
            anchor_pose_min = self.get_point_cloud_min(anchor_point_cloud)

            other_pose = self.get_point_cloud_center(other_point_cloud)
            other_pose_max = self.get_point_cloud_max(other_point_cloud)
            other_pose_min = self.get_point_cloud_min(other_point_cloud)

            self.only_left = False

            if self.pick_place:
                if(anchor_pose[0] < other_pose[0]):
                    right_pred = 1
                    left_pred = 0
                else:
                    right_pred = 0
                    left_pred = 1

                if(anchor_pose[1] < other_pose[1]):
                    front_pred = 1
                    behind_pred = 0
                else:
                    front_pred = 0
                    behind_pred = 1
                #action.append(horizon_pred)
                action.append(right_pred)
                action.append(left_pred)
                #action.append(vertical_pred)
                action.append(front_pred)
                action.append(behind_pred)
            elif self.pushing:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                  
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    action.append(1)
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                  
                else:
                    action.append(0)
                    action.append(0)
                    action.append(0)
                # if(np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < (data_dict['att_objects']['block1']['x_extent'] - data_dict['att_objects']['block2']['x_extent'])/2) and (np.abs(data_dict['objects']['block2']['position'][0][1] - data_dict['objects']['block1']['position'][0][1]) < (data_dict['att_objects']['block1']['y_extent'] - data_dict['att_objects']['block2']['y_extent'])/2):
                #     action.append(1) # form the stack with support relationships yixuan notice need to deal with that carefully when leveraging the stacking relationships
                # else:
                #     action.append(0) # this is actually complex contact
            # yixuan notice, here we assume all the objects with the same size
            # print(data_dict['other_pos'])
            # print(data_dict['anchor_pos'])
            # print(data_dict['att_objects']['block1'])
            if((other_pose[2] - anchor_pose[2]) > 0):
                current_extents = anchor_pose_max - anchor_pose_min
            else:
                current_extents = other_pose_max - other_pose_min
            if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
                if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                    action.append(1)
                    if((other_pose[2] - anchor_pose[2]) > 0): # above
                        action.append(1)
                        action.append(0)
                    else:  # below
                        action.append(0)
                        action.append(1)
                else:
                    action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                action.append(0)
                action.append(0)
                action.append(0)
        return action
    
    def get_relations_sigmoid_6(self, anchor_point_cloud, other_point_cloud):
        action = []
        self.point_cloud_max = True
        if self.point_cloud_max:
            # anchor_pose_max = data_dict['anchor_pos_max']#self.get_point_cloud_max(data_dict['anchor_point_cloud'])
            # anchor_pose_min = data_dict['anchor_pos_min']#self.get_point_cloud_min(data_dict['anchor_point_cloud'])

            # other_pose_max = data_dict['other_pos_max']#self.get_point_cloud_max(data_dict['other_point_cloud'])
            # other_pose_min = data_dict['other_pos_min']#self.get_point_cloud_min(data_dict['other_point_cloud'])
            
            
            # print('anchor_pos', data_dict['anchor_pos'])
            # print('other_pos', data_dict['other_pos'])
            anchor_pose = self.get_point_cloud_center(anchor_point_cloud)
            anchor_pose_max = self.get_point_cloud_max(anchor_point_cloud)
            anchor_pose_min = self.get_point_cloud_min(anchor_point_cloud)

            other_pose = self.get_point_cloud_center(other_point_cloud)
            other_pose_max = self.get_point_cloud_max(other_point_cloud)
            other_pose_min = self.get_point_cloud_min(other_point_cloud)

            # print([anchor_pose_max, anchor_pose_min])
            # print([other_pose_max, other_pose_min])

            self.only_left = False

            if self.pick_place:
                if(anchor_pose[0] < other_pose[0]):
                    right_pred = 1
                    left_pred = 0
                else:
                    right_pred = 0
                    left_pred = 1

                if(anchor_pose[1] < other_pose[1]):
                    front_pred = 1
                    behind_pred = 0
                else:
                    front_pred = 0
                    behind_pred = 1
                #action.append(horizon_pred)
                action.append(right_pred)
                action.append(left_pred)
                #action.append(vertical_pred)
                action.append(front_pred)
                action.append(behind_pred)
            elif self.pushing:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                  
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    #action.append(1)
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                  
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                # if(np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < (data_dict['att_objects']['block1']['x_extent'] - data_dict['att_objects']['block2']['x_extent'])/2) and (np.abs(data_dict['objects']['block2']['position'][0][1] - data_dict['objects']['block1']['position'][0][1]) < (data_dict['att_objects']['block1']['y_extent'] - data_dict['att_objects']['block2']['y_extent'])/2):
                #     action.append(1) # form the stack with support relationships yixuan notice need to deal with that carefully when leveraging the stacking relationships
                # else:
                #     action.append(0) # this is actually complex contact
            # yixuan notice, here we assume all the objects with the same size
            # print(data_dict['other_pos'])
            # print(data_dict['anchor_pos'])
            # print(data_dict['att_objects']['block1'])
            if((other_pose[2] - anchor_pose[2]) > 0):
                current_extents = anchor_pose_max - anchor_pose_min
            else:
                current_extents = other_pose_max - other_pose_min
            if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
                if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                    #action.append(1)
                    if((other_pose[2] - anchor_pose[2]) > 0): # above
                        action.append(1)
                        action.append(0)
                    else:  # below
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                #action.append(0)
                action.append(0)
                action.append(0)
        return action
    
    def get_ground_truth_relations(self, obj_pose_1, obj_extent_1, obj_pose_2, obj_extent_2): # to start, assume no orientation
        action = []
        anchor_pose = obj_pose_1
        anchor_pose_max = []
        anchor_pose_min = []
        #print(len(obj_extent_1))
        # print('obj 1', [obj_pose_1, obj_extent_1])
        # print('obj 2', [obj_pose_2, obj_extent_2])
        for i in range(len(obj_extent_1)):
            anchor_pose_max.append(obj_pose_1[i] + obj_extent_1[i]/2)
            anchor_pose_min.append(obj_pose_1[i] - obj_extent_1[i]/2)

        other_pose = obj_pose_2
        other_pose_max = []
        other_pose_min = []
        for i in range(len(obj_extent_2)):
            other_pose_max.append(obj_pose_2[i] + obj_extent_2[i]/2)
            other_pose_min.append(obj_pose_2[i] - obj_extent_2[i]/2)

        # print('obj 1 min max', [anchor_pose_min, anchor_pose_max])
        # print('obj 2 min max', [other_pose_min, other_pose_max])
        if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]:
            if(anchor_pose[0] < other_pose[0]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)
        else:
            action.append(0)
            action.append(0)
        if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
            if(anchor_pose[1] < other_pose[1]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)                  
        else:
            action.append(0)
            action.append(0)
        if((other_pose[2] - anchor_pose[2]) > 0):
            current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
        else:
            current_extents = np.array(other_pose_max) - np.array(other_pose_min)
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                if((other_pose[2] - anchor_pose[2]) > 0): # above
                    action.append(1)
                    action.append(0)
                else:  # below
                    action.append(0)
                    action.append(1)
            else:
                action.append(0)
                action.append(0)
        else:
            action.append(0)
            action.append(0)
        #print(action)
        return action

    def get_ground_truth_tf_relations(self, anchor_pose, anchor_pose_max, anchor_pose_min, other_pose, other_pose_max, other_pose_min): # to start, assume no orientation
        action = []
        # anchor_pose = obj_pose_1
        # anchor_pose_max = []
        # anchor_pose_min = []
        # #print(len(obj_extent_1))
        # # print('obj 1', [obj_pose_1, obj_extent_1])
        # # print('obj 2', [obj_pose_2, obj_extent_2])
        # for i in range(len(obj_extent_1)):
        #     anchor_pose_max.append(obj_pose_1[i] + obj_extent_1[i]/2)
        #     anchor_pose_min.append(obj_pose_1[i] - obj_extent_1[i]/2)

        # other_pose = obj_pose_2
        # other_pose_max = []
        # other_pose_min = []
        # for i in range(len(obj_extent_2)):
        #     other_pose_max.append(obj_pose_2[i] + obj_extent_2[i]/2)
        #     other_pose_min.append(obj_pose_2[i] - obj_extent_2[i]/2)

        # print('obj 1 min max', [anchor_pose_min, anchor_pose_max])
        # print('obj 2 min max', [other_pose_min, other_pose_max])
        if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]:
            if(anchor_pose[0] < other_pose[0]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)
        else:
            action.append(0)
            action.append(0)
        if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
            if(anchor_pose[1] < other_pose[1]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)                  
        else:
            action.append(0)
            action.append(0)
        if((other_pose[2] - anchor_pose[2]) > 0):
            current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
        else:
            current_extents = np.array(other_pose_max) - np.array(other_pose_min)
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                if((other_pose[2] - anchor_pose[2]) > 0): # above
                    action.append(1)
                    action.append(0)
                else:  # below
                    action.append(0)
                    action.append(1)
            else:
                action.append(0)
                action.append(0)
        else:
            action.append(0)
            action.append(0)
        #print(action)
        return action

    def get_ground_truth_tf_relations_contact(self, contact_arr, anchor_id, other_id, anchor_pose, anchor_pose_max, anchor_pose_min, other_pose, other_pose_max, other_pose_min): # to start, assume no orientation
        action = []
        
        if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]:
            if(anchor_pose[0] < other_pose[0]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)
        else:
            action.append(0)
            action.append(0)
        if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
            if(anchor_pose[1] < other_pose[1]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)                  
        else:
            action.append(0)
            action.append(0)
        if((other_pose[2] - anchor_pose[2]) > 0):
            current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
        else:
            current_extents = np.array(other_pose_max) - np.array(other_pose_min)
        
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04: # define above as all the above
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                if((other_pose[2] - anchor_pose[2]) > 0): # above
                    action.append(1)
                    action.append(0)
                else:  # below
                    action.append(0)
                    action.append(1)
            else:
                action.append(0)
                action.append(0)
        else:
            action.append(0)
            action.append(0)

        sudo_contact = 0
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                sudo_contact = 1
        
        if contact_arr[0][0] == -1: # simple trick to deal with unsaved contact relations
            action.append(sudo_contact)
        else:
            action.append(contact_arr[anchor_id][other_id])
        return action

    def get_ground_truth_tf_relations_contact_pc(self, anchor_pose, anchor_pose_max, anchor_pose_min, other_pose, other_pose_max, other_pose_min): # to start, assume no orientation
        action = []
        
        if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]:
            if(anchor_pose[0] < other_pose[0]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)
        else:
            action.append(0)
            action.append(0)
        if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
            if(anchor_pose[1] < other_pose[1]):
                action.append(1)
                action.append(0)
            else:
                action.append(0)
                action.append(1)                  
        else:
            action.append(0)
            action.append(0)
        if((other_pose[2] - anchor_pose[2]) > 0):
            current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
        else:
            current_extents = np.array(other_pose_max) - np.array(other_pose_min)
        
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04: # define above as all the above
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                if((other_pose[2] - anchor_pose[2]) > 0): # above
                    action.append(1)
                    action.append(0)
                else:  # below
                    action.append(0)
                    action.append(1)
            else:
                action.append(0)
                action.append(0)
        else:
            action.append(0)
            action.append(0)

        sudo_contact = 0
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                sudo_contact = 1
        
        action.append(sudo_contact) # simple trick to deal with unsaved contact relations

        return action
  
    
    def get_relations_sigmoid_7(self, anchor_point_cloud, other_point_cloud):
        action = []
        self.point_cloud_max = True
        if self.point_cloud_max:
            # anchor_pose_max = data_dict['anchor_pos_max']#self.get_point_cloud_max(data_dict['anchor_point_cloud'])
            # anchor_pose_min = data_dict['anchor_pos_min']#self.get_point_cloud_min(data_dict['anchor_point_cloud'])

            # other_pose_max = data_dict['other_pos_max']#self.get_point_cloud_max(data_dict['other_point_cloud'])
            # other_pose_min = data_dict['other_pos_min']#self.get_point_cloud_min(data_dict['other_point_cloud'])
            
            # print([anchor_pose_max, anchor_pose_min])
            # print([other_pose_max, other_pose_min])
            # print('anchor_pos', data_dict['anchor_pos'])
            # print('other_pos', data_dict['other_pos'])
            anchor_pose = self.get_point_cloud_center(anchor_point_cloud)
            anchor_pose_max = self.get_point_cloud_max(anchor_point_cloud)
            anchor_pose_min = self.get_point_cloud_min(anchor_point_cloud)

            other_pose = self.get_point_cloud_center(other_point_cloud)
            other_pose_max = self.get_point_cloud_max(other_point_cloud)
            other_pose_min = self.get_point_cloud_min(other_point_cloud)

            self.only_left = False

            if self.pick_place:
                if(anchor_pose[0] < other_pose[0]):
                    right_pred = 1
                    left_pred = 0
                else:
                    right_pred = 0
                    left_pred = 1

                if(anchor_pose[1] < other_pose[1]):
                    front_pred = 1
                    behind_pred = 0
                else:
                    front_pred = 0
                    behind_pred = 1
                #action.append(horizon_pred)
                action.append(right_pred)
                action.append(left_pred)
                #action.append(vertical_pred)
                action.append(front_pred)
                action.append(behind_pred)
            elif self.pushing:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                  
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                if anchor_pose_max[0] < other_pose_min[0] or other_pose_max[0] < anchor_pose_min[0]: #np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < data_dict['other_extent'][0]/2 + data_dict['anchor_extent'][0]/2:
                    #action.append(1)
                    if(anchor_pose[0] < other_pose[0]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                if anchor_pose_max[1] < other_pose_min[1] or other_pose_max[1] < anchor_pose_min[1]:
                    #action.append(1)
                    if(anchor_pose[1] < other_pose[1]):
                        action.append(1)
                        action.append(0)
                    else:
                        action.append(0)
                        action.append(1)                  
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
                # if(np.abs(data_dict['other_pos'][0] - data_dict['anchor_pos'][0]) < (data_dict['att_objects']['block1']['x_extent'] - data_dict['att_objects']['block2']['x_extent'])/2) and (np.abs(data_dict['objects']['block2']['position'][0][1] - data_dict['objects']['block1']['position'][0][1]) < (data_dict['att_objects']['block1']['y_extent'] - data_dict['att_objects']['block2']['y_extent'])/2):
                #     action.append(1) # form the stack with support relationships yixuan notice need to deal with that carefully when leveraging the stacking relationships
                # else:
                #     action.append(0) # this is actually complex contact
            # yixuan notice, here we assume all the objects with the same size
            # print(data_dict['other_pos'])
            # print(data_dict['anchor_pos'])
            # print(data_dict['att_objects']['block1'])
            if((other_pose[2] - anchor_pose[2]) > 0):
                current_extents = anchor_pose_max - anchor_pose_min
            else:
                current_extents = other_pose_max - other_pose_min
            if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
                if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                    #action.append(1)
                    if((other_pose[2] - anchor_pose[2]) > 0): # above
                        action.append(1)
                        action.append(0)
                    else:  # below
                        action.append(0)
                        action.append(1)
                else:
                    #action.append(0)
                    action.append(0)
                    action.append(0)
            else:
                #action.append(0)
                action.append(0)
                action.append(0)
        return action
    
    
    def get_point_cloud_center(self, v):
        # print(np.min(v[:, :], axis = 0))
        # print(np.max(v[:, :], axis = 0))

        A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        # print(A_1)
        # time.sleep(10)
        return np.array(A_1)
    
    def get_point_cloud_max(self, v):
        A = (np.max(v[:, :], axis = 0)) #np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return np.array(A_1)

    def get_point_cloud_min(self, v):
        A = (np.min(v[:, :], axis = 0)) #np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]
        return np.array(A_1)
    
    
    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    @property
    def number_of_objects(self):
        return len(self.all_point_cloud)
    
    @property
    def number_of_object_pairs(self):
        return len(self.obj_pair_list)
    
    @property
    def stable_object_ids(self):
        assert self.scene_type == 'box_stacking' or self.scene_type == 'box_stacking_node_label', \
            f"Invalid func for scene {self.scene_type}"
        return self.precond_obj_id_list
    
    def get_object_center_list_2(self):
        # return self.robot_all_pair_voxels.get_object_center_list()
        center_list = []
        for k in sorted(self.scene_pos_info.keys()):
            assert self.scene_pos_info[k]['orient'] in (0, 1)
            orient_list = [1, 0] if self.scene_pos_info[k]['orient'] == 0 else [0, 1]
            center_list.append(self.scene_pos_info[k]['pos'] + orient_list)
        return center_list

    def get_object_center_list(self):
        obj_info_list = []
        for k in sorted(self.scene_pos_info.keys()):
            orient = self.scene_pos_info[k]['orient']
            assert orient in (0, 1)
            pos = self.scene_pos_info[k]['pos']
            if orient == 0:
                size_by_axes = [0.048, 0.15, 0.03]
            elif orient == 1:
                size_by_axes = [0.15, 0.048, 0.03]
            bound_1 = [pos[0] - size_by_axes[0]/2.0,
                       pos[1] - size_by_axes[1]/2.0,
                       pos[2]]
            bound_2 = [pos[0] + size_by_axes[0]/2.0,
                       pos[1] + size_by_axes[1]/2.0,
                       pos[2] + size_by_axes[2]]
            
            orient_info = [1, 0] if orient == 0 else [0, 1]
            obj_info_list.append(
                pos + orient_info + bound_1 + bound_2
            )
        return obj_info_list
    
    def get_object_id_to_remove(self):
        return self.box_stacking_remove_obj_id

    def get_stable_object_label_tensor(self):
        label = torch.zeros((self.number_of_objects)).long()
        for i in self.stable_object_ids:
            label[i] = 1
        return label
    
    def create_position_grid(self):
        for v in self.obj_voxels_by_obj_pair_dict.values():
            return v.create_position_grid()
    
    def get_pair_status_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_voxels_status_by_obj_pair_dict[obj_pair_key]
    
    # ==== Functions that return object point cloud paths ====
    
    def get_object_pair_path_at_index(self, obj_pair_index):
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        return self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
    
    def get_all_object_pair_path(self):
        path_list = []
        for obj_pair_key in self.obj_pair_list:
            path = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            path_list.append(path)
        return path_list

    # ==== Functions that return voxel objects ====
    
    def get_object_pair_voxels_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj.parse()[1]
    
    def get_all_object_pair_voxels_0(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        voxel_list_single = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels = self.obj_voxels_by_obj_pair_dict[0][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[0][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[0][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[0]
            relation_list = self.all_relation_list[0]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        
        for obj_id in range(self.total_objects):
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels_single = self.obj_voxels_single[0][obj_id]
            if voxels_single.dtype == 'object':
                voxels_single = np.vstack(voxels_single[:, :]).astype(np.float32)
            voxel_list_single.append(torch.FloatTensor(voxels_single))
        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, self.all_pos_list, orient_list, self.action_1, relation_list, self.all_gt_pose_list, self.all_gt_orientation_list, self.all_gt_extents_list, self.all_gt_extents_range_list, voxel_list_single, self.select_obj_num_range, self.all_bounding_box[0], self.all_rotated_bounding_box[0]

    def get_all_object_pair_voxels_1(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        voxel_list_single = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels = self.obj_voxels_by_obj_pair_dict[1][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[1][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[1][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[1]
            relation_list = self.all_relation_list[1]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        for obj_id in range(self.total_objects):
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels_single = self.obj_voxels_single[1][obj_id]
            if voxels_single.dtype == 'object':
                voxels_single = np.vstack(voxels_single[:, :]).astype(np.float32)
            voxel_list_single.append(torch.FloatTensor(voxels_single))
        
        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, self.all_pos_list, orient_list, self.action_1, relation_list, self.all_gt_pose_list, self.all_gt_orientation_list, self.all_gt_extents_list, self.all_gt_extents_range_list, voxel_list_single, self.select_obj_num_range, self.all_bounding_box[1], self.all_rotated_bounding_box[1]

    def get_all_object_pair_voxels_2(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        voxel_list_single = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            # voxels = self.obj_voxels_by_obj_pair_dict[2][obj_pair_key]
            # voxel_list.append(torch.FloatTensor(voxels))
            # voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[2][obj_pair_key]
            # voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            # voxels_other = self.obj_voxels_by_obj_pair_dict_other[2][obj_pair_key]
            # voxel_list_other.append(torch.FloatTensor(voxels_other))
            voxels = self.obj_voxels_by_obj_pair_dict[2][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[2][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[2][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[2]
            relation_list = self.all_relation_list[2]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        for obj_id in range(self.total_objects):
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels_single = self.obj_voxels_single[2][obj_id]
            if voxels_single.dtype == 'object':
                voxels_single = np.vstack(voxels_single[:, :]).astype(np.float32)
            voxel_list_single.append(torch.FloatTensor(voxels_single))
        
        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, self.all_pos_list, orient_list, self.action_2, relation_list, self.all_gt_pose_list, self.all_gt_orientation_list, self.all_gt_extents_list, self.all_gt_extents_range_list, voxel_list_single, self.select_obj_num_range, self.all_bounding_box[2], self.all_rotated_bounding_box[2]
    
    def get_all_object_pair_voxels_3(self):
        voxel_list, are_obj_far_apart_list = [], []
        pos_list = []
        voxel_list_anchor = []
        voxel_list_other = []
        voxel_list_single = []
        for obj_pair_key in self.obj_pair_list:
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            # voxels = self.obj_voxels_by_obj_pair_dict[2][obj_pair_key]
            # voxel_list.append(torch.FloatTensor(voxels))
            # voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[2][obj_pair_key]
            # voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            # voxels_other = self.obj_voxels_by_obj_pair_dict_other[2][obj_pair_key]
            # voxel_list_other.append(torch.FloatTensor(voxels_other))
            voxels = self.obj_voxels_by_obj_pair_dict[3][obj_pair_key]
            if voxels.dtype == 'object':
                voxels = np.vstack(voxels[:, :]).astype(np.float32)
            voxel_list.append(torch.FloatTensor(voxels))
            voxels_anchor = self.obj_voxels_by_obj_pair_dict_anchor[3][obj_pair_key]
            if voxels_anchor.dtype == 'object':
                voxels_anchor = np.vstack(voxels_anchor[:, :]).astype(np.float32)
            voxel_list_anchor.append(torch.FloatTensor(voxels_anchor))
            voxels_other = self.obj_voxels_by_obj_pair_dict_other[3][obj_pair_key]
            if voxels_other.dtype == 'object':
                voxels_other = np.vstack(voxels_other[:, :]).astype(np.float32)
            voxel_list_other.append(torch.FloatTensor(voxels_other))
            are_obj_far_apart_list.append(0)
            (anchor_idx, other_idx) = obj_pair_key
            # anchor_pos = [np.mean(np.max(self.all_point_cloud[anchor_idx][:,0]), np.min(self.all_point_cloud[anchor_idx][:,0])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,1]), np.min(self.all_point_cloud[anchor_idx][:,1])), np.mean(np.max(self.all_point_cloud[anchor_idx][:,2]), np.min(self.all_point_cloud[anchor_idx][:,2]))] # can get from self.all_point_cloud[anchor_idx] if needed
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            #print(self.all_pos_list)
            pos_list = self.all_pos_list[3]
            relation_list = self.all_relation_list[3]
            orient_list = self.all_orient_list
            #total_point_cloud = np.concatenate((self.all_point_cloud[anchor_idx], self.all_point_cloud[other_idx]), axis = 0)
            # if anchor_pos not in pos_list:
            #     pos_list.append(anchor_pos)
            # if voxels is not None:
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     are_obj_far_apart_list.append(0)
            # else:
            #     assert voxel_obj.objects_are_far_apart
            #     voxels = voxel_obj.get_all_zero_voxels()
            #     voxel_list.append(torch.FloatTensor(voxels))
            #     obj_paths = self.obj_pcd_path_by_obj_pair_dict[obj_pair_key]
            #     # print(f"Objects are far apart {obj_paths[0]} {obj_paths[1]}")
            #     are_obj_far_apart_list.append(0)

        for obj_id in range(self.total_objects):
            #print(obj_pair_key)
            #voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            #_, voxels, anchor_pos, other_pos = voxel_obj.parse()
            voxels_single = self.obj_voxels_single[3][obj_id]
            if voxels_single.dtype == 'object':
                voxels_single = np.vstack(voxels_single[:, :]).astype(np.float32)
            voxel_list_single.append(torch.FloatTensor(voxels_single))
        
        return voxel_list, voxel_list_anchor, voxel_list_other,are_obj_far_apart_list, pos_list, self.all_pos_list, orient_list, self.action_1, relation_list, self.all_gt_pose_list, self.all_gt_orientation_list, self.all_gt_extents_range_list, voxel_list_single, self.select_obj_num_range
    
    def get_obj_num(self):
        return self.total_objects
    
    # ==== Functions that return voxel objects ====

    def get_object_pair_voxel_object_at_index(self, obj_pair_index) -> RobotVoxels:
        obj_pair_key = self.obj_pair_list[obj_pair_index]
        voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
        return voxel_obj
    
    def get_all_object_pair_voxel_object(self):
        voxel_obj_list = []
        for obj_pair_key in self.obj_pair_list:
            voxel_obj = self.obj_voxels_by_obj_pair_dict[obj_pair_key]
            voxel_obj_list.append(voxel_obj)
        return voxel_obj_list
    
    def get_object_pcd_paths(self):
        return []#self.voxel_cluster_path


class RobotBoxStackingAllPairSceneObject(object):
    def __init__(self, scene_path, scene_type, load_voxels_of_removed_obj=False):
        self.scene_path = scene_path
        self.scene_type = scene_type
        self.load_voxels_of_removed_obj = load_voxels_of_removed_obj

        files = os.listdir(os.path.join(scene_path, 'object_cloud'))
        self.num_objects = len([p for p in files if 'cloud_cluster' in p])

        # Read the json file
        info_json_path = os.path.join(scene_path, 'info.json')
        assert os.path.exists(info_json_path), f"JSON file does not exist: {info_json_path}"
        with open(info_json_path, 'r') as json_fp:
            self.scene_info = json.load(json_fp)
        
        assert self.scene_info.get('precond_obj_id') is not None
    
    def create_scenes_by_removing_voxels(self):
        precond_obj_ids = self.scene_info['precond_obj_id']
        scene_obj_list = []
        valid_objs = get_valid_objects_to_remove_for_box_stacking(
            self.scene_path, 
            self.scene_info)
        for obj_id in valid_objs:
            assert self.scene_info.get(str(obj_id)) is not None, f"Missing pos info for obj {obj_id}"
            label = 0 if obj_id in precond_obj_ids else 1
            scene_obj_list.append(RobotAllPairSceneObject(
                self.scene_path, 
                self.scene_type, 
                precond_label=label,
                box_stacking_remove_obj_id=obj_id,
                scene_pos_info=self.scene_info,
                load_voxels_of_removed_obj=self.load_voxels_of_removed_obj))
        
        if not self.load_voxels_of_removed_obj:
            scene_obj_list.append(self.create_scene_with_all_voxels())

        return scene_obj_list
    
    def create_scene_with_all_voxels(self):
        # The entire scene is always stable
        all_pair_scene_obj = RobotAllPairSceneObject(
            self.scene_path,
            self.scene_type,
            precond_label=1,
            box_stacking_remove_obj_id=-1,
            scene_pos_info=self.scene_info,
            load_voxels_of_removed_obj=False)
        return all_pair_scene_obj


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
