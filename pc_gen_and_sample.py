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

        for i_o in range(total_objects):
            points[i_o] = np.array(points[i_o])
        return points #np.array(points1), np.array(points2), np.array(points3)

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