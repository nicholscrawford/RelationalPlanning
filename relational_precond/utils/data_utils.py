import numpy as np

import os
import pickle
import time
import h5py

def get_delta_pose_for_data_info(info):
    before_pose = info['before']['other_pos'] +  info['before']['other_q']
    after_pose = info['after']['other_pos'] + info['after']['other_q']
    delta_pose = [after_pose[i]-before_pose[i] for i in range(len(before_pose))]
    return delta_pose


def get_euclid_dist_matrix_for_data(data_arr, squared=True):
    '''Get euclidean distance matrix for data.'''
    data_dot_prod = np.dot(data_arr, data_arr.T)
    sq_norm = np.diagonal(data_dot_prod)
    dist = sq_norm[None, :] - 2*data_dot_prod + sq_norm[:, None]
    dist = np.maximum(dist, 1e-8)
    if not squared:
        dist = np.sqrt(dist)
    return dist


def get_dist_matrix_for_data(data_list, 
                             save_inter_scene_dist_dir=None,
                             save_inter_scene_dist_file_prefix='',
                             top_K=100, 
                             bottom_K=100):
    inter_scene_dist_by_path_dict = {'path': {}, 'idx': {}}
    
    get_dist_start_time = time.time()
    pos_list = []

    has_info = data_list[0].get('info') is not None
    for i in range(len(data_list)):
        data_i = data_list[i]
        if has_info:
            pos_list.append(get_delta_pose_for_data_info(data_i['info']))
        else:
            pos_list.append(data_i['delta_pose'])
    
    pos_arr = np.array(pos_list)[:, :3]
    for i in range(len(data_list)):
        data_i = data_list[i]
        if has_info:
            data_i_pos = np.array(
                get_delta_pose_for_data_info(data_i['info']))[:3]
        else:
            data_i_pos = np.array(data_i['delta_pose'])[:3]
        data_i_dist = np.linalg.norm(pos_arr - data_i_pos, axis=1)

        top_k_idxs = np.argpartition(data_i_dist, top_K + 1)[:top_K]
        bottom_k_idxs = np.argpartition(-data_i_dist, bottom_K)[:bottom_K]

        assert len(top_k_idxs) == top_K
        assert len(bottom_k_idxs) == bottom_K

        inter_scene_dist_by_path_dict['path'][data_i['path']] = {
            'top': [(data_i_dist[idx], idx, data_list[idx]['path']) 
                     for idx in top_k_idxs],
            'bottom': [(data_i_dist[idx], idx, data_list[idx]['path']) 
                        for idx in bottom_k_idxs],
        }
        inter_scene_dist_by_path_dict['idx'][i] = \
            inter_scene_dist_by_path_dict['path'][data_i['path']]

    get_dist_end_time = time.time()
    print("Get dist time: {:.4f}".format(get_dist_end_time - get_dist_start_time))
    
    # Save the file only if required.
    if save_inter_scene_dist_dir is not None:
        if save_inter_scene_dist_file_prefix is not None:
            if len(save_inter_scene_dist_file_prefix) > 0:
                file_name = '{}_inter_scene_dist.pkl'.format(
                    save_inter_scene_dist_file_prefix)
            else:
                file_name = 'inter_scene_dist.pkl'
        pkl_path = os.path.join(save_inter_scene_dist_dir, file_name)
        with open(pkl_path, 'wb') as pkl_f:
            pickle.dump(inter_scene_dist_by_path_dict, pkl_f, protocol=2)
            print("Did save inter_scene_dist: {}".format(pkl_path))
        
    return inter_scene_dist_by_path_dict


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def recursively_get_dict_from_group(group_or_data):
    d = {}
    if type(group_or_data) == h5py.Dataset:
        return np.array(group_or_data)

    # Else it's still a group
    for k in group_or_data.keys():
        v = recursively_get_dict_from_group(group_or_data[k])
        d[k] = v
    return d

def convert_list_of_array_to_dict_of_array_for_hdf5(arr_list):
    arr_dict = {}
    for i, a in enumerate(arr_list):
        arr_dict[str(i)] = a
    return arr_dict

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    Take an already open HDF5 file and insert the contents of a dictionary
    at the current path location. Can call itself recursively to fill
    out HDF5 files with the contents of a dictionary.
    """
    assert type(dic) is type({}), "must provide a dictionary"
    assert type(path) is type(''), "path must be a string"
    assert type(h5file) is h5py._hl.files.File, "must be an open h5py file"

    for key in dic:
        assert type(key) is type(''), 'dict keys must be strings to save to hdf5'
        did_save_key = False

        if type(dic[key]) in (np.int64, np.float64, type(''), int, float):
            h5file[path + key] = dic[key]
            did_save_key = True
            assert h5file[path + key].value == dic[key], \
                'The data representation in the HDF5 file does not match the ' \
                'original dict.'
        if type(dic[key]) is type([]):
            h5file[path + key] = np.array(dic[key])
            did_save_key = True
        if type(dic[key]) is np.ndarray:
            h5file[path + key] = dic[key]
            did_save_key = True
            # assert np.array_equal(h5file[path + key].value, dic[key]), \
            #     'The data representation in the HDF5 file does not match the ' \
            #     'original dict.'
        if type(dic[key]) is type({}):
            recursively_save_dict_contents_to_group(h5file,
                                                    path + key + '/',
                                                    dic[key])
            did_save_key = True
        if not did_save_key:
            print("Dropping key from h5 file: {}".format(path + key))