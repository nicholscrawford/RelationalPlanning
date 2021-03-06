### Acknowledgement 
"""
The general code base including the model is edited from the code of CoRL 2020 Relational Learning for Skill Preconditions.
Thanks for Mohit Sharma for sharing this code base. 
Utility function for PointConv from : https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py by Wenxuan Wu from OSU.
We put the licenses about these two softwares in the liscense directory.
"""

### Environment setup 

export PYTHONPATH= YOUR DIRECTORY + /code/relational_precond

pip install -r requirements.txt

https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html


### Data Gathering

Current setup requires a 3 object task, with one action performed, like a push, or a pick, so that there's two time stamps. It seems to work best with a push task, where no objects fall from the table, so that they all remain visible. Part of my research will be redesigning to avoid this error. I've added a config where 3 boxes are stacked, and then one is pushed, and they all should remain on the table, and so visible. 

It seems like --log-only-success and --log_only_behavior_intervals are good to use, for saving only useful data. Otherwise these files are closer to a gigabyte of space each, which doesn't allow for large dataset collection very easily. 


### If you only want to learn the basic ideas or our RD-GNN model, refer to /code/relational_precond/model.py


### If you want to go to details of the details of training, dataloader, planning ,and other stuff, then follow the detailed training and test script to get the details. The TRAIN_DIR and TEST_DIR should contain the point cloud, which is the input for our model. The RESULT_DIR would be the place that you will put the trained model in and the PRETRAINED_MODEL is the model you will load in the test. 


## training.
python ./relational_precond/trainer/multi_object_precond/train_multi_object_precond_e2e.py --result_dir RESULT_DIR --train_dir TRAIN_DIR --test_dir TEST_DIR --train_type 'all_object_pairs_gnn_new' --use_backbone_for_emb 0 --z_dim 7 --batch_size 1 --num_epochs 50000 --save_freq_iters 1001 --log_freq_iters 5 --print_freq_iters 1 --test_freq_iters 50 --lr 0.0003 --emb_lr 0.0001 --save_full_3d 1 --weight_precond 1.0 --loss_type 'classif' --classif_num_classes 2 --cuda 1 --start_id 0 --max_size 2 --start_test_id 0 --test_max_size 2 --max_objects 8 --use_multiple_train_dataset False 

## test 
python ./relational_precond/trainer/multi_object_precond/train_multi_object_precond_e2e.py --result_dir RESULT_DIR --train_dir TRAIN_DIR --test_dir TEST_DIR --train_type 'all_object_pairs_gnn_new' --use_backbone_for_emb 0 --z_dim 7 --batch_size 1 --num_epochs 50000 --save_freq_iters 1001 --log_freq_iters 5 --print_freq_iters 1 --test_freq_iters 50 --lr 0.0003 --emb_lr 0.0001 --save_full_3d 1 --weight_precond 1.0 --loss_type 'classif' --classif_num_classes 2 --cuda 1 --start_id 0 --max_size 2 --start_test_id 0 --test_max_size 20 --max_objects 8 --use_multiple_train_dataset False --checkpoint_path PRETRAINED_MODEL --manual_relations True --using_sub_goal True --save_all_planning_info False 

## Plans

In integration, I'd like to be able to generate sequences of arbitrary actions, whether pick/place, push, or etc. So, action params have to generalize or separate models have to be created. A trained model should be able to generate a sequence of future actions that lead to reaching a goal state.

So, given a simulation state and desired relations (which will be provided in config) this should return a sequence of actions to reach that state. This will be rerean, and only the next planned action will be added to the actual plan.