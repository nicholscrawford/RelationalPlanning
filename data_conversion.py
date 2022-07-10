from real_robot_dataloader import AllPairVoxelDataloaderPointCloudSavePoints
from real_robot_dataloader import AllPairVoxelDataloaderPointCloudFarthesetSampling
import pickle


#Missing robot_octree_data in relational_precond/dataloader/

def main():
    conversion_dirs = ["gym_data/push_blocks_data"]
    save = AllPairVoxelDataloaderPointCloudSavePoints(
        #What is config? It seems to not be used at all, besides where config.args is accessed, and then not used after.
        config="",
        train_dir_list=conversion_dirs,
        test_dir_list=conversion_dirs,
        pushing=True
    )

    sample = AllPairVoxelDataloaderPointCloudFarthesetSampling(
        config="",
        train_dir_list=conversion_dirs, 
        test_dir_list=conversion_dirs,
        pushing=True
    )

    """
    #Use to compare/observe data.
    with open("gym_data/demo_000001.pickle", 'rb') as pkl_f:
        dat = pickle.load(pkl_f)
        with open("/home/nichols/code/yixuan_code/dataset_model/3push_high_success_2/demo_000001.pickle", 'rb') as pkl_f2:
            yixuansdat = pickle.load(pkl_f2)
    """

    

if __name__ == "__main__":
    main()