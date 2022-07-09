from real_robot_dataloader import AllPairVoxelDataloaderPointCloudSavePoints
from real_robot_dataloader import AllPairVoxelDataloaderPointCloudFarthesetSampling
import pickle


#Missing robot_octree_data in relational_precond/dataloader/

def main():
    save = AllPairVoxelDataloaderPointCloudSavePoints(
        #What is config? It seems to not be used at all, besides where config.args is accessed, and then not used after.
        config="",
        train_dir_list=["gym_data"],
        test_dir_list=["gym_data"]
    )

    sample = AllPairVoxelDataloaderPointCloudFarthesetSampling(
        config="",
        train_dir_list=["gym_data"], 
        test_dir_list=["gym_data"]
    )

    with open("gym_data/demo_000001.pickle", 'rb') as pkl_f:
        dat = pickle.load(pkl_f)

        #Paused program here, it looks like the original data has been modified correctly.

    

if __name__ == "__main__":
    main()