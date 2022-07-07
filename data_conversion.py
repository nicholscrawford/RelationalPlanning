from real_robot_dataloader import AllPairVoxelDataloaderPointCloudSavePoints
#Missing robot_octree_data in relational_precond/dataloader/

def main():
    dataloader = AllPairVoxelDataloaderPointCloudSavePoints(
        #What is config? It seems to not be used at all, besides where config.args is accessed, and then not used after.
        config="",
        train_dir_list="gym_data",
        test_dir_list="gym_data"
    )

if __name__ == "__main__":
    main()