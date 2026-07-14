from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import json
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def get_all_hdf5_files(directory):
    """
    Get all HDF5 files in a directory and its subdirectories.
    """
    hdf5_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                hdf5_files.append(os.path.join(root, file))
    return hdf5_files

def create_lerobot_dataset_from_hdf5(args):
    dataset = LeRobotDataset.create(
        repo_id=args.dataset_name,
        robot_type="franka",
        fps=10,
        features={
            "observation.image":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.second_image":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.wrist_image":{
                "dtype": "image",
                "shape": (480, 480, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.state":{
                "dtype": "float",
                "shape": (7,),
                "names": ["state"]
            },
            "actions":{
                "dtype": "float",
                "shape": (7,),
                "names": ["actions"]
            },
        },
        image_writer_processes=12,
        image_writer_threads=4
    )
    
    if args.task_list is None:
        tasks = os.listdir(args.dataset_path)
    else:
        tasks = args.task_list
    print("Task to process:", tasks)
    h5py_files = list()
    for task in tasks:
        h5py_files.extend(get_all_hdf5_files(os.path.join(args.dataset_path, task))[:args.max_files])
    print("File numbers:", len(h5py_files))
    for file in h5py_files:
        with h5py.File(file, "r") as f:
            for timestamp in f["data"].keys():
                # Skip flag
                skip_episode = False
                # load episode config
                episode_config_bytes = np.asarray(f["data"][timestamp]["meta_info"]["episode_config"]).astype('S')
                episode_config = episode_config_bytes.item().decode('utf-8')
                episode_config = json.loads(episode_config)
                if episode_config.get("robot") is not None:
                    robot_frame_pos = np.array(episode_config["robot"]["position"])
                else:
                    robot_frame_pos = np.array([0, -0.4, 0.78])
                images = f["data"][timestamp]["observation"]["rgb"][()]
                ee_state = f["data"][timestamp]["observation"]["ee_state"][()]
                q_state = f["data"][timestamp]["observation"]["q_state"][()]
                actions = f["data"][timestamp]["trajectory"]
                ee_pos, ee_quat, gripper = ee_state[:, :3], ee_state[:, 3:7], ee_state[:, 7]
                ee_euler = np.array([quat2euler(q) for q in ee_quat])
                # transform ee_state to robot frame
                ee_pos -= robot_frame_pos
                ee_state = np.concatenate([ee_pos, ee_euler, gripper.reshape(-1, 1)], axis=1)
                assert images.shape[0] == ee_state.shape[0] == q_state.shape[0] == actions.shape[0]
                task_str = np.array(f["data"][timestamp]["instruction"])[0].decode("utf-8")
                for i in range(images.shape[0]):
                    action = actions[i]
                    # Handle both 7D and 8D action formats
                    # 8D: [pos3, euler3, gripper1, gripper2] (VLABench format)
                    # 7D: [pos3, euler3, gripper1] (LeRobot format)
                    if len(action) == 8:
                        # Take first gripper value (they should be the same)
                        action = np.concatenate([action[:6], np.array([action[6]])])
                    # Convert gripper to binary (open/close)
                    if action[-1] > 0.03:
                        action = np.concatenate([action[:6], np.array([1])])
                    else:
                        action = np.concatenate([action[:6], np.array([0])])
                    dataset.add_frame(
                        {
                            "observation.image": images[i][2], # front camera (base_0_rgb)
                            # NOTE: VLABench eval has 3 cameras (base + left_wrist + right_wrist),
                            # but HDF5 trajectories only record 4 cameras (rgb[0..3]). We use the
                            # 4th as wrist, and mirror it to second_image so pi0.5 sees 3 views.
                            "observation.second_image": images[i][3], # left_wrist_0_rgb (same as wrist here)
                            "observation.wrist_image": images[i][3], # right_wrist_0_rgb
                            "observation.state": ee_state[i],
                            "actions": action,
                            "task": task_str,
                        }
                    )
                dataset.save_episode()
    # LeRobot 0.1.0 不提供 consolidate()：norm_stats.json 由 openpi 的
    # scripts/compute_norm_stats.py 单独生成，输出到
    # $assets_base_dir/$config_name/$repo_id/norm_stats.json。
    # 这里只 stop image writer，保证所有 mp4 flush 到磁盘。
    dataset.stop_image_writer()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LeRobot dataset")
    parser.add_argument("--dataset-name", type=str, default="test", help="Name of the dataset")
    parser.add_argument("--dataset-path", type=str, default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench/select_billiards", help="Path to the dataset")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum number of files to process")
    parser.add_argument("--task-list", type=str, nargs="+", default=None, help="List of tasks to process")
    args = parser.parse_args()

    create_lerobot_dataset_from_hdf5(args)