import sys
sys.path.insert(0, "/ssd/qinmaokai/workspace/lerobot")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import json
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from PIL import Image

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
    target_size = (args.resolution, args.resolution)
    dataset = LeRobotDataset.create(
        repo_id=args.dataset_name,
        robot_type="franka",
        fps=10,
        features={
            "observation.image":{
                "dtype": "image",
                "shape": (args.resolution, args.resolution, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.wrist_image":{
                "dtype": "image",
                "shape": (args.resolution, args.resolution, 3),
                "names": ["height", "width", "channels"]
            },
            "observation.state":{
                "dtype": "float",
                "shape": (7,),
                "names": ["state"]
            },
            "action":{
                "dtype": "float",
                "shape": (7,),
                "names": ["action"]
            },
        },
        image_writer_processes=5,
        image_writer_threads=10
    )

    def _maybe_resize(img):
        """Resize a numpy uint8 image to target_size if needed."""
        if isinstance(img, np.ndarray) and img.shape[:2] != target_size:
            pil = Image.fromarray(img)
            pil = pil.resize(target_size, Image.BILINEAR)
            return np.array(pil, dtype=np.uint8)
        return img
    
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
                assert images.shape[0] == ee_state.shape[0] == q_state.shape[0]
                # Handle shape mismatch: actions may have 1 fewer frame than others
                min_len = min(images.shape[0], actions.shape[0])
                images = images[:min_len]
                ee_state = ee_state[:min_len]
                q_state = q_state[:min_len]
                actions = actions[:min_len]
                for i in range(images.shape[0]):
                    action = actions[i]
                    if actions[i][-1] > 0.03:
                        action = np.concatenate([action[:6], np.array([1])])
                    else:
                        action = np.concatenate([action[:6], np.array([0])])
                    dataset.add_frame(
                        {
                            "observation.image": _maybe_resize(images[i][2]), # front camera
                            "observation.wrist_image": _maybe_resize(images[i][3]), # wrist camera
                            "observation.state": ee_state[i],
                            "action": action
                        }
                    )
                dataset.save_episode(task=np.array(f["data"][timestamp]["instruction"])[0].decode("utf-8"))
    dataset.consolidate(run_compute_stats=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a LeRobot dataset")
    parser.add_argument("--dataset-name", type=str, default="test", help="Name of the dataset")
    parser.add_argument("--dataset-path", type=str, default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench/select_billiards", help="Path to the dataset")
    parser.add_argument("--max-files", type=int, default=500, help="Maximum number of files to process")
    parser.add_argument("--task-list", type=str, nargs="+", default=None, help="List of tasks to process")
    parser.add_argument("--resolution", type=int, default=480, help="目标图像分辨率（H×W），默认 480。设为 256 可大幅加快训练（约 3.5x）")
    args = parser.parse_args()

    create_lerobot_dataset_from_hdf5(args)