#!/usr/bin/env python3
"""
Convert VLABench HDF5 trajectories to LabUtopia DP format.

VLABench format: 4 cameras, ee_state (pos3+quat4+gripper1), action (7D)
LabUtopia format: 2 cameras, agent_pose (8D), action (8D)

Conversion needed:
1. Select 2 cameras (front, wrist) from 4 cameras
2. Pad action dimension 7D → 8D (add 0 at end)
3. Rename keys to match LabUtopia naming convention

Usage:
    # Convert single file
    python scripts/convert_to_dp_format.py --input /path/to/episode.h5 --output /path/to/output

    # Convert directory
    python scripts/convert_to_dp_format.py --input /path/to/hdf5_dir --output /path/to/output_dir
"""

import h5py
import numpy as np
import argparse
import os
from pathlib import Path
import json


def convert_vlabench_to_dp_format(vlabench_h5_path, output_dir):
    """
    Convert a single VLABench HDF5 episode to LabUtopia DP format.

    Args:
        vlabench_h5_path: Path to VLABench HDF5 file
        output_dir: Output directory for converted data
    """
    os.makedirs(output_dir, exist_ok=True)

    episodes_data = {}

    # Check if this is a single episode file or multi-episode file
    with h5py.File(vlabench_h5_path, 'r') as f:
        # Try to find episode data
        if 'data' in f:
            # Single episode format with 'data' group
            print(f"Converting single episode from {vlabench_h5_path}...")
            episode_data = f['data']

            # Extract cameras (index 2=front, 3=wrist from 4 cameras)
            if 'rgb' in episode_data:
                rgb_data = episode_data['rgb'][:]  # [T, 4, H, W, C]
                camera_1_rgb = rgb_data[:, 2, :, :, :]  # Front camera
                camera_2_rgb = rgb_data[:, 3, :, :, :]  # Wrist camera
            else:
                raise ValueError("RGB data not found in episode")

            # Extract state and action
            if 'ee_state' in episode_data:
                ee_state = episode_data['ee_state'][:]  # [T, 8] (pos3+quat4+gripper1)
            else:
                raise ValueError("ee_state not found in episode")

            if 'action' in episode_data:
                action = episode_data['action'][:]  # [T, 7] (pos3+euler3+gripper1)
            else:
                raise ValueError("action not found in episode")

            episode_name = os.path.splitext(os.path.basename(vlabench_h5_path))[0]

            # Convert ee_state to agent_pose format (8D)
            # VLABench: [pos3, quat4, gripper1] → 8D
            # LabUtopia: [pos3, quat4, gripper1] → 8D (same!)
            agent_pose = ee_state

            # Convert action: 7D → 8D (pad with 0 at end)
            action_padded = np.pad(action, ((0, 0), (0, 1)), mode='constant')

            episodes_data[episode_name] = {
                'camera_1_rgb': camera_1_rgb,
                'camera_2_rgb': camera_2_rgb,
                'agent_pose': agent_pose,
                'actions': action_padded
            }

        else:
            # Multi-episode format with episode groups
            print(f"Converting multi-episode file {vlabench_h5_path}...")
            for episode_name in f.keys():
                episode = f[episode_name]

                # Extract cameras (index 2=front, 3=wrist from 4 cameras)
                if 'rgb' in episode:
                    rgb_data = episode['rgb'][:]  # [T, 4, H, W, C]
                    camera_1_rgb = rgb_data[:, 2, :, :, :]  # Front camera
                    camera_2_rgb = rgb_data[:, 3, :, :, :]  # Wrist camera
                else:
                    print(f"⚠ Skipping episode {episode_name}: no RGB data")
                    continue

                # Extract state and action
                if 'ee_state' in episode:
                    ee_state = episode['ee_state'][:]  # [T, 8]
                else:
                    print(f"⚠ Skipping episode {episode_name}: no ee_state")
                    continue

                if 'action' in episode:
                    action = episode['action'][:]  # [T, 7]
                else:
                    print(f"⚠ Skipping episode {episode_name}: no action")
                    continue

                # Convert ee_state to agent_pose format (8D)
                agent_pose = ee_state

                # Convert action: 7D → 8D (pad with 0 at end)
                action_padded = np.pad(action, ((0, 0), (0, 1)), mode='constant')

                episodes_data[episode_name] = {
                    'camera_1_rgb': camera_1_rgb,
                    'camera_2_rgb': camera_2_rgb,
                    'agent_pose': agent_pose,
                    'actions': action_padded
                }

    # Write converted data
    output_path = os.path.join(output_dir, "episode_data.hdf5")
    with h5py.File(output_path, 'w') as f_out:
        for ep_name, ep_data in episodes_data.items():
            grp = f_out.create_group(ep_name)
            grp.create_dataset('camera_1_rgb', data=ep_data['camera_1_rgb'], compression='gzip')
            grp.create_dataset('camera_2_rgb', data=ep_data['camera_2_rgb'], compression='gzip')
            grp.create_dataset('agent_pose', data=ep_data['agent_pose'], compression='gzip')
            grp.create_dataset('actions', data=ep_data['actions'], compression='gzip')

    print(f"✓ Converted {len(episodes_data)} episodes to {output_path}")

    # Save metadata
    metadata = {
        'source_file': vlabench_h5_path,
        'num_episodes': len(episodes_data),
        'format': 'LabUtopia DP',
        'camera_layout': ['front (index 2)', 'wrist (index 3)'],
        'action_dim': 8,  # padded from 7D
        'state_dim': 8   # pos3 + quat4 + gripper1
    }

    metadata_path = os.path.join(output_dir, "conversion_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert VLABench HDF5 to LabUtopia DP format')
    parser.add_argument('--input', required=True, help='VLABench HDF5 file or directory')
    parser.add_argument('--output', required=True, help='Output directory for DP format')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        convert_vlabench_to_dp_format(args.input, args.output)
    elif os.path.isdir(args.input):
        # Process all HDF5 files in directory
        h5_files = list(Path(args.input).glob("*.h5")) + list(Path(args.input).glob("*.hdf5"))

        if not h5_files:
            print(f"❌ No HDF5 files found in {args.input}")
            return

        print(f"Found {len(h5_files)} HDF5 files to convert...")

        for i, h5_file in enumerate(h5_files):
            print(f"\n[{i+1}/{len(h5_files)}] Converting {h5_file.name}...")
            output_subdir = os.path.join(args.output, h5_file.stem)
            convert_vlabench_to_dp_format(str(h5_file), output_subdir)

        print(f"\n✓ Conversion complete! Processed {len(h5_files)} files.")
        print(f"Output saved to: {args.output}")
    else:
        print(f"❌ Error: {args.input} is not a valid file or directory")


if __name__ == "__main__":
    main()
