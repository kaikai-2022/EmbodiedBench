import numpy as np
import torch
from collections import deque
from PIL import Image
import sys
sys.path.insert(0, "/ssd/qinmaokai/workspace/lerobot")

from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler
from lerobot.configs.types import PolicyFeature, FeatureType


class ACTPolicy(Policy):
    """
    ACT policy wrapper for VLABench.
    Implements action chunking with replan strategy for zero-shot testing.
    """

    def __init__(
        self,
        pretrained_policy_path: str = None,
        camera_indices: list = [2, 3],
        replan_steps: int = 4,
        device: str = "cuda",
        **kwargs
    ):
        """
        Args:
            pretrained_policy_path: Optional path to pretrained ACT checkpoint
            camera_indices: Camera indices to use [front, wrist]
            replan_steps: Frequency of replanning (default: 4, like OpenPI)
            device: Device to run model on
        """
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.common.policies.act.modeling_act import ACTPolicy as LeRobotACTPolicy

        # Create ACT configuration with proper input/output features
        config = ACTConfig(
            # Input/output features
            # 当前目标分辨率: 256(匹配 checkpoint 训练)
            # 如需改回 480: 1) 改回下方硬编码 shape;2) 删除 process_observation 中的 resize 逻辑
            input_features={
                # 旧(480×480,如需恢复请改回):
                # "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 480)),
                # "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 480)),
                # 当前(256×256,匹配 checkpoint 训练分辨率):
                "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
                "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            },
            # Model parameters
            n_obs_steps=1,
            chunk_size=100,
            n_action_steps=100,
            dim_model=512,
            n_heads=8,
            n_encoder_layers=4,
            n_decoder_layers=1,
            use_vae=True,
            latent_dim=32,
            # Vision backbone
            vision_backbone="resnet18",
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        )

        # Initialize ACT policy
        if pretrained_policy_path and pretrained_policy_path.lower() != "none":
            print(f"Loading ACT policy from {pretrained_policy_path}")
            self.act_policy = LeRobotACTPolicy.from_pretrained(pretrained_policy_path)
        else:
            # Zero-shot mode: use default initialization
            print("Initializing ACT policy with random weights (zero-shot mode)")
            # Create dummy stats for zero-shot testing
            dummy_stats = {
                "observation.image": {
                    # 旧(480×480):
                    # "mean": torch.zeros(3, 480, 480),
                    # "std": torch.ones(3, 480, 480),
                    # 当前(256×256):
                    "mean": torch.zeros(3, 256, 256),
                    "std": torch.ones(3, 256, 256),
                },
                "observation.wrist_image": {
                    # 旧(480×480):
                    # "mean": torch.zeros(3, 480, 480),
                    # "std": torch.ones(3, 480, 480),
                    # 当前(256×256):
                    "mean": torch.zeros(3, 256, 256),
                    "std": torch.ones(3, 256, 256),
                },
                "observation.state": {
                    "mean": torch.zeros(7),
                    "std": torch.ones(7),
                },
                "action": {
                    "mean": torch.zeros(7),
                    "std": torch.ones(7),
                },
            }
            self.act_policy = LeRobotACTPolicy(config, dataset_stats=dummy_stats)

        self.act_policy.to(device)
        self.act_policy.eval()

        self.replan_steps = replan_steps
        self.action_queue = deque(maxlen=replan_steps)
        self.timestep = 0
        self.camera_indices = camera_indices
        self.device = device

        super().__init__(self.act_policy)

    def reset(self):
        """Clear action queue and timestep counter. Called by Evaluator at episode start."""
        self.action_queue.clear()
        self.timestep = 0
        self.act_policy.reset()

    def process_observation(self, obs):
        """
        Convert VLABench observation dict to ACT input batch.

        VLABench format:
            obs["rgb"]: list of 4 cameras [480, 480, 3]
            obs["ee_state"]: [pos3, quat4, gripper1] or [pos3, euler3, gripper1]
            obs["robot_frame"]: [0, -0.4, 0.78]

        ACT format:
            batch["observation.images"]: [n_cam, C, H, W] (torch tensor)
            batch["observation.state"]: [7] (torch tensor)
        Note: images are resized to 256x256 to match checkpoint training resolution.
        """
        # Extract images (front + wrist)
        front_img = obs["rgb"][self.camera_indices[0]]  # index 2
        wrist_img = obs["rgb"][self.camera_indices[1]]  # index 3

        # 当前(256×256): VLABench obs 是 480×480,checkpoint 训练时是 256×256,需 resize
        # 旧逻辑(480×480,如需恢复请删除以下 resize 块):
        # front_tensor = torch.from_numpy(front_img).float() / 255.0
        # front_tensor = front_tensor.permute(2, 0, 1)  # HWC -> CHW
        # wrist_tensor = torch.from_numpy(wrist_img).float() / 255.0
        # wrist_tensor = wrist_tensor.permute(2, 0, 1)
        # images = torch.stack([front_tensor, wrist_tensor], dim=0)  # [2, 3, 480, 480]
        TARGET_SIZE = 256  # 匹配 checkpoint 训练分辨率
        front_pil = Image.fromarray(front_img).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        front_tensor = torch.from_numpy(np.array(front_pil)).float() / 255.0
        front_tensor = front_tensor.permute(2, 0, 1)  # HWC -> CHW

        wrist_pil = Image.fromarray(wrist_img).resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)
        wrist_tensor = torch.from_numpy(np.array(wrist_pil)).float() / 255.0
        wrist_tensor = wrist_tensor.permute(2, 0, 1)

        # Stack images: [2, 3, 256, 256]
        images = torch.stack([front_tensor, wrist_tensor], dim=0)

        # Transform ee_state to robot-relative frame
        ee_state = obs["ee_state"]
        robot_frame = obs.get("robot_frame", np.array([0, -0.4, 0.78]))

        # Handle quaternion vs euler
        if len(ee_state) == 8:  # [pos3, quat4, gripper1]
            pos = ee_state[:3] - robot_frame
            quat = ee_state[3:7]
            euler = quaternion_to_euler(quat)
            gripper = ee_state[7]
        elif len(ee_state) == 7:  # [pos3, euler3, gripper1]
            pos = ee_state[:3] - robot_frame
            euler = ee_state[3:6]
            gripper = ee_state[6]
        else:
            raise ValueError(f"Unexpected ee_state shape: {ee_state.shape}")

        state = np.concatenate([pos, euler, [gripper]])  # 7D

        # Create batch dict
        batch = {
            "observation.image": front_tensor.unsqueeze(0),  # [1, 3, 256, 256]
            "observation.wrist_image": wrist_tensor.unsqueeze(0),  # [1, 3, 256, 256]
            "observation.state": torch.from_numpy(state).float().unsqueeze(0),  # [1, 7]
        }

        return batch

    def predict(self, obs, unnorm_key=None, **kwargs):
        """
        Predict action with chunk management strategy.

        Strategy:
        - Every replan_steps (4), query ACT model for new action chunk
        - Pop one action from queue each step
        - Apply coordinate transformation (add robot_frame)
        - Convert gripper signal to VLABench 2D format
        """
        # Reset check
        if self.timestep == 0:
            self.reset()

        # Replan when queue empty or at replan interval
        if len(self.action_queue) == 0 or self.timestep % self.replan_steps == 0:
            batch = self.process_observation(obs)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Get action chunk from ACT policy
            with torch.no_grad():
                action_chunk = self.act_policy.select_action(batch)  # [chunk_size, 7]

                # For zero-shot: actions are unnormalized predictions
                # Fill queue with first replan_steps actions
                for i in range(min(self.replan_steps, action_chunk.shape[0])):
                    self.action_queue.append(action_chunk[i].cpu().numpy())

        # Pop action from queue
        raw_action = self.action_queue.popleft()

        # Coordinate transformation: add robot_frame offset
        robot_frame = obs.get("robot_frame", np.array([0, -0.4, 0.78]))
        target_pos = raw_action[:3] + robot_frame
        target_euler = raw_action[3:6]

        # Gripper conversion: 1D signal -> 2D gripper state
        # Training data uses [0, 0.04] range (0=closed, 0.04=open)
        gripper_signal = raw_action[6]
        gripper_state = np.ones(2) * 0.04 if gripper_signal >= 0.02 else np.zeros(2)

        self.timestep += 1

        return target_pos, target_euler, gripper_state

    @property
    def name(self):
        return "ACT"
