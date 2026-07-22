import numpy as np
import torch
from collections import deque
import sys

# Add LeRobot to path
sys.path.insert(0, "/ssd/mkqin/workspace/lerobot")

from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler
from lerobot.configs.types import PolicyFeature, FeatureType


class DPPolicy(Policy):
    """
    Diffusion Policy wrapper for VLABench using LeRobot's DiffusionPolicy.

    This implementation reuses the same LeRobot DiffusionPolicy that is trained
    via `lerobot/scripts/train.py --policy.type=diffusion`, ensuring consistency
    with the ACT training pipeline.

    Input/Output format is identical to ACT:
    - observation.image: Front camera [3, 480, 480]
    - observation.wrist_image: Wrist camera [3, 480, 480]
    - observation.state: 7D robot state [pos3, euler3, gripper1]
    - action: 7D action [pos3, euler3, gripper1]
    """

    def __init__(
        self,
        pretrained_policy_path: str = None,
        camera_indices: list = [2, 3],
        replan_steps: int = 4,
        device: str = "cuda",
        n_obs_steps: int = 2,
        horizon: int = 16,
        n_action_steps: int = 8,
        **kwargs
    ):
        """
        Args:
            pretrained_policy_path: Path to pretrained DP checkpoint directory
            camera_indices: Camera indices to use [front, wrist] = [2, 3]
            replan_steps: Frequency of replanning (default: 4)
            device: Device to run model on
            n_obs_steps: Number of observation history steps (LeRobot default: 2)
            horizon: Diffusion model action prediction horizon (LeRobot default: 16)
            n_action_steps: Number of actions to output per inference (LeRobot default: 8)
        """
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig

        self.device = device
        self.camera_indices = camera_indices
        self.replan_steps = replan_steps
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.n_action_steps = n_action_steps

        self.action_queue = deque(maxlen=replan_steps)
        self.timestep = 0

        if pretrained_policy_path and pretrained_policy_path.lower() != "none":
            print(f"Loading LeRobot DiffusionPolicy from {pretrained_policy_path}")
            self.dp_policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
        else:
            print("Initializing DiffusionPolicy with random weights (zero-shot mode)")
            self.dp_policy = self._create_dummy_policy()

        self.dp_policy.to(device)
        self.dp_policy.eval()

        super().__init__(self.dp_policy)

    def _create_dummy_policy(self):
        """Create untrained DP policy for zero-shot testing with dummy normalization."""
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig

        config = DiffusionConfig(
            n_obs_steps=self.n_obs_steps,
            horizon=self.horizon,
            n_action_steps=self.n_action_steps,
            input_features={
                "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 480)),
                "observation.wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 480)),
                "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            },
            output_features={
                "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            },
            # VLABench uses 480x480 images without cropping
            crop_shape=None,
            crop_is_random=False,
            # Vision backbone (no pretrained weights to allow group norm replacement)
            vision_backbone="resnet18",
            pretrained_backbone_weights=None,
            use_group_norm=True,
            # UNet architecture (LeRobot defaults)
            down_dims=(512, 1024, 2048),
            kernel_size=5,
            n_groups=8,
            diffusion_step_embed_dim=128,
            use_film_scale_modulation=True,
            # Noise scheduler
            noise_scheduler_type="DDPM",
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
            clip_sample_range=1.0,
        )

        # Dummy stats for zero-shot (LeRobot requires mean, std, min, max)
        dummy_stats = {
            "observation.image": {
                "mean": torch.zeros(3, 480, 480),
                "std": torch.ones(3, 480, 480),
                "min": torch.zeros(3, 480, 480),
                "max": torch.ones(3, 480, 480),
            },
            "observation.wrist_image": {
                "mean": torch.zeros(3, 480, 480),
                "std": torch.ones(3, 480, 480),
                "min": torch.zeros(3, 480, 480),
                "max": torch.ones(3, 480, 480),
            },
            "observation.state": {
                "mean": torch.zeros(7),
                "std": torch.ones(7),
                "min": -torch.ones(7) * 10,
                "max": torch.ones(7) * 10,
            },
            "action": {
                "mean": torch.zeros(7),
                "std": torch.ones(7),
                "min": -torch.ones(7),
                "max": torch.ones(7),
            },
        }

        return DiffusionPolicy(config, dataset_stats=dummy_stats)

    def reset(self):
        """Clear action queue and timestep counter. Called by Evaluator at episode start."""
        self.action_queue.clear()
        self.timestep = 0
        self.dp_policy.reset()

    def process_observation(self, obs):
        """
        Convert VLABench observation dict to LeRobot DiffusionPolicy input batch.

        VLABench format:
            obs["rgb"]: list of 4 cameras [480, 480, 3]
            obs["ee_state"]: [pos3, quat4, gripper1] or [pos3, euler3, gripper1]
            obs["robot_frame"]: [0, -0.4, 0.78]

        LeRobot format (identical to ACT):
            batch["observation.image"]: [n_cam, C, H, W] (torch tensor)
            batch["observation.wrist_image"]: [n_cam, C, H, W] (torch tensor)
            batch["observation.state"]: [7] (torch tensor)
        """
        # Extract images (front + wrist)
        front_img = obs["rgb"][self.camera_indices[0]]  # index 2
        wrist_img = obs["rgb"][self.camera_indices[1]]  # index 3

        # Convert to torch tensors, normalize to [0, 1]
        front_tensor = torch.from_numpy(front_img).float() / 255.0
        front_tensor = front_tensor.permute(2, 0, 1)  # HWC -> CHW

        wrist_tensor = torch.from_numpy(wrist_img).float() / 255.0
        wrist_tensor = wrist_tensor.permute(2, 0, 1)  # HWC -> CHW

        # Transform ee_state to robot-relative frame
        ee_state = obs["ee_state"]
        robot_frame = obs.get("robot_frame", np.array([0, -0.4, 0.78]))

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

        # Create batch dict (identical to ACT format)
        batch = {
            "observation.image": front_tensor.unsqueeze(0),  # [1, 3, 480, 480]
            "observation.wrist_image": wrist_tensor.unsqueeze(0),  # [1, 3, 480, 480]
            "observation.state": torch.from_numpy(state).float().unsqueeze(0),  # [1, 7]
        }

        return batch

    def predict(self, obs, **kwargs):
        """
        Predict action with diffusion sampling and replay strategy.

        Strategy (identical to ACT):
        - Every replan_steps (4), query DP model for new action chunk
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

            # Run diffusion sampling
            with torch.no_grad():
                action = self.dp_policy.select_action(batch)  # [1, 7] or [7]

            # Handle both shapes: squeeze if needed
            action_np = action.cpu().numpy()
            if action_np.ndim > 1:
                action_np = action_np.squeeze(0)  # [1, 7] -> [7]

            # Fill queue with replan_steps actions (same action repeated)
            for _ in range(self.replan_steps):
                self.action_queue.append(action_np.copy())

        # Pop action from queue
        raw_action = self.action_queue.popleft()

        # Coordinate transformation: add robot_frame offset
        robot_frame = obs.get("robot_frame", np.array([0, -0.4, 0.78]))
        target_pos = raw_action[:3] + robot_frame
        target_euler = raw_action[3:6]

        # Gripper conversion: 1D signal -> 2D gripper state
        gripper_signal = raw_action[6]
        gripper_state = np.ones(2) * 0.04 if gripper_signal >= 0.5 else np.zeros(2)

        self.timestep += 1

        return target_pos, target_euler, gripper_state

    @property
    def name(self):
        return "DP"
