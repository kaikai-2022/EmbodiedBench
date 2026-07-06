import numpy as np
import torch
from collections import deque
import sys
import os
import cv2

# Add LabUtopia to path for DP model imports
sys.path.insert(0, "/ssd/mkqin/workspace/LabUtopia")

from VLABench.evaluation.model.policy.base import Policy
from VLABench.utils.utils import quaternion_to_euler
from policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from policy.model.common.normalizer import LinearNormalizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import hydra
from omegaconf import OmegaConf


class DPPolicy(Policy):
    """
    Diffusion Policy wrapper for VLABench.
    Implements diffusion-based action prediction with replay strategy.
    """

    def __init__(
        self,
        pretrained_policy_path: str = None,
        policy_config_path: str = None,
        normalizer_path: str = None,
        camera_indices: list = [2, 3],
        replan_steps: int = 4,
        device: str = "cuda",
        **kwargs
    ):
        """
        Args:
            pretrained_policy_path: Path to DP model checkpoint (.ckpt or directory)
            policy_config_path: Path to policy config YAML (for architecture)
            normalizer_path: Optional path to normalization statistics
            camera_indices: Camera indices to use [front, wrist]
            replan_steps: Frequency of replanning (default: 4)
            device: Device to run model on
        """
        self.device = device
        self.camera_indices = camera_indices
        self.replan_steps = replan_steps
        self.action_queue = deque(maxlen=replan_steps)
        self.timestep = 0

        # Load or create policy
        self.is_dummy = False
        if pretrained_policy_path and pretrained_policy_path.lower() != "none":
            self.dp_policy = self._load_pretrained_policy(
                pretrained_policy_path,
                policy_config_path,
                normalizer_path
            )
        else:
            # Zero-shot mode: use random policy
            print("⚠ Using zero-shot mode (random actions)")
            self.dp_policy = None
            self.is_dummy = True

        super().__init__(self.dp_policy)

    def _load_pretrained_policy(self, ckpt_path, config_path, normalizer_path):
        """Load pretrained DP model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Load configuration
        if config_path is None:
            # Try to infer from checkpoint directory
            ckpt_dir = os.path.dirname(ckpt_path)
            possible_config_paths = [
                os.path.join(ckpt_dir, ".hydra", "config.yaml"),
                os.path.join(ckpt_dir, "config.yaml"),
                os.path.join(ckpt_dir, "..", "config.yaml"),
            ]
            for path in possible_config_paths:
                if os.path.exists(path):
                    config_path = path
                    break

        if config_path is None or not os.path.exists(config_path):
            print(f"⚠ Warning: Config file not found at {config_path}, using default config")
            # Create minimal config for 7D action space
            config = self._get_default_config()
        else:
            config = OmegaConf.load(config_path)
            policy_config = config.policy

        # Create policy
        policy = hydra.utils.instantiate(policy_config)

        # Load state dict (handle 'model.' prefix and EMA keys)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            elif "ema" not in k:
                new_state_dict[k] = v

        policy.load_state_dict(new_state_dict)
        policy.eval()
        policy.to(self.device)

        # Load or create normalizer
        if normalizer_path and os.path.exists(normalizer_path):
            normalizer = LinearNormalizer()
            normalizer.load_state_dict(torch.load(normalizer_path, map_location=self.device))
            normalizer.to(self.device)
            policy.set_normalizer(normalizer)
        else:
            # Create dummy normalizer
            print("⚠ Warning: Normalizer not found, using dummy stats")
            normalizer = LinearNormalizer()
            policy.set_normalizer(normalizer)

        print(f"✓ Loaded DP policy from {ckpt_path}")
        return policy

    def _get_default_config(self):
        """Create default configuration for 7D action space."""
        shape_meta = {
            'action': {'shape': [7]},
            'obs': {
                'camera_1_rgb': {'shape': [3, 256, 256], 'type': 'rgb'},
                'camera_2_rgb': {'shape': [3, 256, 256], 'type': 'rgb'},
                'agent_pose': {'shape': [7], 'type': 'low_dim'}
            }
        }

        return OmegaConf.create({
            'policy': {
                '_target_': 'policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy',
                'shape_meta': shape_meta,
                'noise_scheduler': {
                    '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                    'num_train_timesteps': 100,
                    'beta_schedule': 'squaredcos_cap_v2'
                },
                'obs_encoder': {
                    '_target_': 'policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder',
                    'shape_meta': shape_meta,
                    'rgb_model': {
                        '_target_': 'policy.model.vision.model_getter.get_resnet',
                        'name': 'resnet18',
                        'weights': None
                    },
                    'resize_shape': [256, 256],
                    'crop_shape': None,
                    'random_crop': False,
                    'use_group_norm': True,
                    'share_rgb_model': False,
                    'imagenet_norm': True
                },
                'horizon': 16,
                'n_action_steps': 10,
                'n_obs_steps': 3,
                'num_inference_steps': 100,
                'obs_as_global_cond': True
            }
        })

    def _create_dummy_policy(self):
        """Create untrained DP policy for zero-shot testing."""
        from policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
        from policy.model.vision.model_getter import get_resnet

        # Minimal shape meta for VLABench (7D action)
        shape_meta = {
            'action': {'shape': [7]},
            'obs': {
                'camera_1_rgb': {'shape': [3, 256, 256], 'type': 'rgb'},
                'camera_2_rgb': {'shape': [3, 256, 256], 'type': 'rgb'},
                'agent_pose': {'shape': [7], 'type': 'low_dim'}
            }
        }

        # Create RGB model
        rgb_model = get_resnet('resnet18', weights=None)

        # Create obs encoder
        obs_encoder = MultiImageObsEncoder(
            shape_meta=shape_meta,
            rgb_model=rgb_model,
            resize_shape=[256, 256],
            crop_shape=None,
            random_crop=False,
            use_group_norm=True,
            share_rgb_model=False,
            imagenet_norm=True
        )

        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2'
        )

        # Create policy with minimal config
        policy = DiffusionUnetImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            obs_encoder=obs_encoder,
            horizon=16,
            n_action_steps=10,
            n_obs_steps=3,
            num_inference_steps=100,
            obs_as_global_cond=True,
        )

        # Create dummy normalizer
        normalizer = LinearNormalizer()
        # Fit with dummy data for all input/output keys
        import torch
        dummy_data = {
            'observation.camera_1_rgb': torch.zeros(1,3,256,256),
            'observation.camera_2_rgb': torch.zeros(1,3,256,256),
            'observation.agent_pose': torch.zeros(1,7),
            'action': torch.zeros(1,7)
        }
        normalizer.fit(dummy_data)
        policy.set_normalizer(normalizer)

        policy.eval()
        policy.to(self.device)

        print("✓ Created dummy DP policy (zero-shot mode)")
        return policy

    def reset(self):
        """Clear action queue and timestep counter."""
        self.action_queue.clear()
        self.timestep = 0
        if hasattr(self.dp_policy, 'reset'):
            self.dp_policy.reset()

    def process_observation(self, obs):
        """
        Convert VLABench observation dict to DP input dict.

        VLABench format:
            obs["rgb"]: list of 4 cameras [480, 480, 3]
            obs["ee_state"]: [pos3, quat4, gripper1] or [pos3, euler3, gripper1]
            obs["robot_frame"]: [0, -0.4, 0.78]

        DP format:
            obs_dict["camera_1_rgb"]: [batch, time, C, H, W]
            obs_dict["camera_2_rgb"]: [batch, time, C, H, W]
            obs_dict["agent_pose"]: [batch, time, 7]

        Note: DP expects 256×256 images, will resize from 480×480.
        Note: DP expects n_obs_steps=3 history, will replicate current obs.
        """
        # Extract images
        front_img = obs["rgb"][self.camera_indices[0]]  # index 2, [480, 480, 3]
        wrist_img = obs["rgb"][self.camera_indices[1]]  # index 3, [480, 480, 3]

        # Resize from 480×480 to 256×256
        front_resized = cv2.resize(front_img, (256, 256))
        wrist_resized = cv2.resize(wrist_img, (256, 256))

        # Convert to torch tensors, normalize to [0, 1]
        front_tensor = torch.from_numpy(front_resized).float() / 255.0  # [256, 256, 3]
        front_tensor = front_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 256, 256]

        wrist_tensor = torch.from_numpy(wrist_resized).float() / 255.0
        wrist_tensor = wrist_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 256, 256]

        # Handle ee_state conversion
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

        # DP expects n_obs_steps=3, replicate current observation
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)  # [1, 1, 7]
        state_tensor = state_tensor.repeat(1, 3, 1)  # [1, 3, 7]

        front_tensor = front_tensor.repeat(1, 3, 1, 1, 1)  # [1, 3, 3, 256, 256]
        wrist_tensor = wrist_tensor.repeat(1, 3, 1, 1, 1)  # [1, 3, 3, 256, 256]

        # Create observation dict
        obs_dict = {
            "camera_1_rgb": front_tensor.to(self.device),
            "camera_2_rgb": wrist_tensor.to(self.device),
            "agent_pose": state_tensor.to(self.device)
        }

        return obs_dict

    def predict(self, obs, **kwargs):
        """
        Predict action with diffusion sampling and replay strategy.

        Strategy:
        - Zero-shot mode: return random actions (for testing)
        - Pretrained mode: run diffusion sampling every replan_steps (4)
        - Pop one action from queue each step
        - Apply coordinate transformation (add robot_frame)
        - Convert gripper signal to VLABench 2D format
        """
        # Zero-shot mode: return random actions
        if self.is_dummy:
            # Generate random action for testing
            delta_pos = np.random.uniform(-0.05, 0.05, 3)
            delta_euler = np.random.uniform(-0.1, 0.1, 3)
            gripper_open = np.random.uniform(0, 1, 1)

            current_ee_state = obs["ee_state"]
            if len(current_ee_state) == 8:
                pos, quat = current_ee_state[:3], current_ee_state[3:7]
                euler = quaternion_to_euler(quat)
            elif len(current_ee_state) == 7:
                pos, euler = current_ee_state[:3], current_ee_state[3:6]
            else:
                raise ValueError(f"Unexpected ee_state shape: {current_ee_state.shape}")

            target_pos = np.array(pos) + delta_pos
            target_euler = euler + delta_euler
            gripper_state = np.ones(2) * 0.04 if gripper_open >= 0.5 else np.zeros(2)

            return target_pos, target_euler, gripper_state

        # Pretrained mode: run diffusion sampling
        # Reset check
        if self.timestep == 0:
            self.reset()

        # Replan when queue empty or at replan interval
        if len(self.action_queue) == 0 or self.timestep % self.replan_steps == 0:
            obs_dict = self.process_observation(obs)

            # Run diffusion sampling
            with torch.no_grad():
                try:
                    prediction = self.dp_policy.predict_action(obs_dict)
                    action_chunk = prediction['action'][0].cpu().numpy()  # [horizon, 7]

                    # Fill queue with first n_action_steps
                    n_actions = min(self.replan_steps, action_chunk.shape[0])
                    for i in range(n_actions):
                        self.action_queue.append(action_chunk[i])
                except Exception as e:
                    print(f"❌ Error during DP prediction: {e}")
                    # Return zero action as fallback
                    return np.zeros(3), np.zeros(3), np.zeros(2)

        # Pop action from queue
        raw_action = self.action_queue.popleft()

        # Coordinate transformation
        robot_frame = obs.get("robot_frame", np.array([0, -0.4, 0.78]))
        target_pos = raw_action[:3] + robot_frame
        target_euler = raw_action[3:6]

        # Gripper conversion
        gripper_signal = raw_action[6]
        gripper_state = np.ones(2) * 0.04 if gripper_signal >= 0.5 else np.zeros(2)

        self.timestep += 1

        return target_pos, target_euler, gripper_state

    @property
    def name(self):
        return "DP"
