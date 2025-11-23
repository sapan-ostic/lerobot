#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run PI0 policy inference on SO101 robot.

This script demonstrates how to run PI0 (a vision-language-action model) on a single-arm
SO101 robot. It loads normalization statistics from a recorded dataset and uses them to
properly denormalize the model's actions.

Usage:
    python pi0_so101_inference.py

Configuration:
    - Edit the constants below to customize robot ports, camera indices, and task
    - The script expects a dataset with SO101 statistics at the specified repo_id
"""

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_ID = "lerobot/pi0_base"
DEVICE = "cuda"  # or "cpu" if no GPU

# Dataset for normalization statistics (your recorded SO101 dataset)
DATASET_REPO_ID = "sapanostic/pen-placement-home"

# Robot configuration
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower_arm"

# Camera configuration (OpenCV indices)
CAMERA_BASE_INDEX = 10
CAMERA_WRIST_INDEX = 12
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Task configuration
TASK_DESCRIPTION = "Pick up the pen on the table"
ROBOT_TYPE = "so101_follower"

# Episode configuration
NUM_EPISODES = 10
STEPS_PER_EPISODE = 200

# ============================================================================
# Main Script
# ============================================================================


def load_and_pad_dataset_stats(dataset_repo_id: str) -> dict:
    """
    Load dataset statistics and pad action dimensions for PI0.

    PI0 expects 32-dimensional actions (designed for bimanual robots),
    but SO101 has 6 DOF. We pad the statistics with neutral values
    (mean=0, std=1) for the unused dimensions.

    Args:
        dataset_repo_id: HuggingFace dataset repository ID

    Returns:
        Dictionary of padded dataset statistics
    """
    print(f"Loading normalization statistics from: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    stats = dataset.meta.stats

    # Get original 6-DOF action statistics
    original_mean = np.array(stats["action"]["mean"])
    original_std = np.array(stats["action"]["std"])

    # Pad to 32 dimensions (PI0's expected action space)
    padded_mean = np.pad(
        original_mean, (0, 32 - len(original_mean)), constant_values=0.0
    )
    padded_std = np.pad(
        original_std, (0, 32 - len(original_std)), constant_values=1.0
    )

    stats["action"]["mean"] = padded_mean
    stats["action"]["std"] = padded_std

    print(f"✓ Loaded stats for {len(stats)} features")
    print("  Action statistics (SO101 6-DOF):")
    print(f"    Mean: {padded_mean[:6].round(2)}")
    print(f"    Std:  {padded_std[:6].round(2)}")
    print("  Padded to 32 dimensions for PI0 (remaining dims: mean=0, std=1)")

    return stats


def setup_robot() -> SO101Follower:
    """
    Initialize and connect to the SO101 robot.

    Returns:
        Connected SO101Follower instance
    """
    print("\nInitializing robot...")

    # Configure cameras
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(
            index_or_path=CAMERA_BASE_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
        "left_wrist_0_rgb": OpenCVCameraConfig(
            index_or_path=CAMERA_WRIST_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
    }

    # Create robot configuration
    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT, id=FOLLOWER_ID, cameras=camera_config
    )

    # Initialize and connect robot
    robot = SO101Follower(robot_cfg)
    robot.connect()

    print(f"✓ Robot connected on {FOLLOWER_PORT}")
    return robot


def setup_policy(dataset_stats: dict, device: str):
    """
    Load PI0 model and create pre/post processors with SO101 statistics.

    Args:
        dataset_stats: Dataset statistics for normalization
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, preprocessor, postprocessor)
    """
    print(f"\nLoading PI0 model from: {MODEL_ID}")

    # Load model
    device_obj = torch.device(device)
    model = PI0Policy.from_pretrained(MODEL_ID)

    # Update model config with device for processor
    model.config.device = str(device_obj)

    # Create processors with SO101 normalization statistics
    preprocess, postprocess = make_pi0_pre_post_processors(
        config=model.config,
        dataset_stats=dataset_stats,
    )

    print(f"✓ Model loaded on {device}")
    return model, preprocess, postprocess


def run_episode(
    robot: SO101Follower,
    model: PI0Policy,
    preprocess,
    postprocess,
    dataset_features: dict,
    device: torch.device,
    episode_num: int,
    max_steps: int,
) -> None:
    """
    Run a single episode of policy inference.

    Args:
        robot: Connected robot instance
        model: PI0 policy model
        preprocess: Preprocessing pipeline
        postprocess: Postprocessing pipeline
        dataset_features: Dataset feature definitions
        device: PyTorch device
        episode_num: Current episode number (for logging)
        max_steps: Maximum steps per episode
    """
    print(f"\n{'='*60}")
    print(f"Episode {episode_num}/{NUM_EPISODES}")
    print(f"{'='*60}")

    for step in range(max_steps):
        # Get observation from robot
        obs = robot.get_observation()

        # Build inference frame with task and robot type
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_features,
            device=device,
            task=TASK_DESCRIPTION,
            robot_type=ROBOT_TYPE,
        )

        # Preprocess observation
        preprocessed_obs = preprocess(obs_frame)

        # Get action from model
        with torch.no_grad():
            action = model.select_action(preprocessed_obs)

        # Postprocess action (denormalize)
        action = postprocess(action)

        # Convert to robot action format
        robot_action = make_robot_action(action, dataset_features)

        # Send action to robot
        robot.send_action(robot_action)

        # Progress indicator
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step + 1}/{max_steps}")

    print(f"✓ Episode {episode_num} completed")


def main():
    """Main execution function."""
    print("="*60)
    print("PI0 Inference on SO101 Robot")
    print("="*60)

    # Load dataset statistics
    dataset_stats = load_and_pad_dataset_stats(DATASET_REPO_ID)

    # Setup robot
    robot = setup_robot()

    # Setup policy
    device = torch.device(DEVICE)
    model, preprocess, postprocess = setup_policy(dataset_stats, DEVICE)

    # Prepare dataset features for observation/action mapping
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(
        robot.observation_features, "observation"
    )
    dataset_features = {**action_features, **obs_features}

    print(f"\nTask: {TASK_DESCRIPTION}")
    print(
        f"Running {NUM_EPISODES} episodes with "
        f"{STEPS_PER_EPISODE} steps each"
    )

    try:
        # Run episodes
        for episode_num in range(1, NUM_EPISODES + 1):
            run_episode(
                robot=robot,
                model=model,
                preprocess=preprocess,
                postprocess=postprocess,
                dataset_features=dataset_features,
                device=device,
                episode_num=episode_num,
                max_steps=STEPS_PER_EPISODE,
            )

        print("\n" + "="*60)
        print(f"✓ All {NUM_EPISODES} episodes completed successfully!")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")

    except Exception as e:
        print(f"\n\n✗ Error occurred: {e}")
        raise

    finally:
        # Cleanup
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("✓ Done")


if __name__ == "__main__":
    main()
