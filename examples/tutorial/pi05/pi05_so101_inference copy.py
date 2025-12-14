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
Run PI0.5 policy inference on SO101 robot.

This script demonstrates how to run PI0.5 (a vision-language-action model) on a single-arm
SO101 robot. It loads normalization statistics from a recorded dataset and uses them to
properly denormalize the model's actions.

Usage:
    python pi05_so101_inference.py

Configuration:
    - Edit the constants below to customize robot ports, camera indices, and task
    - The script expects a dataset with SO101 statistics at the specified repo_id
"""

import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
# The model has a pretrained_model/ subdirectory with the model files
MODEL_ID = "sapanostic/finetuned-pi05-11k-pick-place-pen"
PRETRAINED_MODEL_SUBDIR = "pretrained_model"
DEVICE = "cuda"  # or "cpu" if no GPU

# Dataset for normalization statistics (your recorded SO101 dataset)
DATASET_REPO_ID = "sapanostic/pen-placement-task"

# Robot configuration
FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "so101_follower_arm"

# Camera configuration (OpenCV indices)
CAMERA_WRIST_INDEX = 10
CAMERA_BASE_INDEX = 4
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Task configuration
TASK_DESCRIPTION = "Pick up pen from the table"
ROBOT_TYPE = "so101_follower"

# Episode configuration
NUM_EPISODES = 10
STEPS_PER_EPISODE = 200

# Visualization configuration
DISPLAY_DATA = True  # Set to True to stream data to Rerun viewer

# ============================================================================
# Main Script
# ============================================================================


def load_dataset_stats(dataset_repo_id: str) -> dict:
    """
    Load dataset statistics for SO101.

    PI0.5 outputs 32-dimensional actions (designed for bimanual robots),
    but we slice to SO101's 6 DOF before denormalization, so we only
    need the original 6-DOF statistics.

    Args:
        dataset_repo_id: HuggingFace dataset repository ID

    Returns:
        Dictionary of dataset statistics
    """
    print(f"Loading normalization statistics from: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    stats = dataset.meta.stats

    # Get original 6-DOF action statistics
    action_mean = np.array(stats["action"]["mean"])
    action_std = np.array(stats["action"]["std"])

    print(f"✓ Loaded stats for {len(stats)} features")
    print("  Action statistics (SO101 6-DOF):")
    print(f"    Mean: {action_mean.round(2)}")
    print(f"    Std:  {action_std.round(2)}")

    return stats


def setup_robot() -> SO101Follower:
    """
    Initialize and connect to the SO101 robot.

    Returns:
        Connected SO101Follower instance
    """
    print("\nInitializing robot...")

    # Configure cameras (names must match the model's training data)
    # Model expects: observation.images.wrist and observation.images.side
    camera_config = {
        "side": OpenCVCameraConfig(
            index_or_path=CAMERA_BASE_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
        ),
        "wrist": OpenCVCameraConfig(
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
    Load PI0.5 model and create pre/post processors with SO101 statistics.

    Args:
        dataset_stats: Dataset statistics for normalization
        device: Device to run the model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, preprocessor, postprocessor)
    """
    from huggingface_hub import snapshot_download
    from pathlib import Path
    
    print(f"\nLoading PI0.5 model from: {MODEL_ID}/{PRETRAINED_MODEL_SUBDIR}")
    
    # Download the full repo and get the local path
    repo_path = snapshot_download(repo_id=MODEL_ID)
    
    # Point to the pretrained_model subdirectory
    model_path = Path(repo_path) / PRETRAINED_MODEL_SUBDIR
    
    # Load model from local pretrained_model subdirectory
    device_obj = torch.device(device)
    model = PI05Policy.from_pretrained(str(model_path))

    # Update model config with device for processor
    model.config.device = str(device_obj)

    # Create processors with SO101 normalization statistics
    preprocess, postprocess = make_pi05_pre_post_processors(
        config=model.config,
        dataset_stats=dataset_stats,
    )

    print(f"✓ Model loaded on {device}")
    return model, preprocess, postprocess


def run_episode(
    robot: SO101Follower,
    model: PI05Policy,
    preprocess,
    postprocess,
    dataset_features: dict,
    device: torch.device,
    episode_num: int,
    max_steps: int,
    display_data: bool = False,
) -> None:
    """
    Run a single episode of policy inference.

    Args:
        robot: Connected robot instance
        model: PI0.5 policy model
        preprocess: Preprocessing pipeline
        postprocess: Postprocessing pipeline
        dataset_features: Dataset feature definitions
        device: PyTorch device
        episode_num: Current episode number (for logging)
        max_steps: Maximum steps per episode
        display_data: Whether to stream data to Rerun viewer
    """
    
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

        # Debug: Print raw model output
        if step % 10 == 0:
            print(f"Step {step}: Raw action shape: {action.shape}")
            print(f"  Raw action values: {action[0, :6]}")

        # Slice to SO101's 6-DOF (model outputs action_dim from config)
        action = action[:, :6]

        # Postprocess action (denormalize)
        action = postprocess(action)

        # Debug: Print denormalized action
        if step % 10 == 0:
            print(f"  Denormalized action: {action[0]}")

        # Convert to robot action format
        robot_action = make_robot_action(action, dataset_features)

        # Debug: Print robot action and current state
        if step % 10 == 0:
            # Get current joint positions
            current_pos = robot.get_observation()
            joint_names = list(robot_action.keys())
            
            targets = [f'{robot_action[k]:.2f}' for k in joint_names]
            currents = [
                f'{current_pos[k]:.2f}'
                for k in joint_names
                if k in current_pos
            ]
            print(f"  Target action: {targets}")
            print(f"  Current state: {currents}")
            
            # Calculate difference
            if all(k in current_pos for k in joint_names):
                diffs = [
                    abs(robot_action[k] - current_pos[k])
                    for k in joint_names
                ]
                diffs_str = [f'{d:.2f}' for d in diffs]
                print(f"  Differences:   {diffs_str}")
                print(f"  Max diff: {max(diffs):.2f} degrees")

        # Stream data to Rerun viewer
        if display_data:
            log_rerun_data(observation=obs, action=robot_action)

        # Send action to robot
        robot.send_action(robot_action)


def main():
    """Main execution function."""
    print("="*60)
    print("PI0.5 Inference on SO101 Robot")
    print("="*60)

    # Initialize Rerun viewer for visualization
    if DISPLAY_DATA:
        init_rerun(session_name="pi05_inference")
        print("✓ Rerun viewer initialized")

    # Load dataset statistics
    dataset_stats = load_dataset_stats(DATASET_REPO_ID)

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
                display_data=DISPLAY_DATA,
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
