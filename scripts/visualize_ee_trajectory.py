#!/usr/bin/env python3
"""
Visualize End-Effector Trajectory from LeRobot Dataset.

This script:
1. Loads a LeRobot dataset from HuggingFace
2. Extracts robot joint angles (observation.state)
3. Runs forward kinematics to compute EE poses
4. Creates a side-by-side GIF with front camera view and 3D trajectory
"""

from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

# LeRobot imports
from lerobot.model.so101_kinematics import SO101ForwardKinematics


def compute_ee_trajectory(
    joint_angles: np.ndarray,
    fk: SO101ForwardKinematics,
) -> np.ndarray:
    """Compute end-effector positions for all frames.
    
    Args:
        joint_angles: Array of shape (N, 6) with joint angles in degrees.
        fk: Forward kinematics solver.
    
    Returns:
        Array of shape (N, 3) with EE positions (x, y, z) in meters.
    """
    n_frames = len(joint_angles)
    ee_positions = np.zeros((n_frames, 3))
    
    for i in range(n_frames):
        # Use only the first 5 joints for FK (exclude gripper)
        pose = fk.compute(joint_angles[i, :5])
        ee_positions[i] = pose[:3, 3]
    
    return ee_positions


def create_trajectory_gif(
    front_images: list[np.ndarray],
    ee_positions: np.ndarray,
    output_path: str,
    fps: int = 15,
    episode_indices: np.ndarray | None = None,
):
    """Create a side-by-side GIF with front camera and 3D trajectory.
    
    Args:
        front_images: List of front camera images (H, W, C).
        ee_positions: Array of shape (N, 3) with EE positions.
        output_path: Path to save the GIF.
        fps: Frames per second.
        episode_indices: Episode index for each frame (for coloring).
    """
    n_frames = len(front_images)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(14, 6), facecolor='#1a1a2e')
    
    # Left: Front camera view
    ax_img = fig.add_subplot(121)
    ax_img.set_facecolor('#1a1a2e')
    ax_img.axis('off')
    ax_img.set_title('Front Camera', fontsize=14, color='#e94560', fontweight='bold', pad=10)
    
    # Right: 3D trajectory
    ax_3d = fig.add_subplot(122, projection='3d')
    ax_3d.set_facecolor('#16213e')
    
    # Set 3D plot style
    ax_3d.set_xlabel('X (m)', color='#e94560', fontsize=10)
    ax_3d.set_ylabel('Y (m)', color='#e94560', fontsize=10)
    ax_3d.set_zlabel('Z (m)', color='#e94560', fontsize=10)
    ax_3d.set_title('End-Effector Trajectory', fontsize=14, color='#e94560', fontweight='bold', pad=10)
    
    # Set viewing angle: 45 degrees between X and -Y, Z up
    # azim=-45 looks from the +X, +Y quadrant toward origin
    ax_3d.view_init(elev=25, azim=-45)
    
    # Style 3D axes
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('#0f3460')
    ax_3d.yaxis.pane.set_edgecolor('#0f3460')
    ax_3d.zaxis.pane.set_edgecolor('#0f3460')
    ax_3d.tick_params(colors='#e94560', labelsize=8)
    ax_3d.grid(True, alpha=0.3, color='#0f3460')
    
    # Compute axis limits with some padding
    x_min, x_max = ee_positions[:, 0].min(), ee_positions[:, 0].max()
    y_min, y_max = ee_positions[:, 1].min(), ee_positions[:, 1].max()
    z_min, z_max = ee_positions[:, 2].min(), ee_positions[:, 2].max()
    
    padding = 0.05
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
    
    ax_3d.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax_3d.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax_3d.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    
    # Color episodes differently
    if episode_indices is not None:
        unique_episodes = np.unique(episode_indices)
        cmap = plt.cm.viridis
        colors = [cmap(i / len(unique_episodes)) for i in range(len(unique_episodes))]
        episode_colors = {ep: colors[i] for i, ep in enumerate(unique_episodes)}
    else:
        episode_colors = None
    
    # Initialize plot elements
    img_display = ax_img.imshow(front_images[0])
    
    # Plot full trajectory as faded background
    if episode_indices is not None:
        for ep in np.unique(episode_indices):
            mask = episode_indices == ep
            ep_positions = ee_positions[mask]
            color = episode_colors[ep] if episode_colors else '#0f3460'
            ax_3d.plot(
                ep_positions[:, 0],
                ep_positions[:, 1],
                ep_positions[:, 2],
                color=color,
                alpha=0.15,
                linewidth=1,
            )
    else:
        ax_3d.plot(
            ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            color='#0f3460',
            alpha=0.15,
            linewidth=1,
        )
    
    # Current trajectory line (will be updated)
    (traj_line,) = ax_3d.plot([], [], [], color='#e94560', linewidth=2, alpha=0.8)
    
    # Current point marker
    (current_point,) = ax_3d.plot([], [], [], 'o', color='#00fff5', markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    # Frame counter text
    frame_text = ax_img.text(
        0.02, 0.98, '', transform=ax_img.transAxes,
        fontsize=10, color='#00fff5', verticalalignment='top',
        fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8, edgecolor='#e94560')
    )
    
    plt.tight_layout()
    
    def init():
        img_display.set_array(front_images[0])
        traj_line.set_data_3d([], [], [])
        current_point.set_data_3d([], [], [])
        frame_text.set_text('')
        return img_display, traj_line, current_point, frame_text
    
    def animate(frame_idx):
        # Update image
        img_display.set_array(front_images[frame_idx])
        
        # Get current episode
        if episode_indices is not None:
            current_ep = episode_indices[frame_idx]
            # Find start of current episode
            ep_start = np.where(episode_indices == current_ep)[0][0]
            traj_positions = ee_positions[ep_start:frame_idx + 1]
        else:
            traj_positions = ee_positions[:frame_idx + 1]
        
        # Update trajectory line
        if len(traj_positions) > 0:
            traj_line.set_data_3d(
                traj_positions[:, 0],
                traj_positions[:, 1],
                traj_positions[:, 2]
            )
            
            # Update current point
            current_point.set_data_3d(
                [ee_positions[frame_idx, 0]],
                [ee_positions[frame_idx, 1]],
                [ee_positions[frame_idx, 2]]
            )
        
        # Update frame text
        ep_text = f"Episode: {episode_indices[frame_idx]}" if episode_indices is not None else ""
        frame_text.set_text(f'Frame: {frame_idx}/{n_frames-1}\n{ep_text}')
        
        return img_display, traj_line, current_point, frame_text
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000 / fps, blit=False
    )
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=fps, dpi=100)
    plt.close(fig)
    print(f"GIF saved successfully!")


def build_episode_map(repo_id: str) -> dict:
    """Build mapping from episode index to file info.
    
    Returns:
        Dict mapping episode_index -> {
            'file_index': int,
            'global_start': int,  # Global frame index where episode starts
            'file_start': int,    # Global frame index where file starts
        }
    """
    print("Building episode-to-file mapping...")
    
    mapping = {}
    
    # Use list_repo_files to find all episode metadata chunks
    all_files = list_repo_files(repo_id, repo_type="dataset")
    meta_files = [f for f in all_files if f.startswith("meta/episodes/chunk-000/") and f.endswith(".parquet")]
    
    for meta_file in sorted(meta_files):
        # Extract file index from filename (file-XXX.parquet)
        try:
            file_idx = int(meta_file.split("file-")[-1].split(".")[0])
        except ValueError:
            continue
            
        # Download
        local_path = hf_hub_download(repo_id, meta_file, repo_type="dataset")
        df = pd.read_parquet(local_path)
        
        # Find start of this file (min of dataset_from_index)
        file_start = df["dataset_from_index"].min()
        
        for _, row in df.iterrows():
            ep_idx = row["episode_index"]
            ep_start = row["dataset_from_index"]
            
            mapping[ep_idx] = {
                "file_index": file_idx,
                "global_start": ep_start,
                "file_start": file_start,
                "length": row["length"]
            }
            
    print(f"Mapped {len(mapping)} episodes.")
    return mapping


def extract_video_frames(
    video_path: str, 
    frame_indices: list[int],
    offset: int = 0
) -> list[np.ndarray]:
    """Extract specific frames from a video file using PyAV.
    
    Args:
        video_path: Path to the video file.
        frame_indices: List of frame indices to extract.
        offset: Offset to add to frame indices (to convert to file-relative).
    
    Returns:
        List of frames as numpy arrays (H, W, C) in RGB format.
    """
    target_indices = [idx + offset for idx in frame_indices]
    
    try:
        import av
        
        # Use PyAV for better codec support (including AV1)
        container = av.open(video_path)
        stream = container.streams.video[0]
        
        # Naive approach: iterate and pick
        # Since seeking can be inaccurate for some codecs/containers, linear scan is safer for correctness
        # though slower. Given file sizes (small chunks), this is acceptable.
        
        max_idx = max(target_indices) if target_indices else 0
        frames_map = {}
        current_idx = 0
        
        for frame in container.decode(stream):
            if current_idx > max_idx:
                break
                
            if current_idx in target_indices:
                frames_map[current_idx] = frame.to_ndarray(format='rgb24')
            
            current_idx += 1
        
        container.close()
        
        # Select the requested frames
        result = []
        for idx in target_indices:
            if idx in frames_map:
                result.append(frames_map[idx])
            else:
                # Frame missing or error
                result.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        return result
        
    except ImportError:
        # Fallback to OpenCV
        print("  PyAV not available, falling back to OpenCV...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        cap.release()
        return frames


def create_multi_view_gif(
    front_images: list[np.ndarray],
    top_images: list[np.ndarray],
    wrist_images: list[np.ndarray],
    ee_positions: np.ndarray,
    output_path: str,
    fps: int = 15,
    episode_indices: np.ndarray | None = None,
):
    """Create a 2x2 grid GIF with 3 camera views and 3D trajectory.
    
    Layout:
    [ Front Camera ] [ Top Camera   ]
    [ Wrist Camera ] [ 3D Trajectory]
    """
    n_frames = len(front_images)
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    
    # Setup subplots
    ax_front = fig.add_subplot(221)
    ax_top = fig.add_subplot(222)
    ax_wrist = fig.add_subplot(223)
    ax_3d = fig.add_subplot(224, projection='3d')
    
    # Style image axes
    for ax, title, img_data in [
        (ax_front, 'Front Camera', front_images[0]),
        (ax_top, 'Top Camera', top_images[0]),
        (ax_wrist, 'Wrist Camera', wrist_images[0])
    ]:
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')
        ax.set_title(title, fontsize=14, color='#e94560', fontweight='bold', pad=10)
        ax.imshow(img_data)
    
    # Style 3D axis
    ax_3d.set_facecolor('#16213e')
    ax_3d.set_xlabel('X (m)', color='#e94560', fontsize=10)
    ax_3d.set_ylabel('Y (m)', color='#e94560', fontsize=10)
    ax_3d.set_zlabel('Z (m)', color='#e94560', fontsize=10)
    ax_3d.set_title('End-Effector Trajectory', fontsize=14, color='#e94560', fontweight='bold', pad=10)
    
    # Set viewing angle: 45 degrees between X and -Y, Z up
    ax_3d.view_init(elev=25, azim=-45)
    
    # Style 3D axes
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('#0f3460')
    ax_3d.yaxis.pane.set_edgecolor('#0f3460')
    ax_3d.zaxis.pane.set_edgecolor('#0f3460')
    ax_3d.tick_params(colors='#e94560', labelsize=8)
    ax_3d.grid(True, alpha=0.3, color='#0f3460')
    
    # Compute limits
    padding = 0.05
    x_min, x_max = ee_positions[:, 0].min(), ee_positions[:, 0].max()
    y_min, y_max = ee_positions[:, 1].min(), ee_positions[:, 1].max()
    z_min, z_max = ee_positions[:, 2].min(), ee_positions[:, 2].max()
    
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
    
    ax_3d.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax_3d.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax_3d.set_zlim(z_min - padding * z_range, z_max + padding * z_range)
    
    # Initialize images
    img_front = ax_front.imshow(front_images[0])
    img_top = ax_top.imshow(top_images[0])
    img_wrist = ax_wrist.imshow(wrist_images[0])
    
    # Plot full trajectory background
    ax_3d.plot(
        ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
        color='#0f3460', alpha=0.15, linewidth=1
    )
    
    # Dynamic elements
    (traj_line,) = ax_3d.plot([], [], [], color='#e94560', linewidth=2, alpha=0.8)
    (current_point,) = ax_3d.plot([], [], [], 'o', color='#00fff5', markersize=10, markeredgecolor='white', markeredgewidth=1)
    
    frame_text = ax_front.text(
        0.02, 0.98, '', transform=ax_front.transAxes,
        fontsize=10, color='#00fff5', verticalalignment='top',
        fontfamily='monospace', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8, edgecolor='#e94560')
    )
    
    plt.tight_layout()
    
    def init():
        return img_front, img_top, img_wrist, traj_line, current_point, frame_text
    
    def animate(frame_idx):
        # Update images
        img_front.set_array(front_images[frame_idx])
        img_top.set_array(top_images[frame_idx])
        img_wrist.set_array(wrist_images[frame_idx])
        
        # Update trajectory
        traj_positions = ee_positions[:frame_idx + 1]
        if len(traj_positions) > 0:
            traj_line.set_data_3d(
                traj_positions[:, 0],
                traj_positions[:, 1],
                traj_positions[:, 2]
            )
            current_point.set_data_3d(
                [ee_positions[frame_idx, 0]],
                [ee_positions[frame_idx, 1]],
                [ee_positions[frame_idx, 2]]
            )
        
        frame_text.set_text(f'Frame: {frame_idx}/{n_frames-1}')
        return img_front, img_top, img_wrist, traj_line, current_point, frame_text
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=n_frames, interval=1000 / fps, blit=False
    )
    
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=fps, dpi=80)
    plt.close(fig)
    print(f"GIF saved successfully!")


def process_single_episode(
    ep_idx: int,
    hf_dataset,
    ep_map: dict,
    fk: SO101ForwardKinematics,
    repo_id: str,
    output_dir: str,
    fps: int = 15,
):
    """Process a single episode and create its GIF."""
    print(f"\n{'='*60}")
    print(f"Processing Episode {ep_idx}")
    print(f"{'='*60}")
    
    # Filter dataset by episode index
    episode_indices_all = hf_dataset["episode_index"]
    filtered_indices = [
        i for i, ep in enumerate(episode_indices_all)
        if ep == ep_idx
    ]
    
    if not filtered_indices:
        print(f"  No frames found for episode {ep_idx}")
        return
    
    print(f"  Found {len(filtered_indices)} frames")
    
    # Extract joint angles and frame indices
    joint_angles = []
    frame_indices = []
    
    for idx in filtered_indices:
        sample = hf_dataset[idx]
        state = np.array(sample["observation.state"])
        joint_angles.append(state)
        frame_indices.append(sample["frame_index"])
    
    joint_angles = np.array(joint_angles)
    
    # Find mapping info for this episode
    if ep_idx not in ep_map:
        print(f"  Warning: No file mapping found for episode {ep_idx}. Skipping video download.")
        # Create blank images
        black_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frame_indices]
        front_images = top_images = wrist_images = black_frames
    else:
        info = ep_map[ep_idx]
        file_idx = info["file_index"]
        # Calculate offset: local frame 0 starts at (global_start - file_start)
        # But wait, video file corresponds to file_start.
        # So frame global_start in dataset corresponds to frame (global_start - file_start) in video.
        # Yes.
        offset = info["global_start"] - info["file_start"]
        
        # Helper to download and extract frames
        def get_frames(camera_key: str):
            chunk_idx = 0
            video_filename = f"videos/{camera_key}/chunk-{chunk_idx:03d}/file-{file_idx:03d}.mp4"
            try:
                video_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=video_filename,
                    repo_type="dataset",
                )
                print(f"  Downloaded {camera_key}: {video_path}")
                # We assume the frame_indices in dataset are relative to episode start (0, 1, 2...)
                # So we just need to add the offset to get into the file.
                return extract_video_frames(video_path, frame_indices, offset=offset)
            except Exception as e:
                print(f"  Warning: Could not download {camera_key}: {e}")
                return [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frame_indices]

        # Get frames for all cameras
        front_images = get_frames("observation.images.front")
        top_images = get_frames("observation.images.top")
        wrist_images = get_frames("observation.images.wrist")
    
    # Compute EE trajectory
    ee_positions = compute_ee_trajectory(joint_angles, fk)
    
    # Create multi-view GIF
    output_path = os.path.join(output_dir, f"episode_{ep_idx:03d}_multiview.gif")
    create_multi_view_gif(
        front_images=front_images,
        top_images=top_images,
        wrist_images=wrist_images,
        ee_positions=ee_positions,
        output_path=output_path,
        fps=fps,
    )


def main():
    parser = argparse.ArgumentParser(description="Visualize EE trajectory from LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="sapanostic/so101_offline_eval",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default="/home/sapan-alienware/projects/lerobot/SO101/so101_new_calib.urdf",
        help="Path to SO101 URDF file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to visualize (default: all episodes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for GIFs",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Initialize FK solver
    print(f"Loading URDF from: {args.urdf_path}")
    fk = SO101ForwardKinematics(urdf_path=args.urdf_path)
    
    # Build episode map first
    ep_map = build_episode_map(args.repo_id)
    
    # Load dataset using HuggingFace datasets
    print(f"Loading dataset: {args.repo_id}")
    hf_dataset = load_dataset(args.repo_id, split="train")
    
    # Get all unique episodes
    all_episodes = sorted(set(hf_dataset["episode_index"]))
    print(f"Dataset has {len(all_episodes)} episodes: {all_episodes}")
    
    # Filter by episodes if specified
    episodes_to_use = args.episodes if args.episodes else all_episodes
    print(f"Processing episodes: {episodes_to_use}")
    
    fps = 15  # Default FPS for this dataset
    
    # Process each episode
    for ep_idx in episodes_to_use:
        if ep_idx not in all_episodes:
            print(f"Warning: Episode {ep_idx} not found in dataset, skipping...")
            continue
        
        process_single_episode(
            ep_idx=ep_idx,
            hf_dataset=hf_dataset,
            ep_map=ep_map,
            fk=fk,
            repo_id=args.repo_id,
            output_dir=args.output_dir,
            fps=fps,
        )
    
    print(f"\n{'='*60}")
    print(f"Done! GIFs saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
