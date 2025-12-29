"""
SO101 Robot Kinematics Module.

This module provides forward kinematics utilities for the SO101 robot arm,
including predefined positions and a convenience wrapper around RobotKinematics.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from .kinematics import RobotKinematics

# =============================================================================
# Constants
# =============================================================================

# Default path to SO101 URDF (relative to repo root)
# Users should provide their own path if the default doesn't exist
DEFAULT_SO101_URDF_PATH = Path(__file__).parents[3] / "SO101" / "so101_new_calib.urdf"

# Joint names for SO101 (matches URDF, excludes gripper)
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# End-effector frame name in URDF
SO101_EE_FRAME = "gripper_frame_link"


# =============================================================================
# Predefined Robot Positions
# =============================================================================


class SO101Position(Enum):
    """Predefined SO101 robot positions with joint angles in degrees.

    Each position is a tuple of (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll).
    All angles are in degrees.

    Example:
        >>> from lerobot.model.so101_kinematics import SO101Position
        >>> home = SO101Position.HOME
        >>> print(home.to_array())  # [0. 0. 0. 0. 0.]
        >>> print(home.to_action_dict())  # {'shoulder_pan.pos': 0.0, ...}
    """

    # Home position - all joints at zero
    HOME = (0.0, 0.0, 0.0, 0.0, 0.0)

    # Arm extended forward
    FORWARD_EXTENDED = (0.0, 45.0, -45.0, 0.0, 0.0)

    # Arm pointing up
    VERTICAL_UP = (0.0, 90.0, 0.0, 0.0, 0.0)

    # Arm tucked to the side
    TUCKED = (0.0, -30.0, 90.0, 45.0, 0.0)

    def to_array(self) -> np.ndarray:
        """Convert position to numpy array.

        Returns:
            numpy array of joint angles in degrees.
        """
        return np.array(self.value, dtype=float)

    def to_action_dict(self) -> dict[str, float]:
        """Convert position to action dictionary format with .pos suffix.

        Returns:
            Dictionary mapping joint names (with .pos suffix) to angles.
        """
        return {f"{name}.pos": value for name, value in zip(SO101_JOINT_NAMES, self.value)}


# =============================================================================
# SO101 Forward Kinematics Helper Class
# =============================================================================


class SO101ForwardKinematics:
    """Forward kinematics helper for SO101 robot arm.

    This class wraps RobotKinematics to provide a convenient interface
    for computing forward kinematics on the SO101 robot.

    Example:
        >>> from lerobot.model.so101_kinematics import SO101ForwardKinematics, SO101Position
        >>> fk = SO101ForwardKinematics()
        >>> pose = fk.compute(SO101Position.HOME)
        >>> position = fk.get_ee_position(SO101Position.HOME)
        >>> print(f"EE at: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")

    Attributes:
        urdf_path: Path to the URDF file.
        kinematics: Underlying RobotKinematics solver.
    """

    def __init__(self, urdf_path: str | Path | None = None):
        """Initialize the FK solver.

        Args:
            urdf_path: Path to the SO101 URDF file. If None, uses the default path.

        Raises:
            FileNotFoundError: If the URDF file does not exist.
            ImportError: If placo is not installed.
        """
        if urdf_path is None:
            urdf_path = DEFAULT_SO101_URDF_PATH

        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.kinematics = RobotKinematics(
            urdf_path=str(self.urdf_path),
            target_frame_name=SO101_EE_FRAME,
            joint_names=SO101_JOINT_NAMES,
        )

    def compute(self, joint_angles: np.ndarray | SO101Position) -> np.ndarray:
        """Compute forward kinematics for given joint angles.

        Args:
            joint_angles: Joint angles in degrees. Can be a numpy array or SO101Position enum.

        Returns:
            4x4 homogeneous transformation matrix of the end-effector pose.
        """
        if isinstance(joint_angles, SO101Position):
            joint_angles = joint_angles.to_array()

        return self.kinematics.forward_kinematics(joint_angles)

    def get_ee_position(self, joint_angles: np.ndarray | SO101Position) -> np.ndarray:
        """Get the end-effector position (x, y, z) in meters.

        Args:
            joint_angles: Joint angles in degrees. Can be a numpy array or SO101Position enum.

        Returns:
            3D position vector (x, y, z) in meters.
        """
        pose = self.compute(joint_angles)
        return pose[:3, 3]

    def get_ee_orientation(self, joint_angles: np.ndarray | SO101Position) -> np.ndarray:
        """Get the end-effector orientation as a 3x3 rotation matrix.

        Args:
            joint_angles: Joint angles in degrees. Can be a numpy array or SO101Position enum.

        Returns:
            3x3 rotation matrix.
        """
        pose = self.compute(joint_angles)
        return pose[:3, :3]

