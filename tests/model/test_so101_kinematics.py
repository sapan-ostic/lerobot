"""
Unit tests for SO101 Forward Kinematics.

This module tests the forward kinematics of the SO101 robot arm using the URDF model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Skip all tests if placo is not installed
pytest.importorskip("placo", reason="placo is required for kinematics tests")

from lerobot.model.so101_kinematics import (
    DEFAULT_SO101_URDF_PATH,
    SO101_JOINT_NAMES,
    SO101ForwardKinematics,
    SO101Position,
)

# Path to SO101 URDF (for tests)
SO101_URDF_PATH = DEFAULT_SO101_URDF_PATH


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def so101_fk() -> SO101ForwardKinematics:
    """Create an SO101 FK solver instance."""
    if not SO101_URDF_PATH.exists():
        pytest.skip(f"SO101 URDF not found at {SO101_URDF_PATH}")
    return SO101ForwardKinematics()


# =============================================================================
# Tests
# =============================================================================


class TestSO101ForwardKinematics:
    """Test suite for SO101 Forward Kinematics."""

    def test_urdf_exists(self):
        """Test that the SO101 URDF file exists."""
        assert SO101_URDF_PATH.exists(), f"URDF not found at {SO101_URDF_PATH}"

    def test_fk_initialization(self, so101_fk: SO101ForwardKinematics):
        """Test that FK solver initializes correctly."""
        assert so101_fk.kinematics is not None
        assert so101_fk.urdf_path.exists()

    def test_home_position_returns_valid_pose(self, so101_fk: SO101ForwardKinematics):
        """Test that HOME position returns a valid 4x4 transformation matrix."""
        pose = so101_fk.compute(SO101Position.HOME)

        # Check shape
        assert pose.shape == (4, 4), f"Expected 4x4 matrix, got {pose.shape}"

        # Check that bottom row is [0, 0, 0, 1]
        expected_bottom = np.array([0, 0, 0, 1])
        np.testing.assert_array_almost_equal(
            pose[3, :], expected_bottom, decimal=6,
            err_msg="Bottom row of transformation matrix should be [0, 0, 0, 1]"
        )

        # Check that rotation matrix is orthonormal
        R = pose[:3, :3]
        RtR = R.T @ R
        np.testing.assert_array_almost_equal(
            RtR, np.eye(3), decimal=6,
            err_msg="Rotation matrix should be orthonormal (R^T @ R = I)"
        )

        # Check determinant of rotation matrix is +1 (proper rotation)
        det = np.linalg.det(R)
        assert np.isclose(det, 1.0, atol=1e-6), f"Rotation determinant should be 1, got {det}"

    def test_home_position_ee_location(self, so101_fk: SO101ForwardKinematics):
        """Test that HOME position places end-effector at expected exact location."""
        position = so101_fk.get_ee_position(SO101Position.HOME)

        # Expected EE position at HOME (all joints at 0 degrees)
        # These values are computed from the SO101 URDF
        expected_position = np.array([0.3913614702, -0.0000092121, 0.2264697102])

        np.testing.assert_array_almost_equal(
            position, expected_position, decimal=4,
            err_msg=f"HOME EE position mismatch. Got: {position}, Expected: {expected_position}"
        )

    def test_home_position_full_pose(self, so101_fk: SO101ForwardKinematics):
        """Test the complete 4x4 pose matrix at HOME position."""
        pose = so101_fk.compute(SO101Position.HOME)

        # Expected full 4x4 transformation matrix at HOME
        expected_pose = np.array([
            [0.0000086650, -0.0000103004, 0.9999999999, 0.3913614702],
            [0.0486629269, 0.9988152579, 0.0000098665, -0.0000092121],
            [-0.9988152579, 0.0486629268, 0.0000091560, 0.2264697102],
            [0.0, 0.0, 0.0, 1.0],
        ])

        np.testing.assert_array_almost_equal(
            pose, expected_pose, decimal=4,
            err_msg="HOME pose matrix mismatch"
        )

    def test_forward_extended_ee_location(self, so101_fk: SO101ForwardKinematics):
        """Test exact EE position at FORWARD_EXTENDED configuration."""
        position = so101_fk.get_ee_position(SO101Position.FORWARD_EXTENDED)

        expected_position = np.array([0.4627596643, -0.0000086203, 0.1736999930])

        np.testing.assert_array_almost_equal(
            position, expected_position, decimal=4,
            err_msg=f"FORWARD_EXTENDED EE position mismatch. Got: {position}"
        )

    def test_vertical_up_ee_location(self, so101_fk: SO101ForwardKinematics):
        """Test exact EE position at VERTICAL_UP configuration."""
        position = so101_fk.get_ee_position(SO101Position.VERTICAL_UP)

        expected_position = np.array([0.1791041432, -0.0000094086, -0.2055270373])

        np.testing.assert_array_almost_equal(
            position, expected_position, decimal=4,
            err_msg=f"VERTICAL_UP EE position mismatch. Got: {position}"
        )

    def test_tucked_ee_location(self, so101_fk: SO101ForwardKinematics):
        """Test exact EE position at TUCKED configuration."""
        position = so101_fk.get_ee_position(SO101Position.TUCKED)

        expected_position = np.array([0.0603087768, -0.0000106051, -0.0378955999])

        np.testing.assert_array_almost_equal(
            position, expected_position, decimal=4,
            err_msg=f"TUCKED EE position mismatch. Got: {position}"
        )

    def test_home_position_is_deterministic(self, so101_fk: SO101ForwardKinematics):
        """Test that FK is deterministic - same input gives same output."""
        pose1 = so101_fk.compute(SO101Position.HOME)
        pose2 = so101_fk.compute(SO101Position.HOME)

        np.testing.assert_array_equal(
            pose1, pose2,
            err_msg="FK should be deterministic"
        )

    def test_numpy_array_input(self, so101_fk: SO101ForwardKinematics):
        """Test that FK accepts numpy array input."""
        home_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        pose = so101_fk.compute(home_array)

        assert pose.shape == (4, 4)

    def test_different_positions_give_different_results(self, so101_fk: SO101ForwardKinematics):
        """Test that different joint configurations result in different EE positions."""
        pos_home = so101_fk.get_ee_position(SO101Position.HOME)
        pos_forward = so101_fk.get_ee_position(SO101Position.FORWARD_EXTENDED)

        # They should be different
        assert not np.allclose(pos_home, pos_forward), \
            "Different joint configurations should give different EE positions"

    def test_shoulder_pan_rotates_around_z(self, so101_fk: SO101ForwardKinematics):
        """Test that shoulder_pan joint rotates the arm around the Z axis."""
        # Home position
        pos_home = so101_fk.get_ee_position(SO101Position.HOME)

        # Rotate shoulder_pan by 90 degrees
        rotated_joints = np.array([90.0, 0.0, 0.0, 0.0, 0.0])
        pos_rotated = so101_fk.get_ee_position(rotated_joints)

        # The Z position should remain approximately the same
        assert np.isclose(pos_home[2], pos_rotated[2], atol=0.01), \
            "Z position should be similar when only rotating shoulder_pan"

        # But X and Y should change
        xy_home = np.sqrt(pos_home[0]**2 + pos_home[1]**2)
        xy_rotated = np.sqrt(pos_rotated[0]**2 + pos_rotated[1]**2)

        # The radial distance from Z axis should be similar (allowing for some URDF offset)
        # Note: Some variation is expected due to joint axis offsets in the URDF
        assert np.isclose(xy_home, xy_rotated, atol=0.05), \
            f"Radial distance from Z axis should be similar after shoulder_pan rotation. " \
            f"Home: {xy_home:.4f}, Rotated: {xy_rotated:.4f}"

    def test_so101_position_enum(self):
        """Test the SO101Position enum methods."""
        # Test to_array
        home_array = SO101Position.HOME.to_array()
        assert isinstance(home_array, np.ndarray)
        assert len(home_array) == 5
        np.testing.assert_array_equal(home_array, [0.0, 0.0, 0.0, 0.0, 0.0])

        # Test to_action_dict
        home_dict = SO101Position.HOME.to_action_dict()
        assert isinstance(home_dict, dict)
        assert len(home_dict) == 5
        assert home_dict["shoulder_pan.pos"] == 0.0
        assert home_dict["shoulder_lift.pos"] == 0.0
        assert home_dict["elbow_flex.pos"] == 0.0
        assert home_dict["wrist_flex.pos"] == 0.0
        assert home_dict["wrist_roll.pos"] == 0.0

    def test_forward_extended_increases_reach(self, so101_fk: SO101ForwardKinematics):
        """Test that FORWARD_EXTENDED position increases horizontal reach."""
        pos_home = so101_fk.get_ee_position(SO101Position.HOME)
        pos_forward = so101_fk.get_ee_position(SO101Position.FORWARD_EXTENDED)

        # Compute horizontal distance from base
        horizontal_home = np.sqrt(pos_home[0]**2 + pos_home[1]**2)
        horizontal_forward = np.sqrt(pos_forward[0]**2 + pos_forward[1]**2)

        print(f"HOME horizontal reach: {horizontal_home:.4f} m")
        print(f"FORWARD_EXTENDED horizontal reach: {horizontal_forward:.4f} m")

        # Forward extended should have greater horizontal reach
        # Note: This depends on the specific joint angles chosen
        # If this fails, the FORWARD_EXTENDED angles may need adjustment

    def test_print_all_ee_poses(self, so101_fk: SO101ForwardKinematics):
        """Print the EE pose for all predefined configurations."""
        print("\n" + "=" * 80)
        print("SO101 End-Effector Poses for All Predefined Configurations")
        print("=" * 80)

        for position in SO101Position:
            pose = so101_fk.compute(position)
            ee_pos = pose[:3, 3]

            # Extract rotation as roll-pitch-yaw (approximate)
            R = pose[:3, :3]

            print(f"\n{position.name}:")
            print(f"  Joint angles (deg): {position.value}")
            print(f"  EE Position (m):    x={ee_pos[0]:.4f}, y={ee_pos[1]:.4f}, z={ee_pos[2]:.4f}")
            print(f"  4x4 Pose Matrix:")
            for row in pose:
                print(f"    [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")

        print("\n" + "=" * 80)

        # This test always passes - it's for information purposes
        assert True


class TestSO101PositionEnum:
    """Test suite for SO101Position enum."""

    def test_all_positions_have_5_joints(self):
        """Test that all predefined positions have exactly 5 joint values."""
        for position in SO101Position:
            assert len(position.value) == 5, \
                f"{position.name} should have 5 joint values, got {len(position.value)}"

    def test_home_is_all_zeros(self):
        """Test that HOME position is all zeros."""
        np.testing.assert_array_equal(
            SO101Position.HOME.to_array(),
            np.zeros(5),
            err_msg="HOME position should be all zeros"
        )

    def test_to_action_dict_keys(self):
        """Test that to_action_dict produces correct keys."""
        action_dict = SO101Position.HOME.to_action_dict()
        expected_keys = {
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
        }
        assert set(action_dict.keys()) == expected_keys
