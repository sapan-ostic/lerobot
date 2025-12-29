#!/usr/bin/env python
"""
Unit test script that reads joint angles from the robot and sends commands to hold position.

This script:
1. Connects to the SO101 robot
2. Reads current joint positions
3. Sends those positions as goal commands to hold the position
4. Repeats in a loop at a fixed rate until interrupted

Usage:
    python scripts/test_hold_position.py --port /dev/ttyACM0
    
Press Ctrl+C to stop.
"""

import argparse
import logging
import time
from enum import Enum
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotPosition(Enum):
    """Predefined robot positions with joint angles.
    
    Each position is a tuple of (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
    Values are in degrees if use_degrees=True, otherwise in normalized range [-100, 100].
    """
    HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def to_action_dict(self) -> dict[str, float]:
        """Convert position to action dictionary format."""
        joint_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos", 
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        return {name: value for name, value in zip(joint_names, self.value)}


def parse_args():
    parser = argparse.ArgumentParser(description="Test script to read joints and hold position")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the robot (default: /dev/ttyACM0)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="Control loop rate in Hz (default: 30.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to run in seconds (default: None, runs until Ctrl+C)",
    )
    parser.add_argument(
        "--use-degrees",
        action="store_true",
        help="Use degrees instead of normalized range [-100, 100]",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run calibration if needed (default: skip calibration prompt)",
    )
    parser.add_argument(
        "--go-home",
        action="store_true",
        help="Move robot to HOME position (all zeros) instead of holding current position",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Import robot classes
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    
    # Create robot config
    config = SO101FollowerConfig(
        id="so101_follower_arm",
        port=args.port,
        use_degrees=args.use_degrees,
        disable_torque_on_disconnect=True,
    )
    
    # Create robot instance
    robot = SO101Follower(config)
    
    logger.info(f"Connecting to robot on port {args.port}...")
    
    try:
        # Connect to the robot
        robot.connect(calibrate=args.calibrate)
        logger.info("Robot connected successfully!")
        
        # Read initial position
        obs = robot.get_observation()
        joint_positions = {k: v for k, v in obs.items() if k.endswith(".pos")}
        
        logger.info("Initial joint positions:")
        for joint, pos in joint_positions.items():
            logger.info(f"  {joint}: {pos:.2f}")
        
        # Determine target position
        if args.go_home:
            target_positions = RobotPosition.HOME.to_action_dict()
            logger.info("\nTarget HOME position:")
            for joint, pos in target_positions.items():
                logger.info(f"  {joint}: {pos:.2f}")
        else:
            target_positions = None  # Will use current position
        
        # Hold position loop
        period = 1.0 / args.rate
        start_time = time.time()
        loop_count = 0
        
        mode = "Moving to HOME" if args.go_home else "Holding position"
        logger.info(f"\n{mode} at {args.rate} Hz. Press Ctrl+C to stop.")
        
        while True:
            loop_start = time.perf_counter()
            
            # Read current position
            obs = robot.get_observation()
            current_positions = {k: v for k, v in obs.items() if k.endswith(".pos")}
            
            # Send target positions as goal
            if args.go_home:
                robot.send_action(target_positions)
            else:
                robot.send_action(current_positions)
            
            loop_count += 1
            
            # Print status every second
            elapsed = time.time() - start_time
            if loop_count % int(args.rate) == 0:
                logger.info(f"Running for {elapsed:.1f}s, loop count: {loop_count}")
                for joint, pos in current_positions.items():
                    logger.info(f"  {joint}: {pos:.2f}")
            
            # Check duration limit
            if args.duration is not None and elapsed >= args.duration:
                logger.info(f"Duration limit of {args.duration}s reached.")
                break
            
            # Sleep to maintain rate
            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = period - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        # Disconnect robot
        if robot.is_connected:
            logger.info("Disconnecting robot...")
            robot.disconnect()
            logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()

