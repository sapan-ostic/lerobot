# SO101 Robot Configuration Files

This directory contains configuration files for SO101 robot teleoperation using the LeRobot framework.

## Configuration Files

### `so101_teleop.json`
Complete SO101 teleoperation configuration with cameras enabled.

**Features:**
- RealSense camera support (serial: 751412060299)
- Wrist camera support (index: 6)
- Camera visualization enabled (`display_data: true`)
- 30 FPS teleoperation
- Standard port configuration (/dev/ttyACM0 for leader, /dev/ttyACM1 for follower)

**Usage:**
```bash
python lerobot/scripts/control_robot.py --config_path=configs/so101_teleop.json
```

### `so101_teleop_no_cameras.json`
SO101 teleoperation configuration without cameras for faster performance.

**Features:**
- No camera configuration (empty cameras object)
- Camera visualization disabled (`display_data: false`)
- 30 FPS teleoperation
- Standard port configuration

**Usage:**
```bash
python lerobot/scripts/control_robot.py --config_path=configs/so101_teleop_no_cameras.json
```

## Customization

You can still override any configuration parameter via command line:

```bash
# Change FPS
python lerobot/scripts/control_robot.py \
    --config_path=configs/so101_teleop.json \
    --control.fps=15

# Change RealSense serial number
python lerobot/scripts/control_robot.py \
    --config_path=configs/so101_teleop.json \
    --robot.cameras.realsense.serial_number=YOUR_SERIAL_NUMBER

# Change ports
python lerobot/scripts/control_robot.py \
    --config_path=configs/so101_teleop.json \
    --robot.leader_arms.main.port=/dev/ttyACM2 \
    --robot.follower_arms.main.port=/dev/ttyACM3
```

## Hardware Setup Requirements

### Ports
- **Leader arm**: `/dev/ttyACM0`
- **Follower arm**: `/dev/ttyACM1`

### Cameras (for `so101_teleop.json`)
- **RealSense camera**: Serial number `751412060299`
- **Wrist camera**: Camera index `6`

### Motors
Both leader and follower arms use FeetechMotorsBus with STS3215 motors:
1. shoulder_pan
2. shoulder_lift  
3. elbow_flex
4. wrist_flex
5. wrist_roll
6. gripper

## Troubleshooting

### Camera Issues
If you have camera issues, switch to the no-cameras configuration:
```bash
python lerobot/scripts/control_robot.py --config_path=configs/so101_teleop_no_cameras.json
```

**Camera Type Requirements:**
- RealSense cameras need `"type": "intelrealsense"`
- USB/Webcam cameras need `"type": "opencv"`

**RealSense Troubleshooting:**
- "No device connected": Re-plug camera, check with `realsense-viewer`
- Check `librealsense` and `pyrealsense2` installation
- Update camera firmware if needed

**USB Camera Troubleshooting:**
- Check available cameras: `v4l2-ctl --list-devices`
- Permission issues: `sudo usermod -a -G video $USER`

### Port Issues
Check your actual device ports and update the configuration:
```bash
ls /dev/tty* | grep ACM
```

### RealSense Serial Number
Find your RealSense serial number:
```bash
python -c "
import pyrealsense2 as rs
ctx = rs.context()
for dev in ctx.query_devices():
    print(f'Serial: {dev.get_info(rs.camera_info.serial_number)}')
"
```

## Comparison with CLI Approach

**Using configuration file:**
```bash
python lerobot/scripts/control_robot.py --config_path=configs/so101_teleop.json
```

**Equivalent CLI command:**
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=so101 \
    --robot.cameras.realsense.serial_number=751412060299 \
    --robot.cameras.realsense.fps=30 \
    --robot.cameras.realsense.width=640 \
    --robot.cameras.realsense.height=480 \
    --robot.cameras.realsense.use_depth=false \
    --robot.cameras.realsense.force_hardware_reset=true \
    --robot.cameras.wrist.camera_index=6 \
    --robot.cameras.wrist.fps=30 \
    --robot.cameras.wrist.width=640 \
    --robot.cameras.wrist.height=480 \
    --control.type=teleoperate \
    --control.fps=30 \
    --control.display_data=true
```

The configuration file approach is much cleaner and easier to manage!
