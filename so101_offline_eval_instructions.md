


## Find motor ports
``` 
lerobot-find-port

## 
# Follower: /dev/ttyACM0
# Leader: /dev/ttyACM1
#

```

### Find camera ports
```
lerobot-find-camera

# RealSense D435: 844212071286 (10)
# OpenCV Camera: 4
```

### Teleop without cameras 
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so101_leader_arm
```

### Teleop with cameras 
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 15}, top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 15}, front: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 15} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so101_leader_arm \
    --display_data=true \
    --fps=15
```

### Record dataset

```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=so101_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 15}, top: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 15}, front: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 15} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=so101_leader_arm \
    --display_data=true \
    --dataset.fps=15 \
    --dataset.num_episodes=5 \
    --dataset.repo_id=sapanostic/so101_offline_eval \
    --dataset.single_task="put the green item inside the bowl" \
    --dataset.push_to_hub=True \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=5 \
    --resume=True
```

## Dataset 
The dataset for SO101 can be found at: https://huggingface.co/datasets/sapanostic/so101_offline_eval