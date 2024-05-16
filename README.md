# tabletop-assistant

## Setup

### Pre-requisites

You should have [ROS 2 Iron](https://docs.ros.org/en/iron/index.html) installed on Ubuntu 22.04.

The follow hardware is is used. The use of other hardware will require adapting the codebase.
- [AR4 robot arm](https://www.anninrobotics.com/) with [AR servo gripper](https://www.anninrobotics.com/product-page/servo-gripper-parts-kit)
- [Intel RealSense D435 Depth Camera](https://www.intelrealsense.com/depth-camera-d435/)

You should be able to run [ar4_ros_driver](https://github.com/ycheng517/ar4_ros_driver) 
with the gripper, and successfully perform hand-eye calibration.

### Install

Import dependent repos
```bash
vcs import . --input tabletop-handybot.repos
```

Create a virtual environment, i.e.
```bash
pyenv virtualenv 3.10.12 handybot
```

Go to the `./Grounded-Segment-Anything/Grounded-Segment-Anything/` sub-directory 
and setup [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) 
in the virtualenv. Ensure you can run the [grounded_sam.ipynb](https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/grounded_sam.ipynb)
notebook.

Install Python dependencies not in the ROS Index
```bash
pip install -r requirements.txt
```

Build and source the project
```bash
colcon build
source install/setup.bash
```

### Run

Launch the AR4
```bash
ros2 launch ar_hardware_interface ar_hardware.launch.py calibrate:=True include_gripper:=True
```

Launch all other programs
```bash
ros2 launch tabletop_handybot run.launch.py
```

If all things are looking good, then you can try publishing a prompt to the
`/prompt` topic for the robot to execute, i.e:
```bash
ros2 topic pub --once /prompt std_msgs/msg/String "data: 'put the marker in the container'"
```

If you have a microphone attached to the computer, you can publish a message 
to the `/listen` topic, and then say your prompt.
```bash
ros2 topic pub --once /listen std_msgs/msg/Empty "{}"
```
