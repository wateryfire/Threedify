#!/bin/bash
# file: orbslam.sh
if screen -ls | grep 'roscore' > /dev/null; then
    echo "try stopping conflicted screen process.."
    screen -S roscore -X stuff $'\003'
fi

echo "running ros core daemon.."
screen -dmS roscore roscore

sleep 3

# trap ctrl-c and call ctrl_c() to stop camera
trap ctrl_c INT

function ctrl_c() {
    echo "** stopping ros camera.."
    screen -S roscore -X stuff $'\003'
}

echo "done. view with screen -ls;  go to each session with screen -r name; terminate current process with Ctrl-C; go back with Ctrl-A-D."

echo "start orbslam process.."
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/opt/orb/ThreeDify/Examples/ROS
rosrun ORB_SLAM2 RGBD /opt/orb/ThreeDify/Vocabulary/ORBvoc.bin /opt/orb/ThreeDify/Examples/RGB-D/XtionProLive.yaml /camera/rgb/image_raw /camera/depth/image_raw

