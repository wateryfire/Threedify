#!/bin/bash
# file: orbslam.sh
if screen -ls | grep 'roscamera' > /dev/null; then
    echo "try stopping conflicted screen process.."
    screen -S roscamera -X stuff $'\003'
fi

echo "running ros camera daemon.."
screen -dmS roscamera ros_camera.sh

sleep 7

echo "done. view with screen -ls;  go to each session with screen -r name; terminate current process with Ctrl-C; go back with Ctrl-A-D."

# trap ctrl-c and call ctrl_c() to stop camera
trap ctrl_c INT

function ctrl_c() {
    echo "** stopping ros camera.."
    screen -S roscamera -X stuff $'\003'
}

echo "start orbslam process.."
#export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/opt/orb/ORB_SLAM2/Examples/ROS
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/opt/orb/ThreeDify/Examples/ROS
#rosrun ORB_SLAM2 RGBD_DPT /opt/orb/ORB_SLAM2/Vocabulary/ORBvoc.txt /opt/orb/ORB_SLAM2/test/XtionProLive.yaml /camera/rgb/image_rect_color /camera/depth_registered/sw_registered/image_rect_raw
rosrun ORB_SLAM2 RGBD /opt/orb/ThreeDify/Vocabulary/ORBvoc.bin /opt/orb/ThreeDify/Examples/RGB-D/XtionProLive.yaml /camera/rgb/image_raw /camera/depth/image_raw
#/camera/rgb/image_rect_color /camera/depth/image_rect_raw

