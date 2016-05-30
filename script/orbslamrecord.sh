#!/bin/bash
# file: orbslam.sh
if screen -ls | grep 'roscamera' > /dev/null; then
    echo "try stopping conflicted screen process.."
    ctrl_c()
fi

echo "running ros camera daemon.."
#rm -f /tmp/roscamera.out
screen -dmS roscamera bash
screen -S roscamera -X stuff "roslaunch openni2_launch openni2.launch >/tmp/roscamera.out 2>&1\n"

#while true
#do
#   echo "hello"
#   sleep 2
#done
sleep 7

echo "done. view with screen -ls;  go to each session with screen -r name; terminate current process with Ctrl-C; go back with Ctrl-A-D."

echo "running rosbag record daemon.."
screen -dmS rosrecord bash
screen -S rosrecord -X stuff "cd /opt/orb/data/\n"
screen -S rosrecord -X stuff "rosbag record camera/depth/camera_info camera/depth/image_raw camera/rgb/camera_info camera/rgb/image_raw rosout tf\n"

# trap ctrl-c and call ctrl_c() to stop camera
trap ctrl_c INT

function ctrl_c() {
    echo "** stopping ros camera.."
    screen -S roscamera -X stuff $'\003'
    screen -S roscamera -X stuff "exit\n"
    screen -S rosrecord -X stuff $'\003'
    screen -S rosrecord -X stuff "exit\n"
}

echo "start orbslam process.."
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/opt/orb/ORB_SLAM2/Examples/ROS
rosrun ORB_SLAM2 RGBD_DPT /opt/orb/ORB_SLAM2/Vocabulary/ORBvoc.bin /opt/orb/ORB_SLAM2/test/XtionProLive.yaml /camera/rgb/image_rect_color /camera/depth_registered/sw_registered/image_rect_raw

