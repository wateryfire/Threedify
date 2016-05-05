#!/bin/sh

# convert raw map points to .pcd and .ply file
# @author jie.z

linenumber=$(wc -l MapPoints.txt | grep -o [0-9]*)
/opt/orb/ORB_SLAM2/test/pcdexport/build/Export MapPoints.txt $linenumber
/opt/orb/ORB_SLAM2/test/pcdexport/build/PcdToPly
