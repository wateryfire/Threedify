#!/bin/sh

# convert raw map points to .pcd and .ply file
# @author jie.z

linenumber=$(wc -l $1 | grep -o [0-9]*)
./Export $1 $linenumber
./PcdToPly
