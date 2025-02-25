#! /bin/bash
# limit gpu acsess for applications
# run as 
# gpu_select.sh #to get info
# . gpu_select.sh 4 # to select mig instance, ie virtual gpu nr 4
#
# Additional info: https://servicedesk.surf.nl/wiki/display/WIKI/Interactive+development+GPU+node

export CUDA_VISIBLE_DEVICES=""

if [ -z "$1" ];then
   echo "nvidia-smi"
   nvidia-smi
else
    mig=($(nvidia-smi -L | sed -nr "s|^.*UUID:\s*(MIG-[^)]+)\)|\1|p"))
    export CUDA_VISIBLE_DEVICES=${mig[$1]}
fi
echo "Selected device ${CUDA_VISIBLE_DEVICES}"