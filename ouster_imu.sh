#!/bin/bash

xterm -e bash -c '

rosrun topic_tools relay /os1_node/imu /imu/data_raw

' &

xterm -e bash -c '

rosrun imu_filter_madgwick imu_filter_node _use_mag:=false

' &
