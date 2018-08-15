 #!/bin/bash

xterm -e bash -c '

source ~/ros_ws/ouster_ws/devel/setup.bash;

read -p "Enter Ouster IP: " HOSTNAME;

read -p "Enter Computer IP: " DESTNAME;

roslaunch ouster_ros os1.launch os1_hostname:=$HOSTNAME os1_udp_dest:=$DESTNAME

' &

