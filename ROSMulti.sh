#!/bin/bash

MULTI_SET=0

if [ $ROS_IP ]
then
   MULTI_SET=1
fi

if [ $MULTI_SET -eq 1 ]
then
   read -p "Single 0 Multiple 1: " MULTI_OPT
   if [ $MULTI_OPT -eq 1 ]
   then
      "Added ROS_IP and ROS_MASTER_URI"
      sed -i '/ROS_IP=/d' /home/$USER/.bashrc
      sed -i '/ROS_MASTER_URI=/d' /home/$USER/.bashrc
      read -p "Enter Computer IP Address: " SELF_IP
      read -p "Enter Host Computer: " HOST_IP
      echo "export ROS_IP=$SELF_IP" >> /home/$USER/.bashrc
      echo "export ROS_MASTER_URI=http://$HOST_IP:11311" >> /home/$USER/.bashrc
   else
      echo "Deleted ROS_IP and ROS_MASTER_URI"
      sed -i '/ROS_IP=/d' /home/$USER/.bashrc
      sed -i '/ROS_MASTER_URI=/d' /home/$USER/.bashrc
   fi
else
   read -p "Single 0 Multiple 1: " MULTI_OPT
   if [ $MULTI_OPT -eq 1 ]
   then
      echo "Added ROS_IP and ROS_MASTER_URI"
      read -p "Enter Computer IP Address: " SELF_IP
      read -p "Enter Host Computer: " HOST_IP
      echo "export ROS_IP=$SELF_IP" >> /home/$USER/.bashrc
      echo "export ROS_MASTER_URI=http://$HOST_IP:11311" >> /home/$USER/.bashrc
   else
      echo "Deleted ROS_IP and ROS_MASTER_URI"
   fi
fi
