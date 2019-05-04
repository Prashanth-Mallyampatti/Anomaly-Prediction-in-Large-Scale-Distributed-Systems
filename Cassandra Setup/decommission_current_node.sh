#!/usr/bin/bash

#Author: Prashanth Mallyampatti

#Decommission node

nodetool decommission


#check for success
IP_ADDR=`/sbin/ifconfig ens33 | grep 'inet' | cut -d: -f2 | awk '{ print $2}'`

LINE_NUM=`nodetool gossipinfo | grep -n /$IP_ADDR | cut -c1-1`

LINE_NUM=$((LINE_NUM + 3))

LINE=`nodetool gossipinfo | grep -n 'LEFT' | cut -d: -f1`

if [ "$LINE" == "$LINE_NUM" ]
then
	echo "Node: $IP_ADDR Decommission successful"
else
	echo "Node: $IP_ADDR Decommission unsuccessful"
fi

