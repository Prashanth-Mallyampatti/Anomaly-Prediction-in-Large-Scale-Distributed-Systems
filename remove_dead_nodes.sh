#!/usr/bin/bash

#Author: Prashanth Mallyampatti

#Get the line numbers for nodes that are shutdown and have been left
LINE+=`nodetool gossipinfo | egrep -n 'shutdown|LEFT' | cut -d: -f1`

j=0
if [[ !${LINE[@]} ]]
then
	echo "No Dead Nodes"
else
	for i in ${LINE[@]}
	do
		#each iteration add 1,2,3... to k(here it is added by j) as removing node adds an extra line(REMOVAL_COORDINATOR) in gossipinfo
		k=$((i+8+j))


		#add line numbers and get the host id for nodetool removal
		HOST=`nodetool gossipinfo | grep -n 'HOST_ID' | grep "$k: " | cut -d: -f4`


		#remove the node
		nodetool removenode "$HOST"
	

		#check for success
		if ( `nodetool gossipinfo | grep -n 'removed' | grep "$k: " | cut -d: -f4` )
		then	
			echo "Node: $HOST removed successfully"
		else
			echo "Node: $HOST removed unsuccessfully"
		fi
	
		j=$((j+1))
	done
fi
