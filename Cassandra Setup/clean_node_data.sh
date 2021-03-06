#!/usr/bin/bash

#Author: Prashanth Mallyampatti

echo "Killing existing cassandra deamons in this VM..."

pgrep -f cassandra | xargs sudo kill -9

if sudo rm -r /var/lib/cassandra/* ;
then
	echo "Cleaned Node's Data"
	echo "Execute command: yes Y | bash ./cassandra.sh to join/create the node to the existing/new ring"
else
	echo "Couldn't clean Node's Data"
fi
