#!/usr/bin/bash

#Author: Prashanth Mallyampatti

echo "Killing existing cassandra deamons in this VM"

pgrep -f cassandra | xargs sudo kill -9

echo
echo "Cleaning cassandra data"

sudo rm -r /var/lib/cassandra/*

