#Author: Prashanth Mallyampatti

#!/bin/bash

#Install stress-ng if not already installed

if ! stress-ng --version >/dev/null 2>&1; then
    sudo apt-get install stress-ng
fi

#Injecting External Faults

CPU_CORE=`nproc --all`   #Get the number of cores
CFAULT_TIME=3 		 #time in seconds for which the fault has to run.

get_mem()
{
	FREE_MEMORY=`free -m  | grep ^Mem | tr -s ' ' | cut -d ' ' -f 4`
	TOTAL_MEMORY=`free -m | grep ^Mem | tr -s ' ' | cut -d ' ' -f 2`
	FREE_PERCENT=$((FREE_MEMORY * 100 / $TOTAL_MEMORY))
}
cpu_fault()
{
	echo
	echo **********************Injecting CPU Faults***********************

	for i in 80 85 90 95 97 98 99
	do
		echo
		echo Acheiving \~$i % CPU usage
		echo "--------------------------"
		stress-ng -c $CPU_CORE -l $i -t $CFAULT_TIME
	done
}

mem_fault()
{
	echo
	echo *********************Injecting Memory Faults********************
	
	get_mem
	echo
	echo Acheiving \~$FREE_PERCENT % memory usage 
	echo "-----------------------------"

	#Spawn on 1 CPU and 1 VM(core) thread
	stress-ng -c 1 -l 10 --vm 1 --vm-bytes $FREE_MEMORY'M'
	
	echo $FREE_MEMORY	
}

if [ $1 -eq 1 ] ; then
	cpu_fault

elif [ $1 -eq 2 ] ; then
	sudo kill -9 $(pgrep -f stress-ng)
	mem_fault
fi
