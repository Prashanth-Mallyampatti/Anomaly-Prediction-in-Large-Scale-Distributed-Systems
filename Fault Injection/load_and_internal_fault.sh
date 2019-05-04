#Author: Prashanth Mallyampatti
#!/bin/bash

HOME=/home/ubuntu
HOST=localhost
YCSB_HOME=$HOME/YCSB
YCSB=$YCSB_HOME/bin/ycsb
KEYSPACE=ycsb
TABLE=usertable  # this is same as column-family
CQLSH=/usr/bin/cqlsh
YCSB_SETUP_SCRIPT=$HOME/setup-ycsb.cql
CLEANUP_SCRIPT=$HOME/cleanup-ycsb.cql
DELETE_SCRIPT=$HOME/delete-ycsb.cql


FIELD=10
WORKLOAD=workloads/workloada
RUN_RESULT_STORE=faultinjection_run_result.txt
LOAD_RESULT_STORE=faultinjection_load_result.txt

chmod 777 $YCSB_SETUP_SCRIPT
chmod 777 $CLEANUP_SCRIPT
chmod 777 $DELETE_SCRIPT
chmod 777 $HOME/*

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre/

CPU_CORE=`nproc --all`
if [ $CPU_CORE -eq 4 ] ; then
	set_params()
	{
		THREAD_COUNT=$1
		OPERATION_COUNT=$2
		RECORD_COUNT=$3
	}

	load_workload()
	{
		cd $YCSB_HOME
		$YCSB load cassandra-cql -p hosts=$HOST -threads $THREAD_COUNT -p fieldcount=$FIELD -p operationcount=$OPERATION_COUNT -p recordcount=$RECORD_COUNT -p requestdistribution=zipfian -P $WORKLOAD -s > $LOAD_RESULT_STORE
		cd $HOME
	}

	crash_cassandra()
	{
		#OutOfMemoryException
		set_params 100000 10000000 100000000
		run_workload
	}

	run_workload()
	{
		cd $YCSB_HOME 
		$YCSB run cassandra-cql -p hosts=$HOST -threads $THREAD_COUNT -p fieldcount=$FIELD -p operationcount=$OPERATION_COUNT -p recordcount=$RECORD_COUNT -p requestdistribution=zipfian -P $WORKLOAD -s  > $RUN_RESULT_STORE
    		cd $HOME
	}

	thirty()
	{
		set_params 2 70000 9999
		run_workload
	}

	fifty()
	{
		set_params 3 200000 9999
		run_workload
	}
	sixty()
	{
		set_params 4 120000 99999
		run_workload
	}
	eighty()
	{
		set_params 8 100000 99999
		run_workload
	}
	ninety()
	{
		set_params 128 900000 99999
		run_workload
	}

	
	#Setting dedicated core for workload
	IP_ADDR=`/sbin/ifconfig eth0 | grep 'inet' | cut -d: -f2 | awk '{ print $2}'` 
	#taskset -pc 1 $IP_ADDR

	echo "********************Setting Database*********************"
	echo $YCSB_SETUP_SCRIPT
   	$CQLSH -f $YCSB_SETUP_SCRIPT
	
	if [ $1 -eq 1 ] ; then
		echo "*******************************Running Workload***************************"
		thirty
		fifty
		sixty
		eighty
		ninety

	elif [ $1 -eq 2 ] ; then
		echo "******************************Loading Workload****************************"
		set_params 128 1000 2000000
		load_workload

	elif [ $1 -eq 3 ] ; then
		echo "******************************Dropping Table******************************"
		echo $DELETE_DATA_SCRIPT
		$CQLSH -f $DELETE_DATA_SCRIPT
	
	elif [ $1 -eq 4 ] ; then
		echo "******************************Dropping Keyspace***************************"
		$CQLSH -f $CLEANUP_SCRIPT
	
	elif [ $1 -eq 5 ]; then
		echo "******************************Crashing Cassandra**************************"
		crash_cassandra

	else
		echo "Enter Valid Arguments"
	fi
fi
