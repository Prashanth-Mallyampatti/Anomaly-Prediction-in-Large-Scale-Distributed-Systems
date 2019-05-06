############## YCSB LOAD GENERATOR FOR CASSANDRA ################
# This script provides the basic commands and setup needed to load and run a workload for performance testing on Cassandra database.
# Author : Varsha Nagarajan
#################################################################

#!/bin/bash

#################################################################################
################ CASSANDRA SYSTEM PATH CONFIGURATION ############################
#################################################################################

HOME=/home/ubuntu
HOST=172.31.45.113,172.31.43.172,172.31.46.249  # the load generator ensures that the it is run on all the nodes in the cluster. So specify any one host only
YCSB_HOME=$HOME/ycsb-0.15.0
YCSB=$YCSB_HOME/bin/ycsb
KEYSPACE=ycsb
TABLE=usertable  # this is same as column-family
CQLSH=/usr/bin/cqlsh
YCSB_SETUP_SCRIPT=$HOME/setup-ycsb.cql
CLEANUP_SCRIPT=$HOME/cleanup-ycsb.cql
DELETE_DATA_SCRIPT=$HOME/delete-ycsb.cql

#################################################################################
###################### LOAD GENERATOR CONFIGURATION PARAMS ######################
#################################################################################
#operation count 200000 and record count 9000 gives around 60 % cpu and 25-37 % memory
THREAD_COUNT=3
THREAD_COUNT_LOW_WL=10
FIELD=10
OPERATION_COUNT=200000
WORKLOAD=workloads/workloada
RECORD_COUNT=9999
RUN_RESULT_STORE=workloadA_low_run_result.txt
LOAD_RESULT_STORE=workloadA_low_load_result.txt


#################################################################################
############################# PERMISSIONS #######################################
#################################################################################

#touch YCSB_SETUP_SCRIPT
#chown user:user /*  # This is necessary for files to be accessible by any user running the test
chmod 777 $YCSB_SETUP_SCRIPT
chmod 777 $CLEANUP_SCRIPT
chmod 777 $DELETE_DATA_SCRIPT
chmod 777 $HOME/*

#################################################################################
#################################################################################
#################################################################################

setup_database()
{
    # TODO: Port cqlsh to cqlsh.py style
    echo $YCSB_SETUP_SCRIPT
    $CQLSH -f $YCSB_SETUP_SCRIPT
}

cleanup()
{
    $CQLSH -f $CLEANUP_SCRIPT
}

delete_data()
{
    $CQLSH -f $DELETE_DATA_SCRIPT
    $CQLSH -f $YCSB_SETUP_SCRIPT
}

load_and_run_workload()
{
    cd $YCSB_HOME
    $YCSB load cassandra-cql -p hosts=$HOST -threads $THREAD_COUNT -p fieldcount=$FIELD -p operationcount=$OPERATION_COUNT -p recordcount=$RECORD_COUNT -p requestdistribution=zipfian -P $WORKLOAD -s > $LOAD_RESULT_STORE
    $YCSB run cassandra-cql -p hosts=$HOST -threads $THREAD_COUNT -p fieldcount=$FIELD -p operationcount=$OPERATION_COUNT -p recordcount=$RECORD_COUNT -p requestdistribution=zipfian -P $WORKLOAD -s  > $RUN_RESULT_STORE
    cd $HOME
}

run_workload()
{
    cd $YCSB_HOME
    $YCSB run cassandra-cql -p hosts=$HOST -threads $THREAD_COUNT -p fieldcount=$FIELD -p operationcount=$OPERATION_COUNT -p recordcount=$RECORD_COUNT -p requestdistribution=zipfian -P $WORKLOAD -s  > $RUN_RESULT_STORE
    cd $HOME
}


load_and_run_small_workload()
{
    #echo LOADING WORKLOAD
    cd $YCSB_HOME
    #echo `pwd`
    $YCSB load cassandra-cql -p hosts=$HOST -P $WORKLOAD -s > $LOAD_RESULT_STORE
    #echo RUNNING WORKLOAD
    $YCSB run cassandra-cql -threads $THREAD_COUNT_LOW_WL -p hosts=$HOST -P $WORKLOAD -s  > $RUN_RESULT_STORE
    cd $HOME
}

#delete_data
#rm -rf /var/lib/cassandra/data/ycsb/*

echo SETTING UP DATABASE TO START LOAD GENERATION
setup_database
echo LOADING AND RUNNING WORKLOAD PHASE STARTS
#load_and_run_workload
#load_and_run_small_workload
for i in `seq 1 20`;
do
    load_and_run_workload
done 

#echo DONE. CLEARING ALL ENTRIES AND EXITING
#delete_data


rm -rf /var/lib/cassandra/data/ycsb/*
