#Author: Prashanth Mallyampatti

#!/bin/bash

git clone https://github.com/brianfrankcooper/YCSB.git

cd YCSB/

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/jre/

mvn -pl com.yahoo.ycsb:cassandra-binding -am clean package
