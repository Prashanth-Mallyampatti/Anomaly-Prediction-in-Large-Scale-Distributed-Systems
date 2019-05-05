#!/usr/bin/bash

#Author: Prashanth Mallyampatti

# script for 3 nodes, 2 seeds cassandra set up

HOST1='172.16.49.149'
SEED=("172.16.49.150" "172.16.49.151")


# update IP table for host1
sudo iptables -A INPUT -p tcp -s ${SEED[0]} -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

sudo iptables -A INPUT -p tcp -s ${SEED[1]} -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

echo "HOST1 IP-table successfully set"
echo

#update IP table for host2
sudo iptables -A INPUT -p tcp -s $HOST1 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

sudo iptables -A INPUT -p tcp -s ${SEED[1]} -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

echo "HOST2 IP-table successfully set"
echo

#update IP table for host3
sudo iptables -A INPUT -p tcp -s $HOST1 -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

sudo iptables -A INPUT -p tcp -s ${SEED[0]} -m multiport --dports 7000,9042 -m state --state NEW,ESTABLISHED -j ACCEPT

echo "HOST3 IP-tables successfully set"
echo

#intall JAVA 8 and JNA
#newer java version has a problem with cassandra
echo "Installing JAVA 8"
echo

sudo apt-get -y install  openjdk-8-jre
sudo apt-get -y install  openjdk-8-jdk
sudo update-alternatives --config java

sudo sh -c "echo 'JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java"' >> /etc/environment "
source /etc/environment

echo  "Installing JNA"
echo

sudo apt-get install libjna-java

echo  "Setting JAVA Path"
echo
#setting java path
export JAVA_HOME=/usr/


sudo apt install curl

#Download Debian packages

echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list

curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -

sudo apt-get -y update

#Install Cassandra and Cassandr Tools
sudo apt-get -y install cassandra
sudo apt-get -y install cassandra-tools
sudo service cassandra stop

#remove any data created so as to update the config files
sudo rm -rf /var/lib/cassandra/data

#Get your own IP address
IP_ADDR=`/sbin/ifconfig eth0 | grep 'inet' | cut -d: -f2 | awk '{ print $2}'`


#Update Cassandra Config File

function join { local IFS="$1"; shift; echo -e "$*"; }
ALL_SEEDS=`join "," ${SEED[@]}`

#echo "Joined seeds:"
#echo $ALL_SEEDS

# Update seed ip addresses to the config file
sudo sed -i 's/- seeds: .*/- seeds: "'${ALL_SEEDS[@]}'"/g' /etc/cassandra/cassandra.yaml

#Update listen address to host's IP address
sudo sed -i 's/listen_address: .*/listen_address: '${IP_ADDR}'/g' /etc/cassandra/cassandra.yaml

#Update RPC address to host's IP address
sudo sed -i 's/rpc_address: .*/rpc_address: '${IP_ADDR}'/g' /etc/cassandra/cassandra.yaml

#Remove if any auto_bootstraps added
sudo sed -i '${/auto_bootstrap: .*/d;}' /etc/cassandra/cassandra.yaml

#Add new auto_bootstrap: false for seed nodes and true for non-seed node 
if [[ ${SEED[*]} =~ $IP_ADDR ]]; then
	sudo sh -c "echo 'auto_bootstrap: false' >> /etc/cassandra/cassandra.yaml "
else
	sudo sh -c "echo 'auto_bootstrap: true' >> /etc/cassandra/cassandra.yaml "
fi


sudo service cassandra stop
sudo rm -rf /var/lib/cassandra/data

#Start Cassandra Service
sudo service cassandra start

echo
echo
echo "Cassandra Host: $IP_ADDR setup successful!"
echo
