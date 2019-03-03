#!/usr/bin/bash

#Author: Prashanth Mallymapatti

# script for 3 nodes, 2 seeds cassandra set up

HOST1='172.16.49.140'
SEED=("172.16.49.143" "172.16.49.143")


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

echo "HOST3 IP-tables successfully set\n"
echo


#uninstall existing java
sudo update-alternatives --remove "java" "/usr/lib/jvm/jdk1.8.0_191/bin/java"
sudo update-alternatives --remove "javac" "/usr/lib/jvm/jdk1.8.0_191/bin/javac"
sudo update-alternatives --remove "javaws" "/usr/lib/jvm/jdk1.8.0_191/bin/javaws"

sudo rm -r /usr/lib/jvm/jdk1.8.0_191
sudo apt-get remove openjdk*
sudo apt-get purge --auto-remove openjdk*


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


#uninstall existing cassandra
sudo apt-get remove cassandra
sudo rm -rf /var/lib/cassandra
sudo rm -rf /var/log/cassandra
sudo rm -rf /etc/cassandra


#Install cassandra


sudo apt install curl

echo "deb http://www.apache.org/dist/cassandra/debian 311x main" | sudo tee -a /etc/apt/sources.list.d/cassandra.sources.list

curl https://www.apache.org/dist/cassandra/KEYS | sudo apt-key add -

sudo apt-get -y update

#single cassandra install isn't updating the config files
sudo apt-get -y install cassandra=2.2.6
sudo apt-get -y install cassandra
sudo apt-get -y install cassandra-tools
sudo service cassandra stop

#remove any data created so as to update the config files
sudo rm -rf /var/lib/cassandra/data/system*

#Get your own IP address
IP_ADDR=`/sbin/ifconfig ens33 | grep 'inet' | cut -d: -f2 | awk '{ print $2}'`


#Update Cassandra Config File

function join { local IFS="$1"; shift; echo -e "$*"; }
ALL_SEEDS=`join "," ${SEED[@]}`

#echo "Joined seeds:"
#echo $ALL_SEEDS

# Update seed ip addresses to the config file
sudo sed -i 's/- seeds: "127.0.0.1"/- seeds: "'${ALL_SEEDS[@]}'"/g' /etc/cassandra/cassandra.yaml

#Update listen address to host's IP address
sudo sed -i 's/listen_address: localhost/listen_address: '${IP_ADDR}'/g' /etc/cassandra/cassandra.yaml

#Update RPC address to host's IP address
sudo sed -i 's/rpc_address: localhost/rpc_address: '${IP_ADDR}'/g' /etc/cassandra/cassandra.yaml

#Remove if any auto_bootstraps added
sudo sed -i '${/auto_bootstrap/d;}' /etc/cassandra/cassandra.yaml

#Add new auto_bootstrap: false for seed nodes and true for non-seed node 
if [[ ${SEED[*]} =~ $IP_ADDR ]]; then
	sudo sh -c "echo 'auto_bootstrap: false' >> /etc/cassandra/cassandra.yaml "
else
	sudo sh -c "echo 'auto_bootstrap: true' >> /etc/cassandra/cassandra.yaml "
fi


sudo service cassandra stop
sudo rm -rf /var/lib/cassandra/data/system*

#Start Cassandra Service
sudo service cassandra start

echo
echo
echo "Cassandra Host: $IP_ADDR setup successful!"
echo
