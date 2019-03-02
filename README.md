# ADS
Anamoly prediction

ifconfig for any 2 hosts and update $SEED with those IP addresses. By default I have considered ens33 as network interface, change it(line 78) according to your system's interface.

command to execute the script:

    yes Y | bash ./cassandra.sh
Check status of the cluster(root):

    nodetool status
    
or

    nodetool -p 7199 status
  
Stop/Start cassandra service(root):
    
    sudo service cassandra stop/start
