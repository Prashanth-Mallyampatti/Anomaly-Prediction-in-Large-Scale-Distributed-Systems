# ADS
Anamoly prediction

ifconfig for any 2 hosts and update $SEED with those IP addresses. By default I have considered ens33 as network interface, change it(line 78) according to your system's interface.

command to execute the script:

    yes Y | bash ./cassandra.sh
Check status of the cluster(root):

    nodetool status
    
or

    nodetool -p 7199 status

If any of the above two commands gives :
    
    nodetool: Failed to connect to '127.0.0.1:7199' - ConnectException: 'Connection refused (Connection refused)'.

Go to /etc/cassandra/cassandra-env.sh file, look for:
    
    #JVM_OPTS="$JVM_OPTS -DJava.rmi.server.hostname=<public name>"

Uncomment it and replace public name with 127.0.0.1 , So it should look like:

    JVM_OPTS="$JVM_OPTS -DJava.rmi.server.hostname=127.0.0.1"

and then restart cassandra service by executing the following command(root):

    systemctl restart cassandra



Stop/Start cassandra service(root):
    
    sudo service cassandra stop/start

Never ever remove cassandra once installed. Else the JAVA paths are lost even if cassandra is freshly installed. Same is applicable to JAVA 8.
