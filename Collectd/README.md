# Collectd 


Install `collectd clients` on all nodes of the distributed system and `collectd server` on a monitoring server.

Note : We observed that the current collectd and generic-jmx plugin have been compiled using higher versions of `java` due to which certain features do not work in our current setup. So, in order to reproduce our setup, do replace `collectd` jar and `generic-jmx` jar with [collectd-api.jar](https://github.ncsu.edu/pmallya/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/blob/master/Collectd/collectd-api.jar) and [generic-jmx.jar](https://github.ncsu.edu/pmallya/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/blob/master/Collectd/generic-jmx.jar) respectively which are provided in this repository.

Before starting with the installation, the following symbolic link must be created as collectd is unable to locate the `libjvm.so` file. Based on your java installation, execute the one that seems relevant. If you have been following our repository for the system setup, then we highly recommend you work with the same java versions as given below. Only these versions have been tested for compatibility with the the entire end-to-end system.

    ln -s /usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/libjvm.so /usr/lib/libjvm.so
or
    
    ln -s /usr/lib/jvm/java-8-oracle/jre/lib/amd64/server/libjvm.so /usr/lib/libjvm.so


<br />Collectd client and server installation:

1. Collect Client:

       sudo apt-get update
    
       sudo apt-get install collectd 

       sudo apt-get install collectd-utils


    Replace `collectd.conf` file found in `./etc/collectd/collectd.conf` with the one provided in this repository [collectd_client.conf](https://github.ncsu.edu/pmallya/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/blob/master/Collectd/collectd_client.conf) (Take a backup of the old file).
). 
Rename this new  file to `collectd.conf`.

    1. Provide the IP address of the server in order to send the collected metric data to the `collectd server`. This can be done by the loading the `Network Plugin` of `collectd`. Change the IP address to that of your monitoring server.
    2. Uncomment any of the other metric data, in case you wish to collect that too.

    
    At this stage if you find a `.csv` file (if csv plugin is enabled) and `rrd` folder, your `collectd client` installation was successful.

2. Collectd Server:

       sudo apt-get update
    
       sudo apt-get install collectd 
    
       sudo apt-get install collectd-utils

    Replace `collectd.conf` file found in `./etc/collectd/collectd.conf` with the one provided in this repository [collectd_server.conf](https://github.ncsu.edu/pmallya/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/blob/master/Collectd/collectd_server.conf) (Take a backup of the old file). 
Rename this new  file to `collectd.conf`.


    1. Go to the `Network Plugin` of `collectd` and change the IP address to your monitoring server IP address. Ensure to use
the same network interface, client is able to ping the server at the configured IP.

    At this stage if you find a `.csv` file (if csv plugin is enabled) and `rrd` folder, your `collectd server` installation was successful.
