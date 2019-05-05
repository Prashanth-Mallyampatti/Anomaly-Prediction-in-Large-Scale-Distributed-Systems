Setup 3-node cassandra cluster

<br /> Open these ports: `22`, `7000`, `7001`, `7199`, `9042`, `9160`, `9142` 
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;which are SSH port, Cassandra inter-node cluster communication, Cassandra SSL inter-node cluster communication, Cassandra JMX monitoring port, Cassandra client port, Cassandra client port (Thrift), Default for native_transport_port_ssl respectively.

<br />To deploy Cassandra on AWS add `All traffic` on all ports `0 - 65535`as a rule in the security group for all the instances

<br /> Alternatively you can disable firewall:
    
    sudo apt install ufw
    
    sudo ufw disable
    
    
<br /> Setup 3-node cluster: Choose any 2 hosts as seed nodes and add IP addresses in `$SEED`. By default `eth0` is considered as network interface.

    yes Y | bash ./cassandra.sh
    
<br />Check status of the cluster:

    nodetool status
    
or

    nodetool -p 7199 status
    
<br />Stop/Start cassandra service:
    
    sudo service cassandra stop/start
    

<br />To safely remove a node (decommissioning):

    yes Y | bash ./decommission_current_node.sh

<br />To remove a dead node: 

    yes Y | bash ./remove_dead_nodes.sh
    
<br />To clean dead nodes data:

    yes Y | bash ./clean_node_data.sh
   
