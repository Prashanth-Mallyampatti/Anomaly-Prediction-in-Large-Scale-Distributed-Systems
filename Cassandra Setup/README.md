Setup 3-node cassandra cluster:
    
    yes Y | bash ./cassandra.sh
    
<br />Check status of the cluster:

    nodetool status   
    
or

    nodetool -p 7199 status
    
<br />Stop/Start cassandra service:
    
    sudo service cassandra stop/start
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Choose any 2 hosts as seed nodes and add IP addresses in $SEED. By default ens33 is considered as network interface.

<br />To safely remove a node (decommissioning):

    yes Y | bash ./decommission_current_node.sh

<br />To remove a dead node: 

    yes Y | bash ./remove_dead_nodes.sh
    
<br />To clean dead nodes data:

    yes Y | bash ./clean_node_data.sh
   
