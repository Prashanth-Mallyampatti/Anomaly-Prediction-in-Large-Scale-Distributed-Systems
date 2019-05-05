Setup 3-node cassandra cluster:
    
    yes Y | bash ./cassandra.sh
    
Check status of the cluster:

    nodetool status
    
or

    nodetool -p 7199 status
    
Stop/Start cassandra service:
    
    sudo service cassandra stop/start
    
Choose any 2 hosts as seed nodes and add IP addresses in $SEED. By default ens33 is considered as network interface.

To safely remove a node (decommissioning):

    yes Y | bash ./decommission_current_node.sh

To remove a dead node: 

    yes Y | bash ./remove_dead_nodes.sh
    
To clean dead nodes data:

    yes Y | bash ./clean_node_data.sh
   
