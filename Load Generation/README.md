# Workload Generation

Install YCSB:

    yes Y | bash ./install_ycsb.sh

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Make sure Maven latest build is installed.

<br />Workload Generation:

    yes Y | bash ./workload_generation.sh <argument>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '1' : for CPU workload. Starts from 30% goes till to 90+%. 
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '2' : for memory workload. Achieves a maximum of 37% of Memory usage with cassandra not being crashed.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '3' : to drop table.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '4' : to drop keyspace.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arguemnt '5' : to crash cassandra.


<br />For user specified workload:

    yes Y | bash ./load_generator.sh

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Change YSCB parameter values in lines 27-34 as per requirement.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To load workload : comment out line 75 and 87.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; To run workload  : comment out line 73 and 85.


<br /> To change table name and keyspace names:

&nbsp;&nbsp;&nbsp; Change in 'setup-ycsb.cql', 'cleanup-ycsb.cql', 'delete-ycsb.cql' accordingly.
