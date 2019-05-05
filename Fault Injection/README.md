# Fault injection using YCSB:

    yes Y | bash ./load_and_internal_fault.sh <argument>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '1' : for increasing CPU workload. 
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Starts from 30% goes till to 90+%. CPU usage of 80+% is considered to be a CPU fault.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '2' : for memory workload. Achieves a maximum of 37% of Memory usage with cassandra not being crashed.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '3' : to drop table.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '4' : to drop keyspace.
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;arguemnt '5' : to crash cassandra.

    
# Fault Injection using Stress-ng:

    yes Y | bash ./external_fault.sh <argument>
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '1' : for CPU fault.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;argument '2' : for Memory fault.
