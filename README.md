# Anomaly-Prediction-in-Large-Scale-Distributed-Systems

<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Performance anomaly prediction is crucial for long running, large scale distributed systems. Many existing monitoring systems are built to analyze system logs produced by distributed systems for troubleshooting and problem diagnosis. However, inspection of such logs are non-trivial owing to the difficulty in converting text logs to vectorized data. This becomes infeasible with the increasing scale and complexity of distributed systems. Few other effective methods employ statistical learning to detect performance anomalies. However, most existing schemes assume labelled training data which requires significant human effort to create annotations and can only handle previously seen anomalies. In this paper, we present two anomaly prediction algorithms based on Self Organizing Maps and Long Short-Term Memory networks. We implemented a prototype of our system on Amazon Web Services and conducted extensive experiments to model the system behavior of Cassandra. Our analysis and results show that both these algorithms pose minimal overhead on the system and are able to predict performance anomalies with high accuracy and achieve sufficient lead time in the process.

<br />System Setup, Prediction Model, and Data files have been divided into seperate folders according to their functionalities. Below are the links to each of those:

<br />System Setup:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. [Cassandra System Setup](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/Cassandra%20Setup)
<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. [Prometheus](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/Prometheus)
<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. [CollectD](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/Collectd)

Load Generation:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. [Load Generation](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/Load%20Generation)

Fault Injection:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. [Fault Injection](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/Fault%20Injection)

Models:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. [Long Short-Term Memory](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/LSTM)
<br />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. [Self Organizing Maps](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/SOM)

Data Sets:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. [Data sets](https://github.com/Prashanth-Mallyampatti/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/tree/master/Data_Sets)
