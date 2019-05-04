"""
@author: Rajat Narang

For Online Anomaly Predictions
"""

from dataExtraction import get_data_x_minutes_back_as_dataframe
from numpySOM import load
import numpy as np
from datetime import datetime
import pandas as pd


def get_normal_neighbors(bmu, som, MID_map, p):
    """
    Helper function to return normal neighbors of a given neuron
    
    Input:
    
    bmu: tuple
        the given neuron for which we have to find normal neighbors
        
    som: Object
        Trained SOM object
        
    MID_map: Matrix
        Mean inter-neuron distance map, each cell in this map is the normalised sum of distance
        between a neuron and its neighbors
    
    p: float
        neighborhood area size percentile threshold
    """
    Q = 5
    normal_neighbors = []
    for neigh_size in range(1, len(som.neigx)):
        col_min = max(bmu[0]-neigh_size, 0)
        col_max = min(len(som.neigx)-1, bmu[0]+neigh_size)

        row_min = max(bmu[1]-neigh_size, 0)
        row_max = min(len(som.neigy)-1, bmu[1]+neigh_size)
        for i in range(col_min, col_max+1):
            for j in range(row_min, row_max+1):
                mid = MID_map[i, j]
                if mid < p:
                    if (i, j) not in normal_neighbors:
                        normal_neighbors.append((i, j))
                        Q -= 1
                    if Q <=0:
                        return normal_neighbors
    return normal_neighbors


def anomaly_cause_inference(bmu, som, MID_map, p):
    """
    Tells which metric is the top contributor to the anomaly
    
    Input:
    
    bmu: tuple
        the given neuron for which we have to find normal neighbors
        
    som: Object
        Trained SOM object
        
    MID_map: Matrix
        Mean inter-neuron distance map, each cell in this map is the normalised sum of distance
        between a neuron and its neighbors
    
    p: float
        neighborhood area size percentile threshold
    """
    nearest_normal_neighbors = get_normal_neighbors(bmu, som, MID_map, p)
    cause = []
    for nnn in nearest_normal_neighbors:
        diff = abs(som.weights[bmu] - som.weights[nnn])
        cause.append(np.argmax(diff))
    return max(set(cause), key=cause.count)


def predict_anomaly(dataset, metric_list):
    """
    Primary function which is used to make anomaly prediction
    """
    anomaly_type = ['CPU', 'Memory']
    
    #load the SOM and Min-Max Scalar
    som = load('cassandra_som.pkl')
    sc = load('cassandra_sc.pkl')
    
    # get the mean inter-neuron distance map
    MID_map = som.get_MID_map()
    
    # Find the neighborhood area size threshold
    all_mids = []
    x_dim, y_dim = len(som.neigx), len(som.neigy)
    for i in range(x_dim):
        for j in range(y_dim):
            all_mids.append(MID_map[i][j])
    p = np.percentile(all_mids, 85)
    
    
    X_test = dataset.values
    N = len(X_test)
    cnt = 0
    cpu_alert_sent, memory_alert_sent = False, False
    slo_violation_occured = False
    cpu_slo_cnt, memory_slo_cnt = 0, 0
    cpu_violation_alerted, memory_violation_alerted = False, False
    
    for i in range(N):
        # Check if an SLO violation occurred 
        cpu_violation, memory_violation = False, False
        if (X_test[i][0] > cpu_slo_violation or X_test[i][1] > cpu_slo_violation or X_test[i][2] > cpu_slo_violation):
            cpu_violation = True
        if (X_test[i][3] > memory_slo_violation or X_test[i][4] > memory_slo_violation 
            or X_test[i][5] > memory_slo_violation):
            memory_violation = True
            
        if cpu_violation:
            cpu_slo_cnt += 1
        else:
            cpu_slo_cnt = 0

        if memory_violation:
            memory_slo_cnt += 1
        else:
            memory_slo_cnt = 0
            
        if cpu_slo_cnt == 3 and not cpu_violation_alerted:
            print(str(dataset.index[i])+ ': CPU SLO Violation')
            cpu_violation_alerted = True
            slo_violation_occured = True
        if memory_slo_cnt == 3 and not memory_violation_alerted:
            print(str(dataset.index[i])+ ': Memory SLO Violation')
            memory_violation_alerted = True
            slo_violation_occured = True
        
        #Make prediction: whether or not an anomaly would occur
        X_test_i = [X_test[i]]
        X_test_i_transformed = sc.transform(X_test_i)
        bmu = som.best_matching_unit(X_test_i_transformed)
        mid = MID_map[bmu]
        if mid>p:
            at = anomaly_cause_inference(bmu, som, MID_map, p)//3
            if not cpu_alert_sent or not memory_alert_sent:
                if at == 0 and not cpu_alert_sent:
                    print(str(dataset.index[i])+ ": Anomaly Predicted - "+anomaly_type[at])
                    cpu_alert_sent = True
                if at == 1 and not memory_alert_sent:
                    print(str(dataset.index[i])+ ": Anomaly Predicted - "+anomaly_type[at])
                    memory_alert_sent = True
                    



if __name__ == "__main__":
    dataset = pd.read_csv('cassandra_data_23April.csv', index_col=0)
    
    # Get CPU SLO violation threshold
    metric = 'collectd_cpu_percent'
    all_rows = dataset[metric+'_node1'].append(dataset[metric+'_node2']).append(dataset[metric+'_node3'])
    cpu_slo_violation = np.percentile(all_rows, 99)
    
    # Get Memory SLO violation threshold
    metric = "memory"
    all_rows = dataset[metric+'_node1'].append(dataset[metric+'_node2']).append(dataset[metric+'_node3'])
    memory_slo_violation = np.percentile(all_rows, 99)
    
    # Fetch real time data
    metric_list = ["collectd_cpu_percent", "collectd_memory{memory='used'}", "collectd_memory{memory='cached'}", 
              "collectd_memory{memory='buffered'}"]
    dataset = get_data_x_minutes_back_as_dataframe(metric_list, 2, datetime.now())
    
    # Predict Anomaly
    predict_anomaly(dataset, metric_list)

