"""
@author: Rajat Narang

This file contains all the wrapper functions to extract data from prometheus.
"""

import csv
import requests
from datetime import datetime, timedelta
import pandas as pd
import calendar
import time


def get_df_from_json(json_data, metric_name):
    """
    To convert the data reveived in JSON format to dataframe
    """
    metrics = json_data.get('data', {}).get('result', [])
    node1, node2, node3 = metrics[0], metrics[1], metrics[2]
    node1_values = node1['values']
    node1_timestamp = [datetime.fromtimestamp(val[0]).replace(microsecond=0) for val in node1_values]
    node1_val = [float(val[1]) for val in node1_values]

    node2_values = node2['values']
    node2_timestamp = [datetime.fromtimestamp(val[0]).replace(microsecond=0) for val in node2_values]
    node2_val = [float(val[1]) for val in node2_values]

    node3_values = node3['values']
    node3_timestamp = [datetime.fromtimestamp(val[0]).replace(microsecond=0) for val in node3_values]
    node3_val = [float(val[1]) for val in node3_values]
    
    ser1 = pd.Series(node1_val, index=node1_timestamp, name=metric_name+"_node1")
    ser2 = pd.Series(node2_val, index=node2_timestamp, name=metric_name+"_node2")
    ser3 = pd.Series(node3_val, index=node3_timestamp, name=metric_name+"_node3")
    
    df = pd.concat([ser1, ser2, ser3], axis=1)
    return df


def get_data_range(start_timestamp, end_timestamp, metric, step):
    """
    Given a start timestamp, an end timestamp, metric name and step size, 
    fetch data in this range from Prometheus. Used the endpoint range query here.
    """
    query_params = {
        'query': metric,
        'start': start_timestamp,
        'end': end_timestamp,
        'step': step
    }
    response = requests.get('http://localhost:8080/api/v1/query_range', params=query_params)
    results = response.json()
    df = get_df_from_json(results, metric)
    return df


def get_data_x_minutes_back(metric, x, end_time=None):
    """
    Given a metric name, fetch last x minutes of data from a given time
    
    Input:
    metric: String
        Metric to extract from prometheus
    x: int
        minutes of data to extract
    end_time: datetime object
        
    """
    if not end_time:
        end_time = datetime.now()
    end_timestamp = time.mktime(end_time.timetuple())
    query_params = {
        'query': metric+'['+str(x)+'m]',
        'time': end_timestamp
    }
    response = requests.get('http://localhost:8080/api/v1/query', params=query_params)
    results = response.json()
    return results


def get_data_all(metric):
    """
    Wrapper function to extract all the data from prometheus.
    Get one hour of data at a time, till you fetch all the data
    """
    end_time = datetime.now()
    while True:
        json_response = get_data_x_minutes_back(metric, 60, end_time)
        data = json_response.get('data', {}).get('result', [])
        end_time = end_time - timedelta(seconds=60*60)
        if len(data) > 0:
            break
            
    dataframe = get_df_from_json(json_response, metric)
    while True:
        json_response = get_data_x_minutes_back(metric, 60, end_time)
        data = json_response.get('data', {}).get('result', [])
        if len(data) == 0:
            break
        temp = get_df_from_json(json_response, metric)
        dataframe = temp.append(dataframe)
        end_time = end_time - timedelta(seconds=60*60)
    return dataframe


def clean_dataframe(dataframe):
    """
    Wrapper function to aggregate memory_used, memory_cached and memory_buffered into one
    metric called memory
    
    Input:
    
    dataframe: Pandas dataframe object
        Raw dataframe
    """
    dataframe['memory_node1'] = dataframe["collectd_memory{memory='used'}_node1"]+dataframe["collectd_memory{memory='cached'}_node1"]+dataframe["collectd_memory{memory='buffered'}_node1"]
    dataframe['memory_node2'] = dataframe["collectd_memory{memory='used'}_node2"]+dataframe["collectd_memory{memory='cached'}_node2"]+dataframe["collectd_memory{memory='buffered'}_node2"]
    dataframe['memory_node3'] = dataframe["collectd_memory{memory='used'}_node3"]+dataframe["collectd_memory{memory='cached'}_node3"]+dataframe["collectd_memory{memory='buffered'}_node3"]
    
    del dataframe["collectd_memory{memory='used'}_node1"]
    del dataframe["collectd_memory{memory='used'}_node2"]
    del dataframe["collectd_memory{memory='used'}_node3"]

    del dataframe["collectd_memory{memory='cached'}_node1"]
    del dataframe["collectd_memory{memory='cached'}_node2"]
    del dataframe["collectd_memory{memory='cached'}_node3"]

    del dataframe["collectd_memory{memory='buffered'}_node1"]
    del dataframe["collectd_memory{memory='buffered'}_node2"]
    del dataframe["collectd_memory{memory='buffered'}_node3"]
    return dataframe


def get_all_data_as_dataframe(metric_list):
    """
    Given a metric list, return the dataframe that'll be fetched to the prediction engine
    """
    res = pd.DataFrame()
    for metric in metric_list:
        print(metric)
        df = get_data_all(metric)
        if 'memory' in metric:
            df = df/(1024*1024*1024)
        res = pd.concat([res, df], axis=1)
    res.fillna(method='bfill',  inplace=True)
    res.fillna(method='ffill',  inplace=True)
    return clean_dataframe(res)


def get_data_range_as_dataframe(metric_list, start_timestamp, end_timestamp, step):
    """
    Given a metric list, return the dataframe that'll be fetched to the prediction engine.
    Data here would be in the given range, start_timestamp to end_timestamp
    """
    res = pd.DataFrame()
    for metric in metric_list:
        print(metric)
        df = get_data_range(start_timestamp, end_timestamp, metric, step)
        if 'memory' in metric:
            df = df/(1024*1024*1024)
        res = pd.concat([res, df], axis=1)
    res.fillna(method='bfill',  inplace=True)
    res.fillna(method='ffill',  inplace=True)
    return clean_dataframe(res)


def get_data_x_minutes_back_as_dataframe(metric_list, x, end_time):
    """
    Given a metric list, return the dataframe that'll be fetched to the prediction engine.
    Data here is the last x minutes of data from the given end_time
    """
    res = pd.DataFrame()
    for metric in metric_list:
        print(metric)
        json_response = get_data_x_minutes_back(metric, x, end_time)
        df = get_df_from_json(json_response, metric)
        if 'memory' in metric:
            df = df/(1024*1024*1024)
        res = pd.concat([res, df], axis=1)
    res.fillna(method='bfill',  inplace=True)
    res.fillna(method='ffill',  inplace=True)
    return clean_dataframe(res)



