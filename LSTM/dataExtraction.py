
# coding: utf-8

# In[1]:


import csv
import requests
from datetime import datetime, timedelta
import pandas as pd
import calendar
import time


# In[2]:


def get_df_from_json(json_data, metric_name):
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


# In[3]:


def get_data_range(start_timestamp, end_timestamp, metric, step):
    query_params = {
        'query': metric,
        'start': start_timestamp,
        'end': end_timestamp,
        'step': step
    }
    response = requests.get('http://localhost:9090/api/v1/query_range', params=query_params)
    results = response.json()
    df = get_df_from_json(results, metric)
    return df


# In[4]:


def get_data_x_minutes_back(metric, x, end_time=None):
    if not end_time:
        end_time = datetime.now()
    end_timestamp = time.mktime(end_time.timetuple())
    query_params = {
        'query': metric+'['+str(x)+'m]',
        'time': end_timestamp
    }
    response = requests.get('http://localhost:9090/api/v1/query', params=query_params)
    results = response.json()
    return results


# In[5]:


'''
Get one hour of data at a time, till you fetch all the data
'''
def get_data_all(metric):
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


# In[6]:


def clean_dataframe(dataframe):
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


# In[7]:


def get_all_data_as_dataframe(metric_list):
    res = pd.DataFrame()
    for metric in metric_list:
        #print(metric)
        df = get_data_all(metric)
        if 'memory' in metric:
            df = df/(1024*1024*1024)
        res = pd.concat([res, df], axis=1)
    res.fillna(method='bfill',  inplace=True)
    res.fillna(method='ffill',  inplace=True)
    return clean_dataframe(res)


# In[8]:


#metric_list = ["collectd_cpu_percent", "collectd_memory{memory='used'}", "collectd_memory{memory='cached'}", 
#               "collectd_memory{memory='buffered'}"]
#dataframe = get_all_data_as_dataframe(metric_list)


# In[9]:


#dataframe.to_csv('cassandra_data_23April.csv')


# In[10]:


def get_data_range_as_dataframe(metric_list, start_timestamp, end_timestamp, step):
    res = pd.DataFrame()
    for metric in metric_list:
        #print(metric)
        df = get_data_range(start_timestamp, end_timestamp, metric, step)
        if 'memory' in metric:
            df = df/(1024*1024*1024)
        res = pd.concat([res, df], axis=1)
    res.fillna(method='bfill',  inplace=True)
    res.fillna(method='ffill',  inplace=True)
    return clean_dataframe(res)


# In[11]:


# start_timestamp, end_timestamp, step = 1555652940, 1555663740, 10
# dataframe = get_data_range_as_dataframe(metric_list, start_timestamp, end_timestamp, step)


# In[12]:


def get_data_x_minutes_back_as_dataframe(metric_list, x, end_time):
    res = pd.DataFrame()
    for metric in metric_list:
        #print(metric)
        json_response = get_data_x_minutes_back(metric, x, end_time)
        df = get_df_from_json(json_response, metric)
        if 'memory' in metric:
            df = df/(1024*1024*1024)
        res = pd.concat([res, df], axis=1)
    res.fillna(method='bfill',  inplace=True)
    res.fillna(method='ffill',  inplace=True)
    return clean_dataframe(res)


# In[13]:


#end_time = datetime.now()
#dataframe = get_data_x_minutes_back_as_dataframe(metric_list, 2, end_time)
def main():
    metric_list = ["collectd_cpu_percent", "collectd_memory{memory='used'}", "collectd_memory{memory='cached'}",
            "collectd_memory{memory='buffered'}"]
    #dataframe = get_all_data_as_dataframe(metric_list)
    dataframe = get_data_x_minutes_back_as_dataframe(metric_list, 30, datetime.now())


# In[9]:


    dataframe.to_csv('new_val.csv')

if __name__ == "__main__":
    main()

