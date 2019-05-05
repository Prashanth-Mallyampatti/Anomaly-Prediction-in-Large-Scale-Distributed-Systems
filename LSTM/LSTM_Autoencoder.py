
# coding: utf-8

# ## LSTM Autoencoder Model for Real-time Anomaly Predictions in Distributed Systems

# In[1]:


# keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Bidirectional, Dense, RepeatVector, TimeDistributed, Input
from keras.utils import plot_model

#others
import pandas as pd
import numpy as np
from numpy import array
import pickle
import time
import sys
from collections import Counter
from slack_utils import send_to_slack
from dataExtraction import get_data_x_minutes_back_as_dataframe
from datetime import datetime

#Uncomment these two lines when running the system on terminal server
#import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#sklearn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ## Defining Simple Autoencoder Model

# In[ ]:


# Here both input and output sequence length should be same as we are just trying to reconstruct the sequence

def get_model(timesteps, input_dim):
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(Bidirectional(LSTM(100)))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(input_dim)))
    
    # adam optimizer with lr=0.001 is used.
    # For reconstruction error, mse is the metric we use
    
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    print(model.summary())
    return model


# In[ ]:


# Here both input and output sequence length should be same as we are just trying to reconstruct the sequence
# Sample model that was experimented with while hyperparameter tuning.
# Uncomment if you wish to try out this model

# def get_model(timesteps, input_dim):
#     model = Sequential()
#     model.add(LSTM(500, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))
#     model.add(Bidirectional(LSTM(256)))
#     model.add(RepeatVector(timesteps))
#     model.add(LSTM(100, activation='relu', return_sequences=True))
#     model.add(TimeDistributed(Dense(input_dim)))
    
#     # adam optimizer with lr=0.001 is used.
#     # For reconstruction error, mse is the metric we use
    
#     model.compile(optimizer='adam', loss='mse')
#     print(model.summary())
#     return model


# ## Defining a Predictor + Decoder LSTM Autoencoder model

# In[ ]:


# An advanced model that returns both previous states and current predictions
# An lstm autoencoder model to reconstruct and predict sequence

def model_advanced(look_back, look_ahead, n_features):
    visible = Input(shape=(look_back, n_features))
    encoder = LSTM(100, activation='relu')(visible)
    # define reconstruct decoder
    decoder1 = RepeatVector(look_back)(encoder)
    decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(n_features))(decoder1)
    # define predict decoder
    decoder2 = RepeatVector(look_ahead)(encoder)
    decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(n_features))(decoder2)
    # combine it all together
    model = Model(inputs=visible, outputs=[decoder1, decoder2])
    model.compile(optimizer='adam', loss='mse')
    return model


# ## Defining Standard Scaling Methods

# In[ ]:


def standardize(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler

def normalize(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    return data, scaler


# ## Creating Input/Output Sequences

# In[ ]:


'''
We create subsequences with a defined window size and 50% overlap. 
These sequences are fed to the LSTM Autoencoder model and the reconstruction errors are assessed.
Note: It is very important to reshape this sequence in the exact format as expected by our LSTM Autoencoder model.
Every sequence must be reshaped to have a shape of [number_of_sample, time_steps, num_features]
'''
def create_subsequence(sequence, n_features, window_size=10):
    
    start = 0
    data = list()
    while start < len(sequence):
        end = start + window_size
        
        if end > len(sequence):
            break
            
        chunk = sequence[start:end]
        # Using 50 % overlap for shifting window
        start = start + int(window_size/2)
        data.append(chunk)
	
    
    X = np.array(data)
    print("shape: ", X.shape)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X



def create_subsequences_for_prediction(sequence, n_features, window_size=10):
    
    start = 0
    data = list()
    while start < len(sequence):
        end = start + window_size
        
        if end > len(sequence):
            break
            
        chunk = sequence.iloc[start:end:,]
        #print(chunk.shape)
        # Using 50 % overlap for shifting window
        start = start + window_size
        data.append(chunk)
	
    return data


# ## Reconstruction Error and Threshold Computation

# ### Defining feature-wise reconstruction error computation

# In[ ]:


def calculate_reconstruction_errors(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
    


# In[ ]:


def calculate_reconstruction_errors_list_of_outputs(y_true, y_pred):
    reconstruction_errors = []
    for i, data in enumerate(y_true):
        
        predicted_value = y_pred[i]
        reconstruction_error = mean_squared_error(data, predicted_value)
        reconstruction_errors = np.append(reconstruction_errors, reconstruction_error)

    return reconstruction_errors


# ### Fitting a Gaussian Distribution

# In[ ]:


def fit_gaussian(error_vectors):
      
    # featurewise mean
    mean = np.mean(error_vectors, axis = 0 , dtype=np.float64)
    print("mean: ", mean)
    #computing standard deviation
    std = np.std(error_vectors, axis = 0 , dtype=np.float64)
    print("std: ", std)
    
    return mean, std


# ### Threshold computation

# In[ ]:


# sigma specifies the amount of deviation tolerated
def calculate_threshold(error_vectors, sigma=3):
    
    mean, std = fit_gaussian(get_error_vectors(error_vectors))
    reconstruction_error_threshold_low = mean - (sigma * std)
    reconstruction_error_threshold_high = mean + (sigma * std)
    
    #print(reconstruction_error_threshold_low)
    
    return reconstruction_error_threshold_low, reconstruction_error_threshold_high


# ### Check for anomalous states

# In[ ]:


'''
This method flags large deviations in the reconstruction errors observed in the reconstructed sequence.
If they are beyond the tolerance level identified by the defined threshold, an anomaly is flagged.
'''
def is_anomaly(error_vector, reconstruction_error_threshold_low, reconstruction_error_threshold_high):
    
    result = 0
    is_an_anomaly = False
    node_list = []
    index = 99999
    
    #print("threshold low: ", reconstruction_error_threshold_low)
    #print("threshold high: ", reconstruction_error_threshold_high)
    
    #print("Error: ", error_vector)
    
    num_samples= 1
    for seq in range(num_samples):
        error_array = np.array(error_vector[seq])
        
        #print("error: ", error_vector)
        
        #print("length: ", len(reconstruction_error_threshold_low))
        
        for i in range(len(reconstruction_error_threshold_low)):
        
            score = [1 if(error > reconstruction_error_threshold_high[i]) else 0 for error in error_array[:,i]]
            #print("score: ", score)
            c = Counter(score)
            if c[1] > 0:
                #print("Alert in Node: ", i)
                node_list.append(i)
                if score.index(1) < index:
                    index = score.index(1)
                is_an_anomaly |= True
    

        if is_an_anomaly:
            result = 1
    
    return result, node_list, index


# ## Prediction and Reconstruction error Interpretation

# In[ ]:


'''
This method flattens the errors observed in the window of 
input sequences into a list of error vectors.
'''
def get_error_vectors(errors):
    vectors = []
    for i in range(len(errors)):
        for j in range(errors[i].shape[0]):
            vectors.append(errors[i][j])
    error_vectors = np.array(vectors)
    
    return error_vectors


'''
This method provides a prediction on the reconstruction of input sequence. Additionally,
all reconstruction errors and computed and returned along with the predictions.
It essentially does a model.predict
'''
def predict(X, model, scaler):
    
    #print("Model loaded is: ", model.name)
    reconstructions = model.predict(X, verbose=0)
    predictions = scaler.inverse_transform(reconstructions)
    X = scaler.inverse_transform(X)
    
    #Compute errors in predictions
    errors = np.zeros((predictions.shape[0], predictions.shape[1], predictions.shape[2]))
    for i in range(predictions.shape[0]):
        errors[i] = np.abs(X[i] - predictions[i])
        #print("errors: ", errors[i])
    
    return errors, predictions


# ## MAIN TRAIN AND TEST PIPELINES

# In[ ]:


'''
This method represents the training phase of the algorithm.
Offline training is done on the collected metric data.
'''
def train_main(data_path, val_data_path):
    
    input_dim = 3 # set to the number of nodes
    timesteps = 60
    
    df = pd.read_csv(data_path)
    df_val = pd.read_csv(val_data_path)
 
        
    # Dictionary to store the models corresponding to differemt metrics. The importance of the metric can be configured 
    # by specifying the weight. Ensure that they all sum up to 1. Any additional metric can be added here and the same will
    # be picked up for analysis by our system.
    params = {} 
    params["cpu"] = {"filename" : "data_cpu_final.csv", "weight" : "0.5"}
    params["mem"] = {"filename" : "data_mem_final.csv", "weight" : "0.5"}
    
    
    for metric_type in params.keys():
        
        if(metric_type == "cpu"):
            data = df.iloc[:,1:4]
            val_data = df_val.iloc[:,1:4]
        else:
            data = df.iloc[:, 4:7]
            val_data = df_val.iloc[:,4:7]
        
        data.dropna(inplace = True)
        params = train_pipeline(data, val_data, metric_type, params, data_path, timesteps, input_dim)
    
    # dump the parameters, models and weights into the filesystem
    with open(data_path + "paramsDict", 'wb') as f:
        pickle.dump(params, f)
 

'''
This method encompasses the entire train pipeline
1. Standardize the data
2. Create subsequences and reshape them to be fed into the LSTM Autoencoder model
3. Create a model or load an existing one
4. Fit the model on the training data
5. Make predictions on the validation set to compute threshold or tolerance on reconstruction errors
6. Compute the thresholds on reconstruction error
7. Save the trained model and update the parameter dictionary with the saved model path
'''
def train_pipeline(data, val_data, metric_type, params, data_path, timesteps, input_dim):
    
    print("Inside train-pipeline: ", metric_type)
    
    # standardize
    data_standardized, scaler = standardize(data)
    
    # Create sequences
    X = create_subsequence(data_standardized, input_dim, window_size = timesteps)
    
    # create model 
    model = get_model(timesteps, input_dim)
    model.name = metric_type
    
    # Fit model
    history_callback = model.fit(X, X, batch_size=32, epochs=20, validation_split=0.10)
   
    # Plot the validation and train losses
#         fig = plt.figure()
#         ax = plt.subplot(111)
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         plt.xticks(range(1, 21, 1))
#         y = range(1, 21)
#         c1 = plt.plot(y, np.squeeze(history_callback.history['loss']), color="teal", label="Training")
#         c2 = plt.plot(y, np.squeeze(history_callback.history['val_loss']), color="orange", label="Validation")
#         ax.legend()
#         plt.title("Train vs Validation Loss")
#         plt.savefig(data_path + "trainval.png")
#         plt.show()
    
    # Plot the validation and train accuracies
#         fig = plt.figure()
#         ax = plt.subplot(111)
#         plt.xlabel("Epochs")
#         plt.ylabel("Accuracy")
#         plt.xticks(range(1, 21, 1))
#         y = range(1, 21)
#         c1 = plt.plot(y, np.squeeze(history_callback.history['acc']), color="teal", label="Training")
#         c2 = plt.plot(y, np.squeeze(history_callback.history['val_acc']), color="orange", label="Validation")
#         ax.legend()
#         plt.title("Train vs Validation Accuracy")
#         plt.savefig(data_path + "trainvalAcc.png")
#         plt.show()

    
    # Predict on train
    print("Metric type is :", metric_type)
    errors, reconstructions = predict(val_data, model, scaler)
    
    # Compute Threshold
    sigma = 3
    threshold_low, threshold_high = calculate_threshold(errors, sigma)
    
    # Save model
    model.save(data_path + metric_type + "_model.hdf5")
    
    # Add scaler, thresholds, model save path to the dictionary
    d = params.get(metric_type)
    d["scaler"] = scaler
    d["model_save_path"] = data_path + metric_type
    d["t_low"] = threshold_low
    d["t_high"] = threshold_high
    params[metric_type] = d
    
    return params


'''
This method has two major responsibilities:
1. Make predictions on the new incoming sequences
2. Compare the reconstruction errors alogn with the defined thresholds and flag if its an anomalous sequence.
'''

def test_pipeline(data, metric_type, params, input_dim, timesteps, metric_model):
    
    print("Making Predictions")
    #print("Metric type is :", metric_type)
    #print("Model loaded is: ", metric_model.name)
    
    # Predict on test
    errors, predictions = predict(data, metric_model, params.get(metric_type)["scaler"])
       
    # Check if anomaly
    return is_anomaly(errors, params.get(metric_type)["t_low"], params.get(metric_type)["t_high"])
  

'''
This method encompasses the entire test pipeline
1. Identify the CPU and memory SLO violation thresholds
2. Fetch real-time data from time-series database of Prometheus
3. Load the CPU and Memory LSTM Autoencoder models
4. Create subsequences and reshape them to be fed into the LSTM Autoencoder model
5. Standardize the data
6. Make predictions
7. Keep track of anomaly count. If 40 consecutive anomalous patterns observed, update the model to capture the context drift
8. Compute the new thresholds on reconstruction error
9. Save the trained model and update the parameter dictionary with the saved model path
'''
def test_main(train_data_path, input_dim=3, timesteps=60):
    
    
    # We load the train data to get the 99th percentile value for cpu and 95 percentile value for observed memory usage.
    # We set these values as thresholds for tagging SLO violations.
    
    dataset = pd.read_csv(train_data_path, index_col=0)
    metric = 'collectd_cpu_percent'
    cpu_slo_violation = [np.percentile(dataset[metric+'_node1'], 99), np.percentile(dataset[metric+'_node2'], 99), np.percentile(dataset[metric+'_node3'], 99)]
    metric = "memory"
    memory_slo_violation = [np.percentile(dataset[metric+'_node1'], 95), np.percentile(dataset[metric+'_node2'], 95), np.percentile(dataset[metric+'_node3'], 99)]
    metric_list = ["collectd_cpu_percent", "collectd_memory{memory='used'}", "collectd_memory{memory='cached'}", 
              "collectd_memory{memory='buffered'}"]
    
    print(cpu_slo_violation, memory_slo_violation)
    
    
    # Fetch real-time data from time-series database of Prometheus
    data = get_data_x_minutes_back_as_dataframe(metric_list, 2, datetime.now())
    
    # drop NA if any
    data.dropna(inplace = True)
    
    #load params dict
    params = pickle.load(open(data_path + "paramsDict", "rb" ))
    
    # Loading the CPU and Memory LSTM Autoencoder models
    model_path = params.get("cpu")["model_save_path"]
    cpu_model = load_model(model_path + "_model.hdf5")
    
    model_path = params.get("mem")["model_save_path"]
    mem_model = load_model(model_path + "_model.hdf5")
    
    num_features_in_data = 6
    X = create_subsequences_for_prediction(data, num_features_in_data, window_size = timesteps)
    
    anomaly_count = 0
    slo_count = 0
    start_timestamp = 0
    
    anomaly_score = 0
    metric_dict = {}
    metric_dict_slo = {}
    print("Starting predictions")
    idx = 0
    for data1 in X:
        anomaly_score = 0
        
        for metric_type in params.keys():
            
            if metric_type == "cpu":
                metric_data = data1.iloc[:,0:3]
                violation = cpu_slo_violation
                metric_model = cpu_model
                
            else:
                metric_data = data1.iloc[:,3:6]
                violation = memory_slo_violation
                metric_model = mem_model
            
            #print(data1)
            #print(metric_type)
            #print(metric_data)
            #print(metric_data)
            data_standardized = params.get(metric_type)["scaler"].transform(metric_data)
            data_standardized = data_standardized.reshape((1, timesteps, input_dim))
            
            start = time.time()
            score, node_list, index = test_pipeline(data_standardized, metric_type, params, input_dim, timesteps, metric_model) 
            end = time.time()
            print("Time taken : ", end - start)
            
            
            if(len(node_list) > 0):
                
                d = dict()
                d["Node_list"] = node_list
                d["time"] = data.index[idx + index]
                metric_dict[metric_type] = d
            
            anomaly_score += score * float(params.get(metric_type)["weight"])

        #print("anomaly_score: ", anomaly_score)
        
        # If anomoly score is greater than or equal to 0.5, we flag it as an anomaly
        is_anomaly_score = anomaly_score >= 0.5

        if is_anomaly_score:
            print("ANOMALY PREDICTION MADE. ALERT!")
            print("WATCH OUT FOR:", metric_dict)
            print()
            send_to_slack("ANOMALY PREDICTION MADE. CHECK "+ str(metric_dict))
            anomaly_count += 1
            
        
        # Checking for SLO violations
        metric_dict_slo = {}
        for metric_type in params.keys():
            
            if metric_type == "cpu":
                metric_data = data1.iloc[:,0:3]
                violation = cpu_slo_violation
            else:
                metric_data = data1.iloc[:,3:6]
                violation = memory_slo_violation
                
            index_of_slo_violation = 9999   
            for t in range(len(metric_data)):
                
                if (metric_data.iloc[t, 0] > violation[0]) or (metric_data.iloc[t, 1] > violation[1]) or (metric_data.iloc[t, 2] > violation[2]):
                    index_of_slo_violation = t
                    
            if(index_of_slo_violation < 9999):
                node_list_cpu = list()
                    
                if (metric_data.iloc[index_of_slo_violation, 0] >= violation[0]):
                    node_list_cpu.append(0)
                    
                if (metric_data.iloc[index_of_slo_violation, 1] >= violation[1]):
                    node_list_cpu.append(1)
                    
                if (metric_data.iloc[index_of_slo_violation, 2] >= violation[2]):
                    node_list_cpu.append(2)
                    
                    
                d = dict()
                d["Node_list"] = node_list_cpu
                d["time"] = data.index[idx + index_of_slo_violation]
                metric_dict_slo[metric_type] = d
         
        # Here, we identify if node list is empty or not. Only if its not empty, we shall
        # add it to the list
        add = 0
        for key in metric_dict_slo.keys():
            if(len(metric_dict_slo.get(key)["Node_list"]) > 0):
                add = 1
        
        
        slo_count = slo_count + add
        # If we observe 3 consecutive SLO violations, a violation is notified.
        if slo_count == 3:
            for key in metric_dict_slo.keys():
                if(len(metric_dict_slo.get(key)["Node_list"]) > 0):
                    print("SLO VIOLATION:", metric_dict_slo)
            slo_count = 0
                
                
        # increment index to proceed to the next window   
        idx = idx + timesteps
        
        
        
       # This part of the code handles context drift. If the LSTM Autoencoder model reports an
       # anomaly for 40 consecutive times, we identify this as possible shift in context and hence update the model and
       # threshold values.

       #if anomaly_count > 40:
            #update_model_and_thresholds(params, start_timestamp)

       #if anomaly_count == 1:
            #start_timestamp = data[0]

        #else:
            #anomaly_count = 0

            
            
## Online model update code. 
## Note: this code is experimental. It is currently being tested
def update_model_and_thresholds(params, start_timestamp, data_path=""):
    # get data from utility api, range query from start timestamp to now
    
    data = get_data_range(start_timestamp, datetime.now(), "cpu")
    data.dropna(inplace = True)
    params = train_pipeline(data, data, "cpu", params, data_path, 60, 3)
    
    data = get_data_range(start_timestamp, datetime.now(), "mem")
    data.dropna(inplace = True)
    params = train_pipeline(data, data, "mem", params, data_path, 60, 3)
    
    # dump the parameters, models and weights into the filesystem
    with open(data_path + "paramsDict", 'wb') as f:
        pickle.dump(params, f)
        


# ## Plotting predictions of the LSTM Autoencoder model

# In[ ]:


def plot_prediction(x, y_true, y_pred):
    
    """Plots the predictions.
    
    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """
    
    plt.figure(figsize=(12, 3))
    ax = plt.subplot(111)
    output_dim = x.shape[-1]
    #plt.xticks(range(0, len(past), 1))
    for j in range(output_dim):
        past = x[:, j] 
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j==0 else "_nolegend_"
        label2 = "True future values" if j==0 else "_nolegend_"
        label3 = "Predictions" if j==0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b",
                 label=label1, color="blue")
        plt.plot(range(len(past),
                 len(true)+len(past)), true, "x--b", label=label2, color="teal", markersize=12)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                 label=label3, color="orange")
    plt.legend(loc='best')
    plt.xlabel("Timesteps")
    plt.ylabel("Percentage")
    plt.title("Predictions v.s. true values")
    #ax.set_ylim(ymin=0, ymax = 20, auto = True)
    plt.show()


# ## Plot the model training and validation losses

# In[ ]:


def plot_scores(data1, data2, y, xlabel, ylabel, learning_rate, filename, label1, label2):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    c1 = plt.plot(y, np.squeeze(data1), color="teal", label=label1)
    c2 = plt.plot(y, np.squeeze(data2), color="orange", label=label2)
    ax.legend()
    plt.title("Learning rate =" + str(learning_rate))
    #plt.title("Optimizer = Adam(lr=0.0001)")
    plt.savefig(filename)
    plt.show()


# ## Calling the functions for test or train

# In[ ]:


if __name__ == "__main__":

    if sys.argv[1] == "train":
        
        # check if we have sufficient arguments
        if len(sys.argv) < 4:
        print("Insufficient arguments")
        sys.exit()
        
        data_path = sys.argv[2]
        val_data_path = sys.argv[3]
        train_main(data_path)
        
    if sys.argv[1] == "test":
        
        # check if we have sufficient arguments
        if len(sys.argv) < 3:
        print("Insufficient arguments")
        sys.exit()
        
        data_path = sys.argv[2]
        test_main(data_path)

