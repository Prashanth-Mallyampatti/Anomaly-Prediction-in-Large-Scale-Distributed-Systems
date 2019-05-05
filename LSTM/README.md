# LSTM Autoencoders

Here, we have implemented an `LSTM Autoencoder` based model for real-time anomaly predictions in large scale distributed systems.

Our approach operates on a very simple premise that any anomalous usage pattern is not a normal system behavior. We essentially try to identify instances or patterns in data that deviate from normal behavior. As a solution, we propose an LSTM-Autoencoder model.  In order to model temporal system metrics data, we design an LSTM Autoencoder, which functions very similar to the usual autoencoders, with the only difference being that the encoder and decoder components are LSTM networks instead of multilayer perceptrons. The encoder learns a vector representation of the input time-series and the decoder uses this representation to reconstruct the time-series with the objective function being the minimization of the reconstruction errors.

We jointly train both the encoder and decoder components of the LSTM Autoencoder to reconstruct the input sequences which represent usage patterns of normal system behavior.
In order to identify performance anomalies, the online LSTM Autoencoder model continuously analyzes the stream of resource usage patterns and tries to reconstruct these input sequences. 

The intuition here is that since the algorithm operates in an unsupervised fashion, the model will be trained only on normal usage patterns i.e., patterns which are expected to be seen in the normal functioning of the system. Hence, the LSTM encoder-decoder pair would only have seen normal instances during training and would have learnt to reconstruct them. When fed an anomalous sequence, it might not be able to reconstruct it well, leading to higher reconstruction errors compared to the reconstruction errors for the normal sequences. These reconstruction errors are then used to obtain a likelihood of incoming pattern being anomalous. By identifying a threshold on the reconstruction error tolerance, we make our predictions.


<br />Implementation:

The `LSTM_Autoencoder.ipynb` notebook and `LSTM_Autoencoder.html` contains a detailed explanation of every step along the way starting from the sequence generation, training procedure, threshold determination up till anomaly prediction. The corresponding python code also contains detailed comments. These comments will be sufficient to help you understand the code. Should you face any issues, please send a note to the authors.

<br />Running the training phase:

    python LSTM_Autoencoder.py train <path_to_train_data> <path_to_val_data>

<br />Running the testing and inference part:

    python LSTM_Autoencoder.py test <path_to_train_data>

Ensure that the folder structure is maintained exactly as its done in this repository.
