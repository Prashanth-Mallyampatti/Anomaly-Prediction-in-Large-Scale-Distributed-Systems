# Self Organising Maps

Step 1: Run each cell of the [`trainSOM.ipynb`](https://github.ncsu.edu/rnarang/ADS_SOM/blob/master/trainSOM.ipynb), this will train the model and create a pickle file of the trained model and the iPython notebook. Make sure that the training data is present in the same directory as the IPython notebook.

Step 2: Start the monitoring system. Copy the saved pickle files to the monitoring server. Also, copy the training data CSV file on the server.

Step 3: You may need to install the python libraries which are not present on the server. Simply use `pip` to install the required libraries.

Step 4: Run the [`predictSOM.py`](https://github.ncsu.edu/rnarang/ADS_SOM/blob/master/predictSOM.py) file using the command `python3 predictSOM.py`

Step 5: Create a cron job so that the script runs periodically. The cron job we created made the script run every two minutes.
