"""
@author: Rajat Narang

Python implementation of Solf-Organising Maps(SOM)
"""
import math
import numpy as np
import collections
import pickle

# Function to save the model using pickle
def save(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

#Helper function to load the model
def load(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)


class SOM(object):
    def __init__(self, M, N, noof_features, radius=1.0, learning_rate=0.5, random_seed=None):
        """
        Initialise the self-organising map
        
        Input:
        M: integer
            x dimension of the SOM
            
        N: integer
            y dimension of the SOM
            
        noof_features: integer
            Number of features in the input
            
        radius: integer
            radius of the neighborhood function
        
        learning rate: float
            current learning rate
            
        random seed: float
            random seed to use
            
        """
        self.random_generator = np.random.RandomState(random_seed)

        self.learning_rate = learning_rate
        self.radius = radius
        self.noof_features = noof_features
        
        # Random initialisation of weights
        self.weights = self.random_generator.rand(M, N, noof_features)*2-1

        for i in range(M):
            for j in range(N):
                # Normalise the weights
                norm = self.get_L2Norm(self.weights[i, j])
                self.weights[i, j] = self.weights[i, j]/norm

        self.activation_map = np.zeros((M, N))
        self.neigx = np.arange(M)
        self.neigy = np.arange(N)
        self.decay_function = self.asymptotic_decay
        self.neighborhood = self.gaussian




    def get_L2Norm(self, x):
        """
        returns the L2-norm of the 1-D numpy vector x
        
        Input:
        x: numpy vector
            The vector to normalise
        """
        x_transpose = x.T
        dot_product = np.dot(x, x_transpose)
        sqrt_dot_product = math.sqrt(dot_product)
        return sqrt_dot_product


    def asymptotic_decay(self, learning_rate, current_iteration, max_iterations):
        """
        Decay function
        
        Input:
        
        learning rate: float
            current learning rate
            
        current_iteration: int
        
        max_iterations: int
            maximum number of iterations
        
        """
        denominator = 1 + current_iteration / (max_iterations / 2)
        return learning_rate / denominator


    def activate(self, x):
        """
        activation_map is the matrix in which element i,j is the response of the neuron i,j to x.
        This function updates activation_map
        
        Input:
        
        x: numpy vector
            input vector
        """
        difference = x - self.weights
        itr = np.nditer(self.activation_map, flags=['multi_index'])
        while not itr.finished:
            self.activation_map[itr.multi_index] = self.get_L2Norm(difference[itr.multi_index])
            itr.iternext()

    # Returns the gaussian centered at c with radius equal to "radius"
    def gaussian(self, c, radius):
        d = 2*np.pi*radius*radius
        ax = np.exp(-np.power(self.neigx-c[0], 2)/d)
        ay = np.exp(-np.power(self.neigy-c[1], 2)/d)
        return np.outer(ax, ay)

    
    # Check if the input data id of the correct dimensions
    def check_input_len(self, data):
        data_len = len(data[0])
        if self.noof_features != data_len:
            msg = 'Expected %d features, received %d.' % (self.noof_features, data_len)
            raise ValueError(msg)

    
    def best_matching_unit(self, x):
        """
        Returns the best matching unit for a given sample x
        
        Input: numpy vector
            Sample x
        """
        self.activate(x)
        min_activation = self.activation_map.argmin()
        shape_activation_map = self.activation_map.shape
        return np.unravel_index(min_activation, shape_activation_map)


    def update(self, x, bmu, current_iteration, max_iterations):
        """
        Update the weights of the nodes of the SOM
        
        Input:
        
        x: numpy array
            Sample input
            
        bmu: array
            coordinates of the winning neuron for sample x
            
        current_iteration: int
            current iteration
        
        max_iterations: int
            maximum number of iterations
        """
        LR = self.decay_function(self.learning_rate, current_iteration, max_iterations)
        rad = self.decay_function(self.radius, current_iteration, max_iterations)

        gradient = self.neighborhood(bmu, rad)*LR
        it = np.nditer(gradient, flags=['multi_index'])

        while not it.finished:
            x_w = np.subtract(x, self.weights[it.multi_index])
            self.weights[it.multi_index] = self.weights[it.multi_index] + gradient[it.multi_index] * x_w
            it.iternext()


    def randomly_initialise_weights(self, data):
        """
        Pick random samples from the data to initialise the weights of the SOM
        """
        self.check_input_len(data)
        it = np.nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self.random_generator.randint(len(data))
            self.weights[it.multi_index] = data[rand_i]
            it.iternext()

    # Initialise the weights to span the first two principle components
    def initialise_weights(self, data):
        eigenvalues, eigenvectors= np.linalg.eig(np.cov(np.transpose(data)))
        principal_components = np.argsort(eigenvalues)
        primary_pc = principal_components[0]
        secondary_pc = principal_components[1]
        for i, c1 in enumerate(np.linspace(-1, 1, len(self.neigx))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self.neigy))):
                self.weights[i, j] = c1*eigenvectors[primary_pc] + c2*eigenvectors[secondary_pc]


    def train_random(self, data, total_iterations):
        """
        Trains the SOM by picking random sample from the data
        
        Input:
        
        data: numpy array
            Data Matrix
            
        total_iterations: integer
            number of iterations
        """
        self.check_input_len(data)

        for iteration in range(total_iterations):
            rand_i = self.random_generator.randint(len(data))
            x = data[rand_i]
            bmu = self.best_matching_unit(data[rand_i])
            self.update(x, bmu, iteration, total_iterations)

    # Reutns the mean inter-neuron distance map
    # Each cell in this map is the normalised sum of distance
    # between a neuron and its neighbors
    def get_MID_map(self):
        M, N = self.weights.shape[0], self.weights.shape[1]
        MID_map = np.zeros((M, N))
        itr = np.nditer(MID_map, flags=['multi_index'])
        while not itr.finished:
            for ii in range(itr.multi_index[0]-1, itr.multi_index[0]+2):
                for jj in range(itr.multi_index[1]-1, itr.multi_index[1]+2):
                    if ii >= 0 and ii < M and jj >= 0 and jj < N:
                        w_1 = self.weights[ii, jj, :]
                        w_2 = self.weights[itr.multi_index]
                        MID_map[itr.multi_index] = MID_map[itr.multi_index] + self.get_L2Norm(np.subtract(w_1,w_2))
            itr.iternext()
        MID_map = MID_map/MID_map.max()
        return MID_map

    # This helper functions computes the quantisation error
    # which is the average distance between each sample input and
    # its best matching unit
    def quantization_error(self, data):
        self.check_input_len(data)
        error = 0
        for x in data:
            diff = np.subtract(x, self.weights[self.best_matching_unit(x)])
            error = error + self.get_L2Norm(diff)
        return error/len(data)

    def threshold_based_quantization_error(self, x, threshold1, threshold2):
        """
        This function was used to handle concept drift, this tells whether or not
        during prediction time x(an input sample) has significant information that
        it should be used to update the map.
        """
        bmu = self.best_matching_unit(x)
        bmu_weights = self.weights[bmu]
        QE = np.subtract(x, bmu_weights)
        TEV = [ qe if qe >=threshold1 else 0 for qe in QE]
        QEB = sum(TEV)
        if QEB >= threshold2:
            return 1
        return 0


    def bmu_to_data_points(self, data):
        """
        Returns a dictionary where the key is the coordinates of a neuron and the 
        value is the list of input samples which got mapped to this neuron
        """
        self.check_input_len(data)
        bmu_map = collections.defaultdict(list)
        for x in data:
            bmu_map[self.best_matching_unit(x)].append(x)
        return bmu_map

