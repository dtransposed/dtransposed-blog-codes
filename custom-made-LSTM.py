import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM

## Define parameters ##

timesteps = 30          # Defines how long in time each batch is; one time step is one point of observation in the batch.
no_of_batches = 200     # A batch is an input sequence to the network; it is comprised of one or more points of observation.
no_of_layers = 3        # Number of hidden LSTM layers in the network.
no_of_units = 20        # Number of units in every LSTM layer.

## Define weights matrices ##
'''
This function allows us to extract weights matrices in form of numpy arrays from a model in Keras. 
Key of the dictionary - name of a matrix
e.g. LSTM1_i_W is the matrix W of the input gate for the first LSTM layer.
Value of the key - numpy array associated with the matrix
'''

def import_weights(no_of_layers, hidden_units):
    layer_no = 0
    for index in range(1, no_of_layers+1):
        for matrix_type in ['W', 'U', 'b']:
            if matrix_type != 'b':
                weights_dictionary["LSTM{0}_i_{1}".format(index, matrix_type)] = model_weights[layer_no][:,:hidden_units]
                weights_dictionary["LSTM{0}_f_{1}".format(index, matrix_type)] = model_weights[layer_no][:,hidden_units:hidden_units * 2]
                weights_dictionary["LSTM{0}_c_{1}".format(index, matrix_type)] = model_weights[layer_no][:,hidden_units * 2:hidden_units * 3]
                weights_dictionary["LSTM{0}_o_{1}".format(index, matrix_type)] = model_weights[layer_no][:,hidden_units * 3:]  
                layer_no = layer_no + 1
            else:
                weights_dictionary["LSTM{0}_i_{1}".format(index, matrix_type)] = model_weights[layer_no][:hidden_units]
                weights_dictionary["LSTM{0}_f_{1}".format(index, matrix_type)] = model_weights[layer_no][hidden_units:hidden_units * 2]
                weights_dictionary["LSTM{0}_c_{1}".format(index, matrix_type)] = model_weights[layer_no][hidden_units * 2:hidden_units * 3]
                weights_dictionary["LSTM{0}_o_{1}".format(index, matrix_type)] = model_weights[layer_no][hidden_units * 3:]  
                layer_no = layer_no + 1
                
    weights_dictionary["W_dense"] = model_weights[layer_no]
    weights_dictionary["b_dense"] = model_weights[layer_no + 1]
    
## Define LSTM network ##
'''
Keras_LSTM creates an LSTM network (Keras implementation)
custom_LSTM creates a single LSTM layer ('custom-made' implementation)
'''

class LSTM_Keras(object):  
    def __init__(self, no_hidden_units, timesteps):
        self.timesteps = timesteps
        self.no_hidden_units = no_hidden_units       
        model = Sequential()
        model.add(LSTM(units = self.no_hidden_units, return_sequences = True, input_shape = (self.timesteps, 1)))
        model.add(LSTM(units = self.no_hidden_units, return_sequences = True))
        model.add(LSTM(units = self.no_hidden_units, return_sequences = False))
        model.add(Dense(units = 1))
        self.model = model
    
class custom_LSTM(object):
    
    def __init__(self, timesteps, no_of_units):
        self.timesteps = timesteps
        self.no_hidden_units = no_of_units
        self.hidden = np.zeros((self.timesteps, self.no_hidden_units),dtype = np.float32)
        self.cell_state = np.zeros((self.timesteps, self.no_hidden_units),dtype = np.float32)
        self.output_array=[]
        
    def hard_sigmoid(self, x):
        slope = 0.2
        shift = 0.5
        x = (x * slope) + shift
        x = np.clip(x, 0, 1)
        return x

    def tanh(self, x):
        return np.tanh(x)
        
    def layer(self, xt, Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc):
        ft = self.hard_sigmoid(np.dot(xt, Wf) + np.dot(self.hidden, Uf) + bf)
        it = self.hard_sigmoid(np.dot(xt, Wi) + np.dot(self.hidden, Ui) + bi)
        ot = self.hard_sigmoid(np.dot(xt, Wo) + np.dot(self.hidden, Uo) + bo)
        ct = (ft * self.cell_state)+(it * self.tanh(np.dot(xt, Wc) + np.dot(self.hidden, Uc) + bc))
        ht = ot * self.tanh(ct)
        self.hidden = ht
        self.cell_state = ct
        return self.hidden

    def reset_state(self):
        self.hidden = np.zeros((self.timesteps, self.no_hidden_units),dtype = np.float32)
        self.cell_state = np.zeros((self.timesteps, self.no_hidden_units),dtype = np.float32)
        
    def output(self, x, weights, bias):
        result = np.dot(x, weights)+bias
        self.result=result[0]
        return result[0]
    
    def output_array_append(self):
        self.output_array.append(self.result[0])
            
## Define dense layer ##     
'''
This function takes the output from the last LSTM layer and returns the result from the neural network.
'''
    
def output_calc(x, weights, bias):
    result = np.dot(x, weights)+bias
    return result

## Main ##
    
keras_model = LSTM_Keras(no_of_units,timesteps)                                 #create a model in Keras
model_weights = keras_model.model.get_weights()                                 #get weights of a network from the model
weights_dictionary = {}                                                         #create an empty dictionary
import_weights(no_of_layers, no_of_units)                                       #fill the dictionary with weights matrices from Keras

LSTM_layer_1 = custom_LSTM(timesteps, no_of_units)
LSTM_layer_2 = custom_LSTM(timesteps, no_of_units)
LSTM_layer_3 = custom_LSTM(timesteps, no_of_units)                              #create three LSTM layers

input_to_keras = np.random.randint(100, size = (no_of_batches, timesteps, 1))   #input is a sequence of randomly initialized samples

## Prediction step using custom-made LSTM ##

for batch in range(input_to_keras.shape[0]):

    LSTM_layer_1.reset_state()
    LSTM_layer_2.reset_state()
    LSTM_layer_3.reset_state()
    
    for timestep in range(input_to_keras.shape[1]):
        
        output_from_LSTM_1 = LSTM_layer_1.layer(input_to_keras[batch,timestep,:], weights_dictionary['LSTM1_f_W'], weights_dictionary['LSTM1_i_W'], weights_dictionary['LSTM1_o_W'], weights_dictionary['LSTM1_c_W'],
                                                                                  weights_dictionary['LSTM1_f_U'], weights_dictionary['LSTM1_i_U'], weights_dictionary['LSTM1_o_U'], weights_dictionary['LSTM1_c_U'],
                                                                                  weights_dictionary['LSTM1_f_b'], weights_dictionary['LSTM1_i_b'], weights_dictionary['LSTM1_o_b'], weights_dictionary['LSTM1_c_b'])
        
        output_from_LSTM_2 = LSTM_layer_2.layer(output_from_LSTM_1, weights_dictionary['LSTM2_f_W'], weights_dictionary['LSTM2_i_W'], weights_dictionary['LSTM2_o_W'], weights_dictionary['LSTM2_c_W'],
                                                                    weights_dictionary['LSTM2_f_U'], weights_dictionary['LSTM2_i_U'], weights_dictionary['LSTM2_o_U'], weights_dictionary['LSTM2_c_U'],
                                                                    weights_dictionary['LSTM2_f_b'], weights_dictionary['LSTM2_i_b'], weights_dictionary['LSTM2_o_b'], weights_dictionary['LSTM2_c_b'])
        
        output_from_LSTM_3 = LSTM_layer_3.layer(output_from_LSTM_2, weights_dictionary['LSTM3_f_W'], weights_dictionary['LSTM3_i_W'], weights_dictionary['LSTM3_o_W'], weights_dictionary['LSTM3_c_W'],
                                                                    weights_dictionary['LSTM3_f_U'], weights_dictionary['LSTM3_i_U'], weights_dictionary['LSTM3_o_U'], weights_dictionary['LSTM3_c_U'],
                                                                    weights_dictionary['LSTM3_f_b'], weights_dictionary['LSTM3_i_b'], weights_dictionary['LSTM3_o_b'], weights_dictionary['LSTM3_c_b'])
    
    LSTM_layer_3.output(output_from_LSTM_3, weights_dictionary['W_dense'], weights_dictionary['b_dense'])
    LSTM_layer_3.output_array_append()

## Compare custom-made implementation  ##
result_custom=LSTM_layer_3.output_array
result_keras=keras_model.model.predict(input_to_keras)

plt.plot(result_custom, 'b')
plt.plot(result_keras, 'r')
plt.legend(loc='best')
plt.show()

