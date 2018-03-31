import numpy as np
import matplotlib.pyplot as plt
import seaborn
from mpl_toolkits.mplot3d import Axes3D

class HMM(object):
    
    def __init__(self, transition_matrix_matrix,current_state):
        self.transition_matrix = transition_matrix
        self.current_state = current_state
      
    def filtering(self,observation_matrix):
        new_state = np.dot(observation_matrix,np.dot(self.transition_matrix,self.current_state))
        new_state_normalized = new_state/np.sum(new_state)
        self.current_state = new_state_normalized
        return new_state_normalized
    
    def prediction(self):
        new_state = np.dot(self.transition_matrix,self.current_state)
        new_state_normalized = new_state/np.sum(new_state)
        self.current_state=new_state_normalized
        return new_state_normalized

    def plot_state(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        xpos = [0,0,1,2,2,1]
        ypos = [0,1,1,1,0,0]
        zpos = np.zeros(len(initial_state.shape))
        dx = np.ones(len(initial_state.shape))
        dy = np.ones(len(initial_state.shape))
        dz = self.current_state      
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#ce8900')
        ax1.set_xticks([0., 1., 2.,3.])   
        ax1.set_yticks([0., 1., 2.]) 
        plt.show()
        
    def create_observation_matrix(self,error_rate, no_discrepancies):
        sensor_list=[]
        for number in no_discrepancies:
            probability=(1-error_rate)**(4-number)*error_rate**number
            sensor_list.append(probability)
            observation_matrix = np.zeros((len(sensor_list),len(sensor_list)))
            np.fill_diagonal(observation_matrix,sensor_list)
        return observation_matrix

#   define two models
transition_matrix = np.array([[0.2,0.4,0,0,0,0],
                              [0.8,0.2,0.267,0,0,0],
                              [0,0.4,0.2,0.4,0,0.8],
                              [0,0,0.267,0.2,0.8,0],
                              [0,0,0,0.4,0.2,0],
                              [0,0,0.267,0,0,0.2]]) 
    
initial_state=np.array([1/6,1/6,1/6,1/6,1/6,1/6])

Model = HMM(transition_matrix,initial_state)
Model2 = HMM(transition_matrix,initial_state)

#   create observation matrices
observation_matrix_SWE = Model.create_observation_matrix(0.25,[0,3,4,3,0,0])
observation_matrix_NW = Model.create_observation_matrix(0.25,[3,0,1,2,3,3])
observation_matrix_N = Model.create_observation_matrix(0.25, [4,1,0,1,4,4])
observation_matrix_NE = Model.create_observation_matrix(0.25, [3,2,1,0,3,3])

#   localize of the robot using filtering
state_1 = Model.filtering(observation_matrix_SWE)
Model.plot_state()
state_2 = Model.filtering(observation_matrix_NW)
Model.plot_state()
state_3 = Model.filtering(observation_matrix_N)
Model.plot_state()
state_4 = Model.filtering(observation_matrix_NE)
Model.plot_state()
state_5 = Model.filtering(observation_matrix_SWE)
Model.plot_state()

#   localize of the robot using filtering (three first timesteps) and prediction (two last timesteps)
state_6 = Model2.filtering(observation_matrix_SWE)
Model2.plot_state()
state_7 = Model2.filtering(observation_matrix_NW)
Model2.plot_state()
state_8 = Model2.filtering(observation_matrix_N)
Model2.plot_state()
prediction_1 = Model2.prediction()
Model2.plot_state()
prediction_2 = Model2.prediction()
Model2.plot_state()
