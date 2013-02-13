import numpy as np

#Define the states that an individual can be in, in 
#the order they should be indexed in
states = ["S", "E", "I", "R"]

#Provide a dictionary of IDs 



#Give initial numbers of individuals in each state

#Define the basic transition parameters
eps = 1.0 - np.exp(-0.5) #rate of transition to infectious from latent
gamma = 1.0 - np.exp(-0.5) #rate of transition to recovered from infectious

#Trying a simple SEIR model where transitions
#are governed by the transition matrix, T below
T = np.array([[0.0, 0.0, 0.0, 0.0],
			[0.0, 1.0-eps, eps, 0.0],
			[0.0, 0.0, 1.0-gamma, gamma],
			[0.0, 0.0, 0.0, 1.0]]
			)