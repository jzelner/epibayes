# Model test assuming full knowledge of stuff
# Created 1.29.13

import numpy as np

# starting parameters
t = 10
dt = 1
steps = round(dt*t)


# use model using numpy
def model_numpy(T, ):
	
	pass



def model_theano(T,):
	pass


# def createObservations(N, nStates, T, dt):
# 	observations = [[[] for j in range(N)] for i in range(nStates)]

# 	steps = round(T*dt)
# 	for i in range(steps):
# 		observations[]


# model parameters
beta = .01
gamma = 1/3.0

# Transition Matrix
T = np.array([[1.0, 0.0], [gamma, 1.0-gamma]])



# Observation Matrix
X_S = np.array(
   [[1, 1, 0, 0, 0, 1, 1],
	[0, 0, 1, 1, 1, 1, 0],
	[0, 1, 1, 0, 0, 0, 1]], dtype=int)
#X_S = np.array([[1,0,0],[0,1,0]], dtype=int)

X_I = np.ones(X_S.shape, dtype=int)
X_I = X_I - X_S

X = np.array([X_S, X_I])
#print X

# Adjaceny Matrix with 3 individuals
A = np.array(
	[[0,1,1],[1,0,1],[1,1,0]], dtype=int
	)
#A = np.array([[0,1],[1,0]], dtype=int)

# Transmission Matrix
E_S = np.array(
	[[0,0],[0,0]], dtype=int
	)

E_I = np.array([[0,1],[0,0]], dtype=int)

E = np.array([E_S, E_I], dtype=int)
#print A.shape

print X_S*np.dot(A,X_I)