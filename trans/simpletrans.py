import theano
import theano.tensor as T
import numpy as np


adjmat = np.array([[0., 1.0, 1.0], [0.5, 0., 0.5], [1.0, 1.0, 0.]])
imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])

#This gives individual states over time
smat = np.dstack([np.ones((3,4))-imat,imat])

#This gives the evolution of a particular state
#over time, over rows of individuals
smat2 = smat.reshape(2,4,3)

#This is the transition matrix
tmat = np.array([[0.99, 0.01], [0.5, 0.5]])

#To get the transition rates at each timestep, 
#we do this:
transition_rates = np.dot(smat,tmat)

next_state_probs = transition_rates[:2]*smat[1:]
total_prob = np.sum(np.log(next_state_probs.sum(axis = 2)))


# foimat = np.dot(adjmat,imat)
# expmat = foimat*smat
print(expose(adjmat, imat))