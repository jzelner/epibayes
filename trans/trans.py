#This file defines symbolic tensors and functions for calculating 
#the likelihood of the transmission model

import theano
import theano.tensor as T
from theano import printing
import numpy as np
import time


#The distance matrix gives the pairwise
#distances between locations
distance = T.dmatrix('distance')

#Population is a vector of population
#sizes by village
population = T.dvector('population')
sender_tau = T.dscalar('sender_tau')
receiver_tau = T.dscalar('receiver_tau')
distance_tau = T.dscalar('distance_tau')

gravity_fn = T.outer(population**sender_tau, population**receiver_tau) / (distance ** distance_tau) * (1.0 - T.eye(distance.shape[0], distance.shape[1]))

#This function returns the gravity matrix giving the pairwise weights of 
#interaction between locations, so that the ij-th entry reflects the exposure
#of location i to infections in location j.
gravity = theano.function([population, distance, sender_tau, receiver_tau, distance_tau], gravity_fn)

#The contact matrix gives the weight of 
#contact between pairs of nodes
contact = T.dmatrix('adjacency')
I = T.dmatrix("Infected")
foi = T.dmatrix("FOI")
initPredictors = T.dvector("init_predictors")

#The dot product of the adjacency matrix
#and the underlying infection state matrix gives
#the  force of infection on the 
#susceptible individuals, represented by 1 - I,
#since this is just an SIS model.

#Exposure is a function that takes an adjacency
#matrix and an infection state matrix and returns
#the amount of exposure on each susceptible 
#individual.
exposure = theano.function([contact, I], (T.dot(contact, I))*(1.0-I))


#stateLLStep takes an exposure matrix, initial state occupation probabilities,
#and the states matrix and returns the log likelihood of observing the sequences
#of states in the state matrix
def stateLLStep(lastState, thisState, predictors, g):
	#If we're in state 0 (susceptible), then the rate of transition
	#out is going to be dictated by b or 1-b. If we're in 1 (infectious)
	#the rate of transition out is going to be dictated by g & 1-g.
	logit = predictors + (lastState * g) 
	event_prob = 1.0 / (1.0 + T.exp(-logit))

	#If the state is the same as the last, then nothing happened and the prob
	#is 1-eventprob, otherwise it's the event probability.
	nextProb = T.log(T.switch(T.eq(thisState,0), 1.0 - event_prob, event_prob))

	return nextProb


#Defining symbolic variables for all of the inputs to the 
#likelihood function:
initPredictors = T.dvector("init_predictors")
statePredictors = T.dmatrix("state_predictors")
stateMatrix = T.dmatrix("state_matrix")
g = T.dscalar("state_autocorr")


#Given initial infection probabilities (ip), transition predictors (sp),
#state series (ss), and logit for autocorrelation, this fn returns the 
#likelihood of a single individual's infection series
def stateSeriesProb(ip, sp, ss, g):
	#Get the probability of the first element of the state series
	initProb = 1.0 / (1.0 + T.exp(-ip))
	start_prob = T.log(T.switch(T.eq(ss[0], 0), 1.0 - ip, ip))

	#Define a scan op that goes over the remaining elements of the state
	#series and applies the stateLLStep to each one
	result, updates = theano.map(fn = stateLLStep, non_sequences = [g], sequences = [ss[0:-1], ss[1:], sp])

	return T.sum(result)


#Map op that computes the series likelihood for all series in sequence and 
#returns a vector of log-probabilities
all_series_prob, all_updates = theano.map(fn = stateSeriesProb, non_sequences = [g], sequences = [initPredictors, statePredictors, stateMatrix])

#The total probability is the sum over all locations (i.e., rows)
#in the state matrix
total_prob = T.sum(all_series_prob)

#Wrap this up into a function that returns the total probability when 
#given the predictors, state predictors, state matrix and autocorrelation.
multiSeriesProb = theano.function(inputs = [initPredictors, statePredictors, stateMatrix, g], outputs = total_prob, updates = all_updates)



if __name__ == '__main__':
	
	#imat is a matrix where i,t = 1 indicates that the pathogen was 
	#circulating in location i at time t.
	imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])

	#To get the mirror image of the infection matrix, just subtract
	#it from a matrix of ones
	smat = np.ones((3,4))-imat

	#This gives the pairwise distances between locations
	distance_mat = np.array([[1., 10., 50.], [4., 1., 30.], [10.0, 15.0, 1.]])

	#A vector of location population sizes
	population = np.array([100., 20., 30.])

	#A vector of initial state occupation probabilities, expressed a logits
	initLogit = np.array([-0.05, -0.0005, -0.0001])

	#Obtain the gravity relationships between locations by calling gravity
	grav = gravity(population, distance_mat, 0.5, 0.1, 2.0)

	print("Gravity")
	print(grav)

	print("Infectiousness")
	print(imat)
	
	print("Exposures")
	exposure = expose(grav, imat)

	#Get the probability of the states in imat by calling seriesProb 
	multiSeriesProb(initLogit, exposure, imat, 0.9)
