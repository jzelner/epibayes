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

gravity = theano.function([population, distance, sender_tau, receiver_tau, distance_tau], gravity_fn)

#The contact matrix gives the weight of 
#contact between pairs of nodes
contact = T.dmatrix('adjacency')
I = T.dmatrix("Infected")
foi = T.dmatrix("FOI")
initPredictors = T.dvector("init_predictors")

#The dot product of the adjacency matrix
#and the underlying state matrix gives
#the  force of infection on the 
#susceptible individuals, represented by 1 - I,
#since this is just an SIS model.
exposure = (T.dot(contact, I))*(1.0-I)

#Expose is a function that takes an adjacency
#matrix and an infection state matrix and returns
#the amount of exposure on each susceptible 
#individual.
expose = theano.function([contact, I], exposure)


#StateLogProb is a function that takes the exposure 
#matrix returned by expose, initial state occupation probabilities,
#and the states of individuals in the population
#and returns a log likelihood of the states in the state matrix
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

initPredictors = T.dvector("init_predictors")
statePredictors = T.dmatrix("state_predictors")
stateMatrix = T.dmatrix("state_matrix")
g = T.dscalar("state_autocorr")

# outputs_info = T.as_tensor_variable(np.asarray(0, np.float64))

def stateSeriesProb(ip, sp, ss, g):
	#Get the probability of the first element of the state series
	initProb = 1.0 / (1.0 + T.exp(-ip))
	start_prob = T.log(T.switch(T.eq(ss[0], 0), 1.0 - ip, ip))

	#Define a scan op that goes over the remaining elements of the state
	#series and applies the stateLLStep to each one

	result, updates = theano.map(fn = stateLLStep, non_sequences = [g], sequences = [ss[0:-1], ss[1:], sp])

	return T.sum(result)


all_series_prob, all_updates = theano.map(fn = stateSeriesProb, non_sequences = [g], sequences = [initPredictors, statePredictors, stateMatrix])

total_prob = T.sum(all_series_prob)

seriesProb = theano.function(inputs = [initPredictors, statePredictors, stateMatrix, g], outputs = total_prob, updates = all_updates)




if __name__ == '__main__':
	
	#The adj
	contact_mat = np.array([[0., 1.0, 1.0], [0.5, 0., 0.5], [1.0, 1.0, 0.]])
	imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])
	smat = np.ones((3,4))-imat

	distance_mat = np.array([[1., 10., 50.], [4., 1., 30.], [10.0, 15.0, 1.]])
	population = np.array([100., 20., 30.])
	initLogit = np.array([-0.05, -0.0005, -0.0001])
	grav = gravity(population, distance_mat, 0.5, 0.1, 2.0)

	print("Gravity")
	print(grav)

	print("Infectiousness")
	print(imat)
	
	print("Exposures")
	s = time.time()
	for i in xrange(10000):
		expose(grav, imat)
		seriesProb(initLogit, grav, imat, 0.9)
	e = time.time()
	print(e-s)