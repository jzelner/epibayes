import numpy as np


def randomPathogens(n_pathogens, n_steps, n_loc, p):
	pathogen_presence = [np.reshape(np.random.binomial(1, p, n_pathogens*n_steps), (n_pathogens, n_steps)) for i in xrange(n_locations)]	
	return pathogen_presence

def randomWeights(n_loc):
	weights = np.reshape(np.random.uniform(size = n_locations**2), (n_locations, n_locations))
	return weights

def PairwiseExposure(pathogens, weights):
	#Accumulate each location's exposure in a pre-allocated
	#matrix
	exposures = [np.zeros_like(p) for p in pathogens]
	wt = weights.T

	#Loop through each location and calculate the exposure of each other 
	#place to this one
	for i, p in enumerate(pathogens):
		#Go through each row of weights
		for j, w in enumerate(wt[i,:]):
			if i == j:
				#Make sure not to add the same
				#matrix to itself
				continue
			exposures[j] = exposures[j] + w*p

	return exposures

def SeparatePastAndFutureState(state):
	pastStates = [s[:,0:s.shape[1]] for s in state]
	futureStates = [s[:,1:s.shape[1]-1] for s in state]
	ic = [s[:,0] for s in state]
	return(ic, pastStates, futureStates)

def WithinLocationLag(pathogens, autocorr):
	lag = [np.zeros_like(p) for p in pathogens]
	for i, a in enumerate(autocorr):
		lag[i] = lag[i] + a*pathogens[i]
	return lag

def AddInitialConditions(ic, toLag):
	rVals = []
	for i,m in enumerate(toLag):
		num_row = m.shape[0]
		fr = np.array([[ic[i]] for j in xrange(num_row)])
		rVals.append(np.append(fr, m,1))
	return rVals

def IndependentPredictor(pathogens, predictors):
	ip = [pr*np.ones_like(pathogens[i]) for i,pr in enumerate(predictors)]
	return ip

if __name__ == '__main__':
	#Make a random matrix representing pathogens over time in each village
	n_pathogens = 5
	n_times = 10
	n_locations = 3

	r_path = randomPathogens(n_pathogens, n_times, n_locations, 0.5)
	initial_conditions, last_path, next_path = SeparatePastAndFutureState(r_path)
	w = randomWeights(n_locations)

	pe = PairwiseExposure(last_path, w)
	lag = WithinLocationLag(last_path, np.random.uniform(size = n_locations))
	ip = IndependentPredictor(r_path, np.random.uniform(-5, 5,size = n_locations))

	print(lag)
