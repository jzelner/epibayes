import numpy as np
import pymc
import time

def randomPathogens(n_pathogens, n_steps, n_loc, p):
	pathogen_presence = [np.reshape(np.random.binomial(1, p, n_pathogens*n_steps), (n_pathogens, n_steps)) for i in xrange(n_locations)]	
	return pathogen_presence

def randomWeights(n_loc):
	weights = np.reshape(np.random.uniform(size = n_locations**2), (n_locations, n_locations))
	return weights

def pairwiseExposure(pathogens, weights):
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

def PairwiseExposure(name, pathogens, weights):
	pe = pymc.Deterministic(name = name, doc = "Calculates pairwise exposures between sets of locations.", eval = pairwiseExposure, parents = {"pathogens" : pathogens, "weights" : weights})
	return pe

def separatePastAndFutureState(state):
	pastStates = [s[:,0:s.shape[1]-1] for s in state]
	futureStates = [s[:,1:s.shape[1]] for s in state]
	ic = [s[:,0] for s in state]
	return(ic, pastStates, futureStates)

def SeparatePastAndFutureState(name, state):
	s = pymc.Deterministic(name = name, doc = "Extracts initial conditions, lagged states and future states", eval = separatePastAndFutureState, parents = {"state" : state})
	return s

def withinLocationLag(pathogens, autocorr):
	lag = [np.zeros_like(p) for p in pathogens]
	for i, a in enumerate(autocorr):
		lag[i] = lag[i] + a*pathogens[i]
	return lag

def WithinLocationLag(name, pathogens, autocorr):
	wl = pymc.Deterministic(name = name, doc = "Gets logit for lag within a location", eval = withinLocationLag, parents = {"pathogens":pathogens, "autocorr":autocorr})
	return wl

def independentPredictor(pathogens, predictors):
	ip = [pr*np.ones_like(pathogens[i]) for i,pr in enumerate(predictors)]
	return ip

def IndependentPredictor(name, pathogens, predictors):
	ip = pymc.Deterministic(name = name, doc = "Independent predictors", eval = independentPredictor, parents = {"pathogens":pathogens, "predictors": predictors})
	return ip

def predictedStateProbability(value, ex, ac, iv):
	total_prob = []
	tp = 0.0
	for i,x in enumerate(value):
		tp += pymc.bernoulli_like(x, pymc.invlogit(ex[i] + ac[i] + iv[i]))
	return tp

def PredictedStateProbability(name, ex, ac, iv, value = None):
	ps = pymc.Stochastic(name = name, doc = "Probability of next state", logp = predictedStateProbability, parents = {"ex" : ex, "ac":ac, "iv":iv}, value = value)
	return ps

if __name__ == '__main__':
	#Make a random matrix representing pathogens over time in each village
	n_pathogens = 100
	n_times = 8
	n_locations = 25

	r_path = randomPathogens(n_pathogens, n_times, n_locations, 0.5)
	w = randomWeights(n_locations)
	s = SeparatePastAndFutureState("Separator", r_path)

	initial_conditions, last_path, next_path = (s.value[0], s.value[1], s.value[2])

	pe = PairwiseExposure("Pairwise exposure", last_path, w)
	wl = WithinLocationLag("Lag", last_path, np.random.uniform(size = n_locations))
	ip = IndependentPredictor("Independent predictors", last_path, np.random.uniform(-5, 5,size = n_locations))

	ps = PredictedStateProbability("MANZE", ip, wl, pe, value = next_path)
	print(ps.logp)
