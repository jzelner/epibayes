import numpy as np
import pymc
import time
import random

#generates a list of matrices of length n_locations,
#where each list element is a matrix with n_pathogens rows
#and n_steps columns
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

	return np.array(exposures)

def PairwiseExposure(name, pathogens, weights):
	pe = pymc.Deterministic(name = name, doc = "Calculates pairwise exposures between sets of locations.", eval = pairwiseExposure, parents = {"pathogens" : pathogens, "weights" : weights})
	return pe

def separatePastAndFutureState(state):
	pastStates = np.array([s[:,0:s.shape[1]-1] for s in state])
	futureStates = np.array([s[:,1:s.shape[1]] for s in state])
	ic = np.array([s[:,0] for s in state])
	return {"IC":ic, "PS":pastStates, "FS":futureStates}

def SeparatePastAndFutureState(name, state):
	s = pymc.Deterministic(name = name, doc = "Extracts initial conditions, lagged states and future states", eval = separatePastAndFutureState, parents = {"state" : state})
	return s

def withinLocationLag(pathogens, autocorr):
	lag = [np.zeros_like(p) for p in pathogens]
	for i, a in enumerate(autocorr):
		lag[i] = lag[i] + a*pathogens[i]
	return np.array(lag)

def WithinLocationLag(name, pathogens, autocorr):
	wl = pymc.Deterministic(name = name, doc = "Gets logit for lag within a location", eval = withinLocationLag, parents = {"pathogens":pathogens, "autocorr":autocorr})
	return wl

def independentPredictor(pathogens, predictors):
	ip = np.array([pr*np.ones_like(pathogens[i]) for i,pr in enumerate(predictors)])
	return ip

def IndependentPredictor(name, pathogens, predictors):
	ip = pymc.Deterministic(name = name, doc = "Independent predictors", eval = independentPredictor, parents = {"pathogens":pathogens, "predictors": predictors})
	return ip

def predictedStateProbability(value, ex, ac, iv, fs):
	total_prob = []
	tp = 0.0
	for i,x in enumerate(fs):
		tp += pymc.bernoulli_like(x, pymc.invlogit(ex[i] + ac[i] + iv[i]))
	return tp

#ex = external exposure, ac = within-location autocorrelation,
#iv = independent variables
def PredictedStateProbability(name, ex, ac, iv, fs, value = 0.0):
	ps = pymc.Stochastic(name = name, doc = "Probability of next state", logp = predictedStateProbability, parents = {"ex" : ex, "ac":ac, "iv":iv, "fs" : fs}, value = value, dtype = float, observed = True)
	return ps

def cs_logp(value):
	return 0.0

def CompleteState(name, s):
	cs = pymc.Stochastic(name = name, doc = "Placeholder for total system state that facilitates manipulation by mcmc.", trace = False, cache_depth = 2, parents = {}, value = s, logp = cs_logp)
	return cs


class StateMetropolis(pymc.Metropolis):

	def __init__(self, stochastic, sample_p, unobserved_indices, *args, **kwargs):
		self.sample_p = sample_p
		self.unobserved_indices = unobserved_indices
		self.num_changeable = len(self.unobserved_indices[0])
		self.last_value = np.zeros_like(self.unobserved_indices[0])
		pymc.Metropolis.__init__(self, stochastic, *args, **kwargs)

	def propose(self):
		p = min(max(0.01, self.adaptive_scale_factor*self.sample_p), 0.99)
		
		num_to_sample = max(1, np.random.binomial(self.num_changeable, p))

		self.last_sampled_indices = zip(*random.sample(self.unobserved_indices, num_to_sample))

		self.last_value = self.stochastic.value
		val = np.copy(self.stochastic.value)
		val[self.last_sampled_indices] = 1.0 - self.last_value[self.last_sampled_indices]
		self.stochastic.value = val

	def reject(self):
		self.stochastic.value = self.last_value

	def hastings_factor(self):
		return 0.0


#Want to manipulate the underlying state in a way that doesn't conflict with 
#the observations

if __name__ == '__main__':
	#Make a random matrix representing pathogens over time in each village
	n_pathogens = 100
	n_times = 8
	n_locations = 25

	r_path = randomPathogens(n_pathogens, n_times, n_locations, 0.5)

	cs = CompleteState("CS", r_path)
	w = randomWeights(n_locations)
	s = SeparatePastAndFutureState("Separator", cs)

	initial_conditions, last_path, next_path = (s.value["IC"], s.value["PS"], s.value["FS"])

	pe = PairwiseExposure("Pairwise exposure", last_path, w)
	wl = WithinLocationLag("Lag", last_path, np.random.uniform(size = n_locations))
	ip = IndependentPredictor("Independent predictors", last_path, np.random.uniform(-5, 5,size = n_locations))

	ps = PredictedStateProbability("MANZE", ip, wl, pe, value = next_path)
	print(ps.logp)

	print(s["IC"].value)
