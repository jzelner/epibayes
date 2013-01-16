import pymc
import epibayes
from epibayes.occupancy import trans, obs
import random
import copy
import numpy as np

def grav(population, distance, phi, sender_tau, receiver_tau, distance_tau):
	g = trans.gravity(population, distance, sender_tau, receiver_tau, distance_tau)
	return phi*g

def Gravity(population, distance, phi, sender_tau, receiver_tau, distance_tau):
	g = pymc.Deterministic(eval = grav, name = "gravity", parents = {"population" : population, "distance":distance, "phi": phi, "sender_tau": sender_tau, "receiver_tau" : receiver_tau, "distance_tau":distance_tau}, doc = "Matrix of flows between locations", trace = False, verbose = 0, dtype = float, plot = False, cache_depth = 2)

	return g


def stateObs(value, gravity, init_predictors, predictors, autocorr):
	e = trans.exposure(gravity, value)
	num_col = e.shape[1]
	pr = predictors + e[:,0:num_col-1]
	#print("initial conditions", init_predictors)
	#print("Autocorrelation", autocorr)

	v = trans.multiSeriesProb(init_predictors, pr, value, autocorr)
	#print("STATE LL", v)
	return v

def State(name, state, gravity, init_predictors, predictors, autocorr):
	s = pymc.Stochastic(logp = stateObs, name = name, doc = "Log-likelihood of underlying state variables", value = state, parents = {"gravity": gravity, "init_predictors" : init_predictors, "predictors":predictors, "autocorr":autocorr}, dtype = float, plot = False, trace = False, cache_depth = 2, verbose = 0)

	return s

def matObs(value, state, true_positive, true_negative):
	obsLL =  obs.matObs(state, value, true_positive, true_negative)
	#print("Obs LL" , obsLL)
	return obsLL

def Observation(name, obs, state, true_positive, true_negative):
	o = pymc.Stochastic(name = name, doc = "Mapping from underlying state to observation", logp = matObs, parents = {"state":state, "true_positive" : true_positive, "true_negative": true_negative}, trace = False, observed = True, value = obs, verbose = 0, cache_depth = 2, dtype = float, plot = False)

	return o

def timeConstantPredictor(predictors, num_steps):
	#Tile for all steps except the 0-th
	pr = [p for p in predictors]
	tiledPredictors = np.tile(pr, (num_steps, 1)).T
	return tiledPredictors

def TimeConstantPredictor(predictors, num_steps):
	p = pymc.Deterministic(name = "Time constant predictors", doc = "Expand constant predictors", eval = timeConstantPredictor, parents = {"predictors" : predictors, "num_steps" : num_steps}, trace = False, verbose = 0, cache_depth = 2, dtype = float, plot = False )

	return p


#A subclass of pymc.Metropolis that does simple Metropolis-Hastings sampling
#for the state matrix. The unobserved_indices argument provides the (row, col)
#IDs of observations that can be manipulated by the sampler as a list of tuples.
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
		#print("Proposal probability", p, "sampling", num_to_sample, self.last_sampled_indices, self.unobserved_indices)
		self.last_value = self.stochastic.value
		val = np.copy(self.stochastic.value)
		val[self.last_sampled_indices] = 1.0 - self.last_value[self.last_sampled_indices]
		self.stochastic.value = val

	def reject(self):
		self.stochastic.value = self.last_value

	def hastings_factor(self):
		return 0.0


if __name__ == '__main__':
	import numpy as np
	
	#This gives the pairwise distances between locations
	distance_mat = np.array([[1., 10., 50.], [4., 1., 30.], [10.0, 15.0, 1.]])

	#A vector of location population sizes
	population = np.array([100., 20., 30.])

	#Occupation probabilities for each patch when t = 0
	initLogit = np.array([-0.05, -0.0005, -0.0001])

	#Starting value of state matrix
	imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])
	
	#Value of observation matrix
	omat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])

	autocorr = np.array([0.01, 0.1, 0.05])

	#Create a pymc deterministic that relates population 
	#and distances to the gravity matrix
	g = Gravity(population, distance_mat, 0.001, 0.5, 0.5, 0.5)

	#Expand time constant predictors and initial conditions using a 
	#deterministic
	tc = TimeConstantPredictor(initLogit, np.array([-1.0, -2.0, -3.0]), len(imat[0])-1)


	#Create a pymc stochastic that gives the log-likelihood
	#of the observations, given the rate of false positives and
	#false negatives	
	o = Observation(omat, imat, 0.5, 0.5)

	#A pymc stochastic that gives the log-likelihood of the underlying
	#state configuration, given the gravity matrix, other predictors
	#and the logit for within-patch autocorrelation
	s = State(imat, 0.001*g.value, initLogit, np.zeros_like(imat), autocorr)

	#Try instantiating a StateMetropolis step method with an instance of
	#state
	sm = StateMetropolis(s, 0.1, [(0,0), (1,1)])
