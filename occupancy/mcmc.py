import pymc
import epibayes
from epibayes.occupancy import trans, obs
import random

def Gravity(population, distance, sender_tau, receiver_tau, distance_tau):
	g = pymc.Deterministic(eval = trans.gravity, name = "gravity", parents = {"population" : population, "distance":distance, "sender_tau": sender_tau, "receiver_tau" : receiver_tau, "distance_tau":distance_tau}, doc = "Matrix of flows between locations", trace = False, verbose = 0, dtype = float, plot = False, cache_depth = 2)

	return g

def matObs(value, state, true_positive, true_negative):
	return obs.matObs(state, value, true_positive, true_negative)

def Observation(obs, state, true_positive, true_negative):

	o = pymc.Stochastic(name = "Observation process", doc = "Mapping from underlying state to observation", logp = matObs, parents = {"state":state, "true_positive" : true_positive, "true_negative": true_negative}, trace = False, observed = True, value = obs, verbose = 0, cache_depth = 2, dtype = float, plot = False)

	return o

if __name__ == '__main__':
	import numpy as np
	
	#This gives the pairwise distances between locations
	distance_mat = np.array([[1., 10., 50.], [4., 1., 30.], [10.0, 15.0, 1.]])

	#A vector of location population sizes
	population = np.array([100., 20., 30.])

	#Create a pymc deterministic that relates population 
	#and distances to the gravity matrix
	g = Gravity(population, distance_mat, 0.5, 0.5, 0.5)

	imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])
	omat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])
	
	o = Observation(omat, imat, 0.5, 0.5)
	print(o.logp)