import theano
import theano.tensor as T
import numpy as np

obs_mat = T.dmatrix("observations")
state_mat = T.dmatrix("state")

multi_obs = T.dtensor3("multi_obs")
multi_state = T.dtensor3("multi_state")

tn = T.dscalar("true_negative")
tp = T.dscalar("true_positive")

#sensitivity = proportion of people positive who test
#positive.
#specificity = proportion of those negative who test
#negative

def singleStateObservation(st, obs, tp, tn):
	#st is the current state
	#obs is the current observation
	#lp is the last probability
	#tp is the probability of finding
	#a true positive, i.e. p(positive) | positive_obs
	#tn is the probability of finding a 
	#true negative, p(negative) | negative_obs



	obs_type = T.switch(T.eq(obs, 1.0), tp, tn)
	obs_prob = T.switch(T.eq(obs, -1.0), 0.0, T.log(T.switch(T.eq(st, obs), obs_type, 1.0 - obs_type)))
	return obs_prob


def stateSeriesObservation(ss, obss, tp, tn):
	result, updates = theano.map(fn = singleStateObservation, non_sequences = [tp, tn], sequences = [ss, obss])

	return T.sum(result)

def stateMatObservation(sm, obsm, tp, tn):
	result, updates = theano.map(fn = stateSeriesObservation, non_sequences = [tp, tn], sequences = [multi_state, multi_obs])

	return T.sum(result)

#Map op for iterating over all rows of a matrix
all_series_obs, all_updates = theano.map(fn = stateSeriesObservation, sequences = [state_mat, obs_mat], non_sequences = [tp, tn])

final_prob = T.sum(all_series_obs)

matObs = theano.function(inputs = [state_mat, obs_mat, tp, tn], outputs = final_prob, updates = all_updates)

#Map op for iterating over all matrices in a 3d tensor
all_mat_obs, all_mat_updates = theano.map(fn = stateMatObservation, sequences = [multi_state, multi_obs], non_sequences = [tp, tn])

multi_prob = T.sum(all_mat_obs)

multiMatObs = theano.function(inputs = [multi_state, multi_obs, tp, tn], outputs = multi_prob )

if __name__ == '__main__':
	imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])
	omat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])

	multi_imat = np.array([imat for i in xrange(10000)])
	multi_obs = np.array([omat for i in xrange(10000)])

	print(matObs(imat, omat, 1.0, 1.0))
	print(multiMatObs(multi_imat, multi_obs, 0.6, 0.5))