import theano
import theano.tensor as T
import numpy as np

#Make a transition matrix
initProbs = np.array([0.5, 0.5])
tmat = np.array([[0.7, 0.3], [0.3, 0.7]])
obs = np.array([0, 0, 1, 0, 0])
emission = np.array([[0.9, 0.1], [0.2, 0.8]])
emission = emission.T

obs_vec = T.ivector()
obs_type = T.iscalar()
emission_tensor = T.dmatrix()

#Expression for extracting the right row out of the emission
#matrix and multiplying it by the identity matrix
extract_e = emission_tensor[obs_type] * T.eye(T.shape(emission_tensor)[1], m = T.shape(emission_tensor)[1])

init_probs_tensor = T.dvector()
tmat_tensor = T.dmatrix()

f_ex = (T.dot(T.dot(init_probs_tensor, tmat_tensor), extract_e)).T
f_fn = theano.function([init_probs_tensor, tmat_tensor, emission_tensor, obs_type], f_ex)

final_probs_tensor = T.dvector()
b_ex = (T.dot(tmat_tensor, T.dot(final_probs_tensor, extract_e)))
b_fn = theano.function([tmat_tensor, final_probs_tensor, emission_tensor, obs_type], b_ex)

# forwardProbs, updates = theano.scan(fn = f_fn, outputs_info= T.ones((emission_tensor.shape[0])))


forwardProbs = [initProbs]
for v in obs:
	probVec = f_fn(initProbs, tmat, emission, v)
	probVec = probVec / sum(probVec)
	initProbs = probVec
	forwardProbs.append(initProbs)
print(forwardProbs)

#Now do backwards
backwardProbs = []
lastProb = np.array([1.0, 1.0])
backwardProbs.append(lastProb)
for i in reversed(xrange(len(obs))):
	x = b_fn(tmat, lastProb, emission, obs[i])
	x = (x / sum(x))
	lastProb = x.T
	backwardProbs.append(lastProb)

smoothed = []
for i, b in enumerate(reversed(backwardProbs)):
	sm_n = forwardProbs[i]*b
	smoothed.append(sm_n / sum(sm_n))

print(smoothed)
