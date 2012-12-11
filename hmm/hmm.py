import theano
import theano.tensor as T
import numpy as np

#Make a transition matrix
initProbs = np.array([0.5, 0.5])
tmat = np.array([[0.7, 0.3], [0.3, 0.7]])
obs = np.array([0, 0, 1, 0, 0])
emission = np.array([[0.9, 0.1], [0.2, 0.8]])
emission = emission.T

eprob = T.dvector()
eToObs = theano.function([eprob],eprob * T.eye(T.shape(eprob)[0], m = T.shape(eprob)[0]))

init_probs_tensor = T.dvector()
tmat_tensor = T.dmatrix()
obs_tensor = T.dmatrix()
f_ex = T.dot(T.dot(init_probs_tensor, tmat_tensor), obs_tensor)
f_fn = theano.function([init_probs_tensor, tmat_tensor, obs_tensor], f_ex)

final_probs_tensor = T.dvector()
b_ex = T.dot(tmat_tensor, T.dot(final_probs_tensor, obs_tensor))
b_fn = theano.function([tmat_tensor, final_probs_tensor, obs_tensor], b_ex)

forwardProbs = []
forwardProbs.append(initProbs)
for v in obs:
	obsProb = eToObs(emission[v])
	probVec = f_fn(initProbs, tmat,obsProb)
	probVec = probVec / sum(probVec)
	initProbs = probVec.T
	forwardProbs.append(initProbs)

#Now do backwards
print("Backwards")
backwardProbs = []
lastProb = np.array([1.0, 1.0])
backwardProbs.append(lastProb)
for i in reversed(xrange(len(obs))):
	obsProb = emission[obs[i]]*np.eye(2)
	x = b_fn(tmat, lastProb, obsProb)#np.dot(tmat,np.dot(lastProb, obsProb))
	x = (x / sum(x))
	lastProb = x.T
	backwardProbs.append(lastProb)


smoothed = []
for i, b in enumerate(reversed(backwardProbs)):
	sm_n = forwardProbs[i]*b
	smoothed.append(sm_n / sum(sm_n))
print(smoothed)
