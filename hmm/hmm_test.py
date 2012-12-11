import hmm
import numpy as np

#Make a transition matrix
initProbs = np.array([0.5, 0.5])
tmat = np.array([[0.7, 0.3], [0.3, 0.7]])
obs = np.array([0, 0, 1, 0, 0])
emission = np.array([[0.9, 0.1], [0.2, 0.8]])
emission = emission.T

forwardProbs = []
forwardProbs.append(initProbs)
for v in obs:
	obsProb = emission[v]*np.eye(2)
	probVec = np.dot(np.dot(initProbs,tmat),obsProb)
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
	x = np.dot(tmat,np.dot(lastProb, obsProb))
	x = (x / sum(x))
	lastProb = x.T
	backwardProbs.append(lastProb)


smoothed = []
for i, b in enumerate(reversed(backwardProbs)):
	sm_n = forwardProbs[i]*b
	smoothed.append(sm_n / sum(sm_n))
print(smoothed)
