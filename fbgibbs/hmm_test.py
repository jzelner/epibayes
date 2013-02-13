import numpy as np

#Make a transition matrix
initProbs = np.array([0.5, 0.5])
tmat = np.array([[0.5, 0.5], [0.3, 0.7]])
obs = np.array([0, 0, 1, 0, 0])
emission = np.array([[0.5, 0.5], [0.2, 0.8]])
emission = emission.T

forwardProbs = []
forwardProbs.append(initProbs)
for v in obs:
	obsProb = emission[v]*np.eye(2)
	probVec = np.dot(np.dot(initProbs,tmat),obsProb)
	probVec = probVec / sum(probVec)
	initProbs = probVec.T
	forwardProbs.append(initProbs)

#Now do the stochastic backwards recursion
backwardProbs = []
lastProb = np.array([1.0, 1.0])
backwardProbs.append(lastProb)
sampledStates = []
for i in reversed(xrange(len(obs))):
	if i == len(obs)-1:
		next_state = np.where(np.random.multinomial(1, forwardProbs[i]) == 1)[0][0]
		sampledStates.append(next_state)
	backward_prob = forwardProbs[i]*tmat[next_state][0]
	next_state = np.where(np.random.multinomial(1, backward_prob) == 1)[0][0]
	sampledStates.append(next_state)

	obsProb = emission[obs[i]]*np.eye(2)
	x = np.dot(tmat,np.dot(lastProb, obsProb))
	x = (x / sum(x))
	lastProb = x.T
	backwardProbs.append(lastProb)



