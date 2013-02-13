import numpy as np

#A simple observation series
obs = np.array([0, 1, 2, 3, 3])

#initial state occupation probabilities
#(flat prior over state occupancies when t = 0)
initProbs = np.ones(4)/4

#These are the time-constant transition *rates* (with zeros
#representing non-constant transitions or true zero 
#rate transitions. The probability of remaining in the
#current state is 1-(p(all other transitions)), so this can 
#be left empty as well

#transition rates are specified as logits, with
#current state as the reference category
epsilon = -0.5
gamma = -0.5
tmat = np.array([[0.0, 0.0, 0.0, 0.0], 
				[0.0, 0.0, epsilon, 0.0],
				[0.0, 0.0, 0.0, gamma],
				[0.0, 0.0, 0.0, 0.0]])

#Expand the transition matrix with a time-varying component,
#in this case on the (S -> E) transition

#To start, we'll just sample some random values
inf_rates = np.random.uniform(-1,1., len(obs))

#and put them into the (0,1) spot on the  expanded transition 
#matrices
tv_tmat = np.array([tmat]*(len(obs)))
tv_tmat[:,0,1] = inf_rates

#now convert these to probabilities
pr_tm = []
for tm in tv_tmat:
	ptm = np.copy(tm)
	for i,row in enumerate(ptm):
		#If there are no transitions in the row
		#then it is an absorbing state; move on
		if np.sum(row) == 0:
			ptm[i,i] = 1.0
		else:
			#Exponentiate probabilities for all
			#off-diagonal transition rates > 0
			nz = row != 0
			all_pr = np.exp(row[nz])
			total_pr = 1 + np.sum(all_pr)
			ptm[i,nz] = all_pr / total_pr
			ptm[i,i] = 1. / total_pr
	pr_tm.append(ptm)

#Convert the stacked arrays back into a single numpy array
pr_tm = np.array(pr_tm)

#This is an emission matrix for a simple system where
#underlying states are perfectly observed, i.e. an
# s x s identity matrix where s is the number of states
emission = np.identity(4)
emission = emission.T

#Now, we calculate the forward probabilities with respect to the 
#time-varying transition matrix
forwardProbs = []

#The first element of the forward probabilities
#is just the initial state probabilities
forwardProbs.append(initProbs)

#Now, loop through the observations, accumulating
#forward probabilities as we go
for i,v in enumerate(obs):
	obsProb = emission[v]*np.eye(4)
	probVec = np.dot(np.dot(obsProb,pr_tm[0].T),initProbs)
	print("prvec",probVec)
	probVec = probVec / sum(probVec)
	print(probVec)
	initProbs = probVec
	forwardProbs.append(initProbs)

# #Now do the stochastic backwards recursion
# backwardProbs = []
# lastProb = np.array([1.0, 1.0])
# backwardProbs.append(lastProb)
# sampledStates = []
# for i in reversed(xrange(len(obs))):
# 	if i == len(obs)-1:
# 		next_state = np.where(np.random.multinomial(1, forwardProbs[i]) == 1)[0][0]
# 		sampledStates.append(next_state)
# 	backward_prob = forwardProbs[i]*tmat[next_state][0]
# 	next_state = np.where(np.random.multinomial(1, backward_prob) == 1)[0][0]
# 	sampledStates.append(next_state)

# 	obsProb = emission[obs[i]]*np.eye(2)
# 	x = np.dot(tmat,np.dot(lastProb, obsProb))
# 	x = (x / sum(x))
# 	lastProb = x.T
# 	backwardProbs.append(lastProb)



