import numpy as np
import time
def setup(numObs):
    ''' creates test data based on some number of observations 
        
        Returns:
        -initProbs: uniform initial probability of states at t = 0
        -emission:  emission probability of seeing state s1 given state s2
        -obs:       observation of states (assuming see states perfectly)  
        -pr_tm:     probability of state transmissions from t to t+1
    '''
    
    #A simple observation series
    obs = -1*np.ones(numObs)
    obs[0] = 0
    obs[-1] = 3 

    # initial state occupation probabilities
    # (flat prior over state occupancies when t = 0)
    initProbs = np.ones(4)/4

    # These are the time-constant transition *rates* (with zeros
    # representing non-constant transitions or true zero 
    # rate transitions. The probability of remaining in the
    # current state is 1-(p(all other transitions)), so this can 
    # be left empty as well

    # transition rates are specified as logits, with
    # current state as the reference category
    epsilon = -0.2
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

    return (initProbs, emission, obs, pr_tm)


def forward(initProbs, emission, obs, pr_tm):
    nObsType = len(emission)
    unobsProbs = np.ones(nObsType)/nObsType

    #Now, we calculate the forward probabilities with respect to the 
    #time-varying transition matrix
    forwardProbs = []
    #The first element of the forward probabilities
    #is just the initial state probabilities
    #forwardProbs.append(initProbs)
    #Loop through the observations, accumulating
    #forward probabilities as we go
    for i,v in enumerate(obs):
        if v == -1:
            obsProb = unobsProbs*np.eye(nObsType)
        else:
            obsProb = emission[v]*np.eye(nObsType)

        probVec = np.dot(np.dot(obsProb,pr_tm[i].T),initProbs)
        probVec = probVec / sum(probVec)
        initProbs = probVec
        forwardProbs.append(initProbs)
    forwardProbs = np.array(forwardProbs)
    return forwardProbs

def backward_sample(forwardProbs, tm):
    ''' Do stochastic backwards recursion    '''
    
    sampledStates = []
    for i in reversed(xrange(len(forwardProbs))):
        #For the last observation, we sample from the final step of the forward
        #probability vector, which is the same as sampling from p(X_n | Obs_i \to n),
        #so the sampled state is guaranteed to be consistent with the previous states
        #and the current observation
        if i == len(forwardProbs)-1:
            # choose next state by random draw and seeing which index is chosen
            next_state = np.where(np.random.multinomial(1, forwardProbs[i]) == 1)[0][0]
            sampledStates.append(next_state)
            continue
        
        # For the next steps, we want to sample the state proportional to its probability of
        # generating the observation x probability of transitioning to the next sampled state
        # x probability of being transitioned to, given the distribution over states in the 
        # previous step of the forward prob vector
        ns_p = tm[i][:,next_state] / np.sum(tm[i][:,next_state]) # normalize next state probabilities
        backward_prob = forwardProbs[i]*ns_p
        backward_prob /= np.sum(backward_prob)
        next_state = np.where(np.random.multinomial(1, backward_prob) == 1)[0][0]
        sampledStates.append(next_state)

    sampledStates.reverse()
    return sampledStates

# Take a vector of underlying states, a forward probability vector
# and the current transition matrix, and sample a new vector along
# with the probability of sampling the original vector and the probability
# of sampling the new vector. 
def backward_propose(x,forwardProbs, tm):
    last_val_lprob = 0.0
    this_val_lprob = 0.0
    #Now do the stochastic backwards recursion
    sampledStates = []
    for i in reversed(xrange(len(forwardProbs))):
        # For the last observation, we sample from the final step of the forward
        # probability vector, which is the same as sampling from p(X_n | Obs_i \to n),
        # so the sampled state is guaranteed to be consistent with the previous states
        # and the current observation
        if i == len(forwardProbs)-1:
            next_state = np.where(np.random.multinomial(1, forwardProbs[i]) == 1)[0][0]
            this_val_lprob += np.log(forwardProbs[i][next_state])
            last_val_lprob += np.log(forwardProbs[i][x[i]])
            sampledStates.append(next_state)
            continue

        # For the next steps, we want to sample the state proportional to its probability of
        # generating the the next sampled state
        # x probability of being transitioned to, given the distribution over states in the 
        # previous step of the forward prob vector
        ns_p = tm[i][:,next_state] / np.sum(tm[i][:,next_state])

        #Get the same value for the value we're proposing from
        x_ns_p = tm[i][:,x[i+1]] / np.sum(tm[i][:,x[i+1]])

        backward_prob = forwardProbs[i]*ns_p
        backward_prob /= np.sum(backward_prob)
        backward_prob_x = forwardProbs[i]*x_ns_p
        backward_prob_x /= np.sum(backward_prob_x)

        next_state = np.where(np.random.multinomial(1, backward_prob) == 1)[0][0]

        #Multiply by the probability of obtaining this state given the current one
        this_val_lprob += np.log(backward_prob[next_state])
        last_val_lprob += np.log(backward_prob_x[x[i]])
        sampledStates.append(next_state)

    sampledStates.reverse()
    return (sampledStates, last_val_lprob, this_val_lprob)

def fbg_sample(initProbs, emission, obs, pr_tm):
    forwardProbs = forward(initProbs, emission, obs, pr_tm)
    sampledStates = backward_sample(forwardProbs, pr_tm)
    return sampledStates    

def fbg_propose(x, initProbs, emission, obs, pr_tm):
    forwardProbs = forward(initProbs, emission, obs, pr_tm)
    sampledStates = backward_propose(x,forwardProbs, pr_tm)
    return sampledStates    

def main():
    numObs = 50
    (initProbs, emission, obs, pr_tm) = setup(numObs)
    sampledStates = fbg_sample(initProbs, emission, obs, pr_tm)
    (new_x, lv, tv) = fbg_propose(sampledStates, initProbs, emission, obs, pr_tm)
    print("Obs:", obs)
    print(sampledStates, "->", new_x)
    print("P(x'->x) = %0.2f; P(x->x') = %0.2f, Proposal ratio: %0.2f" %(lv, tv,lv-tv))
if __name__ == '__main__':
    main()
