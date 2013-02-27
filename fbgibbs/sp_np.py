import numpy as np

def setup_states():

    epsilon = 0.9
    gamma = 0.2
    tmat = np.array([[0.0, 0.0, 0.0, 0.0], 
                    [0.0, 0.0, epsilon, 0.0],
                    [0.0, 0.0, 0.0, gamma],
                    [0.0, 0.0, 0.0, 0.0]])

    #Make a simple 4 person system
    x = np.array([[2,2,2,3,3,3],
                [0,0,1,2,2,2],
                [0,0,0,0,1,2],
                [0,0,0,0,0,0]])

    #Repeat it 10 times to represent 10 subgroupings
    num_groups = 50
    x = np.vstack([x]*num_groups)

    #assign group IDs


    #groups = [np.array([0,1]), np.array([2,3])]
    groups = []
    for i in xrange(num_groups):
        groups.append(np.arange(4)+(4*i))

    return (tmat, x, groups)

##########################################
#Exposure functions for time-constant infectiousness
#likely with a modeled endpoint (i.e., SEIR, SIR model)
def mass_action_exposure(x, b, infstate = 2):
    return b*np.sum(x == infstate, axis = 0)

def grouped_exposure(x, b, groups, infstate = 2):
    ge = []
    for g in groups:
        ge.append(b*np.sum(x[g] == infstate, axis = 0 ))
    return np.array(ge)

def group_exposure(x, b0, bg, groups, infstate = 2):
    #get exposure for everyone
    ma_ex = mass_action_exposure(x, b0)

    #get group exposures with mass-action rate
    g_ex_ma = grouped_exposure(x, b0, groups)

    #subtract group exposures from mass-action for group-level
    #top-level exposure
    ###### why is the for loop necessary?????????
    g_ex_ma = np.array([ma_ex - g for g in g_ex_ma])

    #get group exposures with group rate
    g_ex = grouped_exposure(x, bg, groups)

    #sum these with top-level exposures
    return g_ex_ma + g_ex

def corrected_group_exposures(ma_exposure, l1_l0_exposure, l1_exposure):
    corrected_l0 = np.array([ma_exposure - l1_l0 for l1_l0 in l1_l0_exposure])
    return corrected_l0 + l1_exposure


# Takes a matrix of exposures (rows correspond to groups, 
# columns to times) and a static transition matrix consisting
# of transition rates. Returns a list of transition matrices
# consisting of the probabilities of each state transition 
# over time for each group.
def group_tmats(exposure, tmat):
    '''
        Takes a matrix of exposures (rows correspond to groups, columns to times) 
        and a static transition matrix consisting of transition rates. Returns a 
        list of transition matrices consisting of the probabilities of each state 
        transition over time for each group.

        :param exposure: exposure matrix
        :param tmat:     transition matrix (static)
        
        .. todo:: group_tmats is a bottleneck, takes a ton of time for each call
        as of 2.25 it takes .15 seconds per call
    '''
    #print exposure.shape
    #print tmat.shape
    gmats = np.zeros((exposure.shape[0], exposure.shape[1], tmat.shape[0], tmat.shape[1]))
    #gmats = []
    for group,e in enumerate(exposure):
        gmat = np.array([tmat]*len(e))

        #Set the S->E element to the exposure rate
        gmat[:,0,1] = e

        # Get the total rate at which individuals
        # are transitioning out of the states in
        # gmat

        col_totals = np.sum(gmat, axis = 2)
        leave_probs = 1.0 - np.exp(-col_totals) ## is this not the exposure for entire sum???
        #Go through each timeslice, each of which
        #corresponds to an element of the column totals
        for t,ct in enumerate(col_totals):
            #If the row totals to > 0, then divide all
            #of the elements by the total; this gives
            #the proportion of the leave rate associated 
            #with this element
            for j,tot in enumerate(ct):
                lp = leave_probs[t,j]
                if tot > 0:
                    gmat[t,j] /= tot
                    gmat[t,j] *= lp
                gmat[t,j,j] = 1.-lp
        gmats[group] = gmat
    #print np.array(gmats).shape
    
    return np.array(gmats)

#Takes a matrix of states and the transition matrix for individuals
#in that matrix and returns the log-likelihood of 
def sampling_probabilities(x, tmat):
    ''' 
        Takes a matrix of states and the transition matrix for individuals
        in that matrix and returns the log-likelihood of states occurring
        
        :param x:    states
        :param tmat: Transition matrix
        :rtype:      float
        
        .. todo:: Bottleneck, mostly because a ton of call to this function 
    '''


    s_prob = 0.0
    #get the number of columns in the state matrix
    num_col = x.shape[1] # really T
    
    for t in xrange(num_col-1):
        from_states = x[:,t]
        to_states = x[:,t+1]
        #print(tmat)
        #print(t, from_states, to_states)
        # sum up all groups? 
        s_prob += np.sum(np.log(tmat[t,from_states,to_states]))
        
    return s_prob

#takes a vector of exposures and returns num_exposed * (1 - p_inf)
def escape_exposure(n, exposure):
    ''' takes a vector of exposures and returns num_exposed * (1 - p_inf) 
    
    :param n: Number of people
    :param exposure: Exposure over time
    :rtype: float
    '''
    e_prob = n * np.sum(-exposure)
    return e_prob

def main():
    tm, x, groups = setup_states()
    ge = group_exposure(x, 0.009, 0.5, groups)
    gm = group_tmats(ge, tm)
    for i in xrange(100):
        sp = 0.0
        for i,g in enumerate(groups):
            sp += sampling_probabilities(x[g], gm[i])
        print(sp)
    return gm

if __name__ == '__main__':
    main()
