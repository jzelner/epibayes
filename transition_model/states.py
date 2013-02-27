import numpy as np
from collections import defaultdict
from collections import OrderedDict
import json, itertools
from time import time
from itertools import tee
def mapping_to_keyList(mapping):
    ''' Creates dictionary of lists based on unique keys '''
    d = defaultdict(list)
    for k,v in mapping.items():
        d[v].append(k)
    return d

def importData(fname):
    d = json.load(open(fname, 'r'))
    return d
    
# takes an array that is 2 dimensional and makes additional layers for each unique value
def binaryOfIndexed(X, ndim=0):
    
    ndim = max(np.max(np.unique(X)), ndim)
    Z = np.zeros((ndim,) + X.shape, dtype="int")

    for i in range(ndim):
        Z[i][(X==i)] = 1
    return Z

    
def fillInState(state_slice):
    if np.max(state_slice)==-1:
        # if entire person is susceptible
        state_slice[:]=0
        return state_slice
    
    ## modify state_slice to fill in values for fun
    i_index = np.where(state_slice==1)[0][0]
    
    state_slice[0:i_index]=0
    state_slice[i_index+2]=2
    state_slice[i_index+3:]=3
    return state_slice

d = importData("mercury_complete_cabins.json")

# initial values
T = len(d['observations'][0])
N = len(d['personToCabin'])


# matrix of observations
states = np.array(map(lambda x: fillInState(np.array(x)), d['observations']))
states_split= binaryOfIndexed(states)
infected_state = states_split[2].T


# work with state array counting
person_sr_map = dict(map(lambda (k,v): (int(k), int(v)), d['personToCabin'].items())) # person to stateroom mapping
person_sr_pairs = sorted([x for x in person_sr_map.items()])
staterooms = np.array([x[1] for x in person_sr_pairs])

# initial parameters
beta_s = np.array([[.1]*T])
beta_r = np.array([[.2]*T])

#creates a list of people in stateroom sorted order
states_sr_order = infected_state[:,np.argsort(staterooms)]

# create group sums
sr_people_map = mapping_to_keyList(person_sr_map) # creates stateroom -> people in stateroom mapping
sr_people_values = sr_people_map.values()


# T x |staterooms|
# initial stateroom sums
sr_sums = np.array([infected_state[:,x].sum(axis=1) for x in sr_people_values]).T

def calculateSums(sr_sums):

    # sums of state rooms with person i missing T x |people|
    sr_sums_repeated = sr_sums.repeat(map(lambda x : len(x), sr_people_map.values()), axis=1)
    sr_staggered_sums = sr_sums_repeated - infected_state

    # total infecteds sum
    total_infected = np.sum(sr_sums, axis=1, keepdims=True)

    # total infecteds sum staggered (does not include own stateroom) T x |people|
    total_infected_staggered = total_infected.repeat(N, axis=1) - sr_sums_repeated

    lambdas = beta_s*sr_staggered_sums.T  +  beta_r*total_infected_staggered.T
    
    return(lambdas)
    
def testRuntime():
    person = 100
    
    cabin = person_sr_map[person]
    sequence = np.array([0,0,0,0,0,0,1,1,1,1,1,0,0,0,0])
    
    for l in range(len([1]*100)):
        start = time()
        
        originalSequence = infected_state.T[person]
        people_in_cabin = sr_people_map[cabin][:]
        people_in_cabin.remove(person)

        sr_sums[:,cabin] = sr_sums[:,cabin] - originalSequence + sequence
        infected_state.T[person]=sequence
        
        calculateSums(sr_sums)
        end = time()
        print "%20d%20f" % (1, end-start)

        
def calculateNewSums(sequence, person):
    cabin = person_sr_map[person]
    originalSequence = infected_state.T[person]
    people_in_cabin = sr_people_map[cabin][:]
    people_in_cabin.remove(person)

    sr_sums[:,cabin] = sr_sums[:,cabin] - originalSequence + sequence
    infected_state.T[person]=sequence
    
    return calculateSums(sr_sums)

#testRuntime()

lambdas = calculateSums(sr_sums)
lambdas_flat = np.reshape(lambdas, N*T)
epsilon = 1
gamma = 1/3
S=0;E=1;I=2;R=3
Trans = np.array([-lambdas_flat, lambdas_flat, [0]*N*T, [0]*N*T, 
                [0]*N*T, [epsilon]*N*T, [-epsilon]*N*T, [0]*N*T, 
                [0]*N*T, [0]*N*T, [gamma]*N*T, [-gamma]*N*T,
                [0]*N*T, [0]*N*T, [0]*N*T, [1]*N*T]).T.reshape((N, T, 4, 4))

# Calculate Probability Matrix
P = np.zeros((N,T,4, 4))
for i in range(N):
    for t in range(T):
        if T == 0:
            S_S = np.exp(-Trans[i,t,S,E])
        else:
            S_S = np.exp(np.log(P[i,t-1,S,S])-Trans[i,t,S,E])

        S_E = 1 - np.exp(-Trans[i,t,S,E])
        E_E = 1 - np.exp(-Trans[i,t,E,E])
        E_I = np.exp(-Trans[i,t,E,E])
        I_I = 1-np.exp(-Trans[i,t,I,I])
        I_R = np.exp(-Trans[i,t,I,I])
        R_R = 1.
        P[i,t] = np.array([[S_S, S_E, 0. , 0.],
                           [0. , E_E, E_I, 0.],
                           [0. ,  0., I_I, I_R],
                           [0., 0., 0., 1.]])

# return tuples of consecutive states ([(1,2),(2,3),(3,4)] from [1,2,3,4]
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
 
state_pairs = np.array([pairwise(x) for x in states])

state_P = [[ P[i,t,state_pairs[i,t,0], state_pairs[i,t,1]] for t in range(T-1)] for i in range(N)]
print sum(map(lambda x: np.log(x+.000001), np.ravel(state_P)))
