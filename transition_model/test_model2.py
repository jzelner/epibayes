import random
import numpy as np
from itertools import *

# define new exception for invalid parameters
class InvalidParameters(Exception): pass

# Crazy enum function to define constants to states
# http://stackoverflow.com/questions/36932/whats-the-best-way-to-implement-an-enum-in-python
def enum(**enums):
    return type('Enum', (), enums)
#States = enum(S=0, E=1, I=2, R=3) # create enumeration of states



# global variables
### Possible states, assuming SEIR model
S=0; E=1; I=2; R=3
states = [S,E,I,R]

### possible routes
beta_n = 0    # no contact
beta_r = 1    # stateroom transmission
beta_s = 2    # ship transmission
routes = [beta_n, beta_r, beta_s]


def createPeople(N, index, escaped):
    
    X = np.array([sorted([random.choice(states) for t in range(nSteps)]) for i in range(N-(index+escaped))], dtype=int)
    Y = np.array([sorted([random.choice([states[0]]) for t in range(nSteps)]) for i in range(escaped)])
    Z = np.array([sorted([random.choice(states[1:]) for t in range(nSteps)]) for i in range(index)])
    return np.vstack((X,Y,Z))

# takes an array that is 2 dimensional and makes additional layers for each unique value
def binaryOfIndexed(X, ndim):
    Z = np.zeros((ndim,) + X.shape, dtype="int")
    print X
    for i in range(ndim):
        Z[i][(X==i)] = 1
    return Z

    
def indexedOfBinary(X):
    pass

def runModel(A, X, Beta, T_r, N, vector=False):

    ### run tests to check if parameters are correct
    def doChecks():
        if T_r <= 0 or N <= 0:
            raise InvalidParameters("invalid T_r or N values")
        # check shapes of matrices
        if A.shape != (N,N):
            raise InvalidParameters("contact matrix is not NxN")
        if X.shape != (N, T_r):
            raise InvalidParameters("state matrix is not NxT")
        if Beta.shape != (len(routes),T_r):
            raise InvalidParameters("Beta matrix is not RxT")
        # sanity check of values
        if X.max() > max(states):
            raise InvalidParameters("X matrix contains an invalid state")
        
    doChecks()
    ### runModel takes an adjacency matrix and observed states and number of steps
    
    ### initial conditions
    epsilon = 1/3.
    gamma   = 1.

    
    if vector:
        Sigma = binaryOfIndexed(np.array([[0,0,I,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0],
                 [0,0,0,0,0]]), len(states))
        Xv = binaryOfIndexed(X, len(states))
        Av = binaryOfIndexed(A, len(routes))
    else:
    
        # each entry Sigma[i,j] is a set of states that will move a person in state[i] to state[j]
        Sigma = np.array([[set([]), set([I]), set([]), set([])],
                  [set([]), set([]), set([]), set([])],
                  [set([]), set([]), set([]), set([])],
                  [set([]), set([]), set([]), set([])]])

        # lambda function for calculating force of infection from person i in state X[i][t] to state s2 at time t
        force = lambda s2, i, t : sum(( sum( ( Beta[A[i,j],t]*int(X[j,t] == s2) for j in xrange(N) )) for s in Sigma[X[i,t],s2] ))        

        # lambda function for calculating total force of infection on person i
        lambda_all = lambda i, t: sum([ force(s2, i, t) for s2 in states ])

    # T is the state transition matrix 
    T = np.array([[[
            [-lambda_all(i,t), lambda_all(i,t), 0,0],
            [0, epsilon, -epsilon, 0],
            [0, 0, gamma, -gamma],
            [0,0,0,1]] for t in range(T_r)] for i in range(N)], dtype='float32')
    
    # Calculate Probability Matrix
    P = np.zeros((N,T_r,len(states),len(states)))
    for i in range(N):
        for t in range(T_r):
            if t == 0:
                S_S = np.exp(-T[i,t,S,E])
            else:
                S_S = np.exp(np.log(P[i,t-1,S,S])-T[i,t,S,E])

            S_E = 1 - np.exp(-T[i,t,S,E])
            E_E = 1 - np.exp(-T[i,t,E,E])
            E_I = np.exp(-T[i,t,E,E])
            I_I = 1-np.exp(-T[i,t,I,I])
            I_R = np.exp(-T[i,t,I,I])
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
     
    state_pairs = np.array([pairwise(x) for x in X])
    
    state_P = [[ P[i,t,state_pairs[i,t,0], state_pairs[i,t,1]] for t in range(T_r-1)] for i in range(N)]
    print sum(map(lambda x: np.log(x+.000001), np.ravel(state_P)))
    #print state_P
    return {'T':T, 'P':P}

            
            
            
            
### HMM variables
N = 19                       # number of people
index = 3                    # number of index cases
escape = 2                   # number of people still susceptible
T = 10                       # length of observation window (days)
dt = .5                      # stepsize
nSteps = int(T/dt)           # number of steps

# state matrix, NxT, of "actual" states of people (will be guessed in FB)
#X = np.array([sorted([random.choice(states) for t in range(nSteps)]) for i in range(N-1)], dtype=int)
#X = np.vstack((X, sorted([random.choice(states[1:]) for t in range(nSteps)])))
X = createPeople(N,index,escape)

# Adjancy matrix, NxN !!! could be NxNxT if contact types change over time !!!
A = np.array([[random.choice(routes) for t in range(N)] for i in range(N)], dtype=int) 
np.fill_diagonal(A, 0) # fill diagonals with zero so that cannot infect self


binaryOfIndexed(A, len(routes))


print X
print A

# Array of beta values
beta_s0 = .01
beta_r0 = .02
Beta = np.array([[0.0]*nSteps,
                 [beta_s0]*nSteps,
                 [beta_r0]*nSteps])

r = runModel(A,X,Beta,nSteps,N)
