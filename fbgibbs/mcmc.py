

import pymc
import numpy as np
import sp_np 
import random
import hmm_np as hmm
import weibull

###################################
#Stochastic for switching between a Gamma-distributed vector of 
#random effects with mean R and a homogeneous vector in which all
#elements = R

###################################
#Stochastic for state matrix. 
def sm_logp(value):
    ''' Prints unique values of state matrix??? But why does it always return 0.0
    
       .. todo:: What exactly does sm_logp do?  Is this statematrix?  It seems wrong that it 

    '''
	#print("STATE MATRIX", np.unique(value))
	return 0.0

def StateMatrix(name, sm, trace = False, observed = False):
     ''' Creates a pymc stochastic object for the State matrix
   
        :param name: name of state matrix
        :param sm:   numpy state matrix 
        :param trace: not totally sure
        
        :rtype: pymc.Stochastic
    '''
	return pymc.Stochastic(name = name, doc = "State Matrix", value = sm,  cache_depth = 2, parents = {}, trace = trace, logp = sm_logp, dtype = int, observed = observed)

def sm_test():
    ''' Tests and prints value of a stochastic test matrix '''
    x = np.array([[2,2,2,3,3,3],
                [0,0,1,2,2,2],
                [0,0,0,0,1,2],
                [0,0,0,0,0,0]])
    sm = StateMatrix("States", x) 
    print("State Matrix Value:", sm.value)

##################################
#Deterministic for mass-action level exposure
def ma_exposure(sm, b, infstate):
    ''' 
    Determines mass action exposure of state matrix using the sp_np.mass_action_exposure method
    
    :param sm: State matrix of entire population
    :param b:  Beta value that scales exposure value
    :rtype:    float
    
    '''
    return sp_np.mass_action_exposure(sm, b, infstate = infstate)

def MassActionExposure(name, b, sm, infstate = 2, trace = False):
    '''
    Creates a pymc.Deterministic variable for use in sampler (???)
    
    :param name: Name of deterministic variable i guess???
    :param b:    Beta value for mass action exposure
    :param sm:   state matrix 
    :rtype:  pymc.Deterministic variable
    '''
    return pymc.Deterministic(name = name, doc = "MassActionExposure", parents = {"b": b, "sm":sm, "infstate":infstate}, trace = trace, cache_depth = 2, eval = ma_exposure)

def ma_test():
    x = np.array([[2,2,2,3,3,3],
            [0,0,1,2,2,2],
            [0,0,0,0,1,2],
            [0,0,0,0,0,0]])
    ma = MassActionExposure("MA", 0.2, x)
    print("Mass Action Exposure Test:", ma.value)


###################################
#Deterministic for within-group exposures
def grouped_exposures(sm, b, groups, infstate):
    '''
    Calculates exposures based on groups
    
    :param sm:  State matrix 
    :param b:   beta value for state matrix
    :param groups: List of lists of indices of group members
    :param infstate: Infection state that looking for 
    :rtype: Array G x T of exposures per group
    
    '''

	#print("B", b)
	return sp_np.grouped_exposure(sm, b, groups, infstate = infstate)

    
def GroupedExposures(name, b, sm, groups, infstate = 2, trace = False):
    '''  
        Creates a pymc.Deterministic variable of grouped exposures, which can recalculate group values
        
        :param name: pymc Variable name (???)
        :param b:    Beta value of group
        :param sm:   State matrix
        :param groups: List of group indices
        :rtype: pymc.Deterministic
        
    '''

    return pymc.Deterministic(name = name, doc = "Grouped Exposures", parents = {"b":b, "sm":sm, "groups":groups, "infstate":infstate}, trace = trace, cache_depth = 2, eval = grouped_exposures)

def ge_test():
    x = np.array([[2,2,2,3,3,3],
            [0,0,1,2,2,2],
            [0,0,0,0,1,2],
            [0,0,0,0,0,0]])
    groups = [np.array([0,1]), np.array([2,3])]
    ge = GroupedExposures("Grouped", 0.2, x, groups)
    print("Grouped Exposure Values", ge.value)


####################################
#Deterministic for weibull within-group inf profiles
def grouped_weibull_profile(sm, groups, b, q1, x1, q2, x2, infstate):
    # try:
    #     len(b)
    #     print("Grouped", np.mean(1500*b), np.var(1500*b), x1, x2)
    # except TypeError:
    #     print("Grouped", b, x1, x2)
    return weibull.grouped_weibull_exposure(sm, groups, b, q1, x1, q2, x2, infstate = infstate)

def GroupedWeibullExposures(name, sm, groups, b, q1, x1, q2, x2, infstate = 2, trace = False):
    return pymc.Deterministic(name = name, doc = "GroupedWeibullExposure", parents = {"b":b, "q1":q1, "x1":x1, "q2":q2, "x2":x2, "sm":sm, "infstate":infstate, "groups": groups}, trace = trace, cache_depth = 2, eval = grouped_weibull_profile)

def gw_test():
    x = np.array([[2,2,2,2,2,2],
            [0,0,1,2,2,2],
            [0,0,0,0,1,2],
            [0,0,0,0,0,0]])
    q1 = 0.5
    q2 = 0.9
    x1 = 1.0
    x2 = 3.0

    groups = [np.array([0,1]), np.array([2,3])]
    ge = GroupedWeibullExposures("Grouped", x, groups, 0.2, q1, x1, q2, x2)
    print("Grouped Weibull Values", ge.value)    

####################################
#Deterministic for weibull mass-action inf profiles
def ma_weibull_profile(sm, b, q1, x1, q2, x2, infstate):
    return weibull.mass_action_weibull_exposure(sm, b, q1, x1, q2, x2, infstate = infstate)

def MassActionWeibullExposures(name, sm, b, q1, x1, q2, x2, infstate = 2, trace = False):
    return pymc.Deterministic(name = name, doc = "GroupedWeibullExposure", parents = {"b":b, "q1":q1, "x1":x1, "q2":q2, "x2":x2, "sm":sm, "infstate":infstate}, trace = trace, cache_depth = 2, eval = ma_weibull_profile)

def maw_test():
    x = np.array([[2,2,2,2,2,2],
            [0,0,1,2,2,2],
            [0,0,0,0,1,2],
            [0,0,0,0,0,0]])
    q1 = 0.5
    q2 = 0.9
    x1 = 1.0
    x2 = 3.0

    ge = MassActionWeibullExposures("Grouped", x, 0.2, q1, x1, q2, x2)
    print("Mass Action Weibull Values", ge.value)    

###################################
#Deterministic adding and correcting mass-action and group
#exposures
def corrected_gr(ma, l1_l0, l1):
    ''' Corrected Group exposure based
        
        :param ma: Total exposure to every person
        :param l1_l0: level 1 and level 0 group overlap (with exposure based on l1 rate)
        :param l1: level 1 group exposure
        :type ma: pymc.Deterministic
        :type l1_l0: pymc.Deterministic
        :type l1: pymc.Deterministic
        
        :rtype: numpy.array
    
    '''
    return sp_np.corrected_group_exposures(ma, l1_l0, l1)

def CorrectedGroupExposures(name, ma, l1_l0, l1, trace = False):
    ''' Creates pymc.Deterministic variable to update corrected group exposures
    
        :param name: name of pymc variable
        :param ma:    Total exposure to every person
        :type ma:     pymc.Deterministic
        :param l1_l0: Total amount to remove from l0 group exposures (overlap)
        :type l1_l0:  pymc.Deterministic
        :param l1:    Total group exposure at l1
        :type l1:     pymc.Deterministic
        :rtype:       pymc.Deterministic
    '''
    
    return pymc.Deterministic(name = name, doc = "CorrectedGroupExposures", parents = {"ma": ma, "l1_l0":l1_l0, "l1":l1}, eval = corrected_gr, cache_depth = 2, trace = trace)

def cge_test():
    l0_b = 0.02
    l1_b = 0.2

    x = np.array([[2,2,2,3,3,3],
            [0,1,2,2,2,2],
            [0,0,0,1,2,2],
            [0,0,0,0,0,0]])
    groups = [np.array([0,1]), np.array([2,3])]

    ma = MassActionExposure("MA", l0_b, x)
    l0_ge = GroupedExposures("L0_groups", l0_b, x, groups)    
    l1_ge = GroupedExposures("L1_groups", l1_b, x, groups)
    l1_ce = CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
    print("Corrected Group Exposures", l1_ce.value)

#####################################
#Deterministic that that embeds scalar parameters into a transition 
#matrix
def tmatrix(epsilon, gamma):

	tmat = np.array([[0.0, 0.0, 0.0, 0.0], 
					[0.0, 0.0, epsilon, 0.0],
					[0.0, 0.0, 0.0, gamma],
					[0.0, 0.0, 0.0, 0.0]])
	return tmat

def StaticTransitionMatrix(name, epsilon, gamma, trace = False):
    ''' Embeds values of epsilon and gamma into transition matrix 
    
        :param name: pymc name
        :param epsilon: Value of epsilon
        :param gamma:   Value of gamma
        :rtype: pymc.Deterministic
    
    '''

    return pymc.Deterministic(name = name, doc = "TransitionMatrix", parents = {"epsilon": epsilon, "gamma":gamma}, eval = tmatrix,cache_depth = 2, trace = trace)

def tmat_test():
    tm = StaticTransitionMatrix("TM", 0.5, 0.5)
    print("Transition Matrix", tm.value)

#####################################
#Deterministic that that embeds scalar parameters into an SIR transition 
#matrix
def sir_tmatrix(gamma):
	tmat = np.array([[0.0, 0.0, 0.0], 
					[0.0, 0.0, gamma],
					[0.0, 0.0, 0.0]])
	return tmat

def SIRTransitionMatrix(name, gamma, trace = False):
	return pymc.Deterministic(name = name, doc = "TransitionMatrix", parents = {"gamma":gamma}, eval = sir_tmatrix,cache_depth = 2, trace = trace)

def sir_tmat_test():
	tm = SIRTransitionMatrix("TM", 0.5)
	print("Transition Matrix", tm.value)

##################################
#Deterministic that generates transition matrices w/Erlang distributed 
#latent periods w/o a recovery period
def erlang_latent_tmatrix(epsilon, num_stages):
	#This is for transition matrices where infectiousness
	#is an absorbing state, so we have 1 col for transitions
	#from S, then num_stages row/col for transitions from the 
	#states of latency and a final one for I
	num_col = 2 + num_stages
	tmat = np.zeros((num_col, num_col))

	#rate of transition through each sub-stage is 
	#epsilon*num_stages
	t_rate = epsilon*num_stages
	for i in xrange(1, 1+num_stages):
		tmat[i,i] = t_rate

	return tmat

def StaticErlangLatentTmat(name, epsilon, num_stages, trace = False):
	return pymc.Deterministic(name = name, doc = "ErlangLatentOnlyTransitionMatrix", parents = {"epsilon": epsilon, "num_stages":num_stages}, eval = erlang_latent_tmatrix, cache_depth = 2, trace = trace)

def erlang_tmat_test():
	epsilon = 0.5
	num_stages = 4
	tm = StaticErlangLatentTmat("erl", epsilon, num_stages)
	print(tm.value)

###############################################
#Deterministic for exposure pulse
def pulsed_exposure(t, rate, series_length, pulse_duration):
	exposure = np.zeros(series_length)
	for i in xrange(pulse_duration):
		exposure[t+i] = rate
	return exposure

def PulsedExposure(name, t, rate, series_length, pulse_duration = 1, trace = False):
	return pymc.Deterministic(name = name, doc = "PulsedExposure", parents = {"rate": rate, "series_length":series_length, "pulse_duration":pulse_duration, "t":t}, eval = pulsed_exposure, cache_depth = 2, trace = trace)

def pulsed_exposure_test():
	pe = PulsedExposure("PE", 0, 0.5, 10)
	print(pe.value)

####################
#Deterministic to add two series of the same dimension
def series_sum(x1, x2):
	return x1 + x2

def SeriesSum(name, x1, x2, trace = False):
	return pymc.Deterministic(name = name, doc = "PulsedExposure", parents = {"x1": x1, "x2":x2}, eval = series_sum, cache_depth = 2, trace = trace)


##########################################
#Deterministic that generates a list of completed transition
#matrices for groups from the static transition matrix
#and exposures
def full_tmat(stm, exposure):

    return sp_np.group_tmats(exposure, stm)

def FullTransitionMatrix(name, stm, exposure, trace = False):
    ''' pymc.Deterministic that creates list of completed transition 
    matrices for groups from the static transition matrix and exposures
    
        :param name: pymc variable name
        :param stm:  static transmission matrix with embedded values of epsilon and gamma 
        :type stm: pymc.Deterministic
        :param exposure: Corrected Group exposures
        :type exposure: pymc.Deterministic
       
        :rtype: pymc.Deterministic
    
    '''
    
    return pymc.Deterministic(name = name, doc = "FullTransitionMatrix", parents = {"stm": stm, "exposure":exposure}, eval = full_tmat, trace = trace)

def full_tm_test():
    l0_b = 0.9
    l1_b = 0.2
    e = 0.01
    g = 0.5

    x = np.array([[2,2,2,3,3,3],
            [0,1,2,2,2,2],
            [0,0,0,1,2,2],
            [0,0,0,0,0,0]])
    groups = [np.array([0,1]), np.array([2,3])]

    ma = MassActionExposure("MA", l0_b, x)
    l0_ge = GroupedExposures("L0_groups", l0_b, x, groups)    
    l1_ge = GroupedExposures("L1_groups", l1_b, x, groups)
    l1_ce = CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
    stm = StaticTransitionMatrix("TM", e, g)
    ftm = FullTransitionMatrix("Full", stm, l1_ce)
    print("Full Transition Matrix:", ftm.value)

##########################################
#Field potential giving logp of state for groups, given
#exposures

def group_sampling_prob(sm, ftm, groups):
    '''
        :param ftm:     full transition matrix
        :type ftm:      \|Groups\|xTx\|States\|x\|States\|
        :param sm:      state matrix
        :type sm:       \|N_inf\|xT
        :param groups:  lists of people in groups
    '''
    logp = 0.0
    
    
    for group_index, g in enumerate(ftm):
        logp += sp_np.sampling_probabilities(sm[groups[group_index]], g)
    print("GRSP", logp) 
    return logp

def GroupSamplingProbability(name, sm, ftm, groups):
    '''
        Field potential giving logp of state for groups, given exposures

        :param name: pymc variable name
        :param sm:   State matrix
        :type sm:    pymc.Deterministic
        :param ftm:  Full Transition Matrix
        :type ftm:   pymc.Deterministic
        :param groups: lists of people in groups
        
        :rtype: pymc.Potential
        
        .. todo:: Why is it only Group sampling probability? not individual sampling probability?
    
    '''

    return pymc.Potential(name = name, doc = "GroupSamplingProbability", parents = {"sm":sm, "ftm":ftm, "groups":groups}, logp = group_sampling_prob, cache_depth = 2)


def gr_sp_test():
    l0_b = 0.02
    l1_b = 0.2
    e = 0.9
    g = 0.5

    x = np.array([[2,2,2,3,3,3],
            [0,1,2,2,2,2],
            [0,0,0,1,2,2],
            [0,0,0,0,0,0]])
    groups = [np.array([0,1]), np.array([2,3])]

    ma = MassActionExposure("MA", l0_b, x)
    l0_ge = GroupedExposures("L0_groups", l0_b, x, groups)    
    l1_ge = GroupedExposures("L1_groups", l1_b, x, groups)
    l1_ce = CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
    stm = StaticTransitionMatrix("TM", e, g)
    ftm = FullTransitionMatrix("Full", stm, l1_ce)
    gsp = GroupSamplingProbability("GSP", x, ftm)
    print("Group sampling prob", gsp.logp)

############################################
#Field potential for likelihood contribution of 
#uninfected individuals with only l0 contacts
def InfectionEscape(name, n, exposure):
    '''
        Field potential for likelihood contribution of uninfected individuals with only l0 contacts
        
        :param name: pymc variable name
        :param n:    Number of non-infected individuals
        :param exposure:  Mass action exposure over time
        :rtype: pymc.Potential
        
        .. todo:: Doesn't look like sp_np.escape_exposure does not return a log value... problem? 
    '''


    return pymc.Potential(name = name, doc = "InfectionEscape", parents = {"n": n, "exposure": exposure}, cache_depth = 2, logp = sp_np.escape_exposure) 

def inf_escape_test():
    l0_b = 0.02
    n = 100
    x = np.array([[2,2,2,3,3,3],
            [0,1,2,2,2,2],
            [0,0,0,1,2,2],
            [0,0,0,0,0,0]])
    groups = [np.array([0,1]), np.array([2,3])]
    ma = MassActionExposure("MA", l0_b, x)
    ie = InfectionEscape("InfectionEscape", n, ma)
    print("Infection Escape lp:", ie.logp)

############################################
#Field potential for initial infection
def init_infection(p_inf, sm, num_non_inf, new_inf_state):


    #extract the first column of the state matrix,
    #and count the number of values equal to the 
    #infection state
    num_init_inf = np.sum(sm[:,0] == new_inf_state)
    num_init_non_inf = (len(sm[:,0]) - num_init_inf) + num_non_inf
    return (np.log(p_inf)*num_init_inf)+(np.log(1.-p_inf)*num_init_non_inf)

def InitialInfection(name, p_inf, sm, num_non_inf, new_inf_state = 1):
    ''' 
    Field potential for initial infection
    
    :param name: pymc variable name
    :param p_inf:  Initial infection (not sure)
    :type p_inf: pymc.Uniform
    :param sm: State matrix (should not include uninfected individuals)
    :param num_non_inf: Number of individuals that escaped infection
    :param new_inf_state:  Infection state (equivalent to I)
    :rtype: pymc.Potential
    
    '''


    return pymc.Potential(name = name, doc = "InitialInfection", parents = {"p_inf":p_inf, "sm":sm, "num_non_inf": num_non_inf, "new_inf_state": new_inf_state}, logp = init_infection)

def init_inf_test():
    p_inf = 0.02
    x = np.array([[2,2,2,3,3,3],
        [1,1,2,2,2,2],
        [0,0,0,1,2,2],
        [0,0,0,0,0,0]])
    num_non_inf = 100
    init_inf = InitialInfection("init_inf", p_inf, x, num_non_inf)
    print("Initial Infection Logp:", init_inf.logp)

##################################################
#Metropolis-Hastings step method for underlying state
class StateMetropolis(pymc.Metropolis):

    ''' Metropolis-Hastings step method for underlying state '''
	def __init__(self, stochastic, groups, init_probs, emission, obs, tmat, *args, **kwargs):
		self.init_probs = init_probs
		self.emission = emission
		self.obs = obs
		self.tmat = tmat
		#get row indices of individuals whose state can be manipulated
		#at this point, just check and see if the 0th item is a 0; if not
		#then it can be manipulated
		pymc.Metropolis.__init__(self, stochastic, *args, **kwargs)
		self.sample_indices = []
		for i,row in enumerate(self.obs):
			if -1 in row:
				self.sample_indices.append(i)

		#Create a reverse lookup to get the group index from the row index
		self.group_lookup = {}
		for i,g in enumerate(groups):
			for j,m in enumerate(g):
				self.group_lookup[m] = i

		#print(self.group_lookup)


	def propose(self):
         ''' 
            Propose new state matrix
            
            .. todo:: Do we choose in individual person each time? So every time we don't sample every person?
        
        '''
		#print("proposing")

		#copy the state matrix
		self.old_value = np.copy(self.stochastic.value)
		new_value = np.copy(self.stochastic.value)
		#print(self.adaptive_scale_factor)
		num_to_sample = max(1, int(np.round(self.adaptive_scale_factor)))
		#print("Num to sample", num_to_sample)

		#draw an individual to sample
		sample_index = random.sample(self.sample_indices,num_to_sample)

		self.hf = 0.0
		for si in sample_index:
			#grab the observation 
			obs = self.obs[si]

			#get the transition matrix for the sampled individual
			tm = self.tmat.value[self.group_lookup[si]]
			st = new_value[si]

			s_x, lv, tv = hmm.fbg_propose(st, self.init_probs, self.emission, obs, tm)
			#print("Obs:"), obs
			#print("From:", st)
			#print("To:", s_x)
			self.hf += lv-tv
			new_value[si] = s_x
			#print(obs)
			#print(s_x)
		#print("HF =", self.hf)
			
		self.stochastic.value = new_value

	def hastings_factor(self):
		return self.hf

	def reject(self):
		self.stochastic.value = self.old_value


def state_metropolis_test():
    l0_b = 0.02
    l1_b = 0.2
    e = 0.9
    g = 0.5

    obs = np.array([[2,-1,-1,-1,-1,-1],
        [-1,1,2,-1,-1,-1],
        [0,0,0,1,2,-1],
        [0,0,0,0,0,0]])

    x = np.array([[2,2,2,3,3,3],
        [1,1,2,2,2,2],
        [0,0,0,1,2,2],
        [0,0,0,0,0,0]])

    emission = np.identity(4)

    groups = [np.array([0,1]), np.array([2,3])]

    st = StateMatrix("SM", x)
    init_probs = np.ones(4)/4.

    groups = [np.array([0,1]), np.array([2,3])]
    ma = MassActionExposure("MA", l0_b, x)
    l0_ge = GroupedExposures("L0_groups", l0_b, x, groups)    
    l1_ge = GroupedExposures("L1_groups", l1_b, x, groups)
    l1_ce = CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
    stm = StaticTransitionMatrix("TM", e, g)
    ftm = FullTransitionMatrix("Full", stm, l1_ce)

    sm = StateMetropolis(st, groups, init_probs, emission, obs, ftm)
    sm.propose()
    sm.reject()    

##################################################
#Gibbs step method for underlying state
class StateGibbs(pymc.Metropolis):

    def __init__(self, stochastic, groups, init_probs, emission, obs, tmat, *args, **kwargs):
        self.init_probs = init_probs
        self.emission = emission
        self.obs = obs
        self.tmat = tmat
        self._id = "StateGibbs"
        self._tuning_info = []
        #get row indices of individuals whose state can be manipulated
        #at this point, just check and see if the 0th item is a 0; if not
        #then it can be manipulated
        pymc.Metropolis.__init__(self, stochastic, *args, **kwargs)
        self.sample_indices = []
        for i,row in enumerate(self.obs):
            if -1 in row:
                self.sample_indices.append(i)

        #Create a reverse lookup to get the group index from the row index
        self.group_lookup = {}
        for i,g in enumerate(groups):
            for j,m in enumerate(g):
                self.group_lookup[m] = i



    @classmethod
    def competence(self,s):
        return 0

    def current_state(self):
        return {}

    def step(self):
        #copy the state matrix
        new_value = np.copy(self.stochastic.value)
        #draw an individual to sample
        sample_index = random.sample(self.sample_indices,25)

        self.hf = 0.0
        for si in sample_index:
            #grab the observation 
            obs = self.obs[si]

            #get the transition matrix for the sampled individual
            tm = self.tmat.value[self.group_lookup[si]]
            st = new_value[si]


			s_x, lv, tv = hmm.fbg_propose(st, self.init_probs, self.emission, obs, tm)

			# print("Obs:", obs)
			# print("From:", new_value[si])


            new_value[si] = s_x

            # print("To:", new_value[si])

		self.stochastic.value = new_value






def main():
	#Test state matrix stochastic
	sm_test()
	ma_test()
	ge_test()
	cge_test()
	tmat_test()
	full_tm_test()
	gr_sp_test()
	inf_escape_test()
	init_inf_test()
	state_metropolis_test()
	gw_test()
	maw_test()
	erlang_tmat_test()
	pulsed_exposure_test()
	sir_tmat_test()

if __name__ == '__main__':
    main()
