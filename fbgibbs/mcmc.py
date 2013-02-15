import pymc
import numpy as np
import sp_np

###################################
#Stochastic for state matrix. 
def sm_logp(value):
	return 0.0

def StateMatrix(name, sm, trace = False):
	return pymc.Stochastic(name = name, doc = "State Matrix", value = sm, observed = False, cache_depth = 2, parents = {}, trace = trace, logp = sm_logp, dtype = int)

def sm_test():
	x = np.array([[2,2,2,3,3,3],
				[0,0,1,2,2,2],
				[0,0,0,0,1,2],
				[0,0,0,0,0,0]])
	sm = StateMatrix("States", x) 
	print("State Matrix Value:", sm.value)

##################################
#Deterministic for mass-action level exposure
def ma_exposure(sm, b, infstate):
	return sp_np.mass_action_exposure(sm, b, infstate = infstate)

def MassActionExposure(name, b, sm, infstate = 2, trace = False):
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
	return sp_np.grouped_exposure(sm, b, groups, infstate = infstate)

def GroupedExposures(name, b, sm, groups, infstate = 2, trace = False):
	return pymc.Deterministic(name = name, doc = "Grouped Exposures", parents = {"b":b, "sm":sm, "groups":groups, "infstate":infstate}, trace = trace, cache_depth = 2, eval = grouped_exposures)

def ge_test():
	x = np.array([[2,2,2,3,3,3],
			[0,0,1,2,2,2],
			[0,0,0,0,1,2],
			[0,0,0,0,0,0]])
	groups = [np.array([0,1]), np.array([2,3])]
	ge = GroupedExposures("Grouped", 0.2, x, groups)
	print("Grouped Exposure Values", ge.value)

###################################
#Deterministic adding and correcting mass-action and group
#exposures
def corrected_gr(ma, l1_l0, l1):
	return sp_np.corrected_group_exposures(ma, l1_l0, l1)

def CorrectedGroupExposures(name, ma, l1_l0, l1, trace = False):
	return pymc.Deterministic(name = name, doc = "CorrectedGroupExposures", parents = {"ma": ma, "l1_l0":l1_l0, "l1":l1_l0}, eval = corrected_gr, cache_depth = 2, trace = trace)

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
	return pymc.Deterministic(name = name, doc = "TransitionMatrix", parents = {"epsilon": epsilon, "gamma":gamma}, eval = tmatrix,cache_depth = 2, trace = trace)

def tmat_test():
	tm = StaticTransitionMatrix("TM", 0.5, 0.5)
	print("Transition Matrix", tm.value)

##########################################
#Deterministic that generates a list of completed transition
#matrices for groups from the static transition matrix
#and exposures
def full_tmat(stm, exposure):
	return sp_np.group_tmats(exposure, stm)

def FullTransitionMatrix(name, stm, exposure, trace = False):
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
def group_sampling_prob(sm, ftm):
	logp = 0.0
	for g in ftm:
		logp += sp_np.sampling_probabilities(sm, g) 
	return logp

def GroupSamplingProbability(name, sm, ftm):
	return pymc.Potential(name = name, doc = "GroupSamplingProbability", parents = {"sm":sm, "ftm":ftm}, logp = group_sampling_prob, cache_depth = 2)

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

if __name__ == '__main__':
	main()