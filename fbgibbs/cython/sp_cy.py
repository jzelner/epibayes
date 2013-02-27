import epibayes.fbgibbs.mcmc as fbmc
import epibayes.fbgibbs.sp_np as sp_np

import numpy as np
import sp
import pymc
import time
def cy_group_sampling_prob(sm, ftm):
	logp = 0.0
	for g in ftm:
		logp += sp.sampling_probabilities(sm, g)
	return logp

def GroupSamplingProbabilityCy(name, sm, ftm):
	return pymc.Potential(name = name, doc = "GroupSamplingProbability", parents = {"sm":sm, "ftm":ftm}, logp = cy_group_sampling_prob, cache_depth = 2)


x = np.array([[2,2,2,3,3,3],
		[0,1,2,2,2,2],
		[0,0,0,1,2,2],
		[0,0,0,0,0,0]])


y = np.vstack([x]*10)

index = 0
groups = []
for i in xrange(10):
	groups.append(np.array([index,index+1]))
	index += 2


def gr_sp_test(x, groups):
	l0_b = 0.02
	l1_b = 0.2
	e = 0.9
	g = 0.5


	ma = fbmc.MassActionExposure("MA", l0_b, x)
	l0_ge = fbmc.GroupedExposures("L0_groups", l0_b, x, groups)	
	l1_ge = fbmc.GroupedExposures("L1_groups", l1_b, x, groups)
	l1_ce = fbmc.CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
	stm = fbmc.StaticTransitionMatrix("TM", e, g)
	ftm = fbmc.FullTransitionMatrix("Full", stm, l1_ce)
	st = time.time()
	for i in xrange(10):
		z = fbmc.group_sampling_prob(x, ftm.value)
	en = time.time()
	print("Regular Py:", en-st, z)
	#print("Group sampling prob", gsp.logp)


def gr_sp_cy_test(x, groups):
	l0_b = 0.02
	l1_b = 0.2
	e = 0.9
	g = 0.5


	ma = fbmc.MassActionExposure("MA", l0_b, x)
	l0_ge = fbmc.GroupedExposures("L0_groups", l0_b, x, groups)	
	l1_ge = fbmc.GroupedExposures("L1_groups", l1_b, x, groups)
	l1_ce = fbmc.CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
	stm = fbmc.StaticTransitionMatrix("TM", e, g)
	ftm = fbmc.FullTransitionMatrix("Full", stm, l1_ce)
	st = time.time()
	for i in xrange(10):
		z = cy_group_sampling_prob(x, ftm.value)
	en = time.time()
	print("Cython:", en-st, z)

##########################################
#Deterministic that generates a list of completed transition
#matrices for groups from the static transition matrix
#and exposures
def full_tmat(stm, exposure):
	return sp.group_tmats(exposure, stm)


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

	ma = fbmc.MassActionExposure("MA", l0_b, x)
	l0_ge = fbmc.GroupedExposures("L0_groups", l0_b, x, groups)	
	l1_ge = fbmc.GroupedExposures("L1_groups", l1_b, x, groups)
	l1_ce = fbmc.CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
	stm = fbmc.StaticTransitionMatrix("TM", e, g)

	st = time.time()
	for i in xrange(100):
		x = fbmc.full_tmat(stm.value, l1_ce.value)
	en = time.time()
	print("Full Transition Matrix:", en-st)
	return x

def full_tmat(stm, exposure):
	return sp.group_tmats(exposure, stm)


def full_tm_test_cy():
	l0_b = 0.9
	l1_b = 0.2
	e = 0.01
	g = 0.5

	x = np.array([[2,2,2,3,3,3],
			[0,1,2,2,2,2],
			[0,0,0,1,2,2],
			[0,0,0,0,0,0]])
	groups = [np.array([0,1]), np.array([2,3])]

	ma = fbmc.MassActionExposure("MA", l0_b, x)
	l0_ge = fbmc.GroupedExposures("L0_groups", l0_b, x, groups)	
	l1_ge = fbmc.GroupedExposures("L1_groups", l1_b, x, groups)
	l1_ce = fbmc.CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
	stm = fbmc.StaticTransitionMatrix("TM", e, g)

	st = time.time()
	for i in xrange(100):
		x = full_tmat(stm.value, l1_ce.value)
	en = time.time()
	print("Full Transition Matrix Cy:", en-st)
	return x

def full_tm_test_cy_b():
	l0_b = 0.9
	l1_b = 0.2
	e = 0.01
	g = 0.5

	x = np.array([[2,2,2,3,3,3],
			[0,1,2,2,2,2],
			[0,0,0,1,2,2],
			[0,0,0,0,0,0]])
	groups = [np.array([0,1]), np.array([2,3])]

	ma = fbmc.MassActionExposure("MA", l0_b, x)
	l0_ge = fbmc.GroupedExposures("L0_groups", l0_b, x, groups)	
	l1_ge = fbmc.GroupedExposures("L1_groups", l1_b, x, groups)
	l1_ce = fbmc.CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
	stm = fbmc.StaticTransitionMatrix("TM", e, g)
	ftm = fbmc.FullTransitionMatrix("FTM", l1_ce, stm)
	print(ftm.value)

def main():
	gr_sp_test(y,groups)
	gr_sp_cy_test(y, groups)

	x = full_tm_test()
	z = full_tm_test_cy()
	full_tm_test_cy_b()
	# print(x,z)
	# print(np.all(x == z))

if __name__ == '__main__':
	main()