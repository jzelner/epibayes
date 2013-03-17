import numpy as np
import pymc

def group_i(ind_r, onset_times, groups, num_groups, max_time):
	#make a return I matrix with num rows = num_groups and num
	#col equal to max_time, since we don't care about the 
	#force of infection beyond the end of the 2nd to last
	#period
	return_I = np.zeros((num_groups, max_time+1), dtype = int)

	for i,g in enumerate(groups):
		group_onsets = np.bincount(onset_times[g])
		return_I[i,0:len(group_onsets)] = group_onsets
	return return_I

def GroupI(name, ind_r, onset_times, groups, trace = False):
	#calculate the number of groups and maximum size before
	#making the stochastic object
	num_groups = len(groups)
	max_time = np.max(onset_times)

	return pymc.Deterministic(name = name, doc = "GroupI", dtype = int, cache_depth = 2, parents = {"ind_r": ind_r, "onset_times": onset_times, "groups": groups, "num_groups": num_groups, "max_time": max_time}, eval = group_i)


def group_I_test():
	ind_r = np.array([0,1,2,5,1], dtype = int)
	onset_times = np.array([0,1,1,2,1], dtype = int)
	groups = [np.array([0,1], dtype = int), np.array([2,3,4], dtype = int)]

	x = GroupI("Group I", ind_r, onset_times, groups)
	print("GROUP I VALUES", x.value)


def group_s(group_I, group_sizes):
	#make a return I matrix with num rows = num_groups and num
	#col equal to max_time, since we don't care about the 
	#force of infection beyond the end of the 2nd to last
	#period
	return_S = np.ones_like(group_I, dtype = int)
	for i,s in enumerate(group_sizes):
		return_S[i] *= s
	return_S -= np.cumsum(group_I, axis = 1)
	return return_S

def GroupS(name, group_I, group_sizes, trace = False):
	#calculate the number of groups and maximum size before
	#making the stochastic object
	
	return pymc.Deterministic(name = name, doc = "GroupS", dtype = int, cache_depth = 2, parents = {"group_I": group_I, "group_sizes": group_sizes}, eval = group_s)

def group_s_test():
	ind_r = np.array([0,1,2,5,1], dtype = int)
	onset_times = np.array([0,1,1,2,1], dtype = int)
	groups = [np.array([0,1], dtype = int), np.array([2,3,4], dtype = int)]

	gi = GroupI("Group I", ind_r, onset_times, groups)
	group_sizes = np.array([20,10])
	gs = GroupS("Group S", gi, group_sizes)
	print("group S", gs.value)


#take a vector of R0 values and onset times and create 
#a vector of FOI for each day
def group_foi(group_I, group_weights):
	return np.dot(group_weights, group_I)	

def GroupFOI(name, group_I, group_weights):	
	return pymc.Deterministic(name = name, doc = "GroupFOI", dtype = float, cache_depth = 2, parents = {"group_I": group_I, "group_weights": group_weights}, eval = group_foi)

def group_foi_test():
	ind_r = np.array([0,1,2,5,1], dtype = int)
	onset_times = np.array([0,1,1,2,1], dtype = int)
	groups = [np.array([0,1], dtype = int), np.array([2,3,4], dtype = int)]

	num_groups = 2
	max_time = 2

	x = group_i(ind_r, onset_times, groups, num_groups, max_time)
	w = np.array([[0.9, 0.1], [0.1,0.9]])
	gf = GroupFOI("gfoi", x, w)
	print("GROUP FOI", gf.value)

def uniform_outgroup_weights(p_within, num_groups):
	#make weight matrix of size 
	out_group_proportion = 1. - p_within
	per_group_proportion = out_group_proportion / (num_groups-1)
	weights = per_group_proportion * np.ones((num_groups, num_groups))
	q = np.diag_indices_from(weights)
	weights[q] = p_within
	return weights

def UniformOutgroupWeights(name, p_within, num_groups):
	return pymc.Deterministic(name = name, doc = "UniformOutgroupWeights", dtype = float, cache_depth = 2, parents = {"p_within": p_within, "num_groups": num_groups}, eval = uniform_outgroup_weights)

def uniform_outgroup_weights_test():
	p_within = 0.9
	num_groups = 2
	uw = UniformOutgroupWeights("uw", p_within, num_groups)
	print("Uniform outgroup", uw.value)


def main():
	group_I_test()
	group_foi_test()
	uniform_outgroup_weights_test()
	group_s_test()

if __name__ == '__main__':
	main()