import numpy as np
import random
import pymc
import mcmc
import hmm_np as hmm

#############################
#Log-likelihood of observation matrix for incidence
#with imperfectly observed times. Takes a matrix of 
#discretized start times and a dt value and returns
#an expanded matrix in which the times are uniformly
#distributed on an interval from [t/dt,(t+1)/dt]. This is
#useful for data reported on a daily basis, where the 
#actual onset time could be anywhere during that day
def uniform_obs_logp(value, obs, num_steps, prev_state, incidence):
	#value is a random state matrix corresponding to the intervals defined in the 
	#incidence matrix, in which 1 entries are incidence times and 0 are non incidence
	#incidence times is a list of 
	logp = 0.0
	pos_logp = np.log(1. / num_steps)
	for i, obs_row in enumerate(obs):
		t = np.where(obs_row == 1)[0]
		if len(t) == 0:
			continue
		else:
			t = t[0]

		#If there should be an event here, but there isn't, return 
		#ll = -inf
		row = value[i]
		if incidence not in row:
			logp = -np.inf
			return logp

		#Get the range in which the event can occur
		inc_range = (t*num_steps, (t*num_steps)+num_steps)

		#get index of the col containing the incident time
		i_time_index = np.where(row == incidence)[0]
		if len(i_time_index) == 0:
			logp = -np.inf
			return logp
		i_time_index = i_time_index[0]
		if inc_range[0] <= i_time_index and i_time_index < inc_range[1]:
			logp += pos_logp
		else:
			logp = -np.inf

	return logp


def random_uniform_obs(obs, num_steps, prev_state, incidence):
	num_col = num_steps * obs.shape[1]
	num_row = obs.shape[0]
	random_incidence = np.zeros((num_row, num_col), dtype = int)
	inc_indices = np.where(obs == 1)
	row_indices = inc_indices[0]
	col_indices = inc_indices[1]

	for ind,i in enumerate(row_indices):
		j = col_indices[ind]
		s_index = j*num_steps
		e_index = s_index + num_steps
		sample_j = np.random.randint(s_index, e_index)
		part_obs_row = -1 * np.ones(num_col, dtype = int)
		part_obs_row[sample_j] = incidence
		if sample_j > 0:
			part_obs_row[sample_j-1] = prev_state
		random_incidence[i] = part_obs_row

	return random_incidence

def expand_obs_incidence_first(obs, num_steps, prev_state, incidence):
	num_col = num_steps * obs.shape[1]
	num_row = obs.shape[0]
	random_incidence = np.zeros((num_row, num_col), dtype = int)
	inc_indices = np.where(obs == 1)
	row_indices = inc_indices[0]
	col_indices = inc_indices[1]
	for ind,i in enumerate(row_indices):
		j = col_indices[ind]
		s_index = j*num_steps
		e_index = s_index + num_steps
		sample_j = s_index
		part_obs_row = -1 * np.ones(num_col, dtype = int)
		part_obs_row[sample_j] = incidence
		if sample_j > 0:
			part_obs_row[sample_j-1] = prev_state
		random_incidence[i] = part_obs_row

	return random_incidence

def single_random_uniform_obs(obs, num_steps, prev_state, incidence):
	num_col = num_steps * len(obs)
	random_incidence = -1*np.ones(num_col, dtype = int)
	j = np.where(obs == 1)[0][0]

	s_index = j*num_steps
	e_index = s_index + num_steps
	sample_j = np.random.randint(s_index, e_index)
	random_incidence[sample_j] = incidence
	if sample_j > 0:
		random_incidence[sample_j-1] = prev_state
	return random_incidence


def uniform_obs_logp_test():
	x = np.array([[1,0,0,0,0,0],
				[0,1,0,0,0,0],
				[0,1,0,0,0,0]])

	num_steps = 4
	rval = random_uniform_obs(x, num_steps, 1, 2)
	print("RV",rval)
	print("RVLP", uniform_obs_logp(rval, x, num_steps, 1, 2))
	print("Single",single_random_uniform_obs(x[0], 4, 1, 2))

##############################################
#Pymc stochastic mapping the observations to the uncertain observation matrix
def UniformIntervalObservation(name, obs, num_steps, prev_state = 1, incidence = 2, value = None):
	return pymc.Stochastic(name = name, doc = "UniformIntervalObservation", value = value, cache_depth = 2, parents = {"obs":obs, "num_steps":num_steps, "prev_state":prev_state, "incidence":incidence}, logp = uniform_obs_logp, dtype = int)

def uniform_obs_test():
	x = np.array([[1,0,0,0,0,0],
				[0,1,0,0,0,0],
				[0,1,0,0,0,0]])

	num_steps = 4
	uo = UniformIntervalObservation("uo", x, num_steps)
	print(uo.value)
	print(uo.logp)

def expanded_starting_values(obs, num_steps, prev_state, incidence):
	#sample a random starting matrix
	start_obs = expand_obs_incidence_first(obs, num_steps, prev_state, incidence)
	#fill in the entries so that all states after the incident state 
	#are equal to the incidence value. This ensures that the sampler starts
	#on a nonzero part of the likelihood surface
	start_sm = []
	for row in np.copy(start_obs):
		#get index of first incidence value
		if incidence not in row:
			start_sm.append(row)
			continue

		start_incidence = np.where(row == incidence)[0][0]
		row[start_incidence:] = incidence
		if start_incidence >0:
			row[0:start_incidence-1] = 0
		start_sm.append(row)

	return (start_obs, np.array(start_sm))


##################################################
#Metropolis-Hastings step method for block-updating the state matrix
#and the observation matrix at the same time. This is particularly
#important for models w/o hidden states, i.e. where there is a 1->1 mapping
#between underlying state and observation. Without block sampling,
#we'll get ll = -inf if the observation doesn't line up properly with the
#states
class ObservationStateMetropolis(pymc.StepMethod):

	def __init__(self, stochastic, groups, init_probs, emission, tmat, num_steps, prev_state = 1, incidence = 2,verbose = -1, tally = False):
		try:
			len(stochastic)
		except TypeError:
			raise TypeError("ObservationStateMetropolis must receive both the state matrix and observation matrix!")
		self._id = "ObservationStateMetropolis"
		self.init_probs = init_probs
		self.emission = emission
		self.tmat = tmat
		self.obs_stoch = stochastic[0]
		self.state_stoch = stochastic[1]
		self.num_steps = num_steps
		self.prev_state = prev_state
		self.incidence = incidence
		self.rejected = 0
		self.accepted = 0
		self.adaptive_scale_factor = 1.
		#get row indices of individuals whose state can be manipulated
		#at this point, just check and see if the 0th item is a 0; if not
		#then it can be manipulated
		pymc.StepMethod.__init__(self, stochastic, verbose, tally)
		self.sample_indices = []
		for i,row in enumerate(self.obs_stoch.value):
			if -1 in row:
				self.sample_indices.append(i)

		#Create a reverse lookup to get the group index from the row index
		self.group_lookup = {}
		for i,g in enumerate(groups):
			for j,m in enumerate(g):
				self.group_lookup[m] = i


	def propose(self):

		#copy the state matrix
		self.old_obs_value = np.copy(self.obs_stoch.value)
		self.old_state_value = np.copy(self.state_stoch.value)
		
		new_obs_value =  np.copy(self.obs_stoch.value)
		new_state_value = np.copy(self.state_stoch.value)

		#Get the number of rows to sample from
		num_to_sample = max(3, 3*int(np.round(self.adaptive_scale_factor)))

		#draw individual indices to sample
		sample_index = random.sample(self.sample_indices,num_to_sample)
		if self.verbose > 0:
			print("Sampling %d" % num_to_sample)

		self.hf = 0.0
		for si in sample_index:

			#grab the observation from the old observation matrix
			last_obs = self.old_obs_value[si]

			#sample a new observation value
			#print(last_obs, self.num_steps, self.prev_state, self.incidence)
			
			#this needs to be changed to resample the new observation from the original one, i.e. the one
			#in 
			#print("O",self.obs_stoch.parents["obs"])
			new_obs = single_random_uniform_obs(self.obs_stoch.parents["obs"][si], self.num_steps, self.prev_state, self.incidence)

			new_obs_value[si] = new_obs

			#get the transition and current state matrices for the sampled individual
			tm = self.tmat.value[self.group_lookup[si]]
			st = new_state_value[si]

			if self.verbose >=2:
				print("From Obs:", last_obs, np.where(last_obs == 2), len(last_obs))
				print("From State:", st, np.where(st == 2), len(st))

			#get the proposal probability of the current state matrix using the current
			#transition matrix as the proposal density
			lv = hmm.fbg_propose(st, self.init_probs, self.emission, last_obs, tm)[1]

			#propose a new state matrix that goes with the new observation
			s_x, _, tv = hmm.fbg_propose(st, self.init_probs, self.emission, new_obs, tm)

			if self.verbose >=2:
				print("Old ll:", lv, tv)
				print("To   Obs:", new_obs, len(new_obs))
				print("To State:", s_x)

			#set the hastings factor equal to the proposal ratio p(s_t | o_t) / p(s_t+1 | o_t+1)
			self.hf += lv-tv
			new_state_value[si] = s_x
		if self.verbose >=2:
			print("HF =", self.hf)
			
		self.obs_stoch.value = new_obs_value
		self.state_stoch.value = new_state_value

	def hastings_factor(self):
		return self.hf

	def step(self):
		"""
		The default step method applies if the variable is floating-point
		valued, and is not being proposed from its prior.
		"""

		# Probability and likelihood for s's current value:

		if self.verbose>2:
			print_()
			print_(self._id + ' getting initial logp.')

		logp = self.logp_plus_loglike

		if self.verbose>2:
			print_(self._id + ' proposing.')
			
		# Sample a candidate value
		self.propose()

		# Probability and likelihood for s's proposed value:
		try:
		   logp_p = self.logp_plus_loglike
		except pymc.ZeroProbability:

			# Reject proposal
			if self.verbose>2:
				print_(self._id + ' rejecting due to ZeroProbability.')
			self.reject()

			# Increment rejected count
			self.rejected += 1

			if self.verbose>2:
				print_(self._id + ' returning.')
			return

		if self.verbose>2:
			print_('logp_p - logp: ', logp_p - logp)

		HF = self.hastings_factor()

		# Evaluate acceptance ratio
		if np.log(np.random.random()) > logp_p - logp + HF:

			# Revert s if fail
			self.reject()

			# Increment rejected count
			self.rejected += 1
			if self.verbose >= 2:
				print(self._id + ' rejecting')
		else:
			# Increment accepted count
			self.accepted += 1
			if self.verbose >= 2:
				print(self._id + ' accepting')

		if self.verbose > 2:
			print(self._id + ' returning.')

	def reject(self):
		self.obs_stoch.value = self.old_obs_value
		self.state_stoch.value = self.old_state_value

def state_metropolis_test():
	l0_b = 0.02
	l1_b = 0.2
	epsilon = 0.9
	gamma = 0.5
	num_steps = 4

	obs = np.array([[1,0,0,0,0,0],
				[0,1,0,0,0,0],
				[0,1,0,0,0,0],
				[0,0,0,1,0,0]])

	s_obs, ex_sm = expanded_starting_values(obs, num_steps, 1, 2)

	print("ExpandedObs", s_obs)
	print("Expanded Start", ex_sm)

	emission = np.identity(4)

	groups = [np.array([0,1]), np.array([2,3])]

	#Create a stochastic for the uniformly distributed hidden 'observations'
	r_obs = UniformIntervalObservation("uo", obs, num_steps, value = s_obs)

	#Create a stochastic for the state matrix
	sm = mcmc.StateMatrix("sm", ex_sm)

	init_probs = np.ones(4)/4.

	groups = [np.array([0,1]), np.array([2,3])]
	ma = mcmc.MassActionExposure("MA", l0_b, ex_sm)
	l0_ge = mcmc.GroupedExposures("L0_groups", l0_b, ex_sm, groups)	
	l1_ge = mcmc.GroupedExposures("L1_groups", l1_b, ex_sm, groups)
	l1_ce = mcmc.CorrectedGroupExposures("Corr_L1", ma, l0_ge, l1_ge)
	stm = np.array([[0.0, 0.0, 0.0, 0.0], 
					[0.0, 0.0, epsilon, 0.0],
					[0.0, 0.0, 0.0, gamma],
					[0.0, 0.0, 0.0, 0.0]])
	ftm = mcmc.FullTransitionMatrix("Full", stm, l1_ce)
	sm = ObservationStateMetropolis([r_obs, sm], groups, init_probs, emission, ftm, num_steps)
	sm.step()
	# sm.reject()	


def main():
	uniform_obs_logp_test()
	uniform_obs_test()
	state_metropolis_test()

if __name__ == '__main__':
	main()

