import numpy as np
import weibull
import random
import json

N = 1500

b = 1.0

q50 = 1.0
q90 = 5.0
p_init = 0.05

T = 11
dt = 0.5
nSteps = int(T/dt)
e = 200.0 * dt
#make a matrix of zeros with N rows and T/dt cols
X = np.zeros((N, nSteps),dtype = int)

#make a separate matrix tracking the day of infection
#each individual is on, starting with 1
iX = np.zeros((N,nSteps), dtype = float)

#Select the first infections
num_init_inf = np.random.binomial(N, p_init)
indices = np.arange(N)
first_inf = np.array(random.sample(indices, num_init_inf))

X[first_inf, 0] = 2
iX[first_inf, 0] = 1.

#Start looping through time, calculating the force of infection on each step
for t in xrange(nSteps-1):
	print(t)
	inf_t = iX[:,t]
	inf_days = inf_t[np.where(inf_t > 0)]
	before_day = inf_days - 1
	before_dens = weibull.qcdf(before_day, 0.5, q50, 0.9, q90)
	after_dens = weibull.qcdf(inf_days, 0.5, q50, 0.9, q90)
	#print("INFDAYS", zip(inf_days, after_dens-before_dens))
	foi = (b/N) * np.sum((after_dens-before_dens))

	susceptibles = np.where(X[:,t] == 0)[0]
	num_susceptibles = len(susceptibles)

	latent = np.where(X[:,t] == 1)[0]
	num_latent = len(latent)

	infectious = np.where(X[:,t] == 2)[0]
	num_infectious = len(infectious)

	if foi > 0.0:
		infp = 1.0 - np.exp(-foi)
		num_to_infect = np.random.binomial(num_susceptibles, infp)
		
		#sample individuals to infect and update their state
		X[susceptibles,t+1] = 0
		infect_ids = random.sample(susceptibles, num_to_infect)
		X[infect_ids,t+1] = 1

		#now sample latent individuals progressing to infectiousness
		num_onset = 0
		if num_latent > 0:
			num_onset = np.random.binomial(num_latent, 1.0 - np.exp(-e))
			X[latent,t+1] = 1
			onset_ids = random.sample(latent, num_onset)
			X[onset_ids,t+1] = 2

		#update infectious days
		iX[infectious, t+1] = iX[infectious, t] + 1
		X[infectious, t+1] = 2

		if num_onset > 0:
			iX[onset_ids, t+1] = 1.0

#Process simulated data so that infectious individuals are unobs (-1) for all points
#except onset time
num_non_inf = 0
obs = []

for i,x in enumerate(X):
	if x[0] == 2:
		o = list(-1 * np.ones_like(x))
		o[0] = 2
		obs.append(o)
		continue
	if 1 in x:
		onset_index = np.where(x == 2)[0]
		if len(onset_index) == 0:
			num_non_inf += 1
			continue

		o = -1 *np.ones_like(x)
		o[onset_index[0]] = 2
		o[onset_index[0]-1] = 1
		obs.append(list(o))
	else:
		num_non_inf += 1

num_inf = len(obs)

g = {"Groups":{"ID":"Ship", "Members":list(np.arange(num_inf))}}

outputs = {"groups":g, "observations":obs, "l0_non_infected": num_non_inf}

out_file = open("weibull_test_run.json", 'w')
json.dump(outputs, out_file)

