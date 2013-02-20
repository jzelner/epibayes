import numpy as np
import weibull
import random
import json

N = 1500

#make groups of 2
groups = []
offset = 0
for i in xrange(N/2):
	groups.append(np.array([offset,offset+1]))
	offset += 2

b = 1.0
l1_b = 0.15
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
iX = np.zeros((N,nSteps), dtype = int)

#Select the first infections
num_init_inf = np.random.binomial(N, p_init)
indices = np.arange(N)
first_inf = np.array(random.sample(indices, num_init_inf))

X[first_inf, 0] = 2
iX[first_inf, 0] = 1.

#Start looping through time, calculating the force of infection on each step

infection_density = np.zeros(nSteps+1)
for i in xrange(1, nSteps+1):
	print(i)
	infection_density[i] = weibull.qcdf(i, 0.5, q50, 0.9, q90)-weibull.qcdf(i-1, 0.5, q50, 0.9, q90)
print(infection_density)

for t in xrange(nSteps-1):
	print(t)
	inf_t = iX[:,t]
	inf_days = inf_t[np.where(inf_t > 0)]
	infdens = infection_density[inf_days]
	#print("INFDAYS", zip(inf_days, after_dens-before_dens))
	top_foi = (b/N) * np.sum(infdens)

	#Loop through each group, getting the infection density
	ind_density = np.zeros(N)
	for g in groups:
		gfoi = l1_b * np.sum(infection_density[inf_t[g]])
		gfoi_top = b * np.sum(infection_density[inf_t[g]])

		ind_density[g] = top_foi - gfoi_top + gfoi

	susceptibles = np.where(X[:,t] == 0)[0]
	num_susceptibles = len(susceptibles)

	latent = np.where(X[:,t] == 1)[0]
	num_latent = len(latent)

	infectious = np.where(X[:,t] == 2)[0]
	num_infectious = len(infectious)

	infp = 1.0 - np.exp(-ind_density)
	infect_ids = np.arange(N)[np.random.random(N) < infp]
			
	#sample individuals to infect and update their state
	X[susceptibles,t+1] = 0
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

for i,g in enumerate(groups):
	#If both are zeros,then skip; but if one has
	#an infection, we include it.
	g_obs = []
	zgobs = 0
	for x in X[g]:
		if x[0] == 2:
			o = list(-1 * np.ones_like(x))
			o[0] = 2
			g_obs.append(o)
		if 1 in x:
			onset_index = np.where(x == 2)[0]
			if len(onset_index) == 0:
				num_non_inf += 1
				continue

			o = -1 *np.ones_like(x)
			o[onset_index[0]] = 2
			o[onset_index[0]-1] = 1
			g_obs.append(list(o))
		else:
			zgobs += 1

	if len(g_obs) == 0:
		num_non_inf += 2
	elif zgobs == 1:
		g_obs.append(list(np.zeros_like(x)))
		obs.extend(g_obs)
	if zgobs == 0:
		obs.extend(g_obs)
num_inf = len(obs)

g = {"Groups":{"ID":"Ship", "Members":list(np.arange(num_inf))}}

outputs = {"groups":g, "observations":obs, "l0_non_infected": num_non_inf}

out_file = open("nested_weibull_test_run.json", 'w')
json.dump(outputs, out_file)

