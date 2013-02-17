import numpy as np
import pymc
import math
from scipy import stats
import theano
import theano.tensor as T

x = T.dvector("x")
a = T.dscalar("a")
b = T.dscalar("b")

s = (b / a) * ((x / a)**(b-1)) * T.exp(-((x/a)**b))
log_pdf = T.sum(T.log(s))

cdf = 1.0 - T.exp((-(x/a)**b))

PDF = theano.function([x,a,b], s)
CDF = theano.function([x,a,b], cdf)
logPDF = theano.function([x,a,b], log_pdf)

def icdf(q, a, b):
	return a * (-np.log(1.0 - q))**(1.0 / b)

def beta(q1, x1, q2, x2):
	t1 = np.log(-np.log(1.0 - q2))
	t2 = np.log(-np.log(1.0 - q1))
	t3 = np.log(x2) - np.log(x1)

	return (t1 - t2) / t3

def alphaOfBeta(b, x1, q1):
	return x1 / ((-np.log(1.0 - q1))**(1.0 / b))

def weibullFromQuantiles(q1, x1, q2, x2):
	b = beta(q1, x1, q2, x2)
	a = alphaOfBeta(b, x1, q1)
	return (a, b)

def qdist(x, q1, x1, q2, x2):
	try:
		len(x)
	except TypeError:
		x = np.array([x])
	a, b = weibullFromQuantiles(q1, x1, q2, x2)
	return PDF(x, a, b)

def qcdf(x, q1, x1, q2, x2):
	try:
		len(x)
	except TypeError:
		x = np.array([x])
	a, b = weibullFromQuantiles(q1, x1, q2, x2)
	return CDF(x, a, b)

def weibullMean(a, b):
	try:
		return a * math.gamma(1.0 + (1.0 / b))
	except:
		return np.inf

##############################################
#Exposure functions for time-varying infectiousness,
#assume that the infection state is absorbing
def mass_action_weibull_exposure(x, b, q1, x1, q2, x2, infstate = 2):
	if len(x.shape) == 1:
		x = np.array([x])

	maxdur = 0.
	inf_vals = np.array([])
	total_exposure = np.zeros(len(x[0]))
	for row in x:
		inf_times = np.where(row == infstate)[0]
		infdur = len(inf_times)
		if infdur == 0:
			continue
		start_time = inf_times[0]
		if infdur > maxdur:
			inf_vals = np.append(inf_vals, np.diff(qcdf(np.arange(maxdur, infdur+1, dtype = float), q1, x1, q2, x2)))
			maxdur = infdur
		
		total_exposure[start_time:] += inf_vals[0:infdur]

	return b * total_exposure

def grouped_weibull_exposure(x, groups, b, q1, x1, q2, x2, infstate = 2):
	maxdur = 0.
	inf_vals = np.array([])

	ge = []
	for g in groups:
		total_exposure = np.zeros(len(x[0]))
		for row in x[g]:
			inf_times = np.where(row == infstate)[0]
			infdur = len(inf_times)
			if infdur == 0:
				continue
			start_time = inf_times[0]
			if infdur > maxdur:
				inf_vals = np.append(inf_vals, np.diff(qcdf(np.arange(maxdur, infdur+1, dtype = float), q1, x1, q2, x2)))
				maxdur = infdur

			print(inf_vals[0:infdur])
			print(total_exposure[start_time:])
			total_exposure[start_time:] += inf_vals[0:infdur]
		ge.append(b*total_exposure)

	return np.array(ge)




if __name__ == '__main__':
	import time
	(a,b) = weibullFromQuantiles(0.25, 1.0, 0.75, 20.0)
#	qd = qdist(np.array(1, 0.5, 1.0, 0.9, 2.0))
	x = np.array([0,0,0,1,1,1,2,2,2,2])
	x = np.vstack([x]*4)
	groups = [np.array([0,1]), np.array([2,3])]
	s = time.time()
	ex = mass_action_weibull_exposure(x, 1.0, 0.5, 2.0, 0.9, 4.0)
	gex = grouped_weibull_exposure(x, groups, 1.0, 0.5, 2.0, 0.9, 4.0)
	e = time.time()
	print(ex)
	print(gex)
