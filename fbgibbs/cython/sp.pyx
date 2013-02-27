
import numpy as np
cimport numpy as np
cimport cython

DTYPE_f = np.float64
DTYPE_i = np.int
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int_t DTYPE_it
ctypedef np.float64_t DTYPE_ft

@cython.boundscheck(False)
def sampling_probabilities(np.ndarray[DTYPE_it, ndim = 2] x, np.ndarray[DTYPE_ft, ndim = 3] tmat):
	cdef int t
	cdef int i
	cdef int num_col
	cdef int num_row
	cdef DTYPE_ft s_prob = 0.0
	cdef np.ndarray[DTYPE_ft, ndim = 3] log_tmat = np.log(tmat)
	#get the number of columns in the state matrix
	num_col = x.shape[1]
	num_row = x.shape[0]
	for t in range(num_col-1):
		for i in range(num_row):
			s_prob += log_tmat[t,x[i,t],x[i,t+1]]

	return s_prob


#Takes a matrix of exposures (rows correspond to groups, 
#columns to times) and a static transition matrix consisting
#of transition rates. Returns a list of transition matrices
#consisting of the probabilities of each state transition 
#over time for each group.
@cython.boundscheck(False)
def group_tmats(np.ndarray[DTYPE_ft, ndim = 2] exposure, np.ndarray[DTYPE_ft, ndim = 2] tmat):
	#print(exposure)
	gmats = [] 
	cdef int g
	cdef int t
	cdef int j
	cdef int i
	cdef int num_col
	cdef int num_rows
	cdef int num_ct
	cdef DTYPE_ft tot
	cdef DTYPE_ft lp
	num_rows = exposure.shape[0]
	cdef np.ndarray[DTYPE_ft, ndim = 1] e
	cdef np.ndarray[DTYPE_ft, ndim = 1] ct
	cdef np.ndarray[DTYPE_ft, ndim = 3] gmat
	cdef np.ndarray[DTYPE_ft, ndim = 2] leave_probs
	cdef np.ndarray[DTYPE_ft, ndim = 2] col_totals
	cdef int ct_len
	cdef int gm_len
	cdef int exposure_len
	# cdef np.ndarray[DTYPE_ft, ndim = 2] tot
	#Loop through every group's exposures
	for g in range(num_rows):
		e = exposure[g]
		exposure_len = len(e)
		gmat = np.array([tmat]*exposure_len, dtype = np.float64)

		#Set the S->E element to the exposure rate
		for i in range(exposure_len):
			gmat[i,0,1] = e[i]
		# gmat[:,0,1] = e

		#Get the total rate at which individuals
		#are transitioning out of the states in
		#gmat

		col_totals = np.sum(gmat, axis = 2)
		num_col = col_totals.shape[0]
		leave_probs = 1.0 - np.exp(-col_totals)
		# leave_probs = np.ndarray(col_totals.shape, dtype = DTYPE_ft)
		# for i in range(col_totals.shape[0]):
		# 	for j in range(col_totals.shape[1]):
		# 		leave_probs[i,j] = 1.0 - np.exp(-col_totals[i,j])

		num_col = col_totals.shape[0]
		for t in range(num_col):
			ct = col_totals[t]
			#If the row totals to > 0, then divide all
			#of the elements by the total; this gives
			#the proportion of the leave rate associated 
			#with this element
			ct_len = ct.shape[0]
			gm_len = gmat[0,0].shape[0]
			for j in range(ct_len):
				tot = ct[j]
				lp = leave_probs[t,j]
				if tot > 0:
					for i in range(gm_len):
						gmat[t,j,i] = (gmat[t,j,i]/tot)*lp
				gmat[t,j,j] = 1.-lp
		gmats.append(gmat)
	return np.array(gmats)