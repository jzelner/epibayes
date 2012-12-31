import numpy as np
import theano.tensor as T
import theano

def inputs():
	num_row = 10
	B = 0.05
	G = 0.5

	states = np.array([1., 0.])

	contact_states = np.array([[10., 5.] for i in xrange(num_row)])

	transmission_mat = np.array([[[0., B], [0., 0.]] for i in xrange(num_row)])

	transition_mat = np.array([[[0., 0.], [G, 0.]] for i in xrange(num_row)])

	return (B,G,states, contact_states, transition_mat, transmission_mat)


def theanomain():
	
	contact_states = T.dvector("contact_states")

	transition_mat = T.dmatrix("transition_mat")
	transmission_mat = T.dmatrix("transmission_mat")
	
	residual_mat = 1.0 - T.exp(-(transition_mat + (transmission_mat*contact_states)))
	markov_mat = residual_mat + T.diag((1.0 - T.sum(residual_mat, axis = 1)))

	exposureFunc = theano.function([contact_states, transition_mat, transmission_mat], markov_mat)

	_, _, _, cs, tm, trm = inputs()
	print(exposureFunc(cs, tm, trm))

def main():
	B,G,states, contact_states, transition_mat, transmission_mat = inputs()

	for i in xrange(len(contact_states)):
		print(transmission_mat[i]*contact_states[i])
		
	# print(transition_mat+transmission_mat)

	q = 1.0 - np.exp(-(transition_mat + (transmission_mat*contact_states)))
	print(np.sum(q, axis = 2))
	# #Sum the columns of this matrix to get the total probability
	q += np.diag((1.0 - np.sum(q, axis = 2)))
	print(q)



if __name__ == '__main__':
	main()
	# theanomain()