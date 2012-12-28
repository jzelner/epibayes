import numpy as np
import theano.tensor as T
import theano

def inputs():
	B = 0.05
	G = 0.5

	states = np.array([1., 0.])

	contact_states = np.array([10., 5.])

	transmission_mat = np.array([[0., B], [0., 0.]])

	transition_mat = np.array([[0., 0.], [G, 0.]])

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

	q = 1.0 - np.exp(-(transition_mat + (transmission_mat*contact_states)))

	#Sum the columns of this matrix to get the total probability
	q += np.diag((1.0 - np.sum(q, axis = 1)))



if __name__ == '__main__':
	main()
	theanomain()