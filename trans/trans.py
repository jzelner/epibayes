import theano
import theano.tensor as T
import numpy as np

#The distance matrix gives the pairwise
#distances between locations
distance = T.dmatrix('distance')

#Population is a vector of population
#sizes by village
population = T.dvector('population')
sender_tau = T.dscalar('sender_tau')
receiver_tau = T.dscalar('receiver_tau')
distance_tau = T.dscalar('distance_tau')

gravity_fn = T.outer(population**sender_tau, population**receiver_tau) / (distance ** distance_tau) * (1.0 - T.eye(distance.shape[0], distance.shape[1]))

gravity = theano.function([population, distance, sender_tau, receiver_tau, distance_tau], gravity_fn)

#The contact matrix gives the weight of 
#contact between pairs of nodes
contact = T.dmatrix('adjacency')
I = T.dmatrix("Infected")
foi = T.dmatrix("FOI")

#The dot product of the adjacency matrix
#and the underlying state matrix gives
#the  force of infection on the 
#susceptible individuals, represented by 1 - I,
#since this is just an SIS model.
exposure = (T.dot(contact, I))*(1.0-I)

#Expose is a function that takes an adjacency
#matrix and an infection state matrix and returns
#the amount of exposure on each susceptible 
#individual.
expose = theano.function([contact, I], exposure)

if __name__ == '__main__':
	
	#The adj
	contact_mat = np.array([[0., 1.0, 1.0], [0.5, 0., 0.5], [1.0, 1.0, 0.]])
	imat = np.array([[1., 0., 0.,0.], [0., 1., 1.,0.], [0.,1.,0.,1.]])
	smat = np.ones((3,4))-imat

	distance_mat = np.array([[1., 10., 50.], [4., 1., 30.], [10.0, 15.0, 1.]])
	population = np.array([100., 20., 30.])

	grav = gravity(population, distance_mat, 0.5, 0.1, 2.0)

	print(grav)

	print(imat)
	
	print(expose(grav, imat))