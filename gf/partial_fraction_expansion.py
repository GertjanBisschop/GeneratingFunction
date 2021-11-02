import numpy as np
import numba
import scipy.linalg

"""
Partial Fraction Expansion algorithm from
Y. Ma, J. Yu and Y. Wang, "An Easy Pure Algebraic 
Method for Partial Fraction Expansion of Rational 
Functions With Multiple High-Order Poles," in IEEE 
Transactions on Circuits and Systems I: Regular Papers, 
vol. 61, no. 3, pp. 803-810, March 2014, 
doi: 10.1109/TCSI.2013.2283998.
"""

def return_beta(denominator, dtype=np.float64):
	return denominator[:,None]-denominator[None,:].astype(dtype)

def return_binom_coefficients(n):
	return scipy.linalg.pascal(n)

def return_first_two(A, B, m1, m2, max_multiplicity, dtype=np.float64):
	result = np.zeros((2, max_multiplicity), dtype=dtype)
	m1_idxs = np.arange(1, m1+1,dtype=int)
	m2_idxs = np.arange(1, m2+1,dtype=int)
	leading1 = (-np.ones(m1, dtype=int))**(m1-m1_idxs)
	leading2 = (-np.ones(m2, dtype=int))**(m2-m2_idxs)
	c1 = A[m2-1,m1-m1_idxs]/B[0,1]**(m1+m2-m1_idxs)
	c2 = A[m1-1,m2-m2_idxs]/B[1,0]**(m1+m2-m2_idxs)
	result[0,:c1.size] = leading1 * c1
	result[1,:c2.size] = leading2 * c2
	return result

def derive_residues(A, B, multiplicities, max_multiplicity, dtype=np.float64):
	"""
	:param A array binomomial coefficients order len(max_multiplicity)
	:param B array differences between poles
	:param array multiplicities
	:param int max_multiplicities 
	"""
	#check if there are two poles
	num_poles = multiplicities.size
	result = np.zeros((num_poles, max_multiplicity), dtype=dtype)
	result[0:2] = return_first_two(
		A,
		B,
		multiplicities[0], 
		multiplicities[1],
		max_multiplicity,
		dtype
	)
	for n in range(3, num_poles+1):
		result = derive_residues_next(A,B, multiplicities, result, n, dtype)

	return result

def derive_residues_next(A, B, multiplicities, intermediate_result, num_poles, dtype):
	result = np.zeros_like(intermediate_result)
	for i, multiplicity in enumerate(multiplicities[:num_poles-1]):
		for L in range(multiplicity):
			beta = B[i, num_poles-1]**(np.float64(multiplicities[num_poles-1] - (L + 1)))
			leading_ones = (-1)**(L + 1)
			temp = 0
			for j in range(L, multiplicity):
				temp+=intermediate_result[i,j]*A[multiplicities[num_poles-1]-1, j-L]/(B[num_poles-1,i]**(j+1))
			result[i,L] = leading_ones * temp / beta
	
	for L in range(multiplicities[num_poles-1]):
		for i in range(num_poles-1):
			temp = 0
			beta = B[i, num_poles-1]**(np.float64(multiplicities[num_poles-1] - (L + 1)))
			for j in range(multiplicities[i]):
				temp+=intermediate_result[i,j] * A[j, multiplicities[num_poles -1] - (L + 1)]/(B[num_poles-1,i]**(j+1))    
			result[num_poles-1, L]+=temp/beta
	return result

# numba compiled functions 
# jit compilation switched off during testing (see tests_gf.conftest.py)
@numba.njit(numba.float64[:,:](numba.uint64[:,:], numba.float64[:,:],numba.int64,numba.int64,numba.int64))
def return_first_two_numba(A, B, m1, m2, max_multiplicity=None):
	result = np.zeros((2, max_multiplicity), dtype=np.float64)
	m1_idxs = np.arange(1, m1+1,dtype=np.int8)
	m2_idxs = np.arange(1, m2+1,dtype=np.int8)
	leading1 = (-np.ones(m1, dtype=np.int8))**(m1-m1_idxs)
	leading2 = (-np.ones(m2, dtype=np.int8))**(m2-m2_idxs)
	c1 = np.zeros(m1, dtype=np.float64)
	for m1_idx in m1_idxs:
		c1[m1_idx-1] = A[m2-1,m1-m1_idx]/B[0,1]**(m1+m2-m1_idx)
	c2 = np.zeros(m2, dtype=np.float64)
	for m2_idx in m2_idxs:
		c2[m2_idx-1] = A[m1-1,m2-m2_idx]/B[1,0]**(m1+m2-m2_idx)
	result[0,:c1.size] = leading1 * c1
	result[1,:c2.size] = leading2 * c2
	return result

@numba.njit(numba.float64[:,:](numba.uint64[:,:], numba.float64[:,:],numba.int64[:],numba.float64[:,:],numba.int64))
def derive_residues_next_numba(A, B, multiplicities, intermediate_result, num_poles):
	result = np.zeros_like(intermediate_result)
	for i, multiplicity in enumerate(multiplicities[:num_poles-1]):
		for L in range(multiplicity):
			beta = B[i, num_poles-1]**(np.float64(multiplicities[num_poles-1] - (L + 1)))
			leading_ones = (-1)**(L + 1)
			temp = 0
			for j in range(L, multiplicity):
				temp+=intermediate_result[i,j]*A[multiplicities[num_poles-1]-1, j-L]/(B[num_poles-1,i]**(j+1))
			result[i,L] = leading_ones * temp / beta
	
	for L in range(multiplicities[num_poles-1]):
		for i in range(num_poles-1):
			temp = 0
			beta = B[i, num_poles-1]**(np.float64(multiplicities[num_poles-1] - (L + 1)))
			for j in range(multiplicities[i]):
				temp+=intermediate_result[i,j] * A[j, multiplicities[num_poles -1] - (L + 1)]/(B[num_poles-1,i]**(j+1))    
			result[num_poles-1, L]+=temp/beta
	return result

@numba.njit(numba.float64[:,:](numba.uint64[:,:], numba.float64[:,:], numba.int64[:], numba.int64))
def derive_residues_numba(A, B, multiplicities, max_multiplicity):
	#check if there are two poles
	num_poles = multiplicities.size
	result = np.zeros((num_poles, max_multiplicity), np.float64)
	result[0:2] = return_first_two_numba(
		A,
		B,
		multiplicities[0], 
		multiplicities[1],
		max_multiplicity
	)
	for n in range(3, num_poles+1):
		result = derive_residues_next_numba(A,B, multiplicities, result, n)

	return result

# single step functions

def single_n_1(A, B, multiplicities, intermediate_result, num_poles):
	results = np.zeros_like(intermediate_result)
	for i, multiplicity in enumerate(multiplicities[:num_poles-1]):
		for L in range(multiplicity):
			beta = B[i, num_poles-1]**(np.float64(multiplicities[num_poles-1] - (L + 1)))
			leading_ones = (-1)**(L + 1)
			temp = 0
			for j in range(L, multiplicity):
				temp+=intermediate_result[i,j]*A[multiplicities[num_poles-1]-1, j-L]/(B[num_poles-1,i]**(j+1))
			results[i,L] = leading_ones * temp / beta
	return results

def single_n(A, B, multiplicities, intermediate_result, num_poles):
	result = np.zeros_like(intermediate_result[0])
	for L in range(multiplicities[num_poles-1]):
		for i in range(num_poles-1):
			temp = 0
			beta = B[i, num_poles-1]**(np.float64(multiplicities[num_poles-1] - (L + 1)))
			for j in range(multiplicities[i]):
				temp+=intermediate_result[i,j] * A[j, multiplicities[num_poles -1] - (L + 1)]/(B[num_poles-1,i]**(j+1))    
			result[L]+=temp/beta
	return result