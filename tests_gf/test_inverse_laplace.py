import numpy as np
import pytest
import sage.all
import sys, os

from timeit import default_timer as timer

import gf.gf as gflib
import gf.matrix_representation as gfmat
import gf.partial_fraction_expansion as gfpar

@pytest.mark.inverse_laplace
class Test_Paths:

	def test_inverse_laplace(self, return_gf):
		variables_array, (paths_mat, eq_mat), numerical = return_gf
		delta_idx = len(variables_array) - 5
		subset_no_delta = np.arange(len(variables_array))!=delta_idx
		variables_array_no_delta = variables_array[subset_no_delta]
		eq_mat_no_delta = eq_mat[:,:,subset_no_delta]
		time = sage.all.SR.var('T')
		
		if numerical:
			values_array = np.random.rand(len(variables_array)-1)
			params_dict = {k:v for k,v in zip(variables_array_no_delta, values_array)}
			params_dict[time] = 1.0
		else:
			params_dict=dict()
		
		paths_with_delta = gfmat.split_paths_laplace(paths_mat, eq_mat, delta_idx)
		delta_in_nom_all = eq_mat[:, 0, delta_idx]==1
		#for each of those paths, take inverse laplace two ways 

		for path_const, path_to_invert in paths_with_delta:
			if len(path_to_invert)>0:	
				eq = np.prod(gfmat.equations_from_matrix(eq_mat[path_to_invert], variables_array))
				#using custom function
				multiplier_array = eq_mat_no_delta[path_to_invert]
				if numerical:
					eq = eq.subs(params_dict)
					custom_inverse = gflib.inverse_laplace_single_event(multiplier_array, values_array, time, delta_in_nom_all[path_to_invert])
				else:
					custom_inverse = gflib.inverse_laplace_single_event(multiplier_array, variables_array_no_delta, time, delta_in_nom_all[path_to_invert])
				regular_inverse = gflib.return_inverse_laplace(eq, variables_array[delta_idx])
				self.compare_inverse_laplace(regular_inverse, np.sum(custom_inverse), params_dict, numerical)

	def compare_inverse_laplace(self, v1, v2, params_dict, numerical):
		print('v1:', v1)
		print('v2:', v2)
		if numerical: #numerical evaluation needed
			assert np.isclose(float(v1.subs(params_dict)), float(v2.subs(params_dict)))
		else:
			assert v1==v1

	@pytest.fixture(
		scope='class', 
		params=[
			([(1,2,0)], sage.all.SR.var('E'), None, None, False),
			([(1,2,0)], sage.all.SR.var('E'), [(2,1)], sage.all.SR.var('M'), True)
			],
		ids=[
			'DIV', 
			'IM'
			],
		)
	def return_gf(self, request):
		sample_list = [(),('a','a'),('b','b')]
		ancestral_pop = 0
		coalescence_rates = (sage.all.SR.var('c0'), sage.all.SR.var('c1'), sage.all.SR.var('c2'))
		coalescence_rate_idxs = (0, 1, 2)
		k_max = {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict_mat = gfmat.make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=mutype_labels)
		branchtype_dict_chain = gflib.make_branchtype_dict(sample_list, mapping='unrooted', labels=mutype_labels)
		exodus_direction, exodus_rate, migration_direction, migration_rate, numerical = request.param
		
		variables_array = list(coalescence_rates)
		migration_rate_idx, exodus_rate_idx = None, None
		if migration_rate!=None:
			migration_rate_idx = len(variables_array)
			variables_array.append(migration_rate)
		if exodus_rate!=None:
			exodus_rate_idx = len(variables_array)
			variables_array.append(exodus_rate)
		variables_array += [sage.all.SR.var(m) for m in mutype_labels]
		variables_array = np.array(variables_array, dtype=object)

		gfobj = gflib.GFMatrixObject(
			sample_list, 
			coalescence_rate_idxs, 
			branchtype_dict_mat,
			exodus_rate=exodus_rate_idx,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate_idx,
			migration_direction=migration_direction
			)
		gf_mat = list(gfobj.make_gf())

		return (variables_array, gf_mat, numerical)

@pytest.mark.partial_fraction_expansion
class Test_pfe_algorithm:
	def test_3poles(self):
		multiplicities = np.array([2, 3, 2], dtype=np.int8)
		max_multiplicity = np.max(multiplicities)
		poles = np.array([-1,-2, -3], dtype = np.int8)
		B = gfpar.return_beta(poles)
		A = gfpar.return_binom_coefficients(max_multiplicity)
		results = gfpar.derive_residues(A, B, multiplicities, max_multiplicity)
		print(results)
		expected = np.array([
			[-1, 0.25, 0],
			[2, 0, 1],
			[-1, -0.25, 0]], 
			dtype = np.float64)
		assert np.array_equal(results, expected)

	def test_4poles(self):
		multiplicities = np.array([3, 4, 2, 3], dtype=np.int8)
		max_multiplicity = np.max(multiplicities)
		poles = np.array([-1, -2, -3, -4], dtype = np.int8)
		B = gfpar.return_beta(poles)
		A = gfpar.return_binom_coefficients(max_multiplicity)
		results = gfpar.derive_residues(A, B, multiplicities, max_multiplicity)
		print(results)
		expected = np.array([
			[0.18904321, -0.0555556, 0.00925926, 0.0],
			[0.15625, -0.375, 0.0625, -0.125],
			[-0.3125, -0.125, 0.0, 0.0],
			[-0.03279321, -0.01157407, -0.00231481, 0.0]], 
			dtype = np.float64)
		assert np.allclose(results, expected)