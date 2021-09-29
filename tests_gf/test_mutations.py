import itertools
import numpy as np
import pytest
import sage.all
import sys, os

import gf.mutations as mutations
import gf.togimble as togimble
import gf.matrix_representation as gfmat
import gf.partial_fraction_expansion as gfpar
import gf.gf as gflib

@pytest.mark.muts
class Test_mutypetree:
	def test_incomplete_tree(self):
		all_mutypes = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),(0,0,2,2),(0,0,3,2),
			(2,1,3,0),(2,1,2,3)
			]
		)
		root = (0,0,0,0)
		max_k = (2,2,2,2)
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		result = {
				(0,0,0,0): [(0,0,2,2),(1,0,0,0)],
				(0,0,0,2): [(0,0,3,2),],
				(1,0,0,0): [(2,1,0,0),], 
				(2,1,0,0): [(2,1,2,0), (2,1,3,0)],
				(2,1,2,0): [(2,1,2,3)]
				}
		assert result==mutype_tree

	def test_complete_tree(self):
		root = (0,0,0,0)
		max_k = (2,2,2,2)
		all_mutypes = sorted(mutations.return_mutype_configs(max_k))
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		mutype_tree_single_digit = mutations.make_mutype_tree_single_digit(all_mutypes, root, max_k)
		assert mutype_tree == mutype_tree_single_digit

	def test_muts_to_gimble(self):
		gf = 1
		max_k = (2,2,2,2)
		root = (0,0,0,0)
		exclude = [(2,3),]
		restrict_to = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),(0,0,2,2),(0,0,3,2),
			(2,1,3,0),(2,1,2,3)
			]
		)
		mutypes = tuple(f'm_{idx}' for idx in range(len(max_k)))
		gfEvaluatorObj = togimble.gfEvaluator(gf, max_k, mutypes, exclude=exclude, restrict_to=restrict_to)
		expected_all_mutation_configurations = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),
			(2,1,3,0),(2,1,0,0),(2,1,1,0),(2,1,2,0)
			])

		print(gfEvaluatorObj.mutype_tree)
		#types to add:
		#(0,0,0,2),(0,0,1,2),(0,0,2,2),(2,1,2,0),(2,1,2,1),(2,1,2,2),(2,1,0,0),(2,1,1,0),(2,1,2,0)
		expected_result = {
				(0,0,0,0): [(0,0,0,2),(1,0,0,0)],
				(1,0,0,0): [(2,1,0,0),], 
				(2,1,0,0): [(2,1,1,0), (2,1,3,0)],
				(2,1,1,0): [(2,1,2,0),]
				}
		assert gfEvaluatorObj.mutype_tree == expected_result

	def test_list_marginal_idxs(self):
		marginals = np.array([(0,0,3,1),(3,0,1,3)],dtype=np.uint8)
		max_k = (2,2,2,2)
		result = [mutations.list_marginal_idxs(m, max_k) for m in marginals]
		expected_result = [
			[(0,0,0,1), (0,0,1,1),(0,0,2,1)],
			[(0,0,1,0),(0,0,1,1),(0,0,1,2),(1,0,1,0),(1,0,1,1),(1,0,1,2),(2,0,1,0),(2,0,1,1),(2,0,1,2)]
			]
		assert result==expected_result

	def test_add_marginals_restrict_to(self):
		restrict_to = [(0,0,3,1),(3,0,1,3), (0,0,1,0), (0,0,1,1),(1,0,0,0)]
		max_k = (2,2,2,2)
		result = mutations.add_marginals_restrict_to(restrict_to, max_k)
		expected_result = sorted([(0,0,0,1),(0,0,1,1),(0,0,2,1),(0,0,3,1),
			(0,0,1,0),(0,0,1,2),(1,0,1,0),(1,0,1,1),(1,0,1,2),
			(2,0,1,0),(2,0,1,1),(2,0,1,2),(3,0,1,3),(1,0,0,0)])
		assert expected_result == result
#multiplicities, multinomial_coefficients, prod_theta_coeff,
@pytest.mark.muts_events
class Test_Mutations_Events:
	@pytest.mark.parametrize("mut_config, exp_array, exp_coefficient",
		[(np.array([0,0,0,0], dtype=int), np.array([[1, 1, 1, 3, 2], [0, 0, 0, 0, 0]], dtype=int).T, 1),
		(np.array([0,0,0,2], dtype=int), np.array([[1, 1, 1, 3, 2], [0, 0, 0, 0, 0]], dtype=int).T, 1),
		(np.array([0,0,0,1], dtype=int), None, None),
		])
	def test_mutations_as_events1(self, mut_config, exp_array, exp_coefficient):
		max_k_test = np.array([1,1,1,1], dtype=int)
		paths = np.array([
			[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[1,1,1,0],
			[1,0,1,0]
			])
		observed = mutations.expand_single_path(paths, mut_config, max_k_test)
		if exp_coefficient!=None:
			for o in observed:
				print(o[0])
				print(exp_array)
				assert np.array_equal(o[0], exp_array)
				assert o[-1]==exp_coefficient
		else:
			assert len(list(observed))==0
	
	def test_mutations_as_events2(self):
		mut_config = np.array([1,2,0,0], dtype=int)
		max_k_test = np.array([2,2,2,2], dtype=int)
		paths = np.array([
			[2,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[1,1,1,0],
			[1,0,1,0]
			], dtype=int)
		observed = mutations.expand_single_path(paths, mut_config, max_k_test)
		expected = [
				(np.array([[2, 1, 1, 3, 2], [0, 0, 0, 2, 1]], dtype=int).T, 1, 1),
				(np.array([[2, 1, 1, 3, 2], [0, 1, 0, 1, 1]], dtype=int).T, 1, 1),
				(np.array([[2, 1, 1, 3, 2], [0, 2, 0, 0, 1]], dtype=int).T, 1, 1),
				(np.array([[2, 1, 1, 3, 2], [0, 0, 0, 3, 0]], dtype=int).T, 1, 3),
				(np.array([[2, 1, 1, 3, 2], [0, 1, 0, 2, 0]], dtype=int).T, 1, 2),
				(np.array([[2, 1, 1, 3, 2], [0, 2, 0, 1, 0]], dtype=int).T, 1, 1),
				(np.array([[2, 1, 1, 3, 2], [1, 0, 0, 2, 0]], dtype=int).T, 2, 1),
				(np.array([[2, 1, 1, 3, 2], [1, 1, 0, 1, 0]], dtype=int).T, 2, 1),
				(np.array([[2, 1, 1, 3, 2], [1, 2, 0, 0, 0]], dtype=int).T, 2, 1)
			]
		for o, e in zip(observed, expected):
			print(o)
			print(e)
			assert np.array_equal(o[0], e[0])
			assert o[1]==e[1]
			assert o[2]==e[2]

	@pytest.fixture(scope='class')
	def get_info_dict(self):
		#multiplier array should be of shape (num_equations, 2, variables)
		#variables: [c0, c1, E, m1, m2]
		#test_cases that need to be included:
		#1.path without eqs to invert
		#2.path with only eqs to invert
		#3.
		max_k = (1,1)
		all_mutypes = list(mutations.return_mutype_configs(max_k, include_marginals=True))
		info_dict = {
			'multiplier_array' : np.array(
				[
					[[0,0,1,0,0],[1,0,1,1,0]],
					[[0,1,0,0,0],[1,0,1,0,1]],
					[[1,0,0,0,0],[1,0,1,1,1]],
					[[3,0,0,0,0],[3,2,0,4,0]],
					[[0,2,0,0,0],[3,2,0,0,5]],
					[[3,0,0,0,0],[3,0,0,1,6]],
				],dtype=np.uint64
				),
			'all_paths' : [np.array((0,1,2), dtype=int), np.array((3,4,5), dtype=int), np.array((0,1,3,4,5), dtype=int), np.array((0,1,2,3,4), dtype=int)],
			'max_k' : max_k,
			'all_mutypes' : all_mutypes,
			'delta_idx' : 2,
			'max_expansion_size' : 9,
		}
		return info_dict

	def test_prod_with_zeros(self):
		test = np.array(
			[[0, 0, 1, 1],
			[1, 0, 1, 0],
			[0, 0, 2, 2],
			[1, 0, 3, 2],
			[0, 0, 1, 2]]
			, dtype=np.uint64)
		expected = np.array([1,0,6,8], dtype=np.uint64)
		result = mutations.prod_with_zeros.reduce(test, axis=0) 
		assert np.array_equal(expected, result)

	def test_split_off_constant_term(self, get_info_dict):
		result = np.zeros((len(get_info_dict['all_paths']), 2, 4), dtype=np.int64)
		delta_idx = get_info_dict['delta_idx']
		multiplier_array_no_delta = get_info_dict['multiplier_array']
		for idx, path in enumerate(get_info_dict['all_paths']):
			drop_delta = np.delete(get_info_dict['multiplier_array'][path], delta_idx, -1)
			path_numerator_no_delta, _ = np.squeeze(np.vsplit(np.swapaxes(drop_delta, 0, 1), 2))
			result[idx] = mutations.split_off_constant_term(path_numerator_no_delta)
		print(result)
		expected = np.array(
			[
				[[1,1,0,0],[1,1,0,0]],
				[[9,2,0,0],[2,1,0,0]],
				[[9,2,0,0],[2,2,0,0]],
				[[3,2,0,0],[2,2,0,0]]
			], dtype=np.uint64) 
		assert np.array_equal(expected, result)

	def test_constant_term_multiplication(self):
		inputarr = np.array(
				[[3, 0, 0, 0],
				[0, 2, 0, 0],
				[3, 0, 0, 0]], dtype=np.uint64)
		variable_array = np.array([3,2,1,4], dtype=np.uint64)
		#coeffs, exponents = mutations.split_off_constant_term(inputarr)
		constants_pre_expansion = mutations.split_off_constant_term(inputarr)
		result = mutations.eval_constants(1, constants_pre_expansion, variable_array)
		expected = inputarr*variable_array
		expected = np.prod(expected[expected>0])
		assert expected==result
	
	def test_get_unique_idx_tuple_for_path(self, get_info_dict):
		paths = [np.array((0,1,2), dtype=int), np.array((3,4,5), dtype=int), np.array((0,1,3,4,5), dtype=int)]
		max_path_size = max(len(p) for p in paths)
		multiplier_array = get_info_dict['multiplier_array']
		eq_no_delta = multiplier_array[:, 1, get_info_dict["delta_idx"]]==0 
		numerator_no_delta = multiplier_array[:, 0, get_info_dict["delta_idx"]]==0
		eq_dict, eq_idx = {}, 0
		eq_matrix = np.zeros((len(paths)*2**(2+1), max_path_size+1, multiplier_array.shape[-1]), dtype=np.uint64)
		_, eq_idx = mutations.get_unique_idx(np.zeros((0,eq_matrix[eq_idx].shape[-1]), dtype=np.uint64), eq_dict, eq_matrix, eq_idx)
		new_paths = np.zeros((len(paths), 2), dtype=int)
		
		for path_idx, path in enumerate(paths):
			denominator = multiplier_array[path, 1]
			numerator_no_delta_path = np.all(numerator_no_delta[path])
			split_condition = eq_no_delta[path]
			eq_split = (denominator[~split_condition], denominator[split_condition])
			(result_idxs, eq_idx) = mutations.get_unique_idx_tuple_for_path(eq_split, eq_dict, eq_matrix, eq_idx, numerator_no_delta_path, True)
			new_paths[path_idx] = result_idxs
		
		eq_matrix = mutations.trim_zeros(eq_matrix)
		temp1 = np.zeros((3, multiplier_array.shape[-1]), dtype=np.uint64)
		temp2 = multiplier_array[paths[0], 1]
		temp3 = multiplier_array[paths[1], 1]
		temp4 = temp2.copy()
		temp4[-1] = 0
		expected_eq_matrix = np.vstack((temp1, temp2, temp3, temp4)).reshape((4,3,-1))
		assert np.array_equal(expected_eq_matrix, eq_matrix)
		expected_new_paths = np.array([[1,0],[0,2],[3,2]], dtype=np.uint64)
		assert np.array_equal(expected_new_paths, new_paths)
	
	def test_expand_all_paths(self, get_info_dict):
		#delta_idx = get_info_dict['delta_idx']
		#(eq_matrix, constants_multinom_coeffs, constants_pre_expansion, paths_by_mutype) = mutations.expand_all_paths(**get_info_dict)
		#variables_array = np.array([2.0, 4.0, 6.0, 6.0], dtype=np.float64)
		#time = 0.25
		#
		#print('eq_matrix:')
		#print(eq_matrix)
		#print('constants_multinom_coeffs')
		#print(constants_multinom_coeffs)
		#print('constants_pre_expansion')
		#print(constants_pre_expansion)
		#print('paths_by_mutype')
		#print(paths_by_mutype)

		assert True 
		
	def get_expected_expand_all_paths(self, info_dict, time, variables_array):
		m1 = sage.all.SR.var('m1')
		m2 = sage.all.SR.var('m2')
		variables_array_obj = np.hstack((variables_array[:2], np.array([m1, m2])))
		paths = info_dict['all_paths']
		expected_equations_no_mutations = gfmat.equations_from_matrix(info_dict['multiplier_array'], variables_array)
		mutype_tree = mutations.make_mutype_tree(info_dict['all_mutypes'], (0, 0), info_dict['max_k'])
		inverse_laplace_gf = np.zeros(len(paths), dtype=object)	
		rate_dict = {}
		theta = variables_array[-1]
		split_paths = gfmat.split_paths_laplace(paths, info_dict['multiplier_array'], info_dict['delta_idx'])
		for path_idx, (path_with_delta, path_no_delta) in enumerate(split_paths):
			#split path_with_delta, path_no_delta
			delta_in_num = info_dict['multiplier_array'][path_with_delta, 0]>0
			temp_delta = gflib.inverse_laplace_single_event(multiplier_array[path_with_delta], variables_array, time, delta_in_num)
			temp_no_delta = gfmat.equations_from_matrix(info_dict['multiplier_array'][path_no_delta], variables_array)
			inverse_laplace_gf[path_idx] = temp_delta * temp_no_delta
		
		gf = sum(inverse_laplace_gf)
		#calculate derivatives from gf
		expected_ETPs = mutations.make_result_dict_from_mutype_tree_alt(gf, mutype_tree, theta, rate_dict, [m1, m2], info_dict['max_k'])
		return expected_ETPs

@pytest.mark.muts_events_single_pop2
class Test_muts_events_single_pop2:
	@pytest.fixture(scope='class')
	def get_gf_no_mutations(self):
		sample_list = [('a', 'a', 'a'),]
		coalescence_rate_idxs = (0,)
		k_max = {'m_1':2, 'm_2':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict_mat = {'a':0, 'aa':1}
		gfobj = gflib.GFMatrixObject(
			sample_list, 
			coalescence_rate_idxs, 
			branchtype_dict_mat
			)
		gfmat = list(gfobj.make_gf())
		return gfmat

	def test_single_pop(self, get_gf_no_mutations):
		paths, gf = get_gf_no_mutations
		max_k = np.array((3,3), dtype=int)
		root = (0,0)
		all_mutypes = list(mutations.return_mutype_configs(max_k))
		#max_expansion_size = 2, for max_k = (1,1)
		potential_max_mutypes = [(3,3),]
		max_expansion_size = get_max_length(gf[:,1,-max_k.size:], paths, potential_max_mutypes)
		eq_matrix, constants_multinom_coeffs, constants_pre_expansion, paths_by_mutype = mutations.expand_all_paths_no_delta(gf, paths, all_mutypes, max_k, max_expansion_size)

		variable_array = np.array([1.0,3.0], dtype=np.float64)
		eq_split = np.vstack((np.zeros(eq_matrix.shape[0],dtype=bool), np.ones(eq_matrix.shape[0], dtype=bool)))
		eq_split[1, 0] = False
		eq_split = (eq_split[0], eq_split[1])
		results = mutations.eval_gf_with_mutations(paths_by_mutype, constants_multinom_coeffs, constants_pre_expansion, eq_matrix, eq_split, variable_array, 0.0) #still needs to be summed across paths and expansions
		results = np.sum(results, axis=(-1,-2))
		print(results)
		
		m1 = sage.all.SR.var('m1')
		m2 = sage.all.SR.var('m2')
		ordered_mutype_list = [m1, m2]
		num_mutypes = len(ordered_mutype_list)
		alt_variable_array = np.hstack((variable_array[:-1], np.array(ordered_mutype_list)))
		alt_eqs = gfmat.equations_from_matrix(gf, alt_variable_array)
		alt_gf = sum(np.prod(alt_eqs[path]) for path in paths)
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		rate_dict = {b:variable_array[-1] for b in ordered_mutype_list}
		exp_result = make_result_dict_from_mutype_tree_alt(alt_gf, mutype_tree, variable_array[-1], rate_dict, ordered_mutype_list, max_k)
		exp_result = exp_result.astype(np.float64)
		print(exp_result)
		assert np.allclose(exp_result, results)

@pytest.mark.muts_events_single_pop
class Test_muts_events_single_pop:
	@pytest.fixture(scope='class')
	def get_gf_no_mutations(self):
		sample_list = [('a', 'a', 'a'),]
		coalescence_rate_idxs = (0,)
		k_max = {'m_1':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict_mat = {'a':0, 'aa':0}
		gfobj = gflib.GFMatrixObject(
			sample_list, 
			coalescence_rate_idxs, 
			branchtype_dict_mat
			)
		gfmat = list(gfobj.make_gf())
		return gfmat

	def test_single_pop(self, get_gf_no_mutations):
		paths, gf = get_gf_no_mutations
		max_k = np.array((2,), dtype=int)
		root = (0,)
		all_mutypes = list(mutations.return_mutype_configs(max_k))
		#max_expansion_size = 2, for max_k = (1,1)
		potential_max_mutypes = [(2,),]
		max_expansion_size = get_max_length(gf[:,1,-max_k.size:], paths, potential_max_mutypes)
		eq_matrix, constants_multinom_coeffs, constants_pre_expansion, paths_by_mutype = mutations.expand_all_paths_no_delta(gf, paths, all_mutypes, max_k, max_expansion_size)
		
		#build equations with results from expand_all_paths
		variable_array = np.array([1.0,3.0], dtype=np.float64)
		#variable_array = np.array([sage.all.SR.var('c'), sage.all.SR.var('m1')], dtype=object)
		eq_split = np.vstack((np.zeros(eq_matrix.shape[0],dtype=bool), np.ones(eq_matrix.shape[0], dtype=bool)))
		eq_split[1, 0] = False
		eq_split = (eq_split[0], eq_split[1])
		results = mutations.eval_gf_with_mutations(paths_by_mutype, constants_multinom_coeffs, constants_pre_expansion, eq_matrix, eq_split, variable_array, 0.0) #still needs to be summed across paths and expansions
		#results = self.eval_symbolic_equations(paths_by_mutype, constants_multinom_coeffs, constants_pre_expansion, eq_matrix, eq_split, variable_array, 0.0)
		results = np.sum(results, axis=(-1,-2))
		print(results)
		
		m1 = sage.all.SR.var('m1')
		ordered_mutype_list = [m1,]
		num_mutypes = len(ordered_mutype_list)
		alt_variable_array = np.hstack((variable_array[:-num_mutypes], np.array(ordered_mutype_list)))
		alt_eqs = gfmat.equations_from_matrix(gf, alt_variable_array)
		alt_gf = sum(np.prod(alt_eqs[path]) for path in paths)
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		rate_dict = {b:variable_array[-1] for b in ordered_mutype_list}
		exp_result = make_result_dict_from_mutype_tree_alt(alt_gf, mutype_tree, variable_array[-1], rate_dict, ordered_mutype_list, max_k)
		exp_result = exp_result.astype(np.float64)
		print(exp_result)
		assert np.allclose(exp_result, results)

@pytest.mark.muts_events_div
class Test_muts_events_div:
	@pytest.fixture(scope='class')
	def get_gf_no_mutations(self):
		sample_list = [(), ('a', 'a', 'a')]
		coalescence_rate_idxs = (0, 1)
		exodus_rate_idx = 2 
		exodus_direction = [(1,0),]
		k_max = {'m_1':2, 'm_2':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict_mat = {'a':0, 'aa':1}
		gfobj = gflib.GFMatrixObject(
			sample_list, 
			coalescence_rate_idxs, 
			branchtype_dict_mat,
			exodus_direction=exodus_direction,
			exodus_rate=exodus_rate_idx
			)
		gfmat = gfobj.make_gf()
		return (exodus_rate_idx, gfmat)

	def test_div(self, get_gf_no_mutations):
		exodus_rate_idx, (paths, gf) = get_gf_no_mutations
		#print(paths)
		#print(gf)
		max_k = np.array((2,2), dtype=int)
		root = (0,0)
		all_mutypes = list(mutations.return_mutype_configs(max_k))
		#max_expansion_size = 2, for max_k = (1,1)
		potential_max_mutypes = [tuple(max_k),]
		max_expansion_size = get_max_length(gf[:,1,-max_k.size:], paths, potential_max_mutypes)		
		eq_matrix, constants_multinom_coeffs, constants_pre_expansion, paths_by_mutype = mutations.expand_all_paths(gf, paths, all_mutypes, max_k, exodus_rate_idx, max_expansion_size)
		##c0, c1, theta
		variable_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
		time = 1.5
		##eq_split, can be verified using paths by mutype, leave out 0 and -1
		to_invert = np.unique(paths_by_mutype[...,0])
		to_invert = to_invert[to_invert>0]
		not_to_invert = np.unique(paths_by_mutype[...,1])
		not_to_invert = not_to_invert[not_to_invert>0]
		eq_split = np.zeros((2, eq_matrix.shape[0]),dtype=int)
		eq_split[0, to_invert] = 1
		eq_split[1, not_to_invert] = 1
		eq_split = eq_split.astype(bool)
		eq_split = (eq_split[0], eq_split[1])
		max_multiplicity = np.max(eq_matrix[..., -1])+1
		binomial_coefficients = gfpar.return_binom_coefficients(max_multiplicity)
		factorials = 1/np.cumprod(np.arange(1,max_multiplicity))
		factorials = np.hstack((1, factorials))
		results = mutations.eval_gf_with_mutations(paths_by_mutype, constants_multinom_coeffs, constants_pre_expansion, eq_matrix, eq_split, variable_array, time, binomial_coefficients, factorials)
		results = np.sum(results, axis=(-1,-2))
		print('results')
		print(results)
		
		m1 = sage.all.SR.var('m1')
		m2 = sage.all.SR.var('m2')
		ordered_mutype_list = [m1, m2]
		num_mutypes = len(ordered_mutype_list)
		alt_variable_array = np.hstack((variable_array[:-1], np.array(ordered_mutype_list)))
		alt_eqs = equations_from_matrix_with_inverse(gf, paths, alt_variable_array, time, exodus_rate_idx)
		alt_gf = sum(alt_eqs)		
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		rate_dict = {b:variable_array[-1] for b in ordered_mutype_list}
		exp_result = make_result_dict_from_mutype_tree_alt(alt_gf, mutype_tree, variable_array[-1], rate_dict, ordered_mutype_list, max_k)
		exp_result = exp_result.astype(np.float64)
		print('exp_result')
		print(exp_result)
		assert np.allclose(exp_result, results)

def equations_from_matrix_with_inverse(multiplier_array, paths, var_array, time, delta_idx):
	split_paths = gfmat.split_paths_laplace(paths, multiplier_array, delta_idx)
	delta_in_nom_all = multiplier_array[:, 0, delta_idx]==1
	results = np.zeros(len(split_paths), dtype=object)
	subset_no_delta = np.arange(multiplier_array.shape[-1])!=delta_idx
	multiplier_array_no_delta = multiplier_array[:,:,subset_no_delta] 
	for idx, (no_delta, with_delta) in enumerate(split_paths):
		delta_in_nom_list = delta_in_nom_all[with_delta]
		inverse = np.sum(gflib.inverse_laplace_single_event(multiplier_array_no_delta[with_delta], var_array, time, delta_in_nom_list))
		no_inverse = np.prod(gfmat.equations_from_matrix(multiplier_array_no_delta[no_delta], var_array))
		results[idx] = np.prod((inverse, no_inverse))
	return results

def prodmax(t):
	return np.prod([s for s in t if s>0])
	
def get_max_length(multiplier_array, paths, mutypes):
	mutypes = iter(mutypes)
	mutype = next(mutypes)
	max_length = 0
	largest_value = prodmax(mutype)
	while mutype!=None and prodmax(mutype)>=largest_value:
		max_length_for_path = max(len(list(itertools.product(*mutations.expand_all_columns_of_path(multiplier_array[path], mutype)))) for path in paths)
		max_length = max(max_length_for_path, max_length)
		mutype = next(mutypes, None)
	return max_length

def eval_symbolic_equations(paths_by_mutype, constants_multinom_coeffs, constants_pre_expansion, eq_matrix, eq_split, variable_array, time):
	constants = eval_constants_symbolic(constants_multinom_coeffs, constants_pre_expansion, variable_array)
	evaluated_equations = eval_equations_symbolic(eq_matrix, eq_split, variable_array, time)
	evaluated_mutype_paths = eval_mutype_paths_symbolic(paths_by_mutype, evaluated_equations)
	return constants * evaluated_mutype_paths #this needs to be summed across paths and expansions  

def eval_constants_symbolic(constants_multinom_coeffs, constants_pre_expansion, variable_array):
	constants_pre_expansion = constants_pre_expansion[..., 0, :]*(variable_array**constants_pre_expansion[..., 1, :]) 
	constants_pre_expansion = np.apply_along_axis(prod_with_zeros_no_numba, -1, constants_pre_expansion)
	#single constants_pre_expansion can be 0, but never all of them (always need coalescence)
	return constants_pre_expansion * constants_multinom_coeffs

def eval_equations_symbolic(eq_matrix, eq_split, variable_array, time):
	#first entry of eq_matrix should be (1, 1), last entry (0, 0)
	eq_to_invert, eq_not_to_invert = eq_split #should not contain first equation, evaluates to 1
	eq_matrix_results = np.zeros((eq_matrix.shape[0], 2), dtype=object) 
	#eq with only zeros should evaluate to 1
	eq_matrix_results[0] = (1, 1)
	#eq_matrix_results[eq_to_invert, 0] = _eval_equations_to_invert(eq_matrix[eq_to_invert], variable_array, time)
	eq_matrix_results[eq_not_to_invert, 1] = _multiply_with_var_array_symbolic(eq_matrix[eq_not_to_invert], variable_array)
	#add row of zeros for matrix entries without result
	eq_matrix_results = np.vstack((eq_matrix_results, np.zeros((1, 2), dtype=object))) #can be improved!
	return eq_matrix_results

def _multiply_with_var_array_symbolic(eq_matrix, variable_array):
	#note: multiplicity of poles is always 1 less than what it should be
	result = (eq_matrix[..., :-1].dot(variable_array))**(eq_matrix[..., -1]+1)
	#can this contain zeros!? axis=-1 represents max_path_size
	result = 1/np.apply_along_axis(prod_with_zeros_no_numba, -1, result) #multiply factors within single equation
	return result

def eval_mutype_paths_symbolic(paths_by_mutype, evaluated_eqs):
	result = np.zeros_like(paths_by_mutype, dtype = object)
	result[..., 0] = evaluated_eqs[paths_by_mutype[..., 0], 0]
	result[..., 1] = evaluated_eqs[paths_by_mutype[..., 1], 1]
	return np.prod(result, axis=-1) #implementation without taking into account potential floating point precision issues!

def prod_with_zeros_no_numba(arr):
		result = np.prod(arr)
		if result==0:
			subset = [x for x in arr if x!=0]
			if len(subset)>0:
				return np.prod(subset)
			else:
				return 0
		else:
			return result

def make_result_dict_from_mutype_tree_alt(gf, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k, precision=165):
	root = tuple(0 for _ in max_k) #root is fixed
	num_mutypes = len(max_k)
	result = np.zeros(max_k + 2,dtype=object)
	stack = [(root, gf)]
	result[root] = eval_equation(gf, theta, rate_dict, root, precision)
	while stack:
		parent, parent_equation = stack.pop()
		if parent in mutype_tree:
			for child in mutype_tree[parent]:
				mucounts = [m for m, max_k_m in zip(child, max_k) if m<=max_k_m]
				marginal = len(mucounts)<num_mutypes
				child_equation = generate_equation(parent_equation, parent, child, max_k, ordered_mutype_list, marginal)
				stack.append((child, child_equation))
				result[child] = eval_equation(child_equation, theta, rate_dict, mucounts, precision)
	return result

def generate_equation(equation, parent, node, max_k, ordered_mutype_list, marginal):
	if marginal:
		marginals = {branchtype:0 for branchtype, count, max_k_m in zip(ordered_mutype_list, node, max_k) if count>max_k_m}
		return equation.subs(marginals)
	else:
		relative_config = [b-a for a,b in zip(parent, node)]
		partial = mutations.single_partial(ordered_mutype_list, relative_config)
		diff = sage.all.diff(equation, partial)
		return diff

def eval_equation(derivative, theta, ratedict, numeric_mucounts, precision):
	mucount_total = np.sum(numeric_mucounts)
	mucount_fact_prod = np.prod([np.math.factorial(count) for count in numeric_mucounts])
	return (-1*theta)**(mucount_total)/mucount_fact_prod*derivative.subs(ratedict)