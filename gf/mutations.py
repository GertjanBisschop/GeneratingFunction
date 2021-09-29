import itertools
import collections
import copy
import multiprocessing
import numpy as np
import numba
import sage.all
import scipy.special
import sys

from . import gf as gflib
from . import matrix_representation as gfmat
from . import partial_fraction_expansion as gfpfe

def single_partial(ordered_mutype_list, partial):
		return list(gflib.flatten(itertools.repeat(branchtype,count) for count, branchtype in zip(partial, ordered_mutype_list)))

def return_mutype_configs(max_k, include_marginals=True):
	add_to_k = 2 if include_marginals else 1
	iterable_range = (range(k+add_to_k) for k in max_k)
	return itertools.product(*iterable_range)

def make_mutype_tree_single_digit(all_mutypes, root, max_k):
	result = collections.defaultdict(list)
	for idx, mutype in enumerate(all_mutypes):
		if mutype == root:
			pass
		else:
			if any(m>max_k_m for m, max_k_m in zip(mutype, max_k)):
				root_config = tuple(m if m<=max_k_m else 0 for m, max_k_m in zip(mutype, max_k))
				result[root_config].append(mutype)
			else:
				root_config = differs_one_digit(mutype, all_mutypes[:idx])
				result[root_config].append(mutype)
	return result

def make_mutype_tree(all_mutypes, root, max_k):
	result = collections.defaultdict(list)
	for idx, mutype in enumerate(all_mutypes):
		if mutype == root:
			pass
		else:
			if any(m>max_k_m for m, max_k_m in zip(mutype, max_k)):
				root_config = tuple(m if m<=max_k_m else 0 for m, max_k_m in zip(mutype, max_k))
				result[root_config].append(mutype)
			else:
				root_config = closest_digit(mutype, all_mutypes[:idx])
				result[root_config].append(mutype)
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
		partial = single_partial(ordered_mutype_list, relative_config)
		diff = sage.all.diff(equation, partial)
		return diff

def eval_equation(derivative, theta, ratedict, numeric_mucounts, precision):
	mucount_total = np.sum(numeric_mucounts)
	mucount_fact_prod = np.prod([np.math.factorial(count) for count in numeric_mucounts])
	return sage.all.RealField(precision)((-1*theta)**(mucount_total)/mucount_fact_prod*derivative.subs(ratedict))

def differs_one_digit(query, complete_list):
	#complete_set = self.generate_differs_one_set(query)
	#similar_to = next(obj for obj in complete_list[idx-1::-1] if obj in complete_set)
	similar_to = next(obj for obj in complete_list[::-1] if sum_tuple_diff(obj, query)==1)
	return similar_to

def closest_digit(query, complete_list):
	return min(complete_list[::-1], key=lambda x: tuple_distance(query,x))

def tuple_distance(a_tuple, b_tuple):
	dist = [a-b for a,b in zip(a_tuple, b_tuple)]
	if any(x<0 for x in dist):
		return np.inf
	else:
		return abs(sum(dist))

def sum_tuple_diff(tuple_a, tuple_b):
	return sum(b-a for a,b in zip(tuple_a, tuple_b))

############################## mutations as events: expanding gf #################################

def balls_in_boxes(n, m, length, positions):
	"""
	n balls in m boxes
	algorithm adapted from https://stackoverflow.com/a/6609080
	"""
	if m==0:
		if n==0:
			yield np.zeros(length, dtype=int)
		return
	for c in itertools.combinations(range(n + m - 1), m - 1):
		result = np.zeros(length, dtype=int)
		result[positions] = tuple(b - a - 1 for a, b in zip((-1,) + c, c + (n + m - 1,)))
		yield result

def multinomial(params):
	if len(params) == 1:
		return 1
	return scipy.special.comb(np.sum(params), params[-1], exact=True) * multinomial(params[:-1])

@numba.njit()
def multinomial_along_dims(A):
	result = np.zeros(A.shape[0], dtype=np.int64)
	for idx, row in enumerate(A):
		result[idx] = multinomial_numba(row)
	return result

@numba.njit()
def multinomial_numba(lst):
	res = 1
	i = np.sum(lst)
	for a in lst:
		for j in range(1,a+1):
			res *= i
			res //= j
			i -= 1
	return res

def expand_all_columns_of_path(path, num_mutations):
	result_shape = path.shape[0]
	num_positions = np.sum(path>0, axis=0)
	positions = np.argwhere(path.T>0)
	return [balls_in_boxes(mut_count, pos_count, result_shape, pos[:,1]) for mut_count, pos_count, pos in zip(num_mutations, num_positions, np.split(positions,np.cumsum(num_positions))[:-1])]	

def expand_single_path(path, num_mutations, max_k):
	"""
	:param path: array containing branchtype configuration for each factor
	:param num_mutations: array describing mutation types and counts to occur along path
	"""
	marg_bool = num_mutations>max_k
	#result_shape = path.shape[0]
	if np.any(marg_bool):
		path = path.copy()
		path[:, marg_bool] = 0
		num_mutations = num_mutations.copy()
		num_mutations[marg_bool] = 0
	all_generators = expand_all_columns_of_path(path, num_mutations)
	sum_theta_coeff = np.sum(path, axis=1)
	for new_path in itertools.product(*all_generators):
		temp = np.vstack(new_path).T
		multiplicities = np.sum(temp, axis=1) #returns multiplicities
		multinomial_coefficients = np.prod(multinomial_along_dims(temp))
		#product of non-zero theta coeff
		theta_coefficients = path**temp
		mask_mutations_placed = theta_coefficients[temp>0] #if temp>0, then path[temp]>0
		if mask_mutations_placed.size==0:
			prod_theta_coeff = 0
		else:
			prod_theta_coeff = np.prod(mask_mutations_placed) 
		theta_sum_and_multiplicities = np.vstack((sum_theta_coeff, multiplicities)).T
		yield (theta_sum_and_multiplicities, prod_theta_coeff, multinomial_coefficients)

def expand_all_paths(multiplier_array, all_paths, all_mutypes, max_k, delta_idx, max_expansion_size):
	#constants
	num_mutypes = len(max_k)
	num_paths = len(all_paths)
	eq_idx = 0
	max_path_size = np.max([len(path) for path in all_paths])
	#views of multiplier_array
	eq_has_no_delta = multiplier_array[:, 1, delta_idx]==0 #no delta in denominator
	prepare_for_inversion = ~np.all(eq_has_no_delta)
	numerator_has_no_delta = multiplier_array[:, 0, delta_idx]==0 #no delta in numerator
	#arrays to populate with results
	constants_pre_expansion = np.zeros(
		(*max_k+2, num_paths, max_expansion_size, 2, multiplier_array.shape[-1]-num_mutypes), dtype=np.uint64
		) #contains for each mutype and for each path an array with coefficients and an array with multiplicities for all variables
	constants_multinom_coeffs = np.zeros((*max_k+2, num_paths, max_expansion_size), dtype=np.uint64)
	eq_matrix = np.zeros((num_paths*np.prod(max_k+2)*max_expansion_size, max_path_size+1, multiplier_array.shape[-1]-num_mutypes+1), dtype=np.uint8) #last entry of last dimension is multiplicity
	#eq_matrix = list()
	paths_by_mutype = -np.ones((*max_k+2, num_paths, max_expansion_size, 2), dtype=np.int64)
	#dicts
	eq_dict = dict()
	#shape of empty array will be different if no delta_idx!
	_, eq_idx = get_unique_idx(np.zeros((0, multiplier_array.shape[-1]-num_mutypes+1), dtype=np.uint64), eq_dict, eq_matrix, eq_idx)

	for path_idx, path in enumerate(all_paths):
		numerator_has_no_delta_path = np.all(numerator_has_no_delta[path])
		path_numerator_no_delta, path_denominator_no_delta = np.squeeze(np.vsplit(np.swapaxes(np.delete(multiplier_array[path], delta_idx, -1), 0, 1), 2))
		constants_pre_expansion_path = split_off_constant_term(path_numerator_no_delta[:, :-num_mutypes])
		for mutype in all_mutypes:
			num_mutations = sum((m for m,k in zip(mutype, max_k) if m<=k))
			for expanded_path_idx, (theta_sum_and_multiplicities, prod_theta_coeff, multinomial_coefficients) in enumerate(expand_single_path(path_denominator_no_delta[:, -num_mutypes:], np.array(mutype), max_k)):
				#add constants to correct array
				constants_pre_expansion[mutype][path_idx, expanded_path_idx, :, :-1] = constants_pre_expansion_path
				constants_pre_expansion[mutype][path_idx,expanded_path_idx, :, -1] = (prod_theta_coeff, num_mutations)
				constants_multinom_coeffs[(*mutype, path_idx, expanded_path_idx)] = multinomial_coefficients
				#unique equations and make paths for each mutype
				new_eq_for_path_split = factorize_expanded_path(path_denominator_no_delta, theta_sum_and_multiplicities, num_mutypes, eq_has_no_delta[path])
				delta_no_delta_tuple, eq_idx = get_unique_idx_tuple_for_path(new_eq_for_path_split, eq_dict, eq_matrix, eq_idx, numerator_has_no_delta_path, prepare_for_inversion)
				#paths_by_mutype[mutype].append(delta_no_delta_tuple)
				paths_by_mutype[(*mutype, path_idx, expanded_path_idx)] = delta_no_delta_tuple
	
	#num_uniques_to_invert = len(set(t[0] for big_t in paths_by_mutype.values() for t in big_t))
	#print(num_uniques_to_invert)
	
	return (eq_matrix[:eq_idx+1], constants_multinom_coeffs, constants_pre_expansion, paths_by_mutype)

def expand_all_paths_no_delta(multiplier_array, all_paths, all_mutypes, max_k, max_expansion_size):
	#temp function!
	#constants
	num_mutypes = len(max_k)
	num_paths = len(all_paths)
	eq_idx = 0
	max_path_size = np.max([len(path) for path in all_paths])
	#views of multiplier_array
	eq_has_no_delta = np.ones(multiplier_array.shape[0], dtype=bool)
	prepare_for_inversion = ~np.all(eq_has_no_delta)
	#arrays to populate with results
	numerator_has_no_delta = np.ones(multiplier_array.shape[0], dtype=bool) #no delta in numerator
	constants_pre_expansion = np.zeros(
		(*max_k+2, num_paths, max_expansion_size, 2, multiplier_array.shape[-1]-num_mutypes+1), dtype=np.uint64
		) #contains for each mutype and for each path array with coefficients and array with multiplicities for all variables
	constants_multinom_coeffs = np.zeros((*max_k+2, num_paths, max_expansion_size), dtype=np.uint64)
	#eq_matrix = np.zeros((len(all_paths)*2**(num_mutypes+1), max_path_size+1, multiplier_array.shape[-1]-num_mutypes+2), dtype=np.uint8)
	eq_matrix = np.zeros((num_paths*np.prod(max_k+2)*max_expansion_size, max_path_size+1, multiplier_array.shape[-1]-num_mutypes+2), dtype=np.uint8)
	paths_by_mutype = -np.ones((*max_k+2, num_paths, max_expansion_size, 2), dtype=np.int64)
	#dicts
	eq_dict = dict()
	_, eq_idx = get_unique_idx(np.zeros((0, eq_matrix[eq_idx].shape[-1]), dtype=np.uint64), eq_dict, eq_matrix, eq_idx)

	for path_idx, path in enumerate(all_paths):
		numerator_has_no_delta_path = np.all(numerator_has_no_delta[path])
		path_numerator_no_delta, path_denominator_no_delta = np.squeeze(np.vsplit(np.swapaxes(multiplier_array[path], 0, 1), 2))
		constants_pre_expansion_path = split_off_constant_term(path_numerator_no_delta[:, :-num_mutypes])
		for mutype in all_mutypes:
			num_mutations = sum((m for m,k in zip(mutype, max_k) if m<=k))
			for expanded_path_idx, (theta_sum_and_multiplicities, prod_theta_coeff, multinomial_coefficients) in enumerate(expand_single_path(path_denominator_no_delta[:, -num_mutypes:], np.array(mutype), max_k)):
				#add constants to correct array
				constants_pre_expansion[mutype][path_idx, expanded_path_idx, :, :-1] = constants_pre_expansion_path
				constants_pre_expansion[mutype][path_idx, expanded_path_idx, :, -1] = (prod_theta_coeff, num_mutations)
				constants_multinom_coeffs[(*mutype, path_idx, expanded_path_idx)] = multinomial_coefficients
				#unique equations and make paths for each mutype
				new_eq_for_path_split = factorize_expanded_path(path_denominator_no_delta, theta_sum_and_multiplicities, num_mutypes, eq_has_no_delta[path])
				delta_no_delta_tuple, eq_idx = get_unique_idx_tuple_for_path(new_eq_for_path_split, eq_dict, eq_matrix, eq_idx, numerator_has_no_delta_path, prepare_for_inversion)
				#paths_by_mutype[mutype].append(delta_no_delta_tuple)
				paths_by_mutype[(*mutype, path_idx, expanded_path_idx)] = delta_no_delta_tuple

	return (eq_matrix[:eq_idx+1], constants_multinom_coeffs, constants_pre_expansion, paths_by_mutype)

def factorize_expanded_path(path_denominator_no_delta, theta_sum_and_multiplicities, num_mutypes, split_condition):
	new_eq_for_path = np.hstack((path_denominator_no_delta[:, :-num_mutypes], theta_sum_and_multiplicities))		
	return (new_eq_for_path[~split_condition], new_eq_for_path[split_condition])

def get_unique_idx_tuple_for_path(eq_split, eq_dict, eq_matrix, eq_idx, numerator_no_delta_path, prepare_for_inversion):
	new_eq_for_path_delta, new_eq_for_path_no_delta = eq_split
	#deal with new_eq_for_path_delta
	if numerator_no_delta_path and prepare_for_inversion:
		#add zero as pole
		new_eq_for_path_delta = np.vstack((np.zeros_like(new_eq_for_path_delta[-1]), new_eq_for_path_delta))
	unique_idx_delta, eq_idx = get_unique_idx(new_eq_for_path_delta, eq_dict, eq_matrix, eq_idx) 
	#deal with new_eq_for_path_no_delta
	unique_idx_no_delta, eq_idx = get_unique_idx(new_eq_for_path_no_delta, eq_dict, eq_matrix, eq_idx)
	return ((unique_idx_delta, unique_idx_no_delta), eq_idx)

def get_unique_idx(eq, eq_dict, eq_matrix, eq_idx):
	eq_tuple = tuple(sorted(tuple(map(tuple, eq)))) #sorting to avoid fact that same equations in different order map to different hash
	if eq_tuple in eq_dict:
		unique_idx = eq_dict[eq_tuple]
	else:
		eq_dict[eq_tuple] = eq_idx
		eq_matrix[eq_idx, :eq.shape[0]] = eq
		unique_idx = eq_idx
		eq_idx+=1
	return (unique_idx, eq_idx)

def split_off_constant_term(numerators):
	result = np.zeros((2, numerators.shape[-1]), dtype=np.uint64)
	result[0] = prod_with_zeros.reduce(numerators, axis=0)
	result[1] = np.sum(numerators>0, axis=0)
	return result

@numba.vectorize([numba.uint8(numba.uint8, numba.uint8),
				numba.uint16(numba.uint16, numba.uint16),
				numba.uint32(numba.uint32, numba.uint32),
				numba.uint64(numba.uint64, numba.uint64),
				numba.float64(numba.float64, numba.float64)])
def prod_with_zeros(x,y):
	#only return 0 if both x and y are zero
    temp = x*y
    if temp==0:
        if x==0:
            if y==0:
                return 0
            else:
                return y
        else:
            return x
    else:
        return temp

@numba.njit()
def trim_zeros_numba(arr):
	#for 2-d arrays
    temp = np.ones(arr.shape[0], dtype=bool)
    for idx in range(1, arr.shape[0]):
        temp[idx] = ~np.all(arr[idx]==0)
    return arr[temp]

def return_split_condition(paths_by_mutype):
	return (np.unique(paths_by_mutype[..., 0]), np.unique(paths_by_mutype[..., 1]))

########### evaluating equations into values and combining them into a result #################

def eval_constants(constants_multinom_coeffs, constants_pre_expansion, variable_array):
	#constants_pre_expansion dimensions: (*mutype, paths, 2, num_vars)
	constants_pre_expansion = constants_pre_expansion[..., 0, :]*(variable_array**constants_pre_expansion[..., 1, :]) 
	constants_pre_expansion = prod_with_zeros.reduce(constants_pre_expansion, axis=-1)
	#single constants_pre_expansion can be 0, but never all of them (always need coalescence)
	return constants_pre_expansion * constants_multinom_coeffs

def eval_equations(eq_matrix, eq_split, variable_array, time, binom_coefficients, factorials):
	#first entry of eq_matrix should be (1, 1), last entry (0, 0)
	#eq_split should be list containing two arrays with bools indicating how each of the equations should be evaluated
	eq_to_invert, eq_not_to_invert = eq_split #should not contain first equation, evaluates to 1
	eq_matrix_results = np.zeros((eq_matrix.shape[0], 2), dtype=np.float64) 
	#eq with only zeros should evaluate to 1
	eq_matrix_results[0] = (1, 1)
	eq_matrix_results[eq_to_invert, 0] = _eval_equations_to_invert(eq_matrix[eq_to_invert], variable_array, time, binom_coefficients, factorials)
	eq_matrix_results[eq_not_to_invert, 1] = _multiply_with_var_array(eq_matrix[eq_not_to_invert], variable_array)
	#add row of zeros for matrix entries without result
	eq_matrix_results = np.vstack((eq_matrix_results, np.zeros((1, 2)))) #can be improved!
	return eq_matrix_results

def eval_mutype_paths(paths_by_mutype, evaluated_eqs):
	result = np.zeros_like(paths_by_mutype, dtype = np.float64)
	result[..., 0] = evaluated_eqs[paths_by_mutype[..., 0], 0]
	result[..., 1] = evaluated_eqs[paths_by_mutype[..., 1], 1]
	return np.prod(result, axis=-1) #implementation without taking into account potential floating point precision issues!

def eval_gf_with_mutations(paths_by_mutype, constants_multinom_coeffs, constants_pre_expansion, eq_matrix, eq_split, variable_array, time, binom_coefficients=None, factorials=None):
	#variable array is array without delta!
	constants = eval_constants(constants_multinom_coeffs, constants_pre_expansion, variable_array)
	evaluated_equations = eval_equations(eq_matrix, eq_split, variable_array, time, binom_coefficients, factorials)
	evaluated_mutype_paths = eval_mutype_paths(paths_by_mutype, evaluated_equations)
	return constants * evaluated_mutype_paths #this needs to be summed across paths and expansions

def _multiply_with_var_array(eq_matrix, variable_array):
	#note: multiplicity of poles is always 1 less than what it should be
	result = (eq_matrix[..., :-1].dot(variable_array))**(eq_matrix[..., -1]+1)
	#can this contain zeros!? axis=-1 represents max_path_size
	result = 1/prod_with_zeros.reduce(result, axis=-1) #multiply factors within single equation
	return result

def _eq_to_poles_multiplicity(eq_matrix, variable_array):
	#note: multiplicity is always 1 less than what it should be!
	#note poles to be used are -poles
	return (-eq_matrix[..., :-1].dot(variable_array), eq_matrix[..., -1]+1)

def _eval_equations_to_invert(eq_matrix, variable_array, time, binom_coefficients, factorials):
	eq_matrix_shape = eq_matrix.shape[0]
	eq_matrix_temp = np.zeros(eq_matrix_shape, dtype=np.float64)
	for idx in range(eq_matrix_shape):
		#trimming zeros should be temporary solution! Alternatively, simply store array shape for each eq
		temp = trim_zeros_numba(eq_matrix[idx])
		to_invert = _eq_to_poles_multiplicity(temp, variable_array)  
		#remember: multiplicity of poles is always 1 less than what it should be
		eq_matrix_temp[idx] = gflib.inverse_laplace_PFE(*to_invert, time, binom_coefficients, factorials)
	return eq_matrix_temp

############################### old functions ######################################

def dict_to_array(result_dict, shape, dtype=object):
	result = np.zeros(shape, dtype=dtype)
	result[tuple(zip(*result_dict.keys()))] = list(result_dict.values())
	return result

#using fast_callable
def ar_to_fast_callable(ar, variables):
	shape = ar.shape
	temp = np.reshape(ar, -1)
	result = [sage.all.fast_callable(value, vars=variables, domain=sage.all.RealField(165)) if len(value.arguments())>0 else sage.all.RealField(165)(value) for value in temp]
	return result
	
def evaluate_fast_callable_ar(ar, variables, shape):
	result = [v(*variables) if isinstance(v, sage.ext.interpreters.wrapper_rr.Wrapper_rr) else v for v in ar]
	return np.array(result, dtype=np.float64).reshape(shape)

def evaluate_ar(ar, variables_dict):
	shape = ar.shape
	temp = np.reshape(ar, -1)
	result = [v.subs(variables_dict) for v in temp]
	return np.array(result, dtype=np.float64).reshape(shape)

def simple_probK(gf, theta, partials, marginals, ratedict, mucount_total, mucount_fact_prod):
	gf_marginals = gf.subs(marginals)
	derivative = sage.all.diff(gf_marginals,partials)
	return (-1*theta)**(mucount_total)/mucount_fact_prod*derivative.subs(ratedict)

def probK_from_diff(derivative, theta, ratedict, mut_config):
	numeric_mucounts = [value for value in mut_config if value]
	mucount_total = np.sum(numeric_mucounts)
	mucount_fact_prod = np.prod([np.math.factorial(count) for count in numeric_mucounts])
	return (-1*theta)**(mucount_total)/mucount_fact_prod*derivative.subs(ratedict)

def symbolic_prob_per_term_old(gf_term, prob_input_subdict, shape, theta, adjust_marginals=False):
	result = {mutation_configuration:simple_probK(gf_term, theta, **subdict) for mutation_configuration, subdict in prob_input_subdict.items()}
	result_array = dict_to_array(result, shape)
	if adjust_marginals:
		result_array = adjust_marginals(result_array, len(shape))
	return result_array

def symbolic_prob_per_term(gf_term, mutype_tree, shape, theta, rate_dict, ordered_mutype_list, max_k, adjust_marginals=False):
	result = make_result_dict_from_mutype_tree(gf_term, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k)
	result_array = dict_to_array(result, shape)
	if adjust_marginals:
		result_array = adjust_marginals_array(result_array, len(shape))
	return result_array

def process_grouped_terms(gf_terms, dummy_variable, mutype_tree, shape, rate_dict, theta, ordered_mutype_list, max_k, adjust_marginals, parameter_dict):
	if dummy_variable:
		gf_to_use = sum(gflib.inverse_laplace(gf_terms, dummy_variable))
	else:
		gf_to_use = sum(gf_terms)
	if parameter_dict:
		gf_to_use = gf_to_use.subs(parameter_dict)
	result = symbolic_prob_per_term(gf_to_use, mutype_tree, shape, theta, rate_dict, ordered_mutype_list, max_k, adjust_marginals) 
	return result

def make_prob_array(gf, mutype_tree, ordered_mutype_list, max_k, theta, dummy_variable, chunksize=100, num_processes=1, adjust_marginals=False, parameter_dict=None):
	rate_dict = {branchtype:theta for branchtype in ordered_mutype_list}
	#make mutype tree to simplify makeprobK
	shape = tuple(kmax+2 for kmax in max_k)
	#gf is generator for gf prior to taking inverse laplace
	symbolic_prob_array = []
	if num_processes==1:
		symbolic_prob_array = sum([process_grouped_terms(
			gf_terms, dummy_variable, mutype_tree, shape, rate_dict, theta, ordered_mutype_list, max_k, adjust_marginals, parameter_dict) 
				for gf_terms in gflib.split_gf_iterable(gf, chunksize)])
	else:
		process_grouped_terms_specified = functools.partial(
			process_grouped_terms,
			dummy_variable=dummy_variable,
			mutype_tree=mutype_tree,
			shape=shape,
			rate_dict=rate_dict,
			theta=theta,
			ordered_mutype_list=ordered_mutype_list,
			max_k=max_k,
			adjust_marginals=adjust_marginals,
			parameter_dict=parameter_dict
			)
		with multiprocessing.Pool(processes=num_processes) as pool:
			symbolic_prob_array = sum(pool.imap(process_grouped_terms_specified, gflib.split_gf(gf, chunksize)))
	return symbolic_prob_array

def prepare_symbolic_prob_dict(ordered_mutype_list, kmax_by_mutype, theta):
	'''
	deprecated function
	'''
	iterable_range = (range(k+2) for k in kmax_by_mutype)
	all_mutation_configurations = itertools.product(*iterable_range)
	prob_input_dict = collections.defaultdict(dict)
	for mutation_configuration in all_mutation_configurations:
		branchtype_mucount_dict = {mutype:(count if count<=kmax else None) for mutype, count, kmax in zip(ordered_mutype_list, mutation_configuration, kmax_by_mutype)}
		numeric_mucounts = [value for value in branchtype_mucount_dict.values() if value] 
		prob_input_dict[mutation_configuration]['mucount_total'] = sage.all.sum(numeric_mucounts)
		prob_input_dict[mutation_configuration]['mucount_fact_prod'] = sage.all.product(sage.all.factorial(count) for count in numeric_mucounts)
		prob_input_dict[mutation_configuration]['marginals'] = {branchtype:0 for branchtype, count in branchtype_mucount_dict.items() if count==None}
		prob_input_dict[mutation_configuration]['ratedict'] = {branchtype:theta for branchtype, count in branchtype_mucount_dict.items() if isinstance(count, int)}
		prob_input_dict[mutation_configuration]['partials'] = list(flatten(itertools.repeat(branchtype,count) for branchtype, count in branchtype_mucount_dict.items() if isinstance(count, int)))
	return prob_input_dict

def subs_parameters(symbolic_prob_dict, parameter_dict, ordered_mutype_list, kmax_by_mutype, precision=165):
	shape = tuple(kmax+2 for kmax in kmax_by_mutype)
	variables = list(parameter_dict.keys())
	return {mutation_configuration:expression.subs(parameter_dict) for mutation_configuration, expression in symbolic_prob_dict.items()}

def list_marginal_idxs(marginal, max_k):
	marginal_idxs = np.argwhere(marginal>max_k).reshape(-1)
	shape=np.array(max_k, dtype=np.uint8) + 2
	max_k_zeros = np.zeros(shape, dtype=np.uint8)
	slicing = [v if idx not in marginal_idxs else slice(-1) for idx,v in enumerate(marginal[:])]
	max_k_zeros[slicing] = 1
	return [tuple(idx) for idx in np.argwhere(max_k_zeros)]

def add_marginals_restrict_to(restrict_to, max_k):
	marginal_np = np.array(restrict_to, dtype=np.uint8)
	marginal_mutypes_idxs = np.argwhere(np.any(marginal_np>max_k, axis=1)).reshape(-1)
	if marginal_mutypes_idxs.size>0:
		result = []
		#for mut_config_idx in marginal_mutypes_idxs:
		#	print(marginal_np[mut_config_idx])
		#	temp = list_marginal_idxs(marginal_np[mut_config_idx], max_k)
		#	result.append(temp)
		#	print(temp)
		result = [list_marginal_idxs(marginal_np[mut_config_idx], max_k) for mut_config_idx in marginal_mutypes_idxs]
		result = list(itertools.chain.from_iterable(result)) + restrict_to
		result = sorted(set(result))
	else:
		return sorted(restrict_to)
	return result

def adjust_marginals_array(array, dimension):
	new_array = copy.deepcopy(array) #why the deepcopy here?
	for j in range(dimension):
		new_array = _adjust_marginals_array(new_array, dimension, j)
	return new_array

def _adjust_marginals_array(array, dimension, j):
	idxs = np.roll(range(dimension), j)
	result = array.transpose(idxs)
	result[-1] = result[-1] - np.sum(result[:-1], axis=0)
	new_idxs=np.zeros(dimension, dtype=np.uint8)
	new_idxs[np.transpose(idxs)]=np.arange(dimension, dtype=np.uint8)
	return result.transpose(new_idxs)

def make_result_dict_from_mutype_tree_stack(gf, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k, precision=165):
	root = tuple(0 for _ in max_k) #root is fixed
	result = {}
	stack = [root,]
	path_parents = []
	while stack:
		node = stack.pop()
		if node != root:
			parent, parent_gf = path_parents[-1]
			if node in mutype_tree:
				#node is not a leaf
				for child in mutype_tree[node]:
					stack.append(child)
			while node not in mutype_tree[parent]:
				del path_parents[-1]
				parent, parent_gf = path_parents[-1]
			#assert node in mutype_tree[parent]
		else:
			parent, parent_gf = root, gf
			for child in mutype_tree[node]:
					stack.append(child)		
		#determine whether marginal probability or not
		if any(m>max_k_m for m, max_k_m in zip(node, max_k)):
				marginals = {branchtype:0 for branchtype, count, m_max_k in zip(ordered_mutype_list, node, max_k) if count>m_max_k}
				node_mutype = tuple(m if m<=max_k_m else None for m, max_k_m in zip(node, max_k))
				result[node] = sage.all.RealField(precision)(probK_from_diff(parent_gf.subs(marginals), theta, rate_dict, node_mutype))
		else:
			marginals = None
			relative_config = [b-a for a,b in zip(parent, node)]
			partial = single_partial(ordered_mutype_list, relative_config)
			diff = sage.all.diff(parent_gf, partial)
			if node in mutype_tree:
				path_parents.append((node, diff))
			result[node] = sage.all.RealField(precision)(probK_from_diff(diff, theta, rate_dict, node))
	return result

def make_result_dict_from_mutype_tree(gf, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k, precision=165):
	root = tuple(0 for _ in max_k)
	result = {}
	result[root] = probK_from_diff(gf, theta, rate_dict, root)
	intermediate_results = {}
	intermediate_results[root] = gf
	for parent, children in mutype_tree.items():
		for child in children:
			if any(m>max_k_m for m, max_k_m in zip(child, max_k)):
				marginals = {branchtype:0 for branchtype, count, m_max_k in zip(ordered_mutype_list, child, max_k) if count>m_max_k}
				child_mutype = tuple(m if m<=max_k_m else None for m, max_k_m in zip(child, max_k))
				result[child] = sage.all.RealField(precision)(probK_from_diff(intermediate_results[parent].subs(marginals), theta, rate_dict, child_mutype))
			else:
				marginals = None
				relative_config = [b-a for a,b in zip(parent, child)]
				partial = single_partial(ordered_mutype_list, relative_config)
				diff = sage.all.diff(intermediate_results[parent], partial)
				if child in mutype_tree.keys():
					intermediate_results[child] = diff
				result[child] = sage.all.RealField(precision)(probK_from_diff(diff, theta, rate_dict, child))
		del intermediate_results[parent] #once all children have been calculate no need to store original
	return result