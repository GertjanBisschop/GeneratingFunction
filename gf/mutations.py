import itertools
import collections
import copy
import multiprocessing
import numpy as np
import sage.all

from . import gf as gflib

def single_partial(ordered_mutype_list, partial):
		return list(gflib.flatten(itertools.repeat(branchtype,count) for count, branchtype in zip(partial, ordered_mutype_list)))

def return_mutype_configs(max_k, include_marginals=True):
	add_to_k = 2 if include_marginals else 1
	iterable_range = (range(k+add_to_k) for k in max_k)
	return itertools.product(*iterable_range)

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
				root_config = differs_one_digit(mutype, all_mutypes[:idx])
				result[root_config].append(mutype)
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

def make_result_dict_from_mutype_tree_stack(gf, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k, precision=165):
	root = tuple(0 for _ in max_k)
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
		try:		
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
		except TypeError:
			print(parent_gf)
			sys.exit()
	return result

def differs_one_digit(query, complete_list):
	#complete_set = self.generate_differs_one_set(query)
	#similar_to = next(obj for obj in complete_list[idx-1::-1] if obj in complete_set)
	similar_to = next(obj for obj in complete_list[::-1] if sum_tuple_diff(obj, query)==1)
	return similar_to

def sum_tuple_diff(tuple_a, tuple_b):
	return sum(b-a for a,b in zip(tuple_a, tuple_b))
	
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

def dict_to_array(result_dict, shape):
	#result = np.zeros(shape, dtype=np.float64)
	result = np.zeros(shape, dtype=object)
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