import sage.all
import numpy as np
import itertools
import string
import collections
import itertools
import operator
import functools
import copy
import sys
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing
from timeit import default_timer as timer

def powerset(iterable):
	"""returns generator containing all possible subsets of iterable
	Arguments:
		iterable {[type]} -- [description]
	"""
	s=list(iterable)
	return (''.join(sorted(subelement)) for subelement in (itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))))

def flatten(input_list):
	"""Flattens iterable from depth n to depth n-1
	Arguments:
		input_list {[iterable]} -- [description]
	"""
	return itertools.chain.from_iterable(input_list)

def coalesce_lineages(input_tuple, to_join):
	'''Joins lineages and returns resulting tuple of lineages for a single pop
	Arguments:
		input_tuple {[iterable]} -- [iterable containing lineages(str)]
		to_join {[iterable]} -- [containing all lineages to be joined]
	'''
	result=list(input_tuple)
	for lineage in to_join:
		result.remove(lineage)
	result.append(''.join(sorted(flatten(to_join))))
	result.sort()
	return tuple(result)

def coalesce_single_pop(pop_state):
	"""For single population generate all possible coalescence events
	Arguments:
		pop_state {[iterable]} -- [containing all lineages (str) present within pop]
		coal_rate {[sage_var, float]} -- [rate at which coalescence happens in that pop]
	"""
	coal_event_pairs = list(itertools.combinations(pop_state,2))
	#print('pairs:', coal_event_pairs)
	coal_counts = collections.Counter(coal_event_pairs)
	#print('pop_state:', pop_state)
	#print('coal_counts:', coal_counts)
	for lineages, count in coal_counts.items():
		#print('lineages:', lineages)
		result = coalesce_lineages(pop_state, lineages)
		#print('result_lineages:', result)
		yield (count, result)

def make_branchtype_dict(sample_list, mapping='unrooted'):
	'''Maps lineages to their respective mutation type
	Mappings: 'unrooted', 'label'
	'''
	all_branchtypes=list(flatten(sample_list))
	branches = [branchtype for branchtype in powerset(all_branchtypes) if len(branchtype)>0 and len(branchtype)<len(all_branchtypes)]	
	if mapping.startswith('label'):
		branchtype_dict = {branchtype:sage.all.var(f'z_{branchtype}') for branchtype in set(branches)}
	elif mapping=='unrooted': #this needs to be extended to the general thing!
		assert set(all_branchtypes)=={'a', 'b'}
		branchtype_dict=dict()
		for branchtype in powerset(all_branchtypes):
			if len(branchtype)==0 or len(branchtype)==len(all_branchtypes):
				pass
			elif branchtype in ('abb', 'a'):
				branchtype_dict[branchtype] = sage.all.var('m_2') #hetA
			elif branchtype in ('aab', 'b'):
				branchtype_dict[branchtype] = sage.all.var('m_1') #hetB
			elif branchtype == 'ab':
				branchtype_dict[branchtype] = sage.all.var('m_3') #hetAB
			else:
				branchtype_dict[branchtype] = sage.all.var('m_4') #fixed difference
	else:
		ValueError("This branchtype mapping has not been implemented yet.")
	return branchtype_dict

def sort_mutation_types(branchtypes):
	if isinstance(branchtypes, dict):
		return sorted(set(branchtypes.values()), key= lambda x: str(x))
	elif isinstance(branchtypes, list):
		return sorted(set(branchtypes), key= lambda x: str(x))
	else:
		raise ValueError(f'sort_mutation_types not implemented for {type(branchtypes)}')

def inverse_laplace(equation, dummy_variable):
	return (sage.all.inverse_laplace(subequation / dummy_variable, dummy_variable, sage.all.SR.var('T', domain='real'), algorithm='giac') for subequation in equation)

def split_gf(gf, chunksize):
	#splitting gf generator using chunksize
	i = iter(gf)
	piece = sum(itertools.islice(i, chunksize))
	while piece:
		yield piece
		piece = sum(itertools.islice(i, chunksize))

def split_gf_iterable(gf, chunksize):
	#splitting gf generator using chunksize
	i = iter(gf)
	piece = list(itertools.islice(i, chunksize))
	while piece:
		yield piece
		piece = list(itertools.islice(i, chunksize))
	
def simple_probK(gf, theta, partials, marginals, ratedict, mucount_total, mucount_fact_prod):
	gf_marginals = gf.substitute(marginals)
	derivative = sage.all.diff(gf_marginals,partials)
	return (-1*theta)**(mucount_total)/mucount_fact_prod*derivative.substitute(ratedict)

def symbolic_prob_per_term(gf_term, prob_input_subdict, shape, theta, adjust_marginals=False):
	result = {mutation_configuration:simple_probK(gf_term, theta, **subdict) for mutation_configuration, subdict in prob_input_subdict.items()}
	result_array = dict_to_array(result, shape)
	if adjust_marginals==True:
		result_array = adjust_marginals(result_array, len(shape))
	return result_array

def process_grouped_terms(gf_terms, dummy_variable, prob_input_dict, shape, theta, adjust_marginals):
	start_time=timer()
	if dummy_variable:
			gf_to_use = sum(inverse_laplace(gf_terms, dummy_variable))
	else:
		gf_to_use = sum(gf_terms)
	result = symbolic_prob_per_term(gf_to_use, prob_input_dict, shape, theta, adjust_marginals) 
	print(timer()-start_time)
	return result

def make_symbolic_prob_array(gf, ordered_mutype_list, kmax_by_mutype, theta, dummy_variable, chunksize=100, num_processes=1, adjust_marginals=False):
	iterable_range = (range(k+2) for k in kmax_by_mutype)
	all_mutation_configurations = itertools.product(*iterable_range)
	prob_input_dict = prepare_symbolic_prob_dict(ordered_mutype_list, kmax_by_mutype, theta)
	shape = tuple(kmax+2 for kmax in kmax_by_mutype)
	#gf is generator for gf prior to taking inverse laplace
	symbolic_prob_array = []
	if num_processes==1:
		symbolic_prob_array = sum([process_grouped_terms(
			gf_terms, dummy_variable, prob_input_dict, shape, theta, adjust_marginals) 
				for gf_terms in split_gf_iterable(gf, chunksize)])
	else:
		process_grouped_terms_specified = functools.partial(
			process_grouped_terms,
			dummy_variable=dummy_variable,
			prob_input_subdict=prob_input_dict,
			shape=shape,
			theta=theta,
			adjust_marginals=adjust_marginals
			)
		with multiprocessing.Pool(processes=num_processes) as pool:
			symbolic_prob_array = sum(pool.imap(process_grouped_terms_specified, split_gf(gf, chunksize)))
	return symbolic_prob_array

def prepare_symbolic_prob_dict(ordered_mutype_list, kmax_by_mutype, theta):
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

def substitute_parameters(symbolic_prob_dict, parameter_dict, ordered_mutype_list, kmax_by_mutype, precision=165):
	shape = tuple(kmax+2 for kmax in kmax_by_mutype)
	variables = list(parameter_dict.keys())
	return {mutation_configuration:expression.substitute(parameter_dict) for mutation_configuration, expression in symbolic_prob_dict.items()}

def adjust_marginals(array, dimension):
	new_array = copy.deepcopy(array) #why the deepcopy here?
	for j in range(dimension):
		new_array = _adjust_marginals(new_array, dimension, j)
	return new_array

def _adjust_marginals(array, dimension, j):
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

def make_graph(all_paths, name):
	G = nx.DiGraph()
	counter = 0
	mapping_dict = {}
	for single_path in all_paths:
		#in single path, all elements ((),(),('a','a'))
		single_path_str = ['/'.join(','.join(e  if len(line)>0 else X for e in line) for line in pop[1]) for pop in single_path]
		G.add_edges_from(zip(single_path_str[:-1], single_path_str[1:]))
	nx.draw_spring(G, with_labels=True)
	plt.savefig(f"{name}.png")

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

class GFObject:
	def __init__(self, sample_list, coalescence_rates, branchtype_dict, migration_direction=None, migration_rate=None, exodus_direction=None, exodus_rate=None):
		assert len(sample_list) == len(coalescence_rates)
		if sum(1 for pop in sample_list if len(pop)>0)>1:
			assert migration_direction or exodus_direction, 'lineages from different populations cannot coalesce without migration or exodus event.'
		self.sample_list = tuple(tuple(sorted(pop)) for pop in sample_list)
		self.coalescence_rates = coalescence_rates
		self.branchtype_dict = branchtype_dict
		self.migration_direction = migration_direction
		if migration_direction and not migration_rate:
			self.migration_rate = sage.all.var('M') #set domain???
		else:
			self.migration_rate = migration_rate
		self.exodus_direction = exodus_direction
		if exodus_direction and not exodus_rate:
			self.exodus_rate = sage.all.var('E') #set domain?
		else:
			self.exodus_rate = exodus_rate

	def coalescence_events(self, state_list):
		"""Generator returning all possible new population configurations due to coalescence,
		and their respective rates
		Arguments:
			state_list {[list]} -- [list of population tuples containing lineages (str)]
		"""
		result=[]
		for idx, (pop, rate) in enumerate(zip(state_list, self.coalescence_rates)):
			for count, coal_event in coalesce_single_pop(pop):
				modified_state_list = list(state_list)
				modified_state_list[idx] = coal_event
				result.append((count*rate, tuple(modified_state_list)))
				#yield((count, tuple(modified_state_list)))
		return result

	def migration_events(self, state_list):
		"""Returning all possible new population configurations due to migration events, 
		and their respective rates
		BACKWARDS IN TIME
		Arguments:
			state_list {[list]} -- [list of population tuples containing lineages(str)]
		"""
		result = []
		if self.migration_direction:
			for source, destination in self.migration_direction:
				lineage_count = collections.Counter(state_list[source])
				for lineage, count in lineage_count.items():
					temp = list(state_list)
					idx = temp[source].index(lineage)
					temp[source] = tuple(temp[source][:idx] + temp[source][idx+1:])
					temp[destination] = tuple(sorted(list(temp[destination]) + [lineage,])) 
					result.append((count*self.migration_rate, tuple(temp)))
		return result
		
	def exodus_events(self, state_list):
		"""Returning all possible new population configurations due to exodus events,
		and their respective rates
		BACKWARDS IN TIME	
		Arguments:
			state_list {[list]} -- [list of population tuples containing lineages(str)]
		"""
		result = []
		if self.exodus_direction:
			for *source, destination in self.exodus_direction:
				temp = list(state_list)
				sources_joined = tuple(itertools.chain.from_iterable([state_list[idx] for idx in source]))
				if len(sources_joined)>0:
					temp[destination] = tuple(sorted(state_list[destination] + sources_joined))
					for idx in source:
						temp[idx] = ()
					result.append((self.exodus_rate, tuple(temp)))
		return result

	def rates_and_events(self, state_list):
		"""Returning all possible events, and their respective rates
		Arguments:
			state_list {[list]} -- [list of population tuples containing lineages(str)]
		"""
		c = self.coalescence_events(state_list)
		m = self.migration_events(state_list)
		e = self.exodus_events(state_list)
		return c+m+e

	def gf_single_step(self, gf_old, state_list):
		"""Yields single (tail) recursion step for the generating function
		Arguments:
			gf_old {[expression]} -- [result from previous recursion step]
			state_list {[list]} -- [list of population tuples containing lineages(str)]
		"""
		current_branches = list(flatten(state_list))
		numLineages = len(current_branches)
		if numLineages == 1:
			ValueError('gf_single_step fed with single lineage, should have been caught.')
		else:
			outcomes = self.rates_and_events(state_list)
			total_rate = sum([rate for rate, state in outcomes])
			dummy_sum = sum(self.branchtype_dict[b] for b in current_branches)    
			#for rate, new_state_list in outcomes:
			#	yield (gf_old*rate*1/(total_rate + dummy_sum), new_state_list)
			return [(gf_old*rate*1/(total_rate + dummy_sum), new_state_list) for rate, new_state_list in outcomes]
		
	def make_gf(self):
		"""[generator returning generating function]
		Yields:
			[type] -- []
		"""
		stack = [(1, self.sample_list)]
		result=[]
		while stack:
			gf_n, state_list =stack.pop()
			if sum(len(pop) for pop in state_list)==1:		
				yield gf_n
			else:
				for gf_nplus1 in self.gf_single_step(gf_n, state_list):
					stack.append(gf_nplus1)

	def gf_single_step_graph(self, tracking_list, state_list):
		current_branches = list(flatten(state_list))
		numLineages = len(current_branches)
		if numLineages == 1:
			ValueError('gf_single_step fed with single lineage, should have been caught.')
		else:
			outcomes = self.rates_and_events(state_list)
			for rate, new_state_list in outcomes:
				yield (tracking_list[:]+[(rate, new_state_list),], new_state_list)

	def make_gf_graph(self):
		"""[generator returning gf graph]
		Yields:
			[type] -- []
		"""
		stack = [([(1,tuple(self.sample_list))], self.sample_list)]
		result=[]
		while stack:
			tracking_list, state_list =stack.pop()
			if sum(len(pop) for pop in state_list)==1:
				yield tracking_list
			else:
				for new_step in self.gf_single_step_graph(tracking_list, state_list):
					stack.append(new_step)