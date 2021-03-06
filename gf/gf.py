import collections
import copy
import functools
import itertools
import numpy as np
import operator
import sage.all
import string
import sys

from timeit import default_timer as timer

#auxilliary functions
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
	coal_counts = collections.Counter(coal_event_pairs)
	for lineages, count in coal_counts.items():
		result = coalesce_lineages(pop_state, lineages)
		yield (count, result)

#dealing with mutation/branchtypes
def make_branchtype_dict(sample_list, mapping='unrooted', labels=None):
	'''Maps lineages to their respective mutation type
	Mappings: 'unrooted', 'label'
	'''
	all_branchtypes=list(flatten(sample_list))
	branches = [branchtype for branchtype in powerset(all_branchtypes) if len(branchtype)>0 and len(branchtype)<len(all_branchtypes)]	
	if mapping.startswith('label'):
		if labels:
			assert len(branches)==len(labels), "number of labels does not match number of branchtypes"
			branchtype_dict = {branchtype:sage.all.SR.var(label) for branchtype, label in zip(branches, labels)}
		else:
			branchtype_dict = {branchtype:sage.all.SR.var(f'z_{branchtype}') for branchtype in branches}
	elif mapping=='unrooted': #this needs to be extended to the general thing!
		if not labels:
			labels = ['m_1', 'm_2', 'm_3', 'm_4']
		assert set(all_branchtypes)=={'a', 'b'}
		branchtype_dict=dict()
		for branchtype in powerset(all_branchtypes):
			if len(branchtype)==0 or len(branchtype)==len(all_branchtypes):
				pass
			elif branchtype in ('abb', 'a'):
				branchtype_dict[branchtype] = sage.all.SR.var(labels[1]) #hetA
			elif branchtype in ('aab', 'b'):
				branchtype_dict[branchtype] = sage.all.SR.var(labels[0]) #hetB
			elif branchtype == 'ab':
				branchtype_dict[branchtype] = sage.all.SR.var(labels[2]) #hetAB
			else:
				branchtype_dict[branchtype] = sage.all.SR.var(labels[3]) #fixed difference
	else:
		ValueError("This branchtype mapping has not been implemented yet.")
	return branchtype_dict

def sort_mutation_types(branchtypes):
	if isinstance(branchtypes, dict):
		return sorted(set(branchtypes.values()), key= lambda x: str(x))
	elif isinstance(branchtypes, list) or isinstance(branchtypes, tuple):
		return sorted(set(branchtypes), key= lambda x: str(x))
	else:
		raise ValueError(f'sort_mutation_types not implemented for {type(branchtypes)}')

#processing generating function
def inverse_laplace(equation, dummy_variable):
	return (sage.all.inverse_laplace(subequation / dummy_variable, dummy_variable, sage.all.SR.var('T', domain='real'), algorithm='giac') for subequation in equation)

def return_inverse_laplace(equation, dummy_variable):
    return sage.all.inverse_laplace(
        equation / dummy_variable, 
        dummy_variable,
        sage.all.SR.var('T', domain='real'), 
        algorithm='giac'
        )

#representing samples
def sample_to_str(sample_list):
	return '/'.join('_'.join(lineage for lineage in pop) for pop in sample_list)

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
			self.migration_rate = sage.all.SR.var('M') #set domain???
		else:
			self.migration_rate = migration_rate
		self.exodus_direction = exodus_direction
		if exodus_direction and not exodus_rate:
			self.exodus_rate = sage.all.SR.var('E') #set domain?
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

################################## old functions #################################

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