import collections
import itertools
import numpy as np
import sys

from . import gf as gflib
from . import chain_rule as cr

def make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=None, starting_index=0):
	all_branchtypes=list(flatten(sample_list))
	branches = [branchtype for branchtype in powerset(all_branchtypes) if len(branchtype)>0 and len(branchtype)<len(all_branchtypes)]	
	if mapping.startswith('label'):
		if labels:
			assert len(branches)==len(labels), "number of labels does not match number of branchtypes"
			branchtype_dict = {branchtype: idx + starting_index for idx, branchtype in enumerate(branches)}
		else:
			branchtype_dict = {branchtype: idx + starting_index for idx, branchtype in enumerate(branches)}
	elif mapping=='unrooted': #this needs to be extended to the general thing!
		if not labels:
			labels = ['m_1', 'm_2', 'm_3', 'm_4']
		assert set(all_branchtypes)=={'a', 'b'}
		branchtype_dict=dict()
		for branchtype in powerset(all_branchtypes):
			if len(branchtype)==0 or len(branchtype)==len(all_branchtypes):
				pass
			elif branchtype in ('abb', 'a'):
				branchtype_dict[branchtype] = 1 + starting_index #hetA
			elif branchtype in ('aab', 'b'):
				branchtype_dict[branchtype] = 0 + starting_index #hetB
			elif branchtype == 'ab':
				branchtype_dict[branchtype] = 2 + starting_index #hetAB
			else:
				branchtype_dict[branchtype] = 3 + starting_index #fixed difference
	else:
		ValueError("This branchtype mapping has not been implemented yet.")
	return branchtype_dict

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

def equations_from_matrix(multiplier_array, variables_array):
	temp = multiplier_array.dot(variables_array)
	return temp[:,0]/temp[:,1]

class GFObject:
	def __init__(self, sample_list, coalescence_rates, branchtype_dict, migration_direction=None, migration_rate=None, exodus_direction=None, exodus_rate=None):
		assert len(sample_list) == len(coalescence_rates)
		if sum(1 for pop in sample_list if len(pop)>0)>1:
			assert migration_direction or exodus_direction, 'lineages from different populations cannot coalesce without migration or exodus event.'
		self.sample_list = tuple(tuple(sorted(pop)) for pop in sample_list)
		assert all(isinstance(coal_rate, int) for coal_rate in coalescence_rates), "Coalescence rates should be indices."
		self.coalescence_rate_idxs = coalescence_rates #should be indices, can be the same: (0,0,1)
		self.branchtype_dict = branchtype_dict #dict with indices rather than labels
		self.num_branchtypes = len(set(self.branchtype_dict.values()))
		
		num_events = 0
		self.migration_direction = migration_direction
		if migration_direction and not migration_rate:
			raise ValueError('Migration direction provided but no migration rate.')
		else:
			if migration_rate!=None:
				assert isinstance(migration_rate, int), "Migration rate should be an integer index."
				num_events+=1
			self.migration_rate_idx = migration_rate #should be an idx
		self.exodus_direction = exodus_direction
		if exodus_direction and not exodus_rate:
			raise ValueError('Exodus direction provided but no exodus rate.')
		else:
			if exodus_rate!=None:
				num_events+=1
				assert isinstance(exodus_rate, int), "Exodus rate should be an integer index."
			self.exodus_rate_idx = exodus_rate #should be an idx
		self.num_variables = max(coalescence_rates) + 1 + num_events

	def make_gf(self):
		stack = [(list(), self.sample_list),]
		paths =  list()
		eq_list = list()
		eq_idx = 0
		#keeping track of things
		graph_dict = collections.defaultdict(list)
		equation_dict = dict() #key=(parent, child), value=eq_idx
		nodes_visited = list() #list of all nodes visisted
	
		while stack:
			path_so_far, state = stack.pop()
			parent_node = gflib.sample_to_str(state)
			if sum(len(pop) for pop in state)==1:		
				paths.append(path_so_far)
			else:
				if parent_node in nodes_visited:
					#depth first search through graph
					for add_on_path in cr.paths_from_visited_node(graph_dict, parent_node, equation_dict, path_so_far):
						paths.append(add_on_path)		
				else:
					nodes_visited.append(parent_node)
					multiplier_array, new_state_list = self.gf_single_step(state)
					eq_list.append(multiplier_array)
					for new_state in new_state_list:
						child_node = gflib.sample_to_str(new_state)
						path = path_so_far[:]
						path.append(eq_idx)
						graph_dict[parent_node].append(child_node)
						equation_dict[(parent_node, child_node)] = eq_idx
						stack.append((path, new_state))
						eq_idx+=1
		return (paths, np.concatenate(eq_list, axis=0))

	def gf_single_step(self, state_list):
		current_branches = list(gflib.flatten(state_list))
		numLineages = len(current_branches)
		if numLineages == 1:
			ValueError('gf_single_step fed with single lineage, should have been caught.')
	
		# collecting the idxs of branches in state_list
		dummy_sum = [self.branchtype_dict[b] for b in current_branches]
		dummy_array = np.zeros((2, self.num_branchtypes), dtype=np.uint8) 
		dummy_unique, dummy_counts = np.unique(dummy_sum, return_counts=True)
		dummy_array[1, dummy_unique] = dummy_counts
		
		outcomes = self.rates_and_events(state_list)
		multiplier_array = np.zeros((len(outcomes), 2, self.num_variables), dtype=np.uint8)
		new_state_list = list()
		
		for event_idx, (rate_idx, count, new_state) in enumerate(outcomes):
			multiplier_array[event_idx, 0, rate_idx] = count
			new_state_list.append(new_state)
		
		multiplier_array[:, 1] = np.sum(multiplier_array[:,0], axis=0)
		dummy_array = np.tile(dummy_array, (multiplier_array.shape[0],1,1))
		multiplier_array = np.concatenate((multiplier_array, dummy_array), axis=2)
		return (multiplier_array, new_state_list)

	def coalescence_events(self, state_list):
		result = []
		for idx, (pop, rate_idx) in enumerate(zip(state_list, self.coalescence_rate_idxs)):
			for count, coal_event in coalesce_single_pop(pop):
				modified_state_list = list(state_list)
				modified_state_list[idx] = coal_event
				result.append((rate_idx, count, tuple(modified_state_list)))
		return result

	def migration_events(self, state_list):
		result = []
		if self.migration_direction:
			for source, destination in self.migration_direction:
				lineage_count = collections.Counter(state_list[source])
				for lineage, count in lineage_count.items():
					temp = list(state_list)
					idx = temp[source].index(lineage)
					temp[source] = tuple(temp[source][:idx] + temp[source][idx+1:])
					temp[destination] = tuple(sorted(list(temp[destination]) + [lineage,])) 
					result.append((self.migration_rate_idx, count, tuple(temp)))
		return result

	def exodus_events(self, state_list):
		result = []
		if self.exodus_direction:
			for *source, destination in self.exodus_direction:
				temp = list(state_list)
				sources_joined = tuple(itertools.chain.from_iterable([state_list[idx] for idx in source]))
				if len(sources_joined)>0:
					temp[destination] = tuple(sorted(state_list[destination] + sources_joined))
					for idx in source:
						temp[idx] = ()
					result.append((self.exodus_rate_idx, 1, tuple(temp)))
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