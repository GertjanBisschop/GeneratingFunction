import collections
import numpy as np
import sage.all
from sage.symbolic.operators import add_vararg

from . import gf as gflib
from . import mutations

class GFObjectChainRule(gflib.GFObject):
	
	def make_gf(self):
		stack = [(list(), self.sample_list),]
		paths =  list()
		eq_list = list()
		
		#keeping track of things
		graph_dict = collections.defaultdict(list)
		equation_dict = dict() #key=(parent, child), value=eq_idx
		nodes_visited = list() #list of all nodes visisted
	
		while stack:
			path_so_far, state_list = stack.pop()
			parent_node = gflib.sample_to_str(state_list)
			if sum(len(pop) for pop in state_list)==1:		
				paths.append(path_so_far)
			else:
				if parent_node in nodes_visited:
					#depth first search through graph
					for add_on_path in paths_from_visited_node(graph_dict, parent_node, equation_dict, path_so_far):
						paths.append(add_on_path)		
				else:
					nodes_visited.append(parent_node)
					for eq, new_state_list in self.gf_single_step(state_list):
						child_node = gflib.sample_to_str(new_state_list)
						eq_idx = len(eq_list)
						eq_list.append(eq)
						path = path_so_far[:]
						path.append(eq_idx)
						graph_dict[parent_node].append(child_node)
						equation_dict[(parent_node, child_node)] = eq_idx
						stack.append((path, new_state_list))
						
		return (paths, eq_list)
	
	def gf_single_step(self, state_list):
			current_branches = list(gflib.flatten(state_list))
			numLineages = len(current_branches)
			if numLineages == 1:
				ValueError('gf_single_step fed with single lineage, should have been caught.')
			else:
				outcomes = self.rates_and_events(state_list)
				total_rate = sum([rate for rate, state in outcomes])
				dummy_sum = sum(self.branchtype_dict[b] for b in current_branches)    
				return [(rate*1/(total_rate + dummy_sum), new_state_list) for rate, new_state_list in outcomes]
	
def paths_from_visited_node(graph, node, equation_dict, path):
	stack = [(path, node),]
	while stack:
		path, parent = stack.pop()
		children = graph.get(parent, None)
		if children!=None:
			for child in children:
				stack.append((path[:] + [equation_dict[(parent, child)],], child)) 
		else:
			yield path

def gf_from_paths(paths, eq_list, default=None):
	result = [np.prod(eq_list[idxs]) for idxs in paths]
	if len(result)==0 and default!=None:
		return [default,]
	return result

def split_paths(paths, eq_list, dummy_var):
	if dummy_var==None:
		return (paths, [])
	to_invert = []
	constant = []
	for path in paths:
		temp_invert = []
		temp_constant = []
		for eq_idx in path:
			if eq_list[eq_idx].has(dummy_var):
				temp_invert.append(eq_idx)
			else:
				temp_constant.append(eq_idx)
		to_invert.append(sorted(temp_invert))
		constant.append(sorted(temp_constant))
	return (constant, to_invert)

def remap_paths_unique(old_paths):
	unique = np.unique(list(gflib.flatten(old_paths)))
	sorter = np.argsort(unique)
	return (unique, [sorter[np.searchsorted(unique, path, sorter=sorter)] for path in old_paths])

def invert_eq(to_invert_paths, eq_list, dummy_var, num_constant_eq=0, expand_inverted_eq=False):
	#from to_invert: filter out the unique (combinations of) equations:
	unique_equations_to_invert, inverse_idxs = np.unique(to_invert_paths, return_inverse=True)
	if len(unique_equations_to_invert)==0:
		inverse_idxs = np.array([0]*len(to_invert_paths))
	#reassemble unique terms that need to be inverted
	to_invert_reassembled = gf_from_paths(unique_equations_to_invert, eq_list, default=1)
	num_unique_equations = len(to_invert_reassembled)
	inverted = np.empty(num_unique_equations, dtype=object)
	for idx, eq in enumerate(to_invert_reassembled):
		inverted[idx] = gflib.return_inverse_laplace(eq, dummy_var) 
	#inverted = [gflib.return_inverse_laplace(eq, dummy_var) for eq in to_invert_reassembled]
	
	if expand_inverted_eq:
		#each equation to invert becomes one or more
		#numbering should start with num_constant_eq
		idx_inverse = num_constant_eq
		all_inverted = []
		inverted_paths = []
		
		for eq in inverted:
			single_path = []
			if eq == 1:
				pass
			else:
				expanded_eq = sage.all.expand(eq)
				if expanded_eq.operator() == add_vararg:
					for expanded_eq in expanded_eq.operands():
						all_inverted.append(expanded_eq)
						single_path.append(idx_inverse)
						idx_inverse+=1    
				else:
					all_inverted.append(eq)
					single_path.append(idx_inverse)
					idx_inverse+=1
			inverted_paths.append(single_path)
		all_inverted = np.array(all_inverted, dtype=object)
		expand_inverted_paths = [inverted_paths[i] for i in inverse_idxs]
		return (all_inverted, expand_inverted_paths)	
	else:
		inverted_paths = inverse_idxs + num_constant_eq
		expand_inverted_paths = np.reshape(inverted_paths, (-1,1))
		return (inverted, expand_inverted_paths)

def expand_paths_associatively(constant, to_expand):
	assert len(constant)==len(to_expand), 'expand_paths_associatively expects equal lengths for the arguments.'
	for c, te in zip(constant, to_expand):
		if len(te)==0:
			yield np.array(c)
		else:
			for term in te:
				yield np.hstack([c,term])

def all_paths_after_inversion(paths_uninverted, equations_uninverted, dummy_var):
	#split paths into paths with and without dummy_var
	constant_paths, to_invert_paths = split_paths(paths_uninverted, equations_uninverted, dummy_var)
	constant_unique, constant_paths_remapped = remap_paths_unique(constant_paths)
	constant_eq_unique = equations_uninverted[constant_unique]
	#invert_eq step can be improved upon
	all_inverted_eq, expand_inverted_paths = invert_eq(to_invert_paths, equations_uninverted, dummy_var, constant_unique.size)
	all_equations = np.hstack([constant_eq_unique, all_inverted_eq])
	#expand_paths_associatively unnecessary when expand_invert_eq=False
	all_paths = list(expand_paths_associatively(constant_paths_remapped, expand_inverted_paths))
	return (all_equations, all_paths)
