import collections
import numpy as np

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