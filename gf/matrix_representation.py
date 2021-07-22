import itertools
import numpy as np

from . import gf as gflib

def make_branchtype_dict_idxs(sample_list, mapping='unrooted', labels=None, starting_index=0):
	all_branchtypes=list(gflib.flatten(sample_list))
	branches = [branchtype for branchtype in gflib.powerset(all_branchtypes) if len(branchtype)>0 and len(branchtype)<len(all_branchtypes)]	
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
		for branchtype in gflib.powerset(all_branchtypes):
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

def equations_from_matrix(multiplier_array, variables_array):
	temp = multiplier_array.dot(variables_array)
	return temp[:,0]/temp[:,1]