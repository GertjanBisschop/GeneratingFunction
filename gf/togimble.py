import numpy as np
import sage.all

from . import gf as gflib
from . import mutations

def _get_gfObj(sample_list, coalescence_rates, mutype_labels, migration_direction=None, migration_rate=None, exodus_direction=None, exodus_rate=None):
	#labels = gflib.sort_mutation_types(list(k_max.keys()))
	#labels = sorted_mutypes
	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted', labels=mutype_labels)

	gfobj = gflib.GFObject(
		sample_list, 
		coalescence_rates, 
		branchtype_dict, 
		migration_rate=migration_rate, 
		migration_direction=migration_direction, 
		exodus_rate=exodus_rate, 
		exodus_direction=exodus_direction
		)
	return gfobj

def _return_inverse_laplace(gfobj, gf):
	if gfobj.exodus_rate:
		return list(gflib.inverse_laplace(gf, gfobj.exodus_rate))
	else:
		return list(gf)

def get_gf(sample_list, coalescence_rates, mutype_labels, migration_direction=None, migration_rate=None, exodus_direction=None, exodus_rate=None):
	gfobj = _get_gfObj(sample_list, coalescence_rates, mutype_labels, migration_direction, migration_rate, exodus_direction, exodus_rate)
	gf = gfobj.make_gf()
	return _return_inverse_laplace(gfobj, gf)

class gfEvaluator:
	def __init__(self, gf, max_k, mutypes):
		self.gf = gf
		self.max_k = max_k
		self.ordered_mutype_list = [sage.all.var(mutype) for mutype in mutypes]
		all_mutation_configurations = list(mutations.return_mutype_configs(max_k))
		root = tuple(0 for _ in max_k)
		self.mutype_tree = mutations.make_mutype_tree(all_mutation_configurations, root, max_k)

	def evaluate_gf(self, parameter_dict, theta, epsilon=0.0001):
		rate_dict = {branchtype:theta for branchtype in self.ordered_mutype_list}
		try:
			gf = sum(self.gf).subs(parameter_dict)
		except ValueError as e:
			if 'division by zero' in str(e):
				epsilon = sage.all.Rational(epsilon)
				M = sage.all.var('M')
				parameter_dict[M] += parameter_dict[M]*epsilon
				gf = sum(self.gf).subs(parameter_dict)
		ETPs = mutations.make_result_dict_from_mutype_tree_stack(
			gf, 
			self.mutype_tree, 
			theta, rate_dict, 
			self.ordered_mutype_list, 
			self.max_k
			)
		ETPs = mutations.dict_to_array(ETPs, (4,4,4,4))
		ETPs = mutations.adjust_marginals_array(ETPs, len(self.max_k))
		if not np.all(ETPs>0):
			ETPs[ETPs<0] = 0 
		return ETPs.astype(np.float64)

	def validate_parameters(self, parameter_dict, mutypes):
		arguments = self.gf.arguments()
		#theta presence