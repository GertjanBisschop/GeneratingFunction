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
	def __init__(self, gf, max_k, mutypes, precision=165, exclude=None, restrict_to=None):
		self.gf = gf
		self.max_k = max_k
		self.ordered_mutype_list = [sage.all.SR.var(mutype) for mutype in mutypes]
		if restrict_to!=None:
			all_mutation_configurations = mutations.add_marginals_restrict_to(restrict_to, max_k)
		else:
			all_mutation_configurations = mutations.return_mutype_configs(max_k)
		if exclude!=None:
			all_mutation_configurations = [m for m in all_mutation_configurations if all(not(all(m[idx]>0 for idx in e)) for e in exclude)]
		else:
			all_mutation_configurations = list(all_mutation_configurations)
		root = tuple(0 for _ in max_k)
		self.mutype_tree = mutations.make_mutype_tree(all_mutation_configurations, root, max_k)
		self.precision = precision
		
	def evaluate_gf(self, parameter_dict, theta, epsilon=0.0001):
		rate_dict = {branchtype:theta for branchtype in self.ordered_mutype_list}
		try:
			gf = sum(self.gf).subs(parameter_dict)
		except ValueError as e:
			if 'division by zero' in str(e):
				epsilon = sage.all.Rational(epsilon)
				M = sage.all.SR.var('M')
				parameter_dict[M] += parameter_dict[M]*epsilon
				gf = sum(self.gf).subs(parameter_dict)
		ETPs = mutations.make_result_dict_from_mutype_tree_stack(
			gf, 
			self.mutype_tree, 
			theta, rate_dict, 
			self.ordered_mutype_list, 
			self.max_k,
			self.precision
			)
		ETPs = mutations.dict_to_array(ETPs, (4,4,4,4), dtype=np.float64)
		try:
			assert np.all(np.logical_and(ETPs>=0, ETPs<=1))
		except AssertionError:
			print("[-] Some ETPs are not in [0,1]. Increase machine precision in the ini file.")
		ETPs = mutations.adjust_marginals_array(ETPs, len(self.max_k))
		if not np.all(ETPs>0):
			ETPs[ETPs<0] = 0
		ETPs = ETPs.astype(np.float64)
		try:
			assert np.isclose(np.sum(ETPs), 1.0, rtol=1e-4)
		except AssertionError:
			sys.exit(f"[-] sum(ETPs): {np.sum(ETPs)} != 1 (rel_tol=1e-4)")
		return ETPs