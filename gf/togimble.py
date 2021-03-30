import numpy as np
import sage.all

from . import gf as gflib

def _get_gfObj(config):
	sample_list = config['sample_list']
	exodus_rate = sage.all.var('E') if config.get('T') else None
	migration_rate = sage.all.var('M') if config.get('me_A_B') or config.get('me_B_A') else None
	coalescence_rates = (sage.all.var('c0'), sage.all.var('c1'), sage.all.var('c2'))
	labels = gflib.sort_mutation_types(list(config["k_max"].keys()))
	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted', labels=labels)

	gfobj = gflib.GFObject(
		sample_list, 
		coalescence_rates, 
		branchtype_dict, 
		migration_rate=migration_rate, 
		migration_direction=config.get('migration_direction'), 
		exodus_rate=exodus_rate, 
		exodus_direction=config.get('exodus_direction')
		)
	return gfobj

def _return_inverse_laplace(gfobj, gf):
	if gfobj.exodus_rate:
		return list(gflib.inverse_laplace(gf, gfobj.exodus_rate))
	else:
		return list(gf)

def get_gf(config):
	gfobj = _get_gfObj(config)
	gf = gfobj.make_gf()
	return _return_inverse_laplace(gfobj, gf)

class gfEvaluator:
	def __init__(self, gf, max_k, mutypes):
		self.gf = gf
		self.max_k = max_k
		self.ordered_mutype_list = gflib.sort_mutation_types(mutypes)
		all_mutation_configurations = list(gflib.return_mutype_configs(max_k))
		root = tuple(0 for _ in max_k)
		self.mutype_tree = gflib.make_mutype_tree(all_mutation_configurations, root, max_k)

	def evaluate_gf(self, parameter_dict, theta):
		rate_dict = {branchtype:theta for branchtype in self.ordered_mutype_list}
		gf = sum(self.gf).subs(parameter_dict)
		ETPs = gflib.make_result_dict_from_mutype_tree_stack(
			gf, 
			self.mutype_tree, 
			theta, rate_dict, 
			self.ordered_mutype_list, 
			self.max_k
			)
		ETPs = gflib.dict_to_array(ETPs, (4,4,4,4))
		ETPs = gflib.adjust_marginals_array(ETPs, len(self.max_k))
		return ETPs.astype(np.float64)

	def validate_parameters(self, parameter_dict, mutypes):
		arguments = self.gf.arguments()
		#theta presence