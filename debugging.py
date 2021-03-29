import pytest
import gf.gf as gflib
from timeit import default_timer as timer
import sage.all
import numpy as np
import tests.aux_functions as af
import sys, os
import itertools
import resource

all_configs = {
	'IM_AB': {
		"global_info" : {
					'model_file':"models/IM_AB.tsv",
					'mu':3e-9, 
					'ploidy': 2, 
					'sample_pop_ids': ['A','B'], 
					'blocklength': 64,
					'k_max': {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2},
					'reference_pop': 'A_B'
					},
		"sim_configs": [{'Ne_A': 1.3e6 , 'Ne_B': 6e5, 'Ne_A_B': 1.5e6, 'T': 1e7, 'me_A_B':7e-7, 'recombination':0}],
		"gf_vars": {'migration_rate':sage.all.var('M'), 'migration_direction':[(1,2)], 'exodus_rate':sage.all.var('E'), 'exodus_direction':[(1,2,0)], 'ancestral_pop': 0}
		},
	'DIV' : {
		"global_info" : {
					'model_file':"models/DIV.tsv",
					'mu':3e-9, 
					'ploidy': 2, 
					'sample_pop_ids': ['A','B'], 
					'blocklength': 64,
					'k_max': {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2},
					'reference_pop': 'A_B'
					},
		"sim_configs": [{'Ne_A': 1.3e6 , 'Ne_B': 6e5, 'Ne_A_B': 1.5e6, 'T': 1e7, 'recombination':0}],
		"gf_vars": {'exodus_rate':sage.all.var('E'), 'exodus_direction':[(1,2,0)], 'ancestral_pop': 0}
		},
	'MIG_BA' : {
		"global_info" : {
					'model_file':"models/MIG_BA.tsv",
					'mu':3e-9, 
					'ploidy': 2, 
					'sample_pop_ids': ['A','B'], 
					'blocklength': 64,
					'k_max': {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2},
					'reference_pop': 'A'
					},
		"sim_configs": [{'Ne_A': 1.3e6 , 'Ne_B': 6e5, 'me_B_A':7e-7, 'recombination':0}],
		"gf_vars": {'migration_rate':sage.all.var('M'), 'migration_direction':[(2,1)], 'ancestral_pop': 1}
		}
	}


def generate_gf(sample_list, branchtype_dict, global_info, sim_configs, gf_vars):
	sim_config = sim_configs[0]
	exodus_rate = sage.all.var('E') if sim_config.get('T') else None
	migration_rate = sage.all.var('M') if sim_config.get('me_A_B') or sim_config.get('me_B_A') else None
	coalescence_rates = (sage.all.var('c0'), sage.all.var('c1'), sage.all.var('c2'))
	gfobj = gflib.GFObject(
		sample_list, 
		coalescence_rates, 
		branchtype_dict, 
		migration_rate=migration_rate, 
		migration_direction=gf_vars.get('migration_direction'), 
		exodus_rate=exodus_rate, 
		exodus_direction=gf_vars.get('exodus_direction')
		)
	return gfobj

def generate_ETPs(gfobj, branchtype_dict, max_k, global_info, sim_configs, gf_vars):
	gf = gfobj.make_gf()
	sim_config = sim_configs[0]
	ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
	engine='symengine'
	theta_symbolic = sage.all.var('theta') if engine == 'sage' else symengine.Symbol('theta')
	theta = get_theta(global_info, sim_config)
	parameter_dict = get_parameter_dict(global_info, sim_config, gf_vars, gfobj.coalescence_rates, engine=engine)
	all_mutation_configurations = list(gflib.return_mutype_configs(max_k))
	root = tuple(0 for _ in max_k)
	mutype_tree = gflib.make_mutype_tree(all_mutation_configurations, root, max_k)
	prob_array = gflib.make_prob_array(gf, mutype_tree, ordered_mutype_list, max_k, theta, gfobj.exodus_rate, chunksize=100, num_processes=1, adjust_marginals=True, parameter_dict=parameter_dict)
	prob_array = prob_array.astype(np.float64)
	return prob_array

def get_parameter_dict(global_info, sim_config, gf_vars, coalescence_rates):
	parameter_dict = {}
	reference_pop = global_info['reference_pop']
	if gf_vars.get('migration_rate'):
		migration_string = 'me_A_B' if gf_vars['migration_direction'] == [(1,2)] else 'me_B_A'
		parameter_dict[gf_vars['migration_rate']] = sage.all.Rational(2 * sim_config[migration_string] * sim_config[f'Ne_{reference_pop}'])
	if gf_vars.get('exodus_rate'):
		parameter_dict[sage.all.var('T')] = sage.all.Rational(sim_config['T']/(2*sim_config[f'Ne_{reference_pop}']))
	for c, Ne in zip(coalescence_rates,('Ne_A_B', 'Ne_A', 'Ne_B')):
		if Ne in sim_config:
			parameter_dict[c] = sage.all.Rational(sim_config[f'Ne_{reference_pop}']/sim_config[Ne])
		else:
			parameter_dict[c] = 0.0
	return parameter_dict

def get_theta(global_info, sim_config):
	reference_pop = global_info['reference_pop']
	Ne_ref = sim_config[f'Ne_{reference_pop}']
	mu=global_info['mu']
	block_length = global_info['blocklength']
	return 2*sage.all.Rational(Ne_ref*mu)*block_length

def compare_ETPs_model(self, config, ETPs_to_test):
	gimbled_ETPs = np.squeeze(np.load(f'tests/ETPs/{config}.npy'))
	print('test_ETPs:', ETPs_to_test[0,0,0,0])
	print('gimble:', gimbled_ETPs[0,0,0,0])
	assert np.all(np.isclose(gimbled_ETPs, ETPs_to_test))

def test_ETPs():
	config='DIV' #DIV, MIG_BA or IM_AB
	sample_list = [(),('a','a'),('b','b')]
	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
	max_k = (2,2,2,2)
	gfobj = generate_gf(sample_list, branchtype_dict, **all_configs[config])
	ETPs_to_test = generate_ETPs(gfobj, branchtype_dict, max_k, **all_configs[config])
	compare_ETPs_model(config, ETPs_to_test)

def test_diff():
	config = 'DIV'
	#config = 'IM_AB'
	sample_list = [(),('a','a'),('b','b')]
	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
	ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)

	max_k = (2,2,2,2)
	gfobj = generate_gf(sample_list, branchtype_dict, **all_configs[config])
	gf = gfobj.make_gf()

	engine = 'sage'
	theta_symbolic = sage.all.var('theta') if engine == 'sage' else symengine.Symbol('theta')
	global_info = all_configs[config]['global_info']
	sim_config = all_configs[config]['sim_configs'][0]
	gf_vars = all_configs[config]['gf_vars']
	theta = get_theta(global_info, sim_config)
	parameter_dict = get_parameter_dict(global_info, sim_config, gf_vars, gfobj.coalescence_rates)
	rate_dict = {branchtype:theta for branchtype in ordered_mutype_list}
	if engine == 'symengine':
		rate_dict = {symengine.sympify(k):v for k,v in rate_dict.items()}
	result = sum(gflib.inverse_laplace(gf, gfobj.exodus_rate)).subs(parameter_dict)
	all_mutation_configurations = list(gflib.return_mutype_configs(max_k))
	root = tuple(0 for _ in max_k)
	mutype_tree = gflib.make_mutype_tree(all_mutation_configurations, root, max_k)	
	start_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	start_time = timer()
	probs = gflib.make_result_dict_from_mutype_tree_stack(result, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k)
	print(timer()-start_time)
	delta_mem = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) - start_mem
	print(f"{delta_mem/1e6} Mb")
	probs = gflib.dict_to_array(probs, (4,4,4,4)).astype(np.float64)
	#checking against:
	probs_check = gflib.make_result_dict_from_mutype_tree(result, mutype_tree, theta, rate_dict, ordered_mutype_list, max_k)
	probs_check = gflib.dict_to_array(probs_check, (4,4,4,4)).astype(np.float64)
	assert np.all(np.isclose(probs_check, probs))

def show_gf():
	config = 'DIV'
	#config = 'IM_AB'
	sample_list = [(),('a','a'),('b','b')]
	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
	ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
	max_k = (2,2,2,2)
	gfobj = generate_gf(sample_list, branchtype_dict, **all_configs[config])
	gf = gfobj.make_gf()
	theta = sage.all.var('theta')
	result = sum(gflib.inverse_laplace(gf, gfobj.exodus_rate))
	#branchtype_dict = {k:sage.all.Rational(21763905/82088362) for k in ordered_mutype_list}
	branchtype_dict = {k:theta for k in ordered_mutype_list}
	rate_dict = {#theta: sage.all.Rational(21763905/82088362), 
		sage.all.var('c1'): sage.all.Rational(1295774084316924/8165234716690477),
	 	sage.all.var('c2'): sage.all.Rational(16287572079/61401460750), sage.all.var('c0'): sage.all.Rational(1), 
	 	sage.all.var('M'): sage.all.Rational(9263575/173243749),
	 	sage.all.var('T'): sage.all.Rational(309756819/163585742)}
	#rate_dict = {**rate_dict, **branchtype_dict}
	sub_result = result.subs(rate_dict)
	sympy_ordered_mutype_list = [sympy.sympify(x) for x in ordered_mutype_list]
	print(sympy_ordered_mutype_list)
	cse_breakdown = sympy.cse(sympy.sympify(sub_result), ignore=sympy_ordered_mutype_list)
	print(cse_breakdown)
	sys.exit()
	mucount = [0,2,0,3]
	mucount_fact_prod = np.prod([np.math.factorial(count) for count in mucount])  
	mucount_total = np.sum(mucount)
	partials = list(gflib.flatten(itertools.repeat(branchtype,count) for branchtype, count in zip(ordered_mutype_list, mucount) if count>0))
	marginals = {branchtype:0 for branchtype, count in zip(ordered_mutype_list, mucount) if count==0}
	deriv = gflib.simple_probK(result, theta, partials, marginals, rate_dict, mucount_total, mucount_fact_prod)
	#deriv_callable = sympy.lambdify(sympy.Symbol('theta'), sympy.sympify(deriv))
	#deriv_callable(0.26512777779632146)
	#print(deriv)
	#print(sage.all.RealField(165)(deriv))
	#print(np.float64(deriv))

def main():
	test_diff()
	#show_gf()
if __name__ == "__main__":
	main()