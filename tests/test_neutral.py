import pytest
import gf.gf as gflib
from timeit import default_timer as timer
import sage.all
import numpy as np
import tests.aux_functions as af
import sys, os

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

@pytest.mark.aux
class Test_aux:
	@pytest.mark.parametrize("input_lineages, to_join, expected",
		[(('a', 'b', 'c', 'a'), ('a', 'b'),('a', 'ab', 'c')),
		(('a', 'b', 'b'), ('b', 'b'), ('a', 'bb')),
		(('ab', 'ab'), ('ab', 'ab'), ('aabb',))])
	def test_coalesce_lineages(self, input_lineages, to_join, expected):
		output_lineages = gflib.coalesce_lineages(input_lineages, to_join)
		assert sorted(output_lineages) == sorted(expected)

	@pytest.mark.parametrize("sample_list, check",
		[(([(), ('a','b'),('c','d')]),([(1, ((), ('ab',), ('c', 'd'))), (1, ((), ('a', 'b'), ('cd',)))])),
		(([('a', 'b', 'b'),(), ('a',)]),([(2, (('ab', 'b'), (), ('a',))), (1, (('a', 'bb'), (), ('a',)))]))])
	def test_coalescence(self, sample_list, check):
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, exodus_rate=1, exodus_direction=[(1,2,0)])
		result = list(gfobj.coalescence_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	@pytest.mark.parametrize("sample_list, check",
	[(([(), ('a','b'),('c','d')]),([(sage.all.var('c1'), ((), ('ab',), ('c', 'd'))), (sage.all.var('c2'), ((), ('a', 'b'), ('cd',)))])),
	(([('a', 'b', 'b'),(), ('a',)]),([(2*sage.all.var('c0'), (('ab', 'b'), (), ('a',))), (1*sage.all.var('c0'), (('a', 'bb'), (), ('a',)))]))])
	def test_coalescence_rates(self, sample_list, check):
		coalescence_rates = (sage.all.var('c0'), sage.all.var('c1'), sage.all.var('c2'))
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, coalescence_rates, branchtype_dict, exodus_rate=1, exodus_direction=[(1,2,0)])
		result = list(gfobj.coalescence_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	@pytest.mark.parametrize("sample_list, check",
		[(([(), ('a','b'),('c','d')]),([(1, ((), ('b',), ('a','c', 'd'))), (1, ((), ('a',), ('b', 'c', 'd')))])),
		(([(), ('a','a'),('b','b')]),([(2, ((), ('a',), ('a', 'b', 'b')))])),
		(([(), ('a','a', 'c', 'c'),('b','b')]),([(2, ((), ('a','c', 'c'), ('a', 'b', 'b'))),(2, ((), ('a','a','c'), ('b', 'b', 'c')))]))]
		)
	def test_migration(self, sample_list, check):
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=1, migration_direction=[(1,2)])
		result = list(gfobj.migration_events(gfobj.sample_list))
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

	def test_migration_empty(self):
		sample_list = [(), (),('c','d')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=1, migration_direction=[(1,2)])
		result = list(gfobj.migration_events(gfobj.sample_list))
		print('result:', result)
		assert isinstance(result, list)
		assert len(result)==0
		
	def test_exodus_empty(self):
		sample_list = [(), (),('c','d')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=1, migration_direction=[(1,2)])
		result = list(gfobj.exodus_events(gfobj.sample_list))
		print('result:', result)
		assert isinstance(result, list)
		assert len(result)==0

	def test_exodus(self):
		sample_list = [(), ('a','a'),('c','d')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		exodus_rate = sage.all.var('E')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, exodus_rate=exodus_rate, exodus_direction=[(1,2,0)])
		result = list(gfobj.exodus_events(gfobj.sample_list))
		check = [(exodus_rate, (('a', 'a', 'c', 'd'), (), ()))]
		print('result:', result)
		print('check:', check)
		for test, truth in zip(result, check):
			assert all(test[i]==truth[i] for i in range(len(test)))

@pytest.mark.gf
class Test_gf:
	def test_gf_unrooted(self):
		sample_list = [('a','a','b','b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
		gfobj = gflib.GFObject(sample_list, (1,), branchtype_dict)
		result = list(gfobj.make_gf())
		subs_dict = {k:0 for k in set(branchtype_dict.values())}
		assert sum([x.substitute(subs_dict) for x in result])==1

@pytest.mark.simple_models
class Test_gf_simple:
	def test_generate_ETPs(self):
		sample_list = [('a', 'b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
		theta = 0.5 #this is actually theta/2
		coalescence_rates = (1,)
		gfobj = gflib.GFObject(
			sample_list, 
			coalescence_rates, 
			branchtype_dict
			)
		gf=gfobj.make_gf()
		ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
		kmax_by_mutype=(2,2)
		symbolic_prob_array =  gflib.make_prob_array(gf, ordered_mutype_list, kmax_by_mutype, theta, dummy_variable=None, chunksize=100, num_processes=1, adjust_marginals=False)
		result = gflib.evaluate_ar(symbolic_prob_array, {})
		check = np.array([[0.5, 0.125,0.03125,0.66666667],
 				[0.125, 0.0625, 0.0234375, 0.22222222],
 				[0.03125, 0.0234375, 0.01171875, 0.07407407],
 				[0.66666667, 0.22222222, 0.07407407, 1]])
		gflib.adjust_marginals_array(result, len(ordered_mutype_list))
		assert np.allclose(result, check)
		
	#def adjust_marginals(self, result, num_mutypes):
	#	print(result)
	#	final_result = gflib.adjust_marginals(result, num_mutypes)
	#	assert np.sum(final_result) == 1
	#	marginals = np.sum(result[:,:-1],axis=1)+final_result[:,-1]
	#	marginals[-1] = 1
	#	assert np.allclose(marginals, result[:,-1])
		
	def test_probK(self):
		ordered_mutype_list = [sage.all.var('z_a'), sage.all.var('z_b')]
		theta = sage.all.var('theta')/2
		gf = 1/(sum(ordered_mutype_list)+1)
		partials = ordered_mutype_list[:]
		marginals = {}
		ratedict = {mutype:theta for mutype in ordered_mutype_list}
		mucount_total = 2
		mucount_fact_prod = 1 
		probk = gflib.simple_probK(gf, theta, partials, marginals, ratedict, mucount_total, mucount_fact_prod)
		#test probability of seeing 1 mutation on each branch
		assert probk == 1/2*(2*theta)**2/(2*theta+1)**3

@pytest.mark.gimble
class Test_gf_against_gimble:
	def generate_gf(self, sample_list, branchtype_dict, global_info, sim_configs, gf_vars):
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

	def generate_ETPs(self, gfobj, branchtype_dict, max_k, global_info, sim_configs, gf_vars):
		gf = gfobj.make_gf()
		sim_config = sim_configs[0]
		ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
		theta_symbolic = sage.all.var('theta')
		theta = self.get_theta(global_info, sim_config)
		parameter_dict = self.get_parameter_dict(global_info, sim_config, gf_vars, gfobj.coalescence_rates)
		all_mutation_configurations = list(gflib.return_mutype_configs(max_k))
		root = tuple(0 for _ in max_k)
		mutype_tree = gflib.make_mutype_tree(all_mutation_configurations, root, max_k)
		prob_array = gflib.make_prob_array(gf, mutype_tree, ordered_mutype_list, max_k, theta, gfobj.exodus_rate, chunksize=100, num_processes=1, adjust_marginals=True, parameter_dict=parameter_dict)
		prob_array = prob_array.astype(np.float64)
		return prob_array

	def get_parameter_dict(self, global_info, sim_config, gf_vars, coalescence_rates):
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

	def get_theta(self, global_info, sim_config):
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

	#@pytest.mark.parametrize('config', [('IM_AB'),('DIV'), ('MIG_BA')])
	def test_ETPs(self):
		config='DIV' #DIV, MIG_BA or IM_AB
		sample_list = [(),('a','a'),('b','b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
		max_k = (2,2,2,2)
		test = Test_gf_against_gimble()
		gfobj = test.generate_gf(sample_list, branchtype_dict, **all_configs[config])
		ETPs_to_test = test.generate_ETPs(gfobj, branchtype_dict, max_k, **all_configs[config])
		test.compare_ETPs_model(config, ETPs_to_test)

@pytest.mark.topology
class Test_divide_into_equivalence_classes:
	def test_basic(self):
		
		assert False