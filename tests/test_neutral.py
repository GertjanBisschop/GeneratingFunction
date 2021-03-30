import pytest
import gf.gf as gflib
import gf.togimble as tg
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
		"gf_vars": {'sample_list' :[(),('a','a'),('b','b')], 'migration_rate':sage.all.var('M'), 'migration_direction':[(1,2)], 'exodus_rate':sage.all.var('E'), 'exodus_direction':[(1,2,0)], 'ancestral_pop': 0}
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
		"gf_vars": {'sample_list' :[(),('a','a'),('b','b')], 'exodus_rate':sage.all.var('E'), 'exodus_direction':[(1,2,0)], 'ancestral_pop': 0}
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
		"gf_vars": {'sample_list' :[(),('a','a'),('b','b')], 'migration_rate':sage.all.var('M'), 'migration_direction':[(2,1)], 'ancestral_pop': 1}
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

@pytest.mark.topology
class Test_divide_into_equivalence_classes:
	def no_test_basic(self):
		pass

@pytest.mark.gimble
class Test_to_gimble:
	@pytest.mark.parametrize('model', [('IM_AB'), ('DIV'), ('MIG_BA')])
	def test_ETPs(self, model):
		ETPs = self.calculate_ETPs(model)
		self.compare_ETPs_model(model, ETPs)

	def calculate_ETPs(self, model):
		config = all_configs[model]
		config['sim_config'] = config['sim_configs'][0]
		del config['sim_configs']
		joined_config = {**config['sim_config'], **config['gf_vars'], **config['global_info']}
		gf = tg.get_gf(joined_config)
		max_k = (2,2,2,2)
		mutype_labels = config['global_info']['k_max'].keys()
		mutypes = [sage.all.var(mutype) for mutype in mutype_labels]
		gfEvaluatorObj = tg.gfEvaluator(gf, max_k, mutypes)
		coalescence_rates = (sage.all.var(c) for c in ['c0', 'c1', 'c2'])
		parameter_dict = af.get_parameter_dict(coalescence_rates, **config)
		theta = af.get_theta(**config)
		return gfEvaluatorObj.evaluate_gf(parameter_dict, theta)

	def compare_ETPs_model(self, model, ETPs):
		gimbled_ETPs = np.squeeze(np.load(f'tests/ETPs/{model}.npy'))
		assert np.all(np.isclose(gimbled_ETPs, ETPs))