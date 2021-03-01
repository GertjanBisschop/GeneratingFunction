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
	#def test_gf(self):
	#	sample_list = [('a','b','c')]
	#	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='label')
	#	gfobj = gflib.GFObject(sample_list, (1,), branchtype_dict)
	#	result = list(gfobj.make_gf())
	#	subs_dict = {k:0 for k in set(branchtype_dict.values())}
	#	assert all(prob==1/3 for prob in [x.substitute(subs_dict) for x in result])

	def test_gf_unrooted(self):
		sample_list = [('a','a','b','b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
		gfobj = gflib.GFObject(sample_list, (1,), branchtype_dict)
		result = list(gfobj.make_gf())
		subs_dict = {k:0 for k in set(branchtype_dict.values())}
		assert sum([x.substitute(subs_dict) for x in result])==1

	def test_gf_exodus(self):
		sample_list = [(),('a','a'),('b','b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
		exodus_rate=sage.all.var('E')
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, exodus_rate=exodus_rate, exodus_direction=[(1,2,0)])
		gf = list(gfobj.make_gf())
		#subs_dict = {k:0 for k in set(branchtype_dict.values())}
		#assert sum([x.substitute(subs_dict) for x in result])==1
		inverse = sum(gflib.inverse_laplace(gf, exodus_rate))
		ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
		kmax_by_mutype = (2,2,2,2)
		#print(ordered_mutype_list)
		symbolic_prob_dict = gflib.make_symbolic_prob_dict(inverse, ordered_mutype_list, kmax_by_mutype, 0.5)
		parameter_dict = {sage.all.var('T'):1.0}
		substituted_values = gflib.substitute_parameters(symbolic_prob_dict, parameter_dict, ordered_mutype_list, kmax_by_mutype)
		#print(substituted_values)
		result_array = gflib.dict_to_array(substituted_values, tuple(k+2 for k in kmax_by_mutype))
		#print(result_array)
		final_result = gflib.adjust_marginals(result_array, len(ordered_mutype_list))
		print(np.sum(final_result))
		assert False

	def test_divide_by_zero_bug(self):
		sample_list = [(),('a','a'),('b','b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
		exodus_rate = sage.all.var('E')
		migration_rate = sage.all.var('M')
		coalescence_rates = (sage.all.var('c0'), sage.all.var('c1'), sage.all.var('c2'))
		gfobj = gflib.GFObject(sample_list, (1,1,1), branchtype_dict, migration_rate=migration_rate, migration_direction=[(1,2)], exodus_rate=exodus_rate, exodus_direction=[(1,2,0)])
		gf = gfobj.make_gf()
		inverse = sum(gflib.inverse_laplace(gf, exodus_rate))
		ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
		kmax_by_mutype = (2,2,2,2)
		#print(ordered_mutype_list)
		symbolic_prob_dict = gflib.make_symbolic_prob_dict(inverse, ordered_mutype_list, kmax_by_mutype, 1.0)
		coal_rates_dict = {param:rate for param, rate in zip(coalescence_rates, (3.0,1.0,3.0))}
		parameter_dict = {sage.all.var('M'):2.0, sage.all.var('T'):1.0}
		parameter_dict = {**parameter_dict, **coal_rates_dict}
		substituted_values = gflib.substitute_parameters(symbolic_prob_dict, parameter_dict, ordered_mutype_list, kmax_by_mutype)
		#print(substituted_values)
		result_array = gflib.dict_to_array(substituted_values, tuple(k+2 for k in kmax_by_mutype))
		#print(result_array)
		final_result = gflib.adjust_marginals(result_array, len(ordered_mutype_list))
		print(np.sum(final_result))
		assert False

	#def test_gf_all_events(self):
	#	sample_list = [(),('a','a'),('b','b')]
	#	branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
	#	exodus_rate = sage.all.var('E')
	#	migration_rate = sage.all.var('M')
	#	gfobj = gflib.GFObject(
	#		sample_list, 
	#		(1,1,1), 
	#		branchtype_dict, 
	#		exodus_rate=exodus_rate, 
	#		exodus_direction=[(1,2,0)],
	#		migration_rate=migration_rate,
	#		migration_direction=[(1,2)]
	#		)
	#	result = list(gfobj.make_gf())
	#	subs_dict = {k:0 for k in set(branchtype_dict.values())}
	#	assert sum([x.substitute(subs_dict) for x in result])==1
	#	inverse = gflib.inverse_laplace(result, exodus_rate)
	#	branchtype_mucount_dict={k:0 for k in set(branchtype_dict.values())}
	#	derivative = gflib.probK(sum(inverse), branchtype_mucount_dict, 0.5)
	#	print(derivative)
	#	assert False

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
		gf=sum(gfobj.make_gf())
		ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
		kmax_by_mutype=(2,2)
		symbolic_prob_dict = gflib.make_symbolic_prob_dict(gf, ordered_mutype_list, kmax_by_mutype, theta)
		result = gflib.substitute_parameters(symbolic_prob_dict, {}, ordered_mutype_list, kmax_by_mutype)
		result_array = gflib.dict_to_array(result, tuple(k+2 for k in kmax_by_mutype))
		check = np.array([[0.5, 0.125,0.03125,0.66666667],
 				[0.125, 0.0625, 0.0234375, 0.22222222],
 				[0.03125, 0.0234375, 0.01171875, 0.07407407],
 				[0.66666667, 0.22222222, 0.07407407, 1]])
		self.adjust_marginals(result_array, len(ordered_mutype_list))
		assert np.allclose(result_array, check)
		
	def adjust_marginals(self, result, num_mutypes):
		print(result)
		final_result = gflib.adjust_marginals(result, num_mutypes)
		#check = np.array([[0.5, 0.125, 0.03125, 0.01041667],
 		#		[0.125, 0.0625, 0.0234375, 0.01128472],
 		#		[0.03125, 0.0234375, 0.01171875, 0.00766782],
 		#		[0.01041667, 0.01128472, 0.00766782, 0.00766782]])
		assert np.sum(final_result) == 1
		marginals = np.sum(result[:,:-1],axis=1)+final_result[:,-1]
		marginals[-1] = 1
		assert np.allclose(marginals, result[:,-1])
		
	def test_probK(self):
		ordered_mutype_list = [sage.all.var('z_a'), sage.all.var('z_b')]
		theta = sage.all.var('theta')/2
		gf = 1/(sum(ordered_mutype_list)+1)
		probk = gflib.probK(gf, {k:1 for k in ordered_mutype_list}, theta)
		assert probk == 1/2*(2*theta)**2/(2*theta+1)**3

@pytest.mark.graph
class Test_debug:
	def test_gf_graph(self):
		sample_list = [(), ('a', 'a'), ('b', 'b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
		coalescence_rates = (sage.all.var('c0'),sage.all.var('c1'),sage.all.var('c2'))
		gfobj = gflib.GFObject(
			sample_list, 
			coalescence_rates, 
			branchtype_dict,
			migration_direction = [(1,2)],
			migration_rate=sage.all.var('M'),
			exodus_direction=[(1,2,0)],
			exodus_rate=sage.all.var('E')
			)
		gf=list(gfobj.make_gf_graph())
		for g in gf:
			print(g)
		print(len(gf))
		gflib.make_graph(gf, 'IM_AB')
		assert False

@pytest.mark.gimble
class Test_gf_against_gimble:
	def generate_ETPs(self, global_info, sim_configs, gf_vars):
		sim_config = sim_configs[0]
		sample_list = [(),('a','a'),('b','b')]
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted')
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
		start_time = timer()
		gf = gfobj.make_gf()
		if exodus_rate != None:
			inverse = sum(gflib.inverse_laplace(gf, exodus_rate))
		else:
			inverse=sum(gf)
		int_time=timer()
		print(f'gf:{int_time-start_time}')
		ordered_mutype_list = gflib.sort_mutation_types(branchtype_dict)
		kmax_by_mutype = (2,2,2,2)
		theta_symbolic = sage.all.var('theta')
		theta = self.get_theta(global_info, sim_config)
		parameter_dict = self.get_parameter_dict(global_info, sim_config, gf_vars, coalescence_rates)
		parameter_dict[theta_symbolic] = theta
		start_time=timer()
		symbolic_prob_dict = gflib.make_symbolic_prob_dict(inverse, ordered_mutype_list, kmax_by_mutype, theta_symbolic)
		print(f'symbolic prob dict:{timer()-start_time}')
		#make param_dict
		start_time = timer()
		substituted_values = gflib.substitute_parameters(symbolic_prob_dict, parameter_dict, ordered_mutype_list, kmax_by_mutype)
		#print(substituted_values)
		print(f'subst values:{timer()-start_time}')
		start_time = timer()
		result_array = gflib.dict_to_array(substituted_values, tuple(k+2 for k in kmax_by_mutype))
		print(sage.all.RealField(165)(result_array[0,0,0,0]))
		final_result = gflib.adjust_marginals(result_array, len(ordered_mutype_list))
		print('test:',float(sage.all.RealField(165)(final_result[0,0,0,0])))
		print(f'remaining times: {timer()-start_time}')
		return final_result

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

	def compare_ETPs_model(self, config):
		gimbled_ETPs = np.squeeze(np.load(f'tests/ETPs/{config}.npy'))
		ETPs_to_test = self.generate_ETPs(**all_configs[config])
		print('gimble:', gimbled_ETPs[0,0,0,0])
		assert np.all(ETPs_to_test>=0)
		sys.exit()
		af.scatter_loglog(gimbled_ETPs, ETPs_to_test, config)
		assert np.array_equal(gimbled_ETPs, ETPs_to_test)

	#@pytest.mark.parametrize('config', [('IM_AB'),('DIV'), ('MIG_BA')])
	def test_ETPs(self):
		#self.compare_ETPs_model(config)
		#self.compare_ETPs_model('DIV')
		#self.compare_ETPs_model('MIG_BA')
		self.compare_ETPs_model('IM_AB')