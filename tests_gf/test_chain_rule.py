import numpy as np
import pytest
import sage.all
import sys, os

import gf.gf as gflib
import gf.mutations as mutations
import gf.chain_rule as cr

@pytest.mark.chain_rule
class Test_Paths:

	def test_paths_pre_laplace(self, return_gf):
		gf_simple_list, (paths, eq_list) = return_gf	
		self.paths_pre_laplace(paths, eq_list, sum(gf_simple_list))
	
	def paths_pre_laplace(self, paths, eq_list, gf_comparison):
		print('________pre_laplace______')
		print(eq_list)
		print(paths)
		gf_chain = self.gf_from_paths(paths, eq_list)
		comparison = sage.all.simplify(gf_comparison-gf_chain)
		assert comparison==0

	def test_paths_post_laplace(self, return_gf):
		gf_simple_list, (paths, eq_list) = return_gf
		dummy_var =  sage.all.SR.var('E')	
		self.paths_post_laplace(paths, eq_list, dummy_var, gf_simple_list)

	def paths_post_laplace(self, paths, eq_list, dummy_var, gf_comparison):
		print('________post_laplace______')
		eq_list = np.array(eq_list)
		new_eq_list, new_paths = cr.all_paths_after_inversion(paths, eq_list, dummy_var)
		print('new_paths:', new_paths)
		print(new_eq_list)
		eval_eq_list = self.evaluate_eq_list(new_eq_list)
		gf = float(self.gf_from_paths(new_paths, eval_eq_list))
		print(gf)
		inverted_gf_comparison = gflib.inverse_laplace(gf_comparison, dummy_var)
		eval_gf_comparison = float(sum(self.evaluate_eq_list(inverted_gf_comparison))) #check for precision errors
		print(eval_gf_comparison)
		assert np.isclose(eval_gf_comparison, gf)

	def evaluate_eq_list(self, eq_list):
		theta = sage.all.Rational(0.7)
		param_dict = {
			sage.all.SR.var('T'):sage.all.Rational(0.5),
			sage.all.SR.var('c0'):sage.all.Rational(1.0),
			sage.all.SR.var('c1'):sage.all.Rational(2.0),
			sage.all.SR.var('c2'):sage.all.Rational(3.0),
			sage.all.SR.var('M'):sage.all.Rational(0.2),
			sage.all.SR.var('m_1'):theta,
			sage.all.SR.var('m_2'):theta,
			sage.all.SR.var('m_3'):theta,
			sage.all.SR.var('m_4'):theta,
		}
		eval_eq_list = np.array([sage.all.RealField(165)(eq.subs(param_dict)) for eq in eq_list])
		return eval_eq_list

	def gf_from_paths(self, paths, eq_list):
		eq_list = np.array(eq_list, dtype=object)
		result = sum(np.prod(eq_list[idxs]) for idxs in paths)
		return result

	def ln_gf_from_paths(self, paths, eq_list):
		eq_list = np.array(eq_list, dtype=np.float64)
		print('eq_list:', eq_list)
		eq_ln = np.log(eq_list, where=eq_list!=0)
		result = np.zeros(len(eq_ln) ,dtype=np.float64)
		for i, idxs in enumerate(paths):
			result[i] = np.sum(eq_ln[idxs])
		return np.sum(np.exp(result))

	@pytest.fixture(
		scope='class', 
		params=[
			([(1,2,0)], sage.all.SR.var('E'), None, None),
			(None, None, [(2,1)], sage.all.SR.var('M')),
			([(1,2,0)], sage.all.SR.var('E'), [(2,1)], sage.all.SR.var('M'))
			],
		ids=[
			'DIV',
			'MIG', 
			'IM'
			],
		)
	def return_gf(self, request):
		sample_list = [(),('a','a'),('b','b')]
		ancestral_pop = 0
		coalescence_rates = (sage.all.SR.var('c0'), sage.all.SR.var('c1'), sage.all.SR.var('c2'))
		k_max = {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted', labels=mutype_labels)
		
		exodus_direction, exodus_rate, migration_direction, migration_rate = request.param
		gfobj = gflib.GFObject(
			sample_list, 
			coalescence_rates, 
			branchtype_dict,
			exodus_rate=exodus_rate,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate,
			migration_direction=migration_direction
			)
		gf_simple = list(gfobj.make_gf())
		
		gfobj2 = cr.GFObjectChainRule(
			sample_list, 
			coalescence_rates, 
			branchtype_dict,
			exodus_rate=exodus_rate,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate,
			migration_direction=migration_direction
			)
		paths, eq_list = gfobj2.make_gf()
		gf_chain = (paths, eq_list)

		return (gf_simple, gf_chain)

@pytest.mark.expand
class TestExpand:
	def test_invert_eq(self):
		c_p = [np.array([3, 4, 5]), np.array([3]), np.array([]), np.array([3,5]), np.array([4,5])]
		to_invert = [[], [0, 1], [0, 1, 2], [0,1, 2], []]
		sage.all.var(['E', 'c0', 'c1', 'c2', 'm_1', 'm_2', 'm_3', 'm_4', 'M'])
		equations = np.array([
			c1/(E + c1 + c2 + 2*m_1 + 2*m_2), 
    		c2/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		E/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		2*c0/(3*c0 + m_1 + m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		])
		inverted, expand_inverted_paths = cr.invert_eq(to_invert, equations, E, 3, expand_inverted_eq=True)
		print('inv:', inverted)
		print('expand_inverted_paths:', expand_inverted_paths)
		assert expand_inverted_paths == [[], [3, 4, 5, 6, 7, 8], [9], [9], []]
		assert len(expand_inverted_paths) == len(c_p)
		constant_unique, constant_paths_remapped = cr.remap_paths_unique(c_p)
		new_paths = list(cr.expand_paths_associatively(constant_paths_remapped, expand_inverted_paths))
		print('new_paths:', new_paths)
		expected = [np.array([0, 1, 2]), np.array([0, 3]), np.array([0, 4]), np.array([0, 5]), np.array([0, 6]), np.array([0, 7]), np.array([0, 8]), np.array([9]), np.array([0, 2, 9]), np.array([1, 2])]
		assert np.all(np.array_equal(a1,a2) for a1, a2 in zip(new_paths, expected))

	def test_invert_constant(self):
		c_p = [np.array([3, 4, 5]), np.array([3]), np.array([4]), np.array([3,5]), np.array([4,5])]
		to_invert = [[], [], [], [], []]
		sage.all.var(['E', 'c0', 'c1', 'c2', 'm_1', 'm_2', 'm_3', 'm_4', 'M'])
		equations = np.array([
			c1/(E + c1 + c2 + 2*m_1 + 2*m_2), 
    		c2/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		E/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		2*c0/(3*c0 + m_1 + m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		])
		inverted, expand_inverted_paths = cr.invert_eq(to_invert, equations, E, 3, expand_inverted_eq=True)
		print('inv:', inverted)
		print('expand_inverted_paths:', expand_inverted_paths)
		constant_unique, constant_paths_remapped = cr.remap_paths_unique(c_p)
		new_paths = list(cr.expand_paths_associatively(constant_paths_remapped, expand_inverted_paths))
		print('new_paths:', new_paths)
		expected = [np.array([0, 1, 2]), np.array([0]), np.array([1]), np.array([0, 2]), np.array([1, 2])]
		assert len(new_paths) == len(c_p)
		assert np.all(np.array_equal(a1,a2) for a1, a2 in zip(new_paths, expected))

	def test_invert_eq_simple(self):
		c_p = [np.array([3, 4, 5]), np.array([3]), np.array([]), np.array([3,5]), np.array([4,5])]
		to_invert = [[], [0, 1], [0, 1, 2], [0,1, 2], []]
		sage.all.var(['E', 'c0', 'c1', 'c2', 'm_1', 'm_2', 'm_3', 'm_4', 'M'])
		equations = np.array([
			c1/(E + c1 + c2 + 2*m_1 + 2*m_2), 
    		c2/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		E/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		2*c0/(3*c0 + m_1 + m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		])
		inverted, expand_inverted_paths = cr.invert_eq(to_invert, equations, E, 3, expand_inverted_eq=False)
		print('inv:')
		for e in inverted:
			print(e)
		print('expand_inverted_paths:', expand_inverted_paths)
		assert expand_inverted_paths == [[], [3], [4], [4], []]
		assert len(expand_inverted_paths) == len(c_p)
		constant_unique, constant_paths_remapped = cr.remap_paths_unique(c_p)
		new_paths = list(cr.expand_paths_associatively(constant_paths_remapped, expand_inverted_paths))
		print('new_paths:', new_paths)
		expected = [np.array([0, 1, 2]), np.array([0, 3]), np.array([4]), np.array([0, 2, 4]), np.array([1, 2])]
		assert np.all(np.array_equal(a1,a2) for a1, a2 in zip(new_paths, expected))

	def test_invert_constant_simple(self):
		c_p = [np.array([3, 4, 5]), np.array([3]), np.array([4]), np.array([3,5]), np.array([4,5])]
		to_invert = [[], [], [], [], []]
		sage.all.var(['E', 'c0', 'c1', 'c2', 'm_1', 'm_2', 'm_3', 'm_4', 'M'])
		equations = np.array([
			c1/(E + c1 + c2 + 2*m_1 + 2*m_2), 
    		c2/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		E/(E + c1 + c2 + 2*m_1 + 2*m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		2*c0/(3*c0 + m_1 + m_2),
    		1/2*c0/(3*c0 + m_1 + m_2),
    		])
		inverted, expand_inverted_paths = cr.invert_eq(to_invert, equations, E, 3, expand_inverted_eq=False)
		print('inv:', inverted)
		print('expand_inverted_paths:', expand_inverted_paths)
		constant_unique, constant_paths_remapped = cr.remap_paths_unique(c_p)
		new_paths = list(cr.expand_paths_associatively(constant_paths_remapped, expand_inverted_paths))
		print('new_paths:', new_paths)
		expected = [np.array([0, 1, 2]), np.array([0]), np.array([1]), np.array([0, 2]), np.array([1, 2])]
		assert len(new_paths) == len(c_p)
		assert np.all(np.array_equal(a1,a2) for a1, a2 in zip(new_paths, expected))