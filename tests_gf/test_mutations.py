import numpy as np
import pytest
import sys, os

import gf.mutations as mutations
import gf.togimble as togimble

@pytest.mark.muts
class Test_mutypetree:
	def test_incomplete_tree(self):
		all_mutypes = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),(0,0,2,2),(0,0,3,2),
			(2,1,3,0),(2,1,2,3)
			]
		)
		root = (0,0,0,0)
		max_k = (2,2,2,2)
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		result = {
				(0,0,0,0): [(0,0,2,2),(1,0,0,0)],
				(0,0,0,2): [(0,0,3,2),],
				(1,0,0,0): [(2,1,0,0),], 
				(2,1,0,0): [(2,1,2,0), (2,1,3,0)],
				(2,1,2,0): [(2,1,2,3)]
				}
		assert result==mutype_tree

	def test_complete_tree(self):
		root = (0,0,0,0)
		max_k = (2,2,2,2)
		all_mutypes = sorted(mutations.return_mutype_configs(max_k))
		mutype_tree = mutations.make_mutype_tree(all_mutypes, root, max_k)
		mutype_tree_single_digit = mutations.make_mutype_tree_single_digit(all_mutypes, root, max_k)
		assert mutype_tree == mutype_tree_single_digit

	def test_muts_to_gimble(self):
		gf = 1
		max_k = (2,2,2,2)
		root = (0,0,0,0)
		exclude = [(2,3),]
		restrict_to = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),(0,0,2,2),(0,0,3,2),
			(2,1,3,0),(2,1,2,3)
			]
		)
		mutypes = tuple(f'm_{idx}' for idx in range(len(max_k)))
		gfEvaluatorObj = togimble.gfEvaluator(gf, max_k, mutypes, exclude=exclude, restrict_to=restrict_to)
		expected_all_mutation_configurations = sorted([
			(0,0,0,0), (1,0,0,0), (2,1,0,0), (2,1,2,0),
			(2,1,3,0),(2,1,0,0),(2,1,1,0),(2,1,2,0)
			])

		print(gfEvaluatorObj.mutype_tree)
		#types to add:
		#(0,0,0,2),(0,0,1,2),(0,0,2,2),(2,1,2,0),(2,1,2,1),(2,1,2,2),(2,1,0,0),(2,1,1,0),(2,1,2,0)
		expected_result = {
				(0,0,0,0): [(0,0,0,2),(1,0,0,0)],
				(1,0,0,0): [(2,1,0,0),], 
				(2,1,0,0): [(2,1,1,0), (2,1,3,0)],
				(2,1,1,0): [(2,1,2,0),]
				}
		assert gfEvaluatorObj.mutype_tree == expected_result

	def test_list_marginal_idxs(self):
		marginals = np.array([(0,0,3,1),(3,0,1,3)],dtype=np.uint8)
		max_k = (2,2,2,2)
		result = [mutations.list_marginal_idxs(m, max_k) for m in marginals]
		expected_result = [
			[(0,0,0,1), (0,0,1,1),(0,0,2,1)],
			[(0,0,1,0),(0,0,1,1),(0,0,1,2),(1,0,1,0),(1,0,1,1),(1,0,1,2),(2,0,1,0),(2,0,1,1),(2,0,1,2)]
			]
		assert result==expected_result

	def test_add_marginals_restrict_to(self):
		restrict_to = [(0,0,3,1),(3,0,1,3), (0,0,1,0), (0,0,1,1),(1,0,0,0)]
		max_k = (2,2,2,2)
		result = mutations.add_marginals_restrict_to(restrict_to, max_k)
		expected_result = sorted([(0,0,0,1),(0,0,1,1),(0,0,2,1),(0,0,3,1),
			(0,0,1,0),(0,0,1,2),(1,0,1,0),(1,0,1,1),(1,0,1,2),
			(2,0,1,0),(2,0,1,1),(2,0,1,2),(3,0,1,3),(1,0,0,0)])
		assert expected_result == result

@pytest.mark.muts_events
class Test_Mutations_Events:
	@pytest.mark.parametrize("mut_config, exp_array, exp_coefficient",
		[(np.array([0,0,0,0], dtype=int), np.array([0, 0, 0, 0, 0], dtype=int), 1),
		(np.array([0,0,0,2], dtype=int), np.array([0, 0, 0, 0, 0], dtype=int), 1),
		(np.array([0,0,0,1], dtype=int), None, None),
		])
	def test_mutations_as_events1(self, mut_config, exp_array, exp_coefficient):
		max_k_test = np.array([1,1,1,1], dtype=int)
		paths = np.array([
			[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[1,1,1,0],
			[1,0,1,0]
			])
		observed = mutations.expand_single_path(paths, mut_config, max_k_test)
		if exp_coefficient!=None:
			for o in observed:
				assert np.array_equal(o[0], exp_array)
				assert o[1]==exp_coefficient
		else:
			assert len(list(observed))==0

	def test_mutations_as_events2(self):
		mut_config = np.array([1,2,0,0], dtype=int)
		max_k_test = np.array([2,2,2,2], dtype=int)
		paths = np.array([
			[1,0,0,0],
			[0,1,0,0],
			[0,0,1,0],
			[1,1,1,0],
			[1,0,1,0]
			], dtype=int)
		observed = mutations.expand_single_path(paths, mut_config, max_k_test)
		expected = [
				(np.array([0, 0, 0, 2, 1], dtype=int), 1),
				(np.array([0, 1, 0, 1, 1], dtype=int), 1),
				(np.array([0, 2, 0, 0, 1], dtype=int), 1),
				(np.array([0, 0, 0, 3, 0], dtype=int), 3),
				(np.array([0, 1, 0, 2, 0], dtype=int), 2),
				(np.array([0, 2, 0, 1, 0], dtype=int), 1),
				(np.array([1, 0, 0, 2, 0], dtype=int), 1),
				(np.array([1, 1, 0, 1, 0], dtype=int), 1),
				(np.array([1, 2, 0, 0, 0], dtype=int), 1)
			]
		for o, e in zip(observed, expected):
			assert np.array_equal(o[0], e[0])
			assert o[1]==e[1]
