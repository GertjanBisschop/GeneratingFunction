import numpy as np
import pytest
import sage.all
import sys, os

import gf.gf as gflib
import gf.mutations as mutations
import gf.chain_rule as cr

@pytest.mark.chain_rule
class Test_DIV:
	@pytest.mark.parametrize("exodus_direction, exodus_rate, migration_direction, migration_rate",
		[([(1,2,0)], sage.all.SR.var('E'), None, None),
		(None, None, [(2,1)], sage.all.SR.var('M')),
		([(1,2,0)], sage.all.SR.var('E'), [(2,1)], sage.all.SR.var('M'))]
		)
	def test_div(self, exodus_direction, exodus_rate, migration_direction, migration_rate):
		sample_list = [(),('a','a'),('b','b')]
		ancestral_pop = 0
		coalescence_rates = (sage.all.var('c0'), sage.all.var('c1'), sage.all.var('c2'))
		k_max = {'m_1':2, 'm_2':2, 'm_3':2, 'm_4':2}
		mutype_labels, max_k = zip(*sorted(k_max.items()))
		branchtype_dict = gflib.make_branchtype_dict(sample_list, mapping='unrooted', labels=mutype_labels)
		
		#get chain rule gf
		gfobj = cr.GFObjectChainRule(
			sample_list, 
			coalescence_rates, 
			branchtype_dict,
			exodus_rate=exodus_rate,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate,
			migration_direction=migration_direction
			)
		paths, eq_list = gfobj.make_gf()
		gf_chain = self.gf_from_paths(paths, eq_list)
		print(gf_chain)
		#get simple gf
		gfobj = gflib.GFObject(
			sample_list, 
			coalescence_rates, 
			branchtype_dict,
			exodus_rate=exodus_rate,
			exodus_direction=exodus_direction,
			migration_rate=migration_rate,
			migration_direction=migration_direction
			)	
		gf_simple = sum(gfobj.make_gf())
		print(gf_simple)
		comparison = sage.all.simplify(gf_simple-gf_chain)
		print(comparison)
		assert comparison == 0
		
	def gf_from_paths(self, paths, eq_list):
		eq_list = np.array(eq_list, dtype=object)
		result = sum(np.prod(eq_list[idxs]) for idxs in paths)
		return result
