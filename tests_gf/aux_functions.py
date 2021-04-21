import numpy as np
import os
import sage.all
import scipy.stats as stats

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

def chisquare(observed, expected, p_th=0.05, recombination=False, all_sims=False):
    #expected counts need to be larger than 5 for chisquare
    #combining bins with less than 5 counts into one bin.
    obs = np.reshape(observed, -1)
    exp = np.reshape(expected, -1)
    if not (recombination or all_sims):
        assert np.all(obs[exp == 0] == 0), "chisquare error: at least one cell with expected frequency 0 has observed values"
    #bin all values with counts smaller than 5
    binning_idxs = np.digitize(exp, np.array([5]), right=True)
    exp_smaller = np.sum(exp[binning_idxs==0])
    obs_smaller = np.sum(obs[binning_idxs==0])
    exp_binned = exp[binning_idxs==1]
    obs_binned = obs[binning_idxs==1]
    not_zeros = exp_binned > 0
    if sum(not_zeros)<1:
        assert False #expected probabilities are all 0
    else:
        chisq = stats.chisquare(obs_binned[not_zeros], exp_binned[not_zeros])
        print("chisquare value:", chisq)
        print("exp", exp_binned)
        print("obs", obs_binned)
        assert chisq.pvalue > p_th

def chisquare_contingency(observed1, observed2, p_th=0.05):
    obs1 = np.reshape(observed1, -1)
    obs2 = np.reshape(observed2, -1)
    all_obs = np.vstack((obs1, obs2))
    assert not np.all(all_obs==0), 'all counts are 0'
    binning_idxs = np.digitize(all_obs,np.array([5]), right=True)
    binning_across_cols = (binning_idxs==0).any(axis=0)
    obs_all_smaller = np.sum(all_obs[:,binning_across_cols==True],axis=1)
    if np.all(obs_all_smaller==0):
        obs_all_binned = all_obs[:,binning_across_cols==False]
    else:
        obs_all_binned = np.hstack((all_obs[:,binning_across_cols==False], obs_all_smaller.reshape((2,1))))
    chi2, p, dof, ex = stats.chi2_contingency(obs_all_binned)
    print("chisquare value:", p)
    print("obs_all_binned")
    assert p>p_th

def bonferroni(p_threshold, pvals):
    assert min(pvals) > p_threshold / len(pvals)

def scatter_loglog(observed, expected, name, xlabel='gimble', ylabel='to_test'):
    obs = np.reshape(observed, -1)
    exp = np.reshape(expected, -1)
    not_zeros = exp > 0
    assert sum(not_zeros) > 1, "expected probabilities are all 0."
    assert np.all(obs[np.logical_not(not_zeros)]==0)
    subs_obs = obs[not_zeros]
    subs_exp = exp[not_zeros]       
    log_obs = np.full(subs_obs.shape, 0)
    np.log(subs_obs, out=log_obs, where=subs_obs>0)
    log_exp=np.log(subs_exp)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(x=-log_exp, y=-log_obs)
    plt.axhline(y=min_value, color='g', linestyle='dashed')
    plt.axvline(x=min_value, color='g', linestyle='dashed')
    ax.plot([0,min_value+5],[0,min_value+5])
    ax.set_xlim((0,min_value+1))
    ax.set_ylim((0,min_value+1))
    ax.set_xlabel(f'lnP_{xlabel}')
    ax.set_ylabel(f'lnP_{ylabel}')
    ax.set_title(name)
    if not os.path.isdir('tests/output'):
        os.mkdir('tests/output')
    ax.figure.savefig(f'tests/output/scatter_{name}.png', dpi=300)
    plt.clf()

def downsample(A, N):
    distribution = [i for i, j in enumerate(A) for _ in range(j)]
    sample = Counter(random.sample(distribution, N))
    return [sample[i] for i in range(len(A))]

def get_parameter_dict(coalescence_rates, global_info, sim_config, gf_vars):
    parameter_dict = {}
    reference_pop = global_info['reference_pop']
    if gf_vars.get('migration_rate'):
        migration_string = 'me_A_B' if gf_vars['migration_direction'] == [(1,2)] else 'me_B_A'
        parameter_dict[gf_vars['migration_rate']] = sage.all.Rational(2 * sim_config[migration_string] * sim_config[f'Ne_{reference_pop}'])
    if gf_vars.get('exodus_rate'):
        parameter_dict[sage.all.SR.var('T')] = sage.all.Rational(sim_config['T']/(2*sim_config[f'Ne_{reference_pop}']))
    for c, Ne in zip(coalescence_rates,('Ne_A_B', 'Ne_A', 'Ne_B')):
        if Ne in sim_config:
            parameter_dict[c] = sage.all.Rational(sim_config[f'Ne_{reference_pop}']/sim_config[Ne])
        else:
            parameter_dict[c] = 0.0
    return parameter_dict

def get_theta(global_info, sim_config, **kwargs):
    reference_pop = global_info['reference_pop']
    Ne_ref = sim_config[f'Ne_{reference_pop}']
    mu=global_info['mu']
    block_length = global_info['blocklength']
    return 2*sage.all.Rational(Ne_ref*mu)*block_length