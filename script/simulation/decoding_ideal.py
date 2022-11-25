import numpy as np
from importlib import reload
import sys
sys.path.insert(2,'support')
import functions
reload(functions)
import functions as f
import parameters
reload(parameters)
import parameters as p

direc = sys.argv[0].split('script')[0]+'data/'

noise_type = f.poisson_noise
tuning_curve = f.vonmises_tuning_curve

s_estimate_ideal = np.zeros((p.n_trials_ideal,1))
stimuli_all = np.zeros((p.n_trials_ideal,1))
for ss in np.arange(p.n_trials_ideal):
    shown_stimulus = np.random.uniform(p.min_stimulus,p.max_stimulus,1)
    stimuli_all[ss] = shown_stimulus

    pref_stimuli = (f.uniform_distribution(p.n_inputs,p.min_stimulus,p.max_stimulus))[np.newaxis].T
    kappas = np.random.uniform(p.min_kappa,p.max_kappa,size=p.n_inputs)[np.newaxis].T
    weights_ideal = kappas*1

    input_rates = f.calculate_input_rate(p.max_rate,pref_stimuli,kappas,shown_stimulus,tuning_curve,noise_type)
    s_estimate_ideal[ss,0] = f.estimate_stimulus(input_rates,weights_ideal,pref_stimuli)

decoding = np.hstack((stimuli_all,s_estimate_ideal))

np.save(direc+"decoding_ideal.npy",decoding)
