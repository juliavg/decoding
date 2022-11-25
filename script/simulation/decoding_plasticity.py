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
nsim = int(sys.argv[1])

all_seeds = np.arange(nsim)

mean_bias = np.zeros((all_seeds.shape[0],5))
mean_var = np.zeros((all_seeds.shape[0],5))
mean_error = np.zeros((all_seeds.shape[0],5))

for ss,seed_number in enumerate(all_seeds):
    orientations_all = np.linspace(p.min_stimulus,p.max_stimulus,20)[np.newaxis].T

    noise_type = f.poisson_noise
    pref_stimuli_distribution = f.uniform_distribution
    tuning_curve = f.vonmises_tuning_curve

    last_n_stimuli = -int(p.analyze_last*p.stimulus_length/p.dt)

    s_estimate_var = np.zeros((p.n_trials_decoding,len(orientations_all)))
    s_estimate_ideal = np.zeros((p.n_trials_decoding,len(orientations_all)))
    s_estimate_uniform = np.zeros((p.n_trials_decoding,len(orientations_all)))
    s_estimate_random = np.zeros((p.n_trials_decoding,len(orientations_all)))
    s_estimate_cov = np.zeros((p.n_trials_decoding,len(orientations_all)))

    from_idx = ss*p.n_inputs
    to_idx =  (ss+1)*p.n_inputs
    weights_var = np.load(direc+"weights_variance.npy")[from_idx:to_idx][np.newaxis].T
    pref_stimuli_var = np.load(direc+"preferred_stimulus_variance.npy")[from_idx:to_idx][np.newaxis].T
    kappas_var = np.load(direc+"kappas_variance.npy")[from_idx:to_idx][np.newaxis].T
    
    weights_cov = np.load(direc+"weights_covariance.npy")[from_idx:to_idx][np.newaxis].T
    pref_stimuli_cov = np.load(direc+"preferred_stimulus_covariance.npy")[from_idx:to_idx][np.newaxis].T
    kappas_cov = np.load(direc+"kappas_covariance.npy")[from_idx:to_idx][np.newaxis].T

    weights_variance_random = weights_var*1
    np.random.shuffle(weights_variance_random)
    weights_uniform = np.ones(p.n_inputs)[np.newaxis].T

    for oo,shown_orientation in enumerate(orientations_all):
        for tt in np.arange(p.n_trials_decoding):
            input_rates = f.calculate_input_rate(p.max_rate,pref_stimuli_var,kappas_var,shown_orientation,tuning_curve,noise_type)
            s_estimate_var[tt,oo] = f.estimate_stimulus(input_rates,weights_var,pref_stimuli_var)
            s_estimate_ideal[tt,oo] = f.estimate_stimulus(input_rates,kappas_var,pref_stimuli_var)
            s_estimate_uniform[tt,oo] = f.estimate_stimulus(input_rates,weights_uniform,pref_stimuli_var)
            s_estimate_random[tt,oo] = f.estimate_stimulus(input_rates,weights_variance_random,pref_stimuli_var)
            
            input_rates = f.calculate_input_rate(p.max_rate,pref_stimuli_cov,kappas_cov,shown_orientation,tuning_curve,noise_type)
            s_estimate_cov[tt,oo] = f.estimate_stimulus(input_rates,weights_cov,pref_stimuli_cov)

    for ee,estimate in enumerate([s_estimate_var,s_estimate_ideal,s_estimate_uniform,s_estimate_random,s_estimate_cov]):
        b_est,var_est,error_est = f.bias_variance(estimate,orientations_all,p.min_stimulus,p.max_stimulus)
        mean_bias[ss,ee] = np.mean(b_est)
        mean_var[ss,ee] = np.mean(var_est)
        mean_error[ss,ee] = np.mean(error_est)

weights_rec = {}
weights_rec['Variance'] = weights_var
weights_rec['ML'] = kappas_var
weights_rec['Uniform'] = weights_uniform
weights_rec['Shuffled'] = weights_variance_random
weights_rec['kappas_var'] = kappas_var
weights_rec['Covariance'] = weights_cov
weights_rec['kappas_cov'] = kappas_cov

solution = {}
solution['labels'] = ['Variance','ML','Uniform','Shuffled','Covariance']
solution['bias'] = mean_bias
solution['variance'] = mean_var
solution['error'] = mean_error

np.save(direc+"decoding_plasticity.npy",solution)
np.save(direc+"decoding_weights.npy",weights_rec)
