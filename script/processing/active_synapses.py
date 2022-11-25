import numpy as np
import matplotlib.pyplot as plt
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
plasticity_type = sys.argv[1]
n_simulations = int(sys.argv[2])

output_ps = np.load(direc+"output_pref_stimulus_"+plasticity_type+".npy")
stimulation_begin = int(p.warmup_time/p.dt)
n_steps_per_stim = int(p.stimulus_length/p.dt)
bins = np.arange(p.n_bins+1)*(p.max_stimulus-p.min_stimulus)/p.n_bins + p.min_stimulus

n_active_inputs = np.zeros((n_simulations,p.n_bins))
total_input_current = np.zeros((n_simulations,p.n_bins))
mean_weight_active_inputs = np.zeros((n_simulations,p.n_bins))

n_selected = int(1.*p.n_inputs)

for idx in np.arange(n_simulations):
    extension = "_"+plasticity_type+"_"+str(idx)+".npy"
    preferred_stimulus = np.load(direc+"preferred_stimulus"+extension)
    kappas = np.load(direc+"kappas"+extension)
    stimuli = np.load(direc+"stimuli"+extension)
    solution = np.load(direc+"solution"+extension)
    
    selected_inputs = np.random.choice(np.arange(p.n_inputs),n_selected,replace=False)
    selected_inputs.sort()

    solution = solution[stimulation_begin:,:]
    mean_weights = np.zeros((stimuli.shape[0],n_selected))
    for ii in np.arange(stimuli.shape[0]):
        sol_temp = solution[ii*n_steps_per_stim:(ii+1)*n_steps_per_stim,selected_inputs+1]
        mean_weights[ii,:] = np.mean(sol_temp,axis=0)

    active_inputs = np.zeros(mean_weights.shape)
    for ii in np.arange(n_selected):
        active_inputs[:,ii] = f.input_rates(p.max_rate,stimuli,preferred_stimulus[selected_inputs[ii]],kappas[selected_inputs[ii]])

    active_inputs[active_inputs<p.activity_threshold] = 0
    active_inputs[active_inputs>=p.activity_threshold] = 1

    n_active_inputs_idx = np.sum(active_inputs,axis=1)
    total_input_current_idx = np.sum(active_inputs*mean_weights,axis=1)
    mean_weight_active_inputs_idx = total_input_current_idx/n_active_inputs_idx

    delta_stimulus = output_ps[idx]-stimuli
    delta_stimulus = (delta_stimulus+np.pi/2)%(np.pi)-np.pi/2

    n_active_inputs[idx,:] = np.histogram(delta_stimulus,bins=bins,weights=n_active_inputs_idx)[0]/np.histogram(delta_stimulus,bins=bins)[0]
    total_input_current[idx,:] = np.histogram(delta_stimulus,bins=bins,weights=total_input_current_idx)[0]/np.histogram(delta_stimulus,bins=bins)[0]
    mean_weight_active_inputs[idx,:] = np.histogram(delta_stimulus,bins=bins,weights=mean_weight_active_inputs_idx)[0]/np.histogram(delta_stimulus,bins=bins)[0]

n_active_inputs = n_active_inputs[~np.isnan(n_active_inputs).any(axis=1)]
total_input_current = total_input_current[~np.isnan(total_input_current).any(axis=1)]
mean_weight_active_inputs = mean_weight_active_inputs[~np.isnan(mean_weight_active_inputs).any(axis=1)]

extension = "_"+plasticity_type+".npy"
np.save(direc+"total_input_current"+extension,total_input_current)
np.save(direc+"n_active_inputs"+extension,n_active_inputs)
np.save(direc+"mean_weight"+extension,mean_weight_active_inputs)
np.save(direc+"bins_input_curret"+extension,bins[:-1]+np.diff(bins)[0]/2)
