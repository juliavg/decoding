import numpy as np
import scipy.special as sp
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

# Generate arrays
preferred_stimulus_all = np.array([])
delta_po_all = np.array([])
cc_all = np.array([])
kappas_all = np.array([])
weights_all = np.array([])
output_pref_stimulus_all = np.array([])
output_selectivity_all = np.array([])
input_selectivity_all = np.array([])
min_kappa = []
max_kappa = []  

for sim_idx in np.arange(n_simulations):
    # Load data
    extension = "_"+plasticity_type+"_"+str(sim_idx)+".npy"
    preferred_stimulus = np.load(direc+"preferred_stimulus"+extension)
    kappas = np.load(direc+"kappas"+extension)
    solution = np.load(direc+"solution"+extension)
    time_steps = np.load(direc+"time_steps"+extension)
    stimuli = np.load(direc+"stimuli"+extension)

    # Analysis
    
    # Output rate
    rate_bins = np.arange(p.n_stimuli+1)*p.stimulus_length+p.warmup_time
    output_rate = np.histogram(time_steps,bins=rate_bins,weights=solution[:,0])[0]/np.histogram(time_steps,bins=rate_bins)[0]

    # Output selectivity
    stimuli_analysis = stimuli[-int(p.n_stimuli/2):]
    output_rate_analysis = output_rate[-int(p.n_stimuli/2):]
    bins = np.arange(p.tc_bins+1)*np.pi/p.tc_bins-np.pi/2
    output_tuning_curve = np.histogram(stimuli_analysis,weights=output_rate_analysis,bins=bins)[0]/np.histogram(stimuli_analysis,bins=bins)[0]
    output_tuning_curve_norm = np.histogram(stimuli_analysis,weights=output_rate_analysis/np.max(output_rate_analysis),bins=bins)[0]/np.histogram(stimuli_analysis,bins=bins)[0]
    bins = bins[:-1]+np.diff(bins)[0]/2
    output_pref_stimulus,output_selectivity = f.calculate_preferred_and_selectivity(output_tuning_curve,bins)

    # Delta PO
    delta_po = output_pref_stimulus-preferred_stimulus
    delta_po = (delta_po+np.pi/2)%(np.pi)-np.pi/2 

    # Correlation coefficient   
    input_rates = np.zeros((p.n_inputs,stimuli.shape[0]))
    for ss,stimulus in enumerate(stimuli):
        input_rates[:,ss] = f.input_rates(p.max_rate,stimulus,preferred_stimulus,kappas)
    
    cc = np.zeros(p.n_inputs)
    for ii in np.arange(p.n_inputs): 
        cc[ii] = np.corrcoef(input_rates[ii,:],output_rate)[0,1]      
        
    # Input selectivity
    stimuli = np.linspace(p.min_stimulus,p.max_stimulus,500)
    input_selectivity = np.zeros(p.n_inputs)
    for ii in np.arange(p.n_inputs):
       firing_rate = f.input_rates(p.max_rate,stimuli,preferred_stimulus[ii],kappas[ii])
       pref_stimulus,input_selectivity[ii] = f.calculate_preferred_and_selectivity(firing_rate,stimuli) 

    # Theory weight
    min_kappa.append(np.min(kappas))
    max_kappa.append(np.max(kappas))

    preferred_stimulus_all = np.concatenate((preferred_stimulus_all,preferred_stimulus))
    delta_po_all = np.concatenate((delta_po_all,delta_po))
    cc_all = np.concatenate((cc_all,cc))
    kappas_all = np.concatenate((kappas_all,kappas))
    weights_all = np.concatenate((weights_all,np.mean(solution[-10000:,1:],axis=0)))
    output_pref_stimulus_all = np.concatenate((output_pref_stimulus_all,np.array([output_pref_stimulus])))
    output_selectivity_all = np.concatenate((output_selectivity_all,np.array([output_selectivity])))
    input_selectivity_all = np.concatenate((input_selectivity_all,input_selectivity))

    np.save(direc+"bins"+extension,bins)
    np.save(direc+"output_tuning_curve"+extension,output_tuning_curve)
    np.save(direc+"output_tuning_curve_norm"+extension,output_tuning_curve_norm)
    np.save(direc+"stimuli_analysis"+extension,stimuli_analysis)
    np.save(direc+"output_rate_analysis"+extension,output_rate_analysis)

# Theory weight
kappa_theory = np.linspace(min(min_kappa),max(max_kappa),1000)
weight_theory = (1/(4*np.pi**2))*((sp.i0(2*kappa_theory)/sp.i0(kappa_theory)**2)-1)*p.learning_rate/(p.weight_decay_rate)
theory = {}
theory['kappa'] = kappa_theory
theory['weight'] = weight_theory

extension = "_"+plasticity_type+".npy"
np.save(direc+"preferred_stimulus"+extension,preferred_stimulus_all)
np.save(direc+"delta_po"+extension,delta_po_all)
np.save(direc+"cc"+extension,cc_all)
np.save(direc+"kappas"+extension,kappas_all)
np.save(direc+"weights"+extension,weights_all)
np.save(direc+"output_pref_stimulus"+extension,output_pref_stimulus_all)
np.save(direc+"output_selectivity"+extension,output_selectivity_all)
np.save(direc+"input_selectivity"+extension,input_selectivity_all)
np.save(direc+"theory_weight"+extension,theory)
