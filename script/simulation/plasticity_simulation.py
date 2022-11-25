import numpy as np
from scipy.integrate import odeint
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
sim_idx = int(sys.argv[2])
plasticity_type = sys.argv[1]
learning_rule = p.learning_rules[plasticity_type]
extension = "_"+plasticity_type+"_"+str(sim_idx)+".npy"

# Simulation

# Inputs tuning
preferred_stimulus = np.random.uniform(p.min_stimulus,p.max_stimulus,size=p.n_inputs)
kappas = np.random.uniform(p.min_kappa,p.max_kappa,size=p.n_inputs)
stimuli_all = np.random.uniform(p.min_stimulus,p.max_stimulus,size=p.n_stimuli)

# Warmup
time_steps = np.arange(int(p.warmup_time/p.dt))*p.dt
initial_conditions = np.zeros(p.n_inputs+1)
input_rates = p.baseline_input_rate*np.ones(p.n_inputs)
solution = odeint(f.rate_model,
                  initial_conditions,
                  time_steps,
                  args=(learning_rule,
                        input_rates,
                        p.max_rate,
                        p.wmax,
                        p.inhibitory_weight,
                        p.inhibitory_rate,
                        p.tau_r,
                        p.learning_rate,
                        p.weight_decay_rate,
                        p.target_rate[plasticity_type],
                        p.slope))
solution_all = solution*1
time_steps_all = time_steps*1


# Stimulation
current_time = time_steps_all[-1]+p.dt
for stimulus in stimuli_all:
    time_steps = np.arange(int(p.stimulus_length/p.dt))*p.dt+current_time
    input_rates = f.input_rates(p.max_rate,stimulus,preferred_stimulus,kappas)
    
    solution = odeint(f.rate_model,
                      solution_all[-1,:],
                      time_steps,
                      args=(learning_rule,
                            input_rates,
                            p.max_rate,
                            p.wmax,
                            p.inhibitory_weight,
                            p.inhibitory_rate,
                            p.tau_r,
                            p.learning_rate,
                            p.weight_decay_rate,
                            p.target_rate[plasticity_type],
                            p.slope))
    solution_all = np.concatenate((solution_all,solution))
    time_steps_all = np.concatenate((time_steps_all,time_steps))
    current_time = time_steps[-1]+p.dt 

np.save(direc+"preferred_stimulus"+extension,preferred_stimulus)
np.save(direc+"kappas"+extension,kappas)
np.save(direc+"solution"+extension,solution_all)
np.save(direc+"time_steps"+extension,time_steps_all)
np.save(direc+"stimuli"+extension,stimuli_all)
