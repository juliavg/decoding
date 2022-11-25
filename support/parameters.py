from importlib import reload
import numpy as np
import functions
reload(functions)
import functions as f

# Parameters

# Feedforward network
tau_r = 0.001    # s
inhibitory_weight = -1700.    # 
inhibitory_rate = 100.    # Hz
n_inputs = 50    # number
slope = 1e-4

# Stimulus
min_stimulus = -np.pi/2
max_stimulus = np.pi/2

# Tuning curves
min_kappa = 1e-12
max_kappa = 1
max_rate = 125.
tc_bins = 20

# Plasticity
learning_rate = 0.1    # 
weight_decay_rate = learning_rate*0.3
target_rate = {}
target_rate['variance'] = 1/(2*np.pi)
target_rate['covariance'] = 30
wmax = 16000.
learning_rules = {}
learning_rules['variance'] = f.variance_leak_rule
learning_rules['covariance'] = f.covariance_leak_rule

# Stimulation
n_stimuli = 1000
stimulus_length = 0.2    # s
offset = 1.
baseline_input_rate = 20.

dt = 0.001    # s
warmup_time = 200.    # s

# Decoding
analyze_last = 500
n_trials_decoding = 50
n_trials_ideal = 500

# Input current
activity_threshold = 20.
n_bins = 7

# Plot
fontsize = 8
fontsize_title = 10
plot_n_weights = 50
linewidth = 1
hist_bins = 30
bar_width = 0.3
colors = np.array([[155, 193, 188],[252,141,98],[141,160,203],[231,138,195],[166,216,84]])/255.
color_pre = np.array([132,0,50])/255.
color_post = np.array([237,106,90])/255.
color_spine = np.array([155,193,188])/255.

