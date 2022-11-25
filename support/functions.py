import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises
import scipy.special as sp
from scipy.stats import circmean as cm
from scipy.stats import circvar as cv

def calculate_preferred_and_selectivity(rate,stimuli):
    R = np.sum(rate*np.exp(2j*stimuli))/np.sum(rate)
    preferred_stimulus = np.angle(R)/2.
    selectivity = np.abs(R)
    return preferred_stimulus,selectivity

def input_rates(max_rate,stimulus,preferred_stimulus,kappas):
    return max_rate*vonmises.pdf(2*stimulus,loc=2*preferred_stimulus,kappa=kappas)

# Rate model
def current_to_rate(input_current,slope):
    rate = slope*input_current
    return rate.clip(0)
    
def variance_leak_rule(weight,pre,post,r_max,target,learning_rate,weight_decay_rate):
    return learning_rate*(pre/r_max-target)**2 - weight_decay_rate*weight

def covariance_leak_rule(weight,pre,post,r_max,target,learning_rate,weight_decay_rate):
    dwdt = learning_rate*(pre/r_max-target/r_max)*(post/r_max-target/r_max) - weight_decay_rate*weight
    return dwdt

def target_rule(weight,pre,post,target,learning_rate,weight_decay_rate):
    return learning_rate*(target-post)

def rate_model(variables,time,learning_rule,input_rates,max_rate,w_max,inhibitory_weight,inhibitory_rate,tau_r,learning_rate,weight_decay_rate,target_rate,slope):
    output_rate = variables[0]
    excitatory_weights = variables[1:]
    drdt = 1./tau_r*(-output_rate+current_to_rate(np.dot(w_max*excitatory_weights,input_rates)+inhibitory_weight*inhibitory_rate,slope=slope))
    dwdt = learning_rule(excitatory_weights,input_rates,output_rate,max_rate,target_rate,learning_rate,weight_decay_rate)
    dydt = np.zeros(variables.shape)
    dydt[0] = drdt
    dydt[1:] = dwdt
    return dydt
    
    
# Decoding
def generate_noisy_tuning_curve(max_rate,pref_stimulus,selectivity,shown_stimuli,tuning_curve,noise_type):
    rates = tuning_curve(max_rate,pref_stimulus,selectivity,shown_stimuli)
    rates = noise_type(rates)
    return rates.clip(0)

def calculate_input_rate(max_rate,pref_stimuli,selectivity,shown_stimuli,tuning_curve,noise_type):
    n_input_neurons = len(selectivity)
    input_rates = np.zeros((n_input_neurons,shown_stimuli.shape[0]))
    for ii in np.arange(n_input_neurons):
        input_rates[ii,:] = generate_noisy_tuning_curve(max_rate,pref_stimuli[ii],selectivity[ii],shown_stimuli,tuning_curve,noise_type)
    return input_rates
    
def poisson_noise(rates):
    return np.random.poisson(rates,len(rates))

def vonmises_tuning_curve(max_rate,mu,kappa,stimuli):
    rates = max_rate*np.exp(kappa*np.cos(2*(stimuli-mu)))/(2*np.pi*sp.i0(kappa))
    return rates
    
def estimate_stimulus(rates,weights,ps):
    pref_stimuli = ps*2
    num = np.sum(rates*weights*np.sin(pref_stimuli))
    denum = np.sum(rates*weights*np.cos(pref_stimuli))
    angle = np.arctan2(num,denum)
    return angle/2
    
def bias_variance(x_estimate,x_real,low,high):
    bias = cm(x_estimate,axis=0,low=low,high=high)-x_real.T
    bias = (bias+np.pi/2)%(np.pi)-np.pi/2
    variance = cv(x_estimate,axis=0,low=low,high=high)
    error = bias**2+variance
    return bias,variance,error
    
def uniform_distribution(n_inputs,*args):
    min_stimulus = args[0]
    max_stimulus = args[1]
    return np.random.uniform(min_stimulus,max_stimulus,size=n_inputs)


# Plotting
def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
def save_fig(fig,path):
    fig.savefig(path+'.svg')
    plt.close()
    
def show_fig(*args):
    return
