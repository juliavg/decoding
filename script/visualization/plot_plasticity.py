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
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': p.fontsize})
plasticity_type = sys.argv[1]

direc_data = sys.argv[0].split('script')[0]+'data/'
direc_fig = sys.argv[0].split('script')[0]+'figures/'

sim_idx = 0

extension_plasticity = "_"+plasticity_type+".npy"
selectivity = np.load(direc_data+"input_selectivity"+extension_plasticity)
delta_ps = np.load(direc_data+"delta_po"+extension_plasticity)
pref_stimuli = np.load(direc_data+"preferred_stimulus"+extension_plasticity)
kappa = np.load(direc_data+"kappas"+extension_plasticity)
cc = np.load(direc_data+"cc"+extension_plasticity)
final_weights = np.load(direc_data+"weights"+extension_plasticity)
theory = np.load(direc_data+"theory_weight"+extension_plasticity,allow_pickle=True)
theory = theory[()]

extension_seed = "_"+plasticity_type+"_"+str(sim_idx)+".npy"
time_steps = np.load(direc_data+"time_steps"+extension_seed)
solution = np.load(direc_data+"solution"+extension_seed)


# Plot
def make_panel_colorbar():
    fig = plt.figure(figsize=(2,1.5))
    ax = fig.add_axes([0.3,0.25,0.45,0.6])
    axb = fig.add_axes([0.8,0.25,0.02,0.6])    
    return fig,ax,axb

def make_panel():
    fig = plt.figure(figsize=(2,1.5))
    ax = fig.add_axes([0.25,0.25,0.45,0.6])
    return fig,ax

def make_panel_out_hist():
    fig = plt.figure(figsize=(1.5,0.5))
    ax = fig.add_axes([0.25,0.25,0.45,0.6])
    return fig,ax

def make_panel_ps_hist():
    fig = plt.figure(figsize=(0.75,1.))
    ax = fig.add_axes([0.25,0.25,0.45,0.6])
    return fig,ax

def make_panel_tc():
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0.25,0.25,0.45,0.6])
    return fig,ax

def plot_weight_time_series(time_steps,weights,warmup_time,color,path):
    fig,ax = make_panel()
    ax.plot(time_steps,weights,color=color,linewidth=p.linewidth)
    ax.axvline(warmup_time,linestyle='--',color='k')
    ax.set_ylabel('Weight')
    ax.set_xlabel('Time [s]')
    f.spines(ax)
    visualize(fig,path)
    
def plot_weight_selectivity(weight,selectivity,delta_ps,color,path):
    fig,ax = make_panel()
    scatter = ax.scatter(selectivity,weight,color=color,s=1)
    ax.set_xlabel("Selectivity")
    ax.set_ylabel("Weight")
    ax.set_xlim([0,0.5])
    ax.set_xticks([0,0.5])
    f.spines(ax)
    visualize(fig,path)

def plot_selectivity_kappa(selectivity,kappa,color,path):
    fig,ax = make_panel()
    ax.scatter(kappa,selectivity,color=color,s=1)
    ax.set_xlabel(r"Width $\kappa$")
    ax.set_ylabel("Selectivity")
    ax.set_ylim([0,0.5])
    ax.set_yticks([0,0.5])
    f.spines(ax)
    visualize(fig,path)


def plot_weight_ps(weight,delta_ps,selectivity,path):
    fig,ax,axb = make_panel_colorbar()
    scatter = ax.scatter(np.abs(delta_ps),weight,c=selectivity,s=1,vmin=0,vmax=0.5)
    cbar = plt.colorbar(mappable=scatter,cax=axb,ticks=[0,0.5])
    cbar.ax.set_yticklabels(['0','0.5'])
    cbar.ax.set_title("Selectivity",fontsize=p.fontsize)
    ax.set_xticks([0,np.pi/2])
    ax.set_xticklabels(['0',r'$\frac{\pi}{2}$'])
    ax.set_xlabel(r"|$\Delta$PO|")
    ax.set_ylabel("Weight")
    f.spines(ax)
    visualize(fig,path)

def plot_cc_ps(delta_ps,cc,selectivity,path):
    fig,ax,axb = make_panel_colorbar()
    scatter = ax.scatter(np.abs(delta_ps),cc,c=selectivity,s=1,vmin=0,vmax=0.5)
    cbar = plt.colorbar(mappable=scatter,cax=axb,ticks=[0,0.5])
    cbar.ax.set_yticklabels(['0','0.5'])
    cbar.ax.set_title("Selectivity",fontsize=p.fontsize)
    ax.set_xticks([0,np.pi/2])
    ax.set_xticklabels(['0',r'$\frac{\pi}{2}$'])
    ax.set_xlabel(r"|$\Delta$PO|")
    ax.set_ylabel("CC")
    f.spines(ax)
    visualize(fig,path)

def plot_kappa_weight(kappa,weight,theory_kappa,theory_weight,color,path):
    fig,ax = make_panel()
    scatter = ax.scatter(kappa,weight,color=color,s=1,label='sim')
    ax.plot(theory_kappa,theory_weight,color='k',linewidth=p.linewidth,label='theo')
    ax.set_xlabel(r"Width $\kappa$")
    ax.set_ylabel("Weight")
    ax.legend()
    f.spines(ax)
    visualize(fig,path)
    
def plot_tuning_curve(stimuli_rate,rate,stimuli_tuning_curve,tuning_curve,color,path):
    fig,ax = make_panel_tc()
    ax.scatter(stimuli_rate,rate/np.max(rate),color=color,s=1)
    ax.plot(stimuli_tuning_curve,tuning_curve,color='k',linewidth=p.linewidth)
    ax.set_xticks([-np.pi/2,0,np.pi/2])
    ax.set_xticklabels([r'$-\frac{\pi}{2}$','0',r'$\frac{\pi}{2}$'])
    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Normalized firing rate")
    f.spines(ax)
    visualize(fig,path)

def plot_histogram(data,bins,xlabel,color,path):
    fig,ax = make_panel()
    ax.hist(data,bins=bins,color=color,density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('pdf')
    f.spines(ax)
    visualize(fig,path)

def plot_output_histogram(data,bins,xticks,xticklabels,xlabel,color,path):
    fig,ax = make_panel_out_hist()
    ax.hist(data,bins=bins,color=color,density=True)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('pdf')
    f.spines(ax)
    visualize(fig,path)

def plot_ps_histogram(data,bins,xticks,xticklabels,xlabel,color,path):
    fig,ax = make_panel_ps_hist()
    ax.hist(data,bins=bins,color=color,density=True)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('pdf')
    f.spines(ax)
    visualize(fig,path)


visualize = f.save_fig

plot_weight_time_series(time_steps,
                        solution[:,np.random.randint(1,p.n_inputs+1,p.plot_n_weights)],
                        p.warmup_time,
                        p.color_spine,
                        direc_fig+plasticity_type+'_weight_time_series')
                        
plot_weight_selectivity(final_weights,
                        selectivity,
                        delta_ps,
                        p.color_spine,
                        direc_fig+plasticity_type+'_weight_selectivity')
                        
plot_selectivity_kappa(selectivity,
                        kappa,
                        p.color_pre,
                        direc_fig+plasticity_type+'_selectivity_kappa')
                        
plot_weight_ps(final_weights,
               delta_ps,
               selectivity,
               direc_fig+plasticity_type+'_weight_ps')
               
plot_cc_ps(delta_ps,
           cc,
           selectivity,
           direc_fig+plasticity_type+'_cc_ps')
           
plot_kappa_weight(kappa,
                  final_weights,
                  theory['kappa'],
                  theory['weight'],
                  p.color_spine,
                  direc_fig+plasticity_type+'_kappa_weight')
                  
plot_histogram(final_weights,
               p.hist_bins,
               'Weight',
               p.color_spine,
               direc_fig+plasticity_type+'_weights_histogram')
               
plot_histogram(kappa,
               p.hist_bins,
               r'Width $\kappa$',
               p.color_pre,
               direc_fig+plasticity_type+'_kappa_histogram')
               
plot_histogram(selectivity,
               p.hist_bins,
               'Selectivity',
               p.color_pre,
               direc_fig+plasticity_type+'_selectivity_histogram')
               
plot_ps_histogram(pref_stimuli,
                  p.hist_bins,
                  [-np.pi/2,0,np.pi/2],
                  [r"$-\frac{\pi}{2}$","0",r"$\frac{\pi}{2}$"],
                  'Post PO',
                  p.color_pre,
                  direc_fig+plasticity_type+'_ps_histogram')
                  
plot_ps_histogram(np.abs(delta_ps),
                  p.hist_bins,
                  [0,np.pi/2],
                  ["0",r"$\frac{\pi}{2}$"],
                  r"|$\Delta$PO|",
                  p.color_post,
                  direc_fig+plasticity_type+'_delta_ps_histogram')

for seed in np.arange(10):
    extension_seed = "_"+plasticity_type+"_"+str(seed)+".npy"
    tuning_curve = np.load(direc_data+"output_tuning_curve_norm"+extension_seed)
    bins_tc = np.load(direc_data+"bins"+extension_seed)
    output_rate = np.load(direc_data+"output_rate_analysis"+extension_seed)
    stimuli = np.load(direc_data+"stimuli_analysis"+extension_seed)
    plot_tuning_curve(stimuli,output_rate,bins_tc,tuning_curve,p.color_post,direc_fig+plasticity_type+'_tuning_curve_'+str(seed))


output_ps = np.load(direc_data+"output_pref_stimulus"+extension_plasticity)
output_selectivity = np.load(direc_data+"output_selectivity"+extension_plasticity)

plot_output_histogram(output_ps,p.hist_bins,[-np.pi/2,0,np.pi/2],[r"$-\frac{\pi}{2}$","0",r"$\frac{\pi}{2}$"],r"Post PO",p.color_post,direc_fig+plasticity_type+'_output_ps')
plot_output_histogram(output_selectivity,p.hist_bins,[0,0.5,1],[0,0.5,1],r"Post selectivity",p.color_post,direc_fig+plasticity_type+'_output_selectivity')

plt.show()
