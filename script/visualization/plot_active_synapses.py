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

extension = "_"+plasticity_type+".npy"
total_input_current = np.load(direc_data+"total_input_current"+extension)
n_active_inputs = np.load(direc_data+"n_active_inputs"+extension)
mean_weight_active_inputs = np.load(direc_data+"mean_weight"+extension)
bins = np.load(direc_data+"bins_input_curret"+extension)

# Plot
def make_panel():
    fig = plt.figure(figsize=(1,1.))
    ax = fig.add_axes([0.25,0.25,0.7,0.7])
    return fig,ax

def plot_modulation(bins,data,xticks,xticklabels,xlabel,ylabel,color,path):
    fig,ax = make_panel()
    ax.errorbar(bins,np.mean(data,axis=0),yerr=np.std(data,axis=0)/np.sqrt(data.shape[0]),color=color)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    f.spines(ax)
    visualize(fig,path)
    
visualize = f.save_fig

plot_modulation(bins,
                total_input_current,
                [-np.pi/2,0,np.pi/2],
                [r"$-\frac{\pi}{2}$","0",r"$\frac{\pi}{2}$"],
                'Stimulus relative to post PO',
                'Total input current',
                p.color_pre,
                direc_fig+'input_current_'+plasticity_type)

plot_modulation(bins,
                n_active_inputs,
                [-np.pi/2,0,np.pi/2],
                [r"$-\frac{\pi}{2}$","0",r"$\frac{\pi}{2}$"],
                'Stimulus relative to post PO',
                'Number of\nactive inputs',
                p.color_pre,
                direc_fig+'number_active_'+plasticity_type)
                
plot_modulation(bins,
                mean_weight_active_inputs,
                [-np.pi/2,0,np.pi/2],
                [r"$-\frac{\pi}{2}$","0",r"$\frac{\pi}{2}$"],
                'Stimulus relative to post PO',
                'Mean weight of\nactve inputs',
                p.color_pre,
                direc_fig+'mean_weight_'+plasticity_type)

plt.show()
