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

direc_data = sys.argv[0].split('script')[0]+'data/'
direc_fig = sys.argv[0].split('script')[0]+'figures/'

matplotlib.rcParams.update({'font.size': p.fontsize})

solution = np.load(direc_data+"decoding_plasticity.npy",allow_pickle=True)
weights = np.load(direc_data+"decoding_weights.npy",allow_pickle=True)
ideal = np.load(direc_data+"decoding_ideal.npy",allow_pickle=True)

solution = solution[()]
labels = solution['labels']
bias = solution['bias']
variance = solution['variance']
error = solution['error']
weights = weights[()]


# Plot
def make_panel():
    fig = plt.figure(figsize=(1.5,1.5))
    ax = fig.add_axes([0.5,0.45,0.45,0.5])
    return fig,ax

def make_panel_weight():
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_axes([0.6,0.4,0.35,0.35])
    return fig,ax
        
def plot_bars(array,width,color,xlabels,ylabel,path):
    fig,ax = make_panel()
    ax.bar(np.arange(len(labels))*width,np.mean(array,axis=0),yerr=np.std(array,axis=0)/np.sqrt(100),width=width,color=color,edgecolor='k')
    ax.set_xticks(np.arange(len(labels))*width)
    ax.set_xticklabels(xlabels,rotation='vertical')
    ax.set_ylabel(ylabel)
    f.spines(ax)
    visualize(fig,path)
    
def plot_weights(kappas,weights,color,title,path):
    fig,ax = make_panel_weight()
    ax.scatter(kappas,weights,color=color,s=1)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("Weight")
    ax.set_title(title,fontsize=p.fontsize)
    f.spines(ax)
    visualize(fig,path)
    
def plot_ideal(stimuli,s_estimate,path):
    fig,ax = make_panel()
    ax.scatter(stimuli,s_estimate,color='grey',s=1)
    ax.plot([p.min_stimulus,p.max_stimulus],[p.min_stimulus,p.max_stimulus],color='k')
    ax.set_xlabel(r"Stimulus $\theta$")
    ax.set_ylabel(r"Decoded stimulus $\hat{\theta}$")
    f.spines(ax)
    visualize(fig,path)

visualize = f.save_fig

plot_bars(np.abs(bias),p.bar_width,p.colors,labels,'|Bias|',direc_fig+'decoding_bias')
plot_bars(variance,p.bar_width,p.colors,labels,'Variance',direc_fig+'decoding_variance')
plot_bars(error,p.bar_width,p.colors,labels,'Error',direc_fig+'decoding_error')
plot_ideal(ideal[:,0],ideal[:,1],direc_fig+'decoding_ideal')

for ll,label in enumerate(labels[:-1]):
    plot_weights(weights['kappas_var'],weights[label],p.colors[ll],label,direc_fig+'decoding_weights_'+label)
plot_weights(weights['kappas_cov'],weights[labels[-1]],p.colors[-1],labels[-1],direc_fig+'decoding_weights_'+labels[-1])

plt.show()

