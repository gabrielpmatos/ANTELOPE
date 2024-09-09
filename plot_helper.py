#!/usr/bin/env python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import colors
from math import ceil
from scipy.stats import pearsonr

tag = "_lhco"
plot_dir = 'plots/lhco/'

def my_metric(s,b):
    return np.sqrt(2*((s+b)*np.log(1+s/b)-s))

def detect_outliers(x):
  z = np.abs(stats.zscore(x))
  print(max(z))
  x_smooth = x[z<40]
  n_removed = len(x)-len(x_smooth)
  print(n_removed, " outliers removed")
  return x_smooth, n_removed

def plot_loss(h, model="", loss='loss'):
  #print(h.history)
  plt.plot(h.history[loss])
  plt.plot(h.history['val_'+loss])
  plt.title(model+' '+loss)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'.png')
  plt.clf()
  print("Saved loss plot for ", model, loss)

def plot_saved_loss(h, model="", loss='loss'):
  plt.plot(h[loss])
  plt.plot(h['val_'+loss])
  plt.title(model+' '+loss)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.yscale('log')
  plt.legend(['train', 'val'], loc='upper left')
  plt.savefig(plot_dir+loss+'VsEpoch_'+model+'_'+tag+'log.png')
  plt.clf()
  print("Saved loss plot for ", model, loss)

def plot_var(x_dict, x_cut1, x_cut2, key):
  #bmax = max(max(x_orig),max(y_orig))
  #bmin = min(min(x_orig),min(y_orig))
  #bins=np.histogram(np.hstack((x_dict[key],x_cut1[key])),bins=20)[1]
  bins= np.linspace(0,8000,20)
  fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
  h1 = ax[0].hist(x_dict[key], bins=bins, weights=1.39e8*x_dict['weight'], alpha=0.5, label="Full bkg", color='dimgray')
  h2 = ax[0].hist(x_cut1[key], bins=bins, weights=1.39e8*x_cut1['weight'], histtype='step',  label="Cut - 50%", color = 'mediumblue')
  h3 = ax[0].hist(x_cut2[key], bins=bins, weights=1.39e8*x_cut2['weight'], histtype='step', label="Cut - 2%", color = 'forestgreen')
  ax[0].set_yscale('log')
  ax[0].set_ylabel('Events')
  ax[0].legend()
  ax[0].set_title(key + "; 50% and 2% Cuts")
  #plt.subplot(2,1,2)
  ax[1].plot(bins[:-1],np.ones(len(bins)-1), linestyle='dashed', color = 'dimgray')
  ax[1].plot(bins[:-1],2*h2[0]/h1[0], drawstyle='steps', color='mediumblue')
  ax[1].plot(bins[:-1],50*h3[0]/h1[0], drawstyle='steps', color='forestgreen')
  ax[1].set_ylim(0,2)
  ax[1].set_xlabel('GeV')
  ax[1].set_ylabel('Ratio * (1/cut)')
  plt.savefig(plot_dir+key+'_'+tag+'.png')
  plt.clf()
  print("Saved cut distribution for", key)

def make_roc(fpr,tpr,auc,model=""):
  plt.plot(fpr,tpr,label="AUC = %0.2f" % auc)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("SVJ "+model+" ROC")
  plt.legend()
  plt.savefig(plot_dir+'roc_'+model+'_'+tag+'.png')
  plt.clf()
  print("Saved ROC curve for model", model)

def make_sic(fpr,tpr,auc,bkg, model=""):
  y = tpr[1:]/np.sqrt(fpr[1:])
  good = (y != np.inf) & (tpr[1:] > 0.08)
  ymax = max(y[good])
  ymax_i = np.argmax(y[good])
  sigEff = tpr[1:][good][ymax_i]
  qcdEff = fpr[1:][good][ymax_i]
  score_cut = np.percentile(bkg,100-(qcdEff*100))
  print("Max improvement: ", ymax)
  print("Sig eff: ", sigEff)
  print("Bkg eff: ", qcdEff)
  print("Score selection: ", score_cut)
  plt.plot(tpr[1:],y,label="AUC = %0.2f" % auc)
  plt.axhline(y=1, color='0.8', linestyle='--')
  plt.xlabel("Signal Efficiency (TPR)")
  plt.ylabel("Signal Sensitivity ($TPR/\sqrt{FPR}$)")
  plt.title("Significance Improvement Characteristic: "+model )
  plt.legend()
  plt.savefig(plot_dir+'sic_'+model+'_'+tag+'.png')
  plt.clf()
  print("Saved SIC for", model)
  return {'sicMax':ymax, 'sigEff': sigEff, 'qcdEff': qcdEff, 'score_cut': score_cut}

def make_grid_plot(values,title,method):
  #values must be 4 X 10

  fig,ax = plt.subplots(1,1)
  if (method.find("compare") != -1): img = ax.imshow(values, cmap='PiYG', vmin=-1, vmax=3)#norm=colors.LogNorm(vmin=0.1,vmax=10))
  else:
    if (title == "qcdEff"): img = ax.imshow(values,norm=colors.LogNorm(vmin=1e-7,vmax=1e-1))
    elif (title == "sigEff"): img = ax.imshow(values,vmin=-0.1,vmax=0.7)
    elif (title == "sensitivity_Inclusive" or title == "sensitivity_mT"): img = ax.imshow(values, norm=colors.LogNorm(vmin=1e-5,vmax=1.5))
    elif (title == "auc"): img = ax.imshow(values, vmin=0.8, vmax=0.9)
    elif (title == "sicMax"): img = ax.imshow(values, vmin=-2, vmax=20)
    else: img = ax.imshow(values, cmap='Wistia')

  # add text to table
  for (j,i),label in np.ndenumerate(values):
    if label == 0.0: continue
    if title == "qcdEff" or title == "sensitivity_Inclusive" or title == "sensitivity_mT": ax.text(i,j,'{0:.1e}'.format(label),ha='center', va='center', fontsize = 'x-small')
    elif title == "score_cut": ax.text(i,j,'{0:.3f}'.format(label),ha='center', va='center', fontsize = 'x-small')
    else: ax.text(i,j,'{0:.2f}'.format(label),ha='center', va='center', fontsize = 'x-small')

  # x-y labels for grid 
  x_label_list = ['1.0', '1.25', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '5.0', '6.0']
  y_label_list = ['0.2', '0.4', '0.6', '0.8']
  ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
  ax.set_xticklabels(x_label_list)
  ax.set_xlabel('Z\' Mass [TeV]')
  ax.set_yticks([0,1,2,3])
  ax.set_yticklabels(y_label_list)
  ax.set_ylabel('$R_{inv}$')
  
  ax.set_title(method+"; "+title)
  plt.savefig(plot_dir+'table_'+method+'_'+title+'_'+tag+'.png')
  print("Saved grid plot for", title)

def make_single_roc(rocs,aucs,ylabel):
  plt.plot(rocs[0],rocs[1],label=str(np.round(r,4))+", $\sigma$="+str(sigs)+": AUC="+str(np.round(aucs,3)))
  plt.xlabel('fpr')
  plt.ylabel(Ylabel)
  plt.title('ROC: '+saveTag)
  plt.figtext(0.7,0.95,"size="+str(sizeeach)+", nvars="+str(nInputs))
  plt.legend()
  plt.savefig(saveTag+'_roc_aucs_'+Ylabel.replace("/","")+'.pdf')
  plt.clf()

def plot_score(bkg_score, sig_score, remove_outliers=True, xlog=True, extra_tag=""):
  if remove_outliers:
    bkg_score,nb = detect_outliers(bkg_score)
    sig_score,ns = detect_outliers(sig_score)
  #bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=80)[1]
  #bkg_score = np.absolute(bkg_score)
  #sig_score = np.absolute(sig_score)
  bmax = max(max(bkg_score),max(sig_score))
  bmin = min(min(bkg_score),min(sig_score))
  if xlog and bmin == 0: bmin = 1e-9
  if xlog: bins = np.logspace(np.log10(bmin),np.log10(bmax),80)
  else: bins=np.histogram(np.hstack((bkg_score,sig_score)),bins=80)[1]
  #bins = np.linspace(500,4000,80)
  #plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg (-"+str(nb)+")", density=True)
  #plt.hist(sig_score, bins=bins, alpha=0.5, label="sig(-"+str(ns)+")", density=True)
  plt.hist(bkg_score, bins=bins, alpha=0.5, label="bkg", density=True)
  plt.hist(sig_score, bins=bins, alpha=0.5, label="sig", density=True)
  if xlog: plt.xscale('log')
  plt.yscale('log')
  plt.legend()
  plt.title("Anomaly Score " + extra_tag)
  plt.xlabel('Loss')
  plt.savefig(plot_dir+'score_'+extra_tag+'_'+tag+'.png')
  plt.clf()
  print("Saved score distribution for", extra_tag)

def correlation_plot(data1, data2, data1_name, data2_name, bin_dict, title):
  corr, _ = pearsonr(data1, data2)
  print("Pearson correlation coeff: ", corr)

  fig, ax = plt.subplots()
  binsx = bin_dict[data1_name]
  binsy = bin_dict[data2_name]
  h = ax.hist2d(data1,data2,bins=[binsx,binsy],norm=colors.LogNorm())
  fig.colorbar(h[3], ax=ax)
  ax.set_xlabel(data1_name)
  ax.set_ylabel(data2_name)
  ax.set_title(f'{title}: Corr = {corr:.2f}')
  plt.savefig(plot_dir+'corr_'+data1_name+'_'+data2_name+'_'+title+'_'+tag+'.png')
  plt.clf()
  print("Saved 2D plot of", data1_name, data2_name)

def plot_phi(phis,name,extra_tag):
  nphis = phis.shape[1]
  nevents = phis.shape[0]
  idx = [i for i in range(nphis)]*nevents

  print(extra_tag)
  phiT = phis.T
  print("n zeros = ", len(np.where(~phiT.any(axis=1))[0]))
  phis = phis.flatten()
  nbinsx = 10
  bin_width = max(phis)/nbinsx
  print("max: ", max(phis))
  print("bin_width", bin_width)
  phis[phis==0] = -bin_width 

  fig, ax = plt.subplots()
  h = ax.hist2d(phis,idx,bins=[nbinsx+1,nphis],norm=colors.LogNorm())
  print("xedges", h[1])
  fig.colorbar(h[3], ax=ax)
  ax.set_xlabel('Value')
  ax.set_ylabel('Index')
  ax.set_title('PFN Set Representation - '+name)
  plt.savefig(plot_dir+'phi2D_'+name+'_'+extra_tag+'_'+tag+'.png')
  plt.clf()
  print("Saved 2D plot of phi-rep for", extra_tag)

def plot_inputs(bkg, sig, variable_array):
  for i in range(len(variable_array)):
    plt.subplot(2,2,i%4+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(bkg[:,i], bins=30, alpha=0.5, density=True)  
    plt.hist(sig[:,i], bins=30, alpha=0.5, density=True)
    plt.title(variable_array[i])
    if (i%4 == 3):
      plt.savefig(plot_dir+'input_vars_'+str(i)+tag+'.png')
      plt.clf()

def plot_jz_input(vrs):
  mt = vrs[:,0]
  jz_vals = vrs[:,1]
  d = [mt]
  w = [np.ones(len(mt))]
  labels = ["All"]
  print(mt)
  print(jz_vals)
  for jz in range(364704, 364706):
    selection = jz_vals == jz #PS
    print(selection)
    dt = mt[selection]
    print(len(dt))
    #selection = (qcd["mcChannelNumber"] == jz) & (qcd["jet2_Width"]<0.05) #CR
    #selection = (qcd["mcChannelNumber"] == jz) & (qcd["jet2_Width"]>0.05) & (qcd["score"]<0.6) #VR
    #selection = (qcd["mcChannelNumber"] == jz) & (qcd["jet2_Width"]>0.05) & (qcd["score"]>0.6) #SR
    d.append(dt)
    w.append(np.ones(len(dt)))
    labels.append(str(jz))
 
  plot_single_variable(d,w,labels, "mT_jj", logy=True)
  
def plot_single_variable(hists, weights, h_names, title, logy=False):
  nbins=100
  hists_flat=np.concatenate(hists)
  bin_min=np.min(hists_flat)
  bin_max=np.max(hists_flat)
  bins=np.linspace(bin_min,bin_max,nbins)
  if(title=="mT_jj"): bins=np.linspace(1500,2500,100)
  if(title=="rT"): bins=np.linspace(0,1.0, nbins)
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  for data,weight,name in zip(hists,weights,h_names):
    if name == "All": h,b,_ = ax.hist(data, bins=bins, histtype='step', label=name, density=False, weights=weight, color='black')
    else: h,b,_ = ax.hist(data, bins=bins, histtype='step', label=name, density=False, weights=weight)
  plt.legend(loc='upper right', fontsize='x-small')
  if (logy): ax.set_yscale("log")
  plt.title(title)
  #ax.set_xticks(np.arange(2000,8001,1000))
  #ax.set_xticks(np.arange(1600,8000,200), minor=True)
  #ax.set_yticks(np.logspace(-3,3,num=7))
  #print(np.logspace(-3,3,num=7))
  #yminor = [(10**j)*i for j in range(-3,4) for i in range(2,9,2)]
  #print(yminor)
  #ax.set_yticks(yminor, minor=True)
  #ax.grid(which='minor', alpha=0.2) 
  #ax.grid(which='major', alpha=0.5) 
  plt.savefig(plot_dir+'hist_'+title.replace(" ","")+'_'+tag+'.png')
  plt.clf()
  print("Saved plot",title)

def plot_simple_ratio(hists, weights, h_names, title, logy=False):
  colors = ['black', 'firebrick', 'darkgreen','limegreen','darkblue', 'deepskyblue' ]
  #colors = ['firebrick', 'darkgreen','limegreen' ]
  nbins=50
  f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
  hists_flat=np.concatenate(hists)
  bin_min=np.min(hists_flat)
  bin_max=np.max(hists_flat)
  bins=np.linspace(bin_min,bin_max,nbins)
  if (title == 'mT_jj'): bins=np.linspace(1500,5000,nbins)
  x_bins = bins[1:]
  for data,weight,name,i in zip(hists,weights,h_names, range(len(hists))):
    y,_, _=axs[0].hist(data, bins=bins, label=f'{name} ({np.sum(weight):0.2e})', density=True, histtype='step', weights=weight, color=colors[i])
    mid = 0.5*(bins[1:] + bins[:-1])
    axs[0].errorbar(mid, y, yerr=np.sqrt(y)/np.sum(weight), fmt='none')
    if i ==0:
      y0=y # make sure the first of hists list has the most number of events
      continue
    else:
      axs[1].scatter(x_bins,y/y0, marker="+", color=colors[i])
  axs[1].set_ylim(0.5,1.5)  
  axs[1].set_ylabel('Ratio')
  plt.tick_params(axis='y', which='minor') 
  plt.grid()
 
  axs[0].set_ylabel('Event Number')
  if (logy): axs[0].set_yscale("log")
  axs[0].legend(loc='upper center', fontsize='x-small')
  axs[0].set_title(title)
  plt.savefig(plot_dir+'ratioSimple_'+title.replace(" ","")+'_'+tag+'.png')
  plt.clf()
  print("Saved simple rato plot",title)
   
def plot_ratio(hists, weights, h_names, title, logy=False, cumsum=False):
  colors = ['black', 'darkblue', 'deepskyblue', 'firebrick', 'orange']

  f, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1]})
  hists_flat=np.concatenate(hists)
  nbins=50
  bin_min=np.min(hists_flat)
  bin_max=np.max(hists_flat)
  #bins=np.linspace(bin_min,bin_max,nbins)
  #gap=(bin_max-bin_min)*0.05
  bins=np.linspace(0.0,0.25,26)
  #bins=np.linspace(1000,6500,50)
  #x_bins=bins[:-1]+ 0.5*(bins[1:] - bins[:-1])
  x_bins = bins[1:]
  hists=list(hists)
  for data,weight,name,i in zip(hists,weights,h_names, range(len(hists))):
    y,_, _=axs[0].hist(data, bins=bins, label=f'{name}', density=True, histtype='step', weights=weight, color=colors[i])
    #print(i, len(bins), len(y), bins, y) 
    #if i ==len(hists)-1:
    if i ==0:
      y0=y # make sure the first of hists list has the most number of events
      print("y0: ", y0)
      if cumsum:
        cdf0 = np.cumsum(y0[::-1])[::-1]
        print("cdf0: ", cdf0)
      continue
    if cumsum:
      cdf = np.cumsum(y[::-1])[::-1]
      vals = my_metric(cdf,cdf0)
      maxval = max(vals)
      binmax = bins[np.argmax(vals)]
      axs[1].scatter(x_bins,vals, marker="+", color=colors[i], label=f'SR > {binmax:.2f}')
      print("Best cut for ", name, " is ", binmax)
    else:
      axs[1].scatter(x_bins,my_metric(y,y0), marker="+", color=colors[i], label=f'{max(my_metric(y,y0)):.1e}')

  #axs[1].set_ylim(0.5,3.0)  
  axs[1].set_ylabel('Fig of Merit')
  axs[1].legend(loc='lower right', fontsize='x-small')
  plt.tick_params(axis='y', which='minor') 
  plt.grid()
 
  axs[0].set_ylabel('Event Number')
  if (logy): axs[0].set_yscale("log")
  axs[0].legend(loc='upper right', fontsize='x-small')
  #axs[1].legend(loc='upper right')
  axs[0].set_title(title)
  plt.savefig(plot_dir+'ratio_'+title.replace(" ","")+'_'+tag+'.png')
  plt.clf()
  print("Saved rato plot",title)

def get_nTracks(x):
  n_tracks = []
  for i in range(x.shape[0]):
    tracks = x[i,:,:].any(axis=1)
    tracks = tracks[tracks == True]
    n_tracks.append(len(tracks))
  return n_tracks
 
def plot_nTracks(bkg, extra_tag=""):
  bkg_tracks = get_nTracks(bkg)
  #bins=np.histogram(np.hstack((bkg_tracks,sig_tracks)),bins=60)[1]
  bins = np.arange(0,160,1)
  plt.hist(bkg_tracks,alpha=0.5, label=extra_tag, histtype='step', bins=bins, density=False)
  plt.title("nTracks "+extra_tag)
  plt.legend()
  plt.tight_layout()
  plt.savefig(plot_dir+'nTracks_'+extra_tag+tag+'.png')
  plt.clf()
  print("Saved plot of nTracks")

def plot_vectors(train,sig,extra_tag):
  variable_array = ["pT", "eta", "phi"]
  if (len(train.shape) == 3):
    train = train.reshape(train.shape[0], train.shape[1] * train.shape[2])
  if (len(sig.shape) == 3):
    sig = sig.reshape(sig.shape[0], sig.shape[1] * sig.shape[2])

  for i in range(3):
    train_v = train[:,i::3].flatten()
    #test_v = test[:,i::4].flatten()
    sig_v = sig[:,i::3].flatten()
    bins=np.histogram(np.hstack((train_v,sig_v)),bins=60)[1]
    if(bins[-1] > 3000): bins = np.arange(0,3000,50)
    plt.subplot(4,2,i+1)
    plt.tight_layout(h_pad=1, w_pad=1)
    plt.hist(train_v, alpha=0.5, label="bkg", bins=bins, density=False, histtype='step')
    #plt.hist(test_v, alpha=0.5, label="test", bins=bins, density=True, color='lightskyblue')
    plt.hist(sig_v, alpha=0.5, label="sig", bins=bins, density=False, histtype='step')
    plt.yscale('log')
    plt.title(variable_array[i])
    if i == 1: plt.legend()
  plt.savefig(plot_dir+'inputs_'+extra_tag+'_'+tag+'.png')
  plt.clf()
  print("Saved inputs plot (", extra_tag, ")")



