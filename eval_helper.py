import numpy as np
from reader import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import MaxAbsScaler
from plot_helper import *
from models import *
import json

def getTwoJetSystem(x_events,input_file, extraVars=[], use_weight=True, idx_range=[]):
    getExtraVars = len(extraVars) > 0
    
    track_array = ["pt", "eta", "phi"]

    print("Reading in data...")
    bkg_in = read_vectors_h5(input_file, x_events, track_array, idx_range=idx_range)

    if getExtraVars: 
        vars_bkg = read_flat_vars(input_file, x_events, extraVars, use_weight=use_weight, idx_range=idx_range)
        print("vars_bkg", vars_bkg) 
   
    print("Loaded data")
    print("Selecting tracks...")
    bkg, selection = apply_TrackSelection(bkg_in)

    # select events which have both valid leading and subleading jet tracks
    if getExtraVars:
        vars_bkg = vars_bkg[selection]    
        print("vars_bkg[selection]", vars_bkg) 

    if getExtraVars: return bkg, vars_bkg
    else: return bkg

def check_weights(x_events):
    #bkg_nw1 = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 10000, ["jet1_pt"], use_weight=False)
    bkg_nw = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 500000, ["mT_jj"], use_weight=True)
    bkg_w = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 100000, ["mT_jj"], use_weight=True)
    sig_nw = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 10000, ["mT_jj"], use_weight=True)
    #sig_nw2 = read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 5000, ["jet1_pt"], use_weight=True)
    plot_single_variable([bkg_nw,bkg_w, sig_nw], ["QCD - 500k", "QCD - 100k", "QCD - 10k"], "mT Stat Check", logy=True) 

def get_dPhi(x1,x2):
    dPhi = x1 - x2
    if(dPhi > 3.14):
        dPhi -= 2*3.14
    elif(dPhi < -3.14):
        dPhi += 2*3.14
    return dPhi

def remove_zero_padding(x):
    #x has shape (nEvents, nSteps, nFeatures)
    #x_out has shape (nEvents, nFeatures)
    x_nz = np.any(x,axis=2) #find zero padded steps
    x_out = x[x_nz]

    return x_out

def reshape_3D(x, nTracks, nFeatures):
    print(x[4])
    x_out = x.reshape(x.shape[0],nTracks,nFeatures)
    print(x_out[4])
    return x_out

def pt_sort(x):
    for i in range(x.shape[0]):
        ev = x[i]
        x[i] = ev[ev[:,0].argsort()]
    #y = x[:,-60:,:]
    return x

def apply_TrackSelection(x_raw):
    """
    Ensures at least 3 tracks per event
    """
    selection = np.count_nonzero(x_raw[:,:,0], axis=1) >= 3
    return x_raw[selection], selection

def apply_StandardScaling(x_raw, scaler=MinMaxScaler(), doFit=True):
    x= np.zeros(x_raw.shape)
    
    x_nz = np.any(x_raw,axis=len(x_raw.shape)-1) #find zero padded events 
    x_scale = x_raw[x_nz] #scale only non-zero jets
    #scaler = StandardScaler()
    if (doFit): scaler.fit(x_scale) 
    x_fit = scaler.transform(x_scale) #do the scaling
    
    x[x_nz]= x_fit #insert scaled values back into zero padded matrix
    
    return x, scaler

def apply_EventScaling(x_raw):
    
    x = np.copy(x_raw) #copy

    x_totals = x_raw.sum(axis=1) #get sum total pt, eta, phi, E for each event
    x[:,:,0] = (x_raw[:,:,0].T/x_totals[:,0]).T  #divide each pT entry by event pT total
    x[:,:,3] = (x_raw[:,:,3].T/x_totals[:,3]).T  #divide each E entry by event E total

    return x

def get_multi_loss(model_svj, x_test, y_test):
    bkg_total_loss = []
    sig_total_loss = []
    bkg_kld_loss = []
    sig_kld_loss = []
    bkg_reco_loss = []
    sig_reco_loss = []
    nevents = min(len(y_test),len(x_test))
    step_size = 1
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        yt = y_test[i:i+step_size]
      
        # NOTE - unclear why they are printed in this order, but it seems to be the case
        x_loss,x_reco,x_kld = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
        y_loss,y_reco,y_kld = model_svj.evaluate(yt, batch_size = step_size, verbose=0)
      
        bkg_total_loss.append(x_loss)
        sig_total_loss.append(y_loss)
        bkg_kld_loss.append(x_kld)
        sig_kld_loss.append(y_kld)
        bkg_reco_loss.append(x_reco)
        sig_reco_loss.append(y_reco)
        if i%100 == 0: print("Processed", i, "events")

    return bkg_total_loss, sig_total_loss, bkg_kld_loss, sig_kld_loss, bkg_reco_loss, sig_reco_loss

def get_multi_loss_each(model_svj, x_test):
    bkg_total_loss = []
    bkg_kl_loss = []
    bkg_reco_loss = []
    nevents = len(x_test)
    step_size = 1
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        # NOTE - unclear why they are printed in this order, but it seems to be the case
        x_loss,x_reco,x_kl = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
      
        bkg_total_loss.append(x_loss)
        bkg_kl_loss.append(x_kl)
        bkg_reco_loss.append(x_reco)
        if i%100 == 0: print("Processed", i, "events")

    bkg_total_loss,  bkg_kl_loss,  bkg_reco_loss = np.array(bkg_total_loss),  np.array( bkg_kl_loss), np.array(bkg_reco_loss)
    return bkg_total_loss, bkg_kl_loss,  bkg_reco_loss


def get_single_loss(model_svj, x_test, y_test):
    bkg_loss = []
    sig_loss = []
    nevents = min(len(y_test),len(x_test))
    step_size = 4
    for i in range(0,nevents, step_size):
        xt = x_test[i:i+step_size]
        yt = y_test[i:i+step_size]
    
        x_loss = model_svj.evaluate(xt, batch_size = step_size, verbose=0)
        y_loss = model_svj.evaluate(yt, batch_size = step_size, verbose=0)
        
        bkg_loss.append(x_loss)
        sig_loss.append(y_loss)
        if i%100 == 0: print("Processed", i, "events")

    return bkg_loss, sig_loss

def transform_loss(bkg_loss, sig_loss, make_plot=False, plot_tag=''):
    nevents = len(sig_loss)
    truth_sig = np.ones(nevents)
    truth_bkg = np.zeros(nevents)
    truth_labels = np.concatenate((truth_bkg, truth_sig))
    eval_vals = np.concatenate((bkg_loss,sig_loss))
    eval_min = min(eval_vals)
    eval_max = max(eval_vals)-eval_min
    eval_transformed = [(x - eval_min)/eval_max for x in eval_vals]
    bkg_transformed = [(x - eval_min)/eval_max for x in bkg_loss]
    sig_transformed = [(x - eval_min)/eval_max for x in sig_loss]
    if make_plot:
        plot_score(bkg_transformed, sig_transformed, False, False, plot_tag+'_Transformed')
    return truth_labels, eval_vals 

def transform_loss_sig(bkg_loss, sig_loss, make_plot=False, plot_tag=''):
    nevents = len(sig_loss)
    truth_sig = np.ones(nevents)
    truth_bkg = np.zeros(nevents)
    truth_labels = np.concatenate((truth_bkg, truth_sig))
    eval_vals = np.concatenate((bkg_loss,sig_loss))
    eval_transformed = [1/(1+np.exp(-np.log10(x))) for x in eval_vals]
    bkg_transformed = [1/(1+np.exp(-np.log10(x))) for x in bkg_loss]
    sig_transformed = [1/(1+np.exp(-np.log10(x))) for x in sig_loss]
    #eval_transformed = [np.log10(x) for x in eval_vals]
    #bkg_transformed = [np.log10(x) for x in bkg_loss]
    #sig_transformed = [np.log10(x) for x in sig_loss]
    if make_plot:
        plot_score(bkg_loss, sig_loss, False, False, plot_tag+'_Orig')
        plot_score(bkg_transformed, sig_transformed, False, False, plot_tag+'_TransformedLog')
    return truth_labels, eval_vals 

def vrnn_transform(bkg_loss, sig_loss, make_plot=False, plot_tag=''):
    train_mean = np.mean(bkg_loss)
    bkg_loss_p = [1-x/(2*train_mean) for x in bkg_loss]
    sig_loss_p = [1-x/(2*train_mean) for x in sig_loss]
    if make_plot:
        plot_score(bkg_loss_p, sig_loss_p, False, False, plot_tag+'_MeanShift')
    return bkg_loss_p, sig_loss_p    

def getSignalSensitivityScore(bkg_loss, sig_loss, percentile=95):
    nSigAboveThreshold = np.sum(sig_loss > np.percentile(bkg_loss, percentile))
    return nSigAboveThreshold / len(sig_loss)

def applyScoreCut(loss,test_array,cut_val):
    return test_array[loss>cut_val] 

def do_roc(bkg_loss, sig_loss, plot_tag, make_transformed_plot=False):
    truth_labels, eval_vals = transform_loss_sig(bkg_loss, sig_loss, make_plot=make_transformed_plot, plot_tag=plot_tag) 
    fpr, tpr, trh = roc_curve(truth_labels, eval_vals) #[fpr,tpr]
    auc = roc_auc_score(truth_labels, eval_vals)
    print("AUC - "+plot_tag+": ", auc)
    make_roc(fpr,tpr,auc,plot_tag)
    sic_vals = make_sic(fpr,tpr,auc,bkg_loss,plot_tag)
    sic_vals['auc'] = auc
    return sic_vals

def do_grid_plots(sic_vals, title):
    with open("dsids_grid_locations.json", "r") as f:
      dsid_coords = json.load(f)
    dsids = list(sic_vals.keys())
    vals = list(sic_vals[dsids[0]].keys())
    for val in vals:
        values = np.zeros([4,10])
        for dsid in dsids:
            loc = tuple(dsid_coords[str(dsid)])
            values[loc] = sic_vals[dsid][val]
        make_grid_plot(values, val, title)
