import numpy as np
import json
from root_to_numpy import *
from tensorflow import keras
from tensorflow import saved_model
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from plot_helper import *
from models import *
from models_archive import *
from eval_helper import *
import h5py

## ---------- USER PARAMETERS ----------
## Model options:
##    "AE", "VAE", "PFN_AE", "PFN_VAE"
pfn_models = ['PFN_LHCO_v2']
vae_models = ["vANTELOPE_LHCO_v2"]
arch_dir = "/data/users/gpm2117/SVJ/svj-vae/architectures_saved/lhco/"
arch_dir_vae = "/data/users/gpm2117/SVJ/svj-vae/architectures_saved/lhco/"
data_path = "/data/users/gpm2117/SVJ/antelope_datasets/antelope-datasets/hdf5s/lhco/"
#pfn_model = 'PFNv6'
x_events = -1
myCernID = "ebusch"

## evaluate bkg
for pfn_model in pfn_models:
  for vae_model in vae_models:

    ## ---------- Load graph model ----------
    graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
    graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
    graph.compile()
    
    ## ---------- Load AE model ----------
    encoder = keras.models.load_model(arch_dir_vae+vae_model+'_encoder_arch')
    decoder = keras.models.load_model(arch_dir_vae+vae_model+'_decoder_arch')
    vae = VAE(encoder,decoder)

    vae.get_layer('encoder').load_weights(arch_dir_vae+vae_model+'_encoder_weights.h5')
    vae.get_layer('decoder').load_weights(arch_dir_vae+vae_model+'_decoder_weights.h5')

    vae.compile(optimizer=keras.optimizers.Adam())

    files = ["lhco_bb2", "lhco_bb1", "lhco_bb3", "lhco_qcd", "lhco_two_prong", "lhco_three_prong"]
    #files = ["lhco_bb1", "lhco_bb3"]
    #files = ["lhco_qcd", "lhco_two_prong", "lhco_three_prong", "lhco_bb2"] 

    ## evaluate all files
    for myFile in files:
      print("-------> Evaluating", myFile)
      outfile = myFile + "_evald_v2_fullstat.h5"
      print(outfile)
  
      my_variables = []

      if myFile == "lhco_qcd":
        bkg2 = np.load(pfn_model + "_test.npy")
        bkg2_key = np.load(pfn_model + "_test_key.npy")
        bkg2 = bkg2[bkg2_key[:, 0] == 1]

      elif myFile == "lhco_two_prong":
        bkg2 = np.load(pfn_model + "_test.npy")
        bkg2_key = np.load(pfn_model + "_test_key.npy")
        bkg2 = bkg2[bkg2_key[:, 1] == 1]
      else:
        bkg2 = getTwoJetSystem(x_events, data_path + myFile + ".h5", my_variables, False)

      """
      if "bb1" or "bb2" in myFile:
        key = np.loadtxt(f"/data/users/gpm2117/SVJ/antelope_datasets/antelope-datasets/hdf5s/lhco/masterkeys/{myFile}.masterkey")
        bkg2 = bkg2[key != 0]
        myFile += "_signal"          
      """

      scaler = load(arch_dir+pfn_model+'_scaler.bin')
      bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
 
      phi_bkg = graph.predict(bkg2)
      pred_phi_bkg = vae.predict(phi_bkg)['reconstruction']

      # AE loss
      bkg_loss, _, _ = get_multi_loss_each(vae, phi_bkg)
      bkg_loss = np.log10(bkg_loss)
      bkg_loss = 1/(1 + np.exp(-bkg_loss))
      
      my_variables.insert(0,"anomaly_score")
      save_bkg = bkg_loss[:, None]

      #print(save_bkg)
      ds_dt = np.dtype({'names':my_variables,'formats':[(float)]*len(my_variables)})
      rec_bkg = np.rec.array(save_bkg, dtype=ds_dt)

      with h5py.File(outfile,"w") as h5f:
        dset = h5f.create_dataset("data",data=rec_bkg)
      print("Saved hdf5 for", outfile)

