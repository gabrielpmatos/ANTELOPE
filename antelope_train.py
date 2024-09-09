import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import *
from reader import *
from plot_helper import *
from eval_helper import *

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Example usage
encoding_dim = 32
latent_dim = 12
phi_dim = 64
nepochs=50
batchsize_ae=32

pfn_model = 'PFN_LHCO_v2'
ae_model = 'vANTELOPE_LHCO_v2'
arch_dir = "./architectures_saved/lhco/"

################### Train the AE ###############################
graph = keras.models.load_model(arch_dir+pfn_model+'_graph_arch')
graph.load_weights(arch_dir+pfn_model+'_graph_weights.h5')
graph.compile()

## AE events
x_events = 100000
y_events = 20000 #20%
#p bkg_file = "../v8.1/skim3.user.ebusch.QCDskim.root"
#p sig_file = "../v8.1/skim3.user.ebusch.SIGskim.root"
bkg_file = "/data/users/gpm2117/SVJ/antelope_datasets/antelope-datasets/hdf5s/lhco/lhco_qcd_fullstat.h5"
sig_file = "/data/users/gpm2117/SVJ/antelope_datasets/antelope-datasets/hdf5s/lhco/lhco_two_prong.h5"

bkg2 = getTwoJetSystem(x_events * 10,bkg_file,[],use_weight=False)
sig2 = getTwoJetSystem(y_events,sig_file,[],use_weight=False)

bkg2 = bkg2[x_events : x_events * 2]

scaler = load(arch_dir+pfn_model+'_scaler.bin')
bkg2,_ = apply_StandardScaling(bkg2,scaler,False)
sig2,_ = apply_StandardScaling(sig2,scaler,False)
plot_vectors(bkg2,sig2,"vANTELOPE")

phi_bkg = graph.predict(bkg2)
phi_sig = graph.predict(sig2)

#plot_score(phi_bkg[:,11], phi_sig[:,11], False, False, "phi_11_raw")

phi_evalb, phi_testb, _, _ = train_test_split(phi_bkg, phi_bkg, test_size=sig2.shape[0])
plot_phi(phi_evalb,"Train","PFN_phi_train_raw")
plot_phi(phi_testb,"Test","PFN_phi_test_raw")
plot_phi(phi_sig,"Signal","PFN_phi_sig_raw")

# We should NOT scale the phis
"""
eval_max = np.amax(phi_evalb)
eval_min = np.amin(phi_evalb)
sig_max = np.amax(phi_sig)
print("Min: ", eval_min)
print("Max: ", eval_max)
if (sig_max > eval_max): eval_max = sig_max
print("Final Max: ", eval_max)

phi_evalb = (phi_evalb - eval_min)/(eval_max-eval_min)
phi_testb = (phi_testb - eval_min)/(eval_max-eval_min)
phi_sig = (phi_sig - eval_min)/(eval_max-eval_min)
"""

#phi_evalb, phi_scaler = apply_StandardScaling(phi_evalb)
#phi_testb, _ = apply_StandardScaling(phi_testb,phi_scaler,False)
#phi_sig, _ = apply_StandardScaling(phi_sig,phi_scaler,False)

plot_phi(phi_evalb,"train","PFN_phi_train")
plot_phi(phi_testb,"test","PFN_phi_test")
plot_phi(phi_sig,"sig","PFN_phi_sig")

ae = get_vae(phi_dim,encoding_dim,latent_dim)

h2 = ae.fit(phi_evalb,
    phi_evalb, 
    epochs=nepochs,
    batch_size=batchsize_ae,
    #validation_split=0.2,
    validation_data=(phi_testb, phi_testb),
    verbose=1)

# # simple ae
# ae.save(arch_dir+ae_model)
# print("saved model")

#complex ae
ae.get_layer('encoder').save_weights(arch_dir+ae_model+'_encoder_weights.h5')
ae.get_layer('decoder').save_weights(arch_dir+ae_model+'_decoder_weights.h5')
ae.get_layer('encoder').save(arch_dir+ae_model+'_encoder_arch')
ae.get_layer('decoder').save(arch_dir+ae_model+'_decoder_arch')
with open(arch_dir+ae_model+'_history.json', 'w') as f:
    json.dump(h2.history, f)
print("Saved model")


######## EVALUATE SUPERVISED ######
# # --- Eval plots 
# 1. Loss vs. epoch 
plot_loss(h2, ae_model, 'loss')

#2. Get loss
#bkg_loss, sig_loss = get_single_loss(ae, phi_testb, phi_sig)
"""
pred_phi_bkg = ae.predict(phi_testb)['reconstruction']
pred_phi_sig = ae.predict(phi_sig)['reconstruction']
bkg_loss = keras.losses.mse(phi_testb, pred_phi_bkg)
sig_loss = keras.losses.mse(phi_sig, pred_phi_sig)
"""

bkg_all_loss, sig_all_loss, bkg_kl_loss, sig_kl_loss, bkg_reco_loss, sig_reco_loss = get_multi_loss(ae, phi_testb, phi_sig)
bkg_losses = [bkg_all_loss, bkg_kl_loss, bkg_reco_loss]
sig_losses = [sig_all_loss, sig_kl_loss, sig_reco_loss]
loss_types = ["all", "kl", "reco"]

for bkg_loss, sig_loss, loss_type in zip(bkg_losses, sig_losses, loss_types):
    print(loss_type)
    #plot_score(bkg_loss, sig_loss, False, True, ae_model + f"_{loss_type}")
   
    # # 3. Signal Sensitivity Score
    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    do_roc(bkg_loss, sig_loss, ae_model + f"_{loss_type}", True)

    print("Taking log of score...")
    bkg_loss = np.log10(bkg_loss)
    sig_loss = np.log10(sig_loss)
    bkg_loss = sigmoid(bkg_loss)
    sig_loss = sigmoid(sig_loss)
    score = getSignalSensitivityScore(bkg_loss, sig_loss)
    print("95 percentile score = ",score)
    # # 4. ROCs/AUCs using sklearn functions imported above  
    #do_roc(bkg_loss, sig_loss, ae_model+f'_{loss_type}_log_sig', True)
    #plot_score(bkg_loss, sig_loss, False, True, ae_model + f"_{loss_type}_log_sig")

