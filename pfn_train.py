# svj_pfn.py
# svj_antelope.py
# plot_helper.py

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from joblib import dump, load
from models import *
from root_to_numpy import *
from plot_helper import *
from eval_helper import *

# Example usage
nevents = 100000
num_elements = 160
element_size = 3
encoding_dim = 32
latent_dim = 4
phi_dim = 64
nepochs= 100
batchsize_pfn=500
batchsize_ae=32

pfn_model = 'PFN_LHCO_v2'
arch_dir = "./architectures_saved/lhco/"

## Load leading two jets
bkg_file = "/data/users/gpm2117/SVJ/antelope_datasets/antelope-datasets/hdf5s/lhco/lhco_qcd_fullstat.h5"
sig_file = "/data/users/gpm2117/SVJ/antelope_datasets/antelope-datasets/hdf5s/lhco/lhco_two_prong.h5"
#plot_single_variable([bkgpt], [np.ones(len(bkgpt))], ["totalBKG"],"met_met",True)

bkg = getTwoJetSystem(nevents * 10, bkg_file,use_weight=False)
sig = getTwoJetSystem(nevents,sig_file,use_weight=False)

# NOTE: to keep samples orthogonal, we pick only the first 100k of QCD
bkg = bkg[ : nevents]

print("Total bkg shape:", bkg.shape)
print("Sig shape:", sig.shape)

# Plot inputs
plot_vectors(bkg,sig,"PFNinput")
#check_weights(nevents)
plot_nTracks(bkg, "bkg")
plot_nTracks(sig, "sig")

# Create truth target
input_data = np.concatenate((bkg,sig),axis=0)

truth_bkg = np.zeros(bkg.shape[0])
truth_sig = np.ones(sig.shape[0])

truth_1D = np.concatenate((truth_bkg,truth_sig))
truth = tf.keras.utils.to_categorical(truth_1D, num_classes=2)

print("Training shape, truth shape")
print(input_data.shape, truth.shape)

# Load the model
pfn,graph_orig = get_full_PFN([num_elements,element_size], phi_dim)
#pfn = get_dnn(160)

# Split the data NOTE
x_train, x_test, y_train, y_test = train_test_split(input_data, truth, test_size=0.2)
#X_train, X_val, Y_train, Y_val = train_test_split(x_eval, y_eval, test_size=0.2)

# Save to use the test set later
np.save(pfn_model + "_test.npy", x_test)
np.save(pfn_model + "_test_key.npy", y_test)

# Fit scaler to training data, apply to testing data
x_train, scaler = apply_StandardScaling(x_train)
dump(scaler, arch_dir+pfn_model+'_scaler.bin', compress=True) #save the scaler
x_test,_ = apply_StandardScaling(x_test,scaler,False)

# Check the scaling & test/train split
bkg_train_scaled = x_train[y_train[:,0] == 1]
sig_train_scaled = x_train[y_train[:,0] == 0]
bkg_test_scaled = x_test[y_test[:,0] == 1]
sig_test_scaled = x_test[y_test[:,0] == 0]
plot_vectors(bkg_train_scaled,sig_train_scaled,"PFNtrain")
plot_vectors(bkg_test_scaled,sig_test_scaled,"PFNtest")

# Train
h = pfn.fit(x_train, y_train,
    epochs=nepochs,
    batch_size=batchsize_pfn,
    #validation_split=0.2,
    validation_data=(x_test, y_test),
    verbose=1)

# Save the model
pfn.get_layer('graph').save_weights(arch_dir+pfn_model+'_graph_weights.h5')
pfn.get_layer('classifier').save_weights(arch_dir+pfn_model+'_classifier_weights.h5')
pfn.get_layer('graph').save(arch_dir+pfn_model+'_graph_arch')
pfn.get_layer('classifier').save(arch_dir+pfn_model+'_classifier_arch')
with open(arch_dir+pfn_model+'_history.json', 'w') as f:
    json.dump(h.history, f)
print("Saved model")

## PFN training plots
# 1. Loss vs. epoch 
plot_loss(h, pfn_model, 'loss')
# 2. Score 
preds = pfn.predict(x_test)
bkg_score = preds[:,1][y_test[:,1] == 0]
sig_score = preds[:,1][y_test[:,1] == 1]
plot_score(bkg_score, sig_score, False, False, pfn_model)
n_test = min(len(sig_score),len(bkg_score))
bkg_score = bkg_score[:n_test]
sig_score = sig_score[:n_test]
do_roc(bkg_score, sig_score, pfn_model, False)
