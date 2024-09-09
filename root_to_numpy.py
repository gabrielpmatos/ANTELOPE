import uproot
import h5py
import numpy as np
import awkward as ak

def get_spaced_elements(arr_len,nElements):
    return np.round(np.linspace(0,arr_len-1, nElements)).astype(int)

def get_weighted_elements(tree, nEvents):
    weight_array=["weight"]
    my_weight_array = tree.arrays(weight_array, library = "np")
    my_weight_array = np.abs(my_weight_array[weight_array[0]])
    np.random.seed(1)
    idx = np.random.choice( my_weight_array.size,size= nEvents, p=my_weight_array/float(my_weight_array.sum()),replace=False) # IMPT that replace=False so that event is picked only once
    return idx

def read_flat_vars(infile, nEvents, variable_array, use_weight=True, idx_range=[]):
    file = uproot.open(infile)
    
    tree = file["PostSel"]
    
    # Read flat branches from nTuple
    my_array = tree.arrays(variable_array, library="np")
    if (nEvents == -1): nEvents = len(my_array[variable_array[0]])
    if (use_weight):
        idx = get_weighted_elements(tree, nEvents)
    else:
        if idx_range == []: idx = get_spaced_elements(len(my_array[variable_array[0]]),nEvents)
        else: idx = idx_range
    #print('Flat variable index:', idx.shape, idx)
    selected_array = np.array([val[idx] for _,val in my_array.items()]).T

    return selected_array

def read_vectors(infile, nEvents, jet_array, use_weight=True, idx_range=[]):
    file = uproot.open(infile)
    
    max_jets = 80

    if(infile.find("Small") != -1): myTree = "outTree"
    else: myTree = "PostSel"
    tree = file[myTree]

    # Read vector branches from nTuple
    my_jet_array = tree.arrays(jet_array, library = "np")
    if (nEvents == -1): nEvents = len(my_jet_array[jet_array[0]])
    if (use_weight):
        idx = get_weighted_elements(tree, nEvents)
    else:
        if idx_range == []: idx = get_spaced_elements(len(my_jet_array[jet_array[0]]),nEvents)
        else: idx = idx_range

    #print('Vector variable index:', idx.shape, idx)
    selected_jet_array = np.array([val[idx] for _,val in my_jet_array.items()]).T

    # create jet matrix
    padded_jet_array = np.zeros((len(selected_jet_array),max_jets,len(jet_array)))
    for jets,zeros in zip(selected_jet_array,padded_jet_array):
        jet_ar = np.stack(jets, axis=1)[:max_jets,:]
        zeros[:jet_ar.shape[0], :jet_ar.shape[1]] = jet_ar

    return padded_jet_array

def read_vectors_h5(infile, nEvents, track_variables, idx_range=[]):
    """
    Use if you want to read events from .h5 file
    Will return you an array with total number of events nEvents in format nEvents x nTracks x track_variable
    NOTE maybe want to add option to include weights? Not sure I understand those from Delphes at the moment...
    """
    print(infile)
    f = h5py.File(infile, "r")

    # Get evenly spaced nEvents number of elements from array
    if (nEvents == -1): nEvents = len(f[track_variables[0]][:])
    if idx_range == []: idx = get_spaced_elements(len(f[track_variables[0]][:]),nEvents)
    else: idx = idx_range

    # The format of this is nEvents x nTracks x track variable
    track_array = []
    for track_variable in track_variables:
        track_array.append(f[track_variable][:])

    track_array = np.stack(track_array, axis=2)

    # Select evenly spaced events
    track_array = track_array[idx]

    return track_array
    
def main():
    read_flat_vars("../v12.5/user.ebusch.QCD.root", 500000, ['mT_jj', 'met_met'])
    read_flat_vars("../v8.1/user.ebusch.QCDskim.mc20e.root", 200, ['mT_jj', 'met_met'])
    read_vectors("../v8.1/user.ebusch.QCDskim.mc20e.root", 100, ['jet0_GhostTrack_pt'])

if __name__ == '__main__':
    main()
