# ANomaly deTEction on particLe flOw latent sPacE (ANTELOPE)

Code for training & evaluating the ANTELOPE architecture.

## Files

Input files can be found from the [LHC Olympics](https://lhco2020.github.io/homepage/) ([R&D](https://zenodo.org/records/6466204), [black boxes](https://zenodo.org/records/4536624)).

## Conda environment

Create a conda enviroment from the environment.yaml file. Key packages are:
`h5py tensorflow keras numpy matplotlib pandas scikit-learn uproot`

## Running & Evaluating

Training and evaluating are done in two separate steps for ANTELOPE:

### 1) PFN

Training is done with the file `pfn_train.py`, in the conda environment.
```
python pfn_train.py
```

You can save your trained network, and evaluate using the `pfn_evaluate.py` script.
```
python pfn_evaluate.py
```

### 2) VAE on PFN Latent Space (ANTELOPE)

Training is done with the file `antelope_train.py`, in the conda environment.
```
python antelope_train.py
```

You can save your trained network, and evaluate using the `antelope_evaluate.py` script.
```
python antelope_evaluate.py
```

**Note**: plot path is hardcoded in `plot_heler.py`

## File descriptions
`models.py`: model architectures  
`pfn_train.py`: train PFN  
`antelope_train.py`: train ANTELOPE (requires trained PFN)  
`pfn_evaluate`: evaluate trained PFN model on more data and save HDF5s  
`antelope_evaluate.py`: evaluate trained ANTELOPE model on more data and save HDF5s  
`eval_helper.py`: functions used in training and evaluation, including to read in and apply selections to inputs  
`plot_helper.py`: plotting scripts  
`reader.py`: functions to load data from hdf5 (or nTuple) inputs  
