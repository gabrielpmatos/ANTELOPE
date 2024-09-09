import pandas as pd
import numpy as np
import h5py
import argparse

save_dir = "./hdf5s"
n_events = 1000000 #1M
n_particles = 700
n_vars = 3

def convert(infile, outfile, n_tracks=160):
   df = pd.read_hdf(infile)
   arrays = df.to_numpy()
   # NOTE: comment next line out if you don't want to split by signal/background
   arrays = arrays[arrays[:, -1] == 0]
   arrays = arrays[: n_events, : n_particles * n_vars]
   arrays = arrays.reshape(-1, n_particles, n_vars)

   pt = arrays[:, :, 0]
   eta = arrays[:, :, 1]
   phi = arrays[:, :, 2]

   # Sorts by particle pT in descending order
   argsorted = np.argsort(pt, axis=1)[:,::-1]

   # Keeps only hardest 160 tracks
   pt = np.take_along_axis(pt, argsorted, axis=1)[:, : n_tracks]
   eta = np.take_along_axis(eta, argsorted, axis=1)[:, : n_tracks]
   phi = np.take_along_axis(phi, argsorted, axis=1)[:, : n_tracks]

   with h5py.File(f"{save_dir}/{outfile}.h5", "w") as f:
       for var_name, var in zip(["pt", "eta", "phi"], [pt, eta, phi]):
           f.create_dataset(var_name, data=var, compression="gzip")

# ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", help="Path to input HDF5 LHCO file to convert to ANTELOPE format")
    parser.add_argument("--outfile", default="outfile", help="Name of output HDF5 file")
    args = parser.parse_args()

    if args.infile is None:
        print("Please provide the path to an input HDF5 file")
        parser.print_help()
        exit()

    infile, outfile = args.infile, args.outfile
    convert(infile, outfile)

if __name__ == "__main__":
    main()

