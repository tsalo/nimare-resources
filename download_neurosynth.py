"""
Download Neurosynth dataset and add abstracts.
"""
import os
from neurosynth.base.dataset import download
import nimare

# Download Neurosynth
if not os.path.isfile('resources/database.txt'):
    download('resources/', unpack=True)

# Convert Neurosynth database files to NiMARE Dataset
dset = nimare.io.convert_neurosynth_to_dataset(
    'resources/database.txt',
    'resources/features.txt')

# Add article abstracts to Dataset and save to file
dset = nimare.extract.download_abstracts(dset, 'tsalo006@fiu.edu')
dset.save('resources/neurosynth_with_abstracts.pkl.gz')
