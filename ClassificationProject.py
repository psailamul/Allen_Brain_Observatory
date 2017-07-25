# Classiication project
# Neural Response recorded from the same mouse in 3 regions  - VISl, VISal, VISp - at depth = 175um

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
from allensdk.brain_observatory.natural_scenes import NaturalScenes
import pprint
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_stimulus_table(stim_table, title):
    fstart = stim_table.start.min()
    fend = stim_table.end.max()
    
    fig = plt.figure(figsize=(15,1))
    ax = fig.gca()
    for i, trial in stim_table.iterrows():    
        x1 = float(trial.start - fstart) / (fend - fstart)
        x2 = float(trial.end - fstart) / (fend - fstart)            
        ax.add_patch(patches.Rectangle((x1, 0.0), x2 - x1, 1.0, color='r'))
    ax.set_xticks((0,1))
    ax.set_xticklabels((fstart, fend))
    ax.set_yticks(())
    ax.set_title(title)
    ax.set_xlabel("frames")
    
def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        # if possible, save metadata too ex stim table : metadata = data_set.get_metada()

def load_object(filename):
    with open(filename, 'rb') as inputfile: 
        return pickle.load( inputfile );
        
#import pdb; pdb.set_trace()
##############################################################################################
# PROOBLEM with download NWB file -->  https://github.com/AllenInstitute/AllenSDK/issues/22 
# http://api.brain-map.org/api/v2/data/OphysExperiment/501498760.xml?include=well_known_files
# http://api.brain-map.org/api/v2/well_known_file_download/514422179 --> then change file name to 5014987600.nwb

# Experiment Container ID 
EXP_CONTAINER_ID = [511510640, 511510715, 511510736] 
EXP_CONTAINER_ID=511510667 # Try with one region first
EXP_ID = 514422179 #501498760

download_time = time.time()
exps = boc.get_ophys_experiments(experiment_container_ids=[EXP_CONTAINER_ID]) 
print("Experiments for experiment_container_id %d: %d\n" % (EXP_CONTAINER_ID, len(exps)))
pprint.pprint(exps) #Check metadata
print "Download complete: Time %s" %(time.time() - download_time)

download_time = time.time()
data_set = boc.get_ophys_experiment_data(EXP_ID)  
pprint.pprint(data_set.get_metadata())
