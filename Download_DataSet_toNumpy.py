# Download Data from Allen Institute's brain_observatory and change to numpy type
# Input: None
#
# Output:
##############################################################################################

##############################################################################################
# Refer to this tutorial
# https://alleninstitute.github.io/AllenSDK/_static/examples/nb/brain_observatory_stimuli.html 
# Experiment_container_id: 511510667
# Experiment id: 501498760
##############################################################################################


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
        
import pdb; pdb.set_trace()
##############################################################################################
# PROOBLEM with download NWB file -->  https://github.com/AllenInstitute/AllenSDK/issues/22 
# http://api.brain-map.org/api/v2/data/OphysExperiment/501498760.xml?include=well_known_files
# http://api.brain-map.org/api/v2/well_known_file_download/514422179 --> then change file name to 5014987600.nwb

# Download Experiment Container for specific ID
EXP_CONTAINER_ID=511510667
EXP_ID = 514422179 #501498760

download_time = time.time()
exps = boc.get_ophys_experiments(experiment_container_ids=[EXP_CONTAINER_ID]) 
print("Experiments for experiment_container_id %d: %d\n" % (EXP_CONTAINER_ID, len(exps)))
pprint.pprint(exps) #Check metadata
print "Download complete: Time %s" %(time.time() - download_time)

download_time = time.time()
data_set = boc.get_ophys_experiment_data(EXP_ID)  
pprint.pprint(data_set.get_metadata())
print "Download complete: Time %s" %(time.time() - download_time)

##############################################################################################
# Natural Scene
##############################################################################################
# Show Image for natural scenes
scene_nums = [0, 83,117]

# read in the array of images
scenes = data_set.get_stimulus_template('natural_scenes') # ---> Here = scenes data

# display a couple of the scenes
fig, axes = plt.subplots(1,len(scene_nums))
for ax,scene in zip(axes, scene_nums):
    ax.imshow(scenes[scene,:,:], cmap='gray')
    ax.set_axis_off()
    ax.set_title('scene %d' % scene)
plt.show()

# To find where each scene occur
#data_set = boc.get_ophys_experiment_data(501498760)

# the natural scenes stimulus table describes when each scene is on the screen
stim_table = data_set.get_stimulus_table('natural_scenes')  ## --- Column : Frame Start End     Start = Time? or  ID? (Me : probably time)

# build up a mask of trials for which one of a list of scenes is visible
trial_mask = stim_table.frame == -2
for scene in scene_nums:
    trial_mask |= (stim_table.frame == scene)
stim_table = stim_table[trial_mask]

# plot the trials
plot_stimulus_table(stim_table, "scenes %s " % scene_nums)



# Analyze natural scene stimulus
import time
download_time = time.time()
#ns = NaturalScenes(data_set)
print("done analyzing natural scenes") # Take about 5 min
print "Time %s" %(time.time() - download_time)
#Download saved Input  
ns=load_object("NS_ECI511510667_Exp501498760.pkl")

#the natural scenes stimulus table describes when each scene is on the screen
scene_nums = np.arange(ns.shape[0])
stim_table = data_set.get_stimulus_table('natural_scenes')

# build up a mask of trials for which one of a list of scenes is visible
trial_mask = stim_table.frame == -2
for scene in scene_nums:
    trial_mask |= (stim_table.frame == scene)
stim_table = stim_table[trial_mask] # saved ---> natural scene info

# plot the trials
plot_stimulus_table(stim_table, "scenes %s " % scene_nums)


##################################
# Get Neural Response data 
##################################

#get dF/F trace
timestamp, df_f = data_set.get_dff_traces() # --> dff of all cells

#Note : for df_f ---> len(df_f) = # of cells  then df_f[0] = traces for that cell --> unit of time? 
#print dF/F trace 

all_cells_id =data_set.get_cell_specimen_ids() # List of cells ID of all cells in this experiment 

#specimen_id = 517425074 # --> for one cell

cell_id = all_cells_id[0]
time, raw_traces = data_set.get_fluorescence_traces(cell_specimen_ids=[cell_id])
_, neuropil_traces = data_set.get_neuropil_traces(cell_specimen_ids=[cell_id])
_, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=[cell_id])
_, dff_traces = data_set.get_dff_traces(cell_specimen_ids=[cell_id])

# plot raw and corrected ROI trace
plt.figure(figsize=(14,4))
plt.title("Raw Fluorescence Trace")
plt.plot(time, raw_traces[0])
plt.show()

plt.figure(figsize=(14,4))
plt.title("Neuropil-corrected Fluorescence Trace")
plt.plot(time, corrected_traces[0])
plt.show()


plt.figure(figsize=(14,4))
plt.title("dF/F Trace")
# warning: dF/F can occasionally be one element longer or shorter 
# than the time stamps for the original traces.
plt.plot(time[:len(df_f[0])], df_f[0])
plt.show()



########################################################################
# Function to cut and find neuron response for each exp
# Summary : 
#	data_set = nwb file 
#	ns = image data
# 	stim_table = time stamp for each natural scene trial
# 	df_f = df/f trace
# choice 1) cut df_f for each df/f trace
# or     2) look at the presprocess data in data_set 
########################################################################

# Cell Response 
# Computes the mean response for each cell to each stimulus condition. 
# Return is a (# scenes, # cells, 3) np.ndarray.  ---> note : contain scenes = -1 which = spontaneous activity
# The final dimension contains the mean response to the condition (index 0), 
# standard error of the mean of the response to the condition (index 1), 
# and the number of trials with a significant (p < 0.05) response to that condition (index 2).
# Returns:	Numpy array storing mean responses.

download_time = time.time()
NS_Response =NaturalScenes.get_response(ns)
print "Download complete: Time %s" %(time.time() - download_time)




# References 
# Info For Experiment B  : contain 3 stimulus - Static Grating, Natural Scene, Natural movie 1
"""
Experiment_container_id 511510667:
Experiment id: 501498760,

Experiments for experiment_container_id 511510667: 3

[{'age_days': 144.0,
  'cre_line': u'Cux2-CreERT2',
  'donor_name': u'222420',
  'experiment_container_id': 511510667,
  'id': 501574836,
  'imaging_depth': 275,
  'reporter_line': u'Ai93(TITL-GCaMP6f)',
  'session_type': u'three_session_A',
  'specimen_name': u'Cux2-CreERT2;Camk2a-tTA;Ai93-222420',
  'targeted_structure': u'VISp'},
 {'age_days': 144.0,
  'cre_line': u'Cux2-CreERT2',
  'donor_name': u'222420',
  'experiment_container_id': 511510667,
  'id': 501773889,
  'imaging_depth': 275,
  'reporter_line': u'Ai93(TITL-GCaMP6f)',
  'session_type': u'three_session_C',
  'specimen_name': u'Cux2-CreERT2;Camk2a-tTA;Ai93-222420',
  'targeted_structure': u'VISp'},
 {'age_days': 144.0,
  'cre_line': u'Cux2-CreERT2',
  'donor_name': u'222420',
  'experiment_container_id': 511510667,
  'id': 501498760,
  'imaging_depth': 275,
  'reporter_line': u'Ai93(TITL-GCaMP6f)',
  'session_type': u'three_session_B',
  'specimen_name': u'Cux2-CreERT2;Camk2a-tTA;Ai93-222420',
  'targeted_structure': u'VISp'}]
  """