
# coding: utf-8

# # 1. Import Libraries

# In[3]:
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
from allensdk.brain_observatory.natural_scenes import NaturalScenes
import pprint
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# In[4]:

# Functions
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


# In[5]:

# Function to download info if file not found


# In[6]:

# Download Experiment Container for specific ID
EXP_CONTAINER_ID=511510667
EXP_ID = 514422179 #501498760

download_time = time.time()
data_set = boc.get_ophys_experiment_data(EXP_ID)  
pprint.pprint(data_set.get_metadata())
print "Download complete: Time %s" %(time.time() - download_time)


# In[7]:

all_cells_id =data_set.get_cell_specimen_ids() # List of cells ID of all cells in this experiment 
total_cells_num = np.shape(all_cells_id)[0]
print(total_cells_num) # Total Cells = 212 


# # Natural Scenes

# In[8]:

download_time = time.time()
#Download saved Analysis in Natural Scenes  
#ns = NaturalScenes(data_set)
ns=load_object("NS_ECI511510667_Exp501498760.pkl") # ns is NaturalScenes object 
# ns.stim_table --> [Frame,start time,  end time]
print "Time %s" %(time.time() - download_time)


# In[12]:

# Pandas Data frame  
# Fields
# scene_ns (scene number),response_reliability_ns,peak_dff_ns (peak dF/F),ptest_ns, p_run_ns,run_modulation_ns,time_to_peak_ns,duration_ns
NS_peak = NaturalScenes.get_peak(ns) # peak = peak response for each cell to a particular scene
NS_peak.keys()


# In[21]:

#  a (# scenes, # cells, 3) np.ndarray.  ---> note : contain scenes = -1 which = spontaneous activity (not certain)
#  (index 0) The final dimension contains the mean response to the condition, 
#  (index 1) standard error of the mean of the response to the condition, 
#  (index 2) the number of trials with a significant (p < 0.05) response to that condition

NS_response = NaturalScenes.get_response(ns) 

NS_response.shape # Problem : the shape is #scene +1 , #cell +1


# In[20]:

import scipy.misc as spm

#get all input images (118)
s_img_size = 31
ns_template = data_set.get_stimulus_template('natural_scenes')
num_image = ns_template.shape[0]
tr_input = np.zeros([num_image, s_img_size,s_img_size])
inputscenes = np.zeros([num_image, s_img_size*s_img_size]) #[118,961]

import ipdb; ipdb.set_traces()
#response per image
output_response = NS_response[1:119,1:213,0]

#unique input image
for scene,i in zip(ns_template,np.arange(num_image)):
    tmp=spm.imresize(ns_template[i,:,:],[s_img_size,s_img_size]) # down sampling to 31x31
    tr_input[i,:,:]=tmp
    inputscenes[i,:]=np.reshape(tmp,[s_img_size*s_img_size]) #vectorized the image


#all trials with image repetition (118*50 = 5900)
stim_table = data_set.get_stimulus_table('natural_scenes')
trial_mask = stim_table.frame != -1 # -1 is black screen , removed 
stim_table = stim_table[trial_mask]
n_trial = stim_table.shape[0]
inputscenes_big = np.zeros([n_trial,s_img_size*s_img_size])
for fr,i in zip(stim_table.frame, np.arange(n_trial)):
	inputscenes_big[i,:] = inputscenes[fr,:].copy()
#inputscenes_big --> [5900,961]

#response of all trials (118*50 =5900)
output_response_big = np.zeros([n_trial,total_cells_num])
msr = ns.mean_sweep_response
#output
for i,col in zip(np.arange(n_trial), msr.columns):
	if col !='dx':		
		cutmsr= msr[col]
		cutmsr = cutmsr.values[trial_mask]
		output_response_big[:,i]=cutmsr
#output_response_big =5900 *212

#Save current data
np.save('all_output_response_118_212.npy',output_response)
np.save('all_input_scenes_118_961.npy',inputscenes)
np.save('all_output_response_big_5900_212.npy',output_response_big)
np.save('all_input_scenes_big_5900_961.npy',inputscenes_big)



# Split the data to train and test set
np.random.seed(31)

train_idx=np.random.permutation(118)
train_idx_big=np.random.permutation(5900)


in_train_set= inputscenes[train_idx[:100],:]; out_train_set= output_response[train_idx[:100],:]; 
in_test_set= inputscenes[train_idx[100:],:]; out_test_set= output_response[train_idx[100:],:]; 

in_train_set_big= inputscenes_big[train_idx_big[:5000],:]; out_train_set_big= output_response_big[train_idx_big[:5000],:]; 
in_test_set_big= inputscenes_big[train_idx_big[5000:],:]; out_test_set_big= output_response_big[train_idx_big[5000:],:]; 

FLAG = False
if FLAG:
	np.save('Allen_train_output.npy',out_train_set)
	np.save('Allen_train_input.npy',in_train_set)
	np.save('Allen_test_output.npy',out_test_set)
	np.save('Allen_test_input.npy',in_test_set)
	np.save('Allen_train_output_big.npy',out_train_set_big)
	np.save('Allen_train_input_big.npy',in_train_set_big)
	np.save('Allen_test_output_big.npy',out_test_set_big)
	np.save('Allen_test_input_big.npy',in_test_set_big)