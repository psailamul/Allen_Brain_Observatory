# Download Data from Allen Institute's brain_observatory and change to numpy type
# #############################################################################################


from config import Allen_Brain_Observatory_Config
config=Allen_Brain_Observatory_Config()
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file=config.data_loc+'boc/manifest.json')
from allensdk.brain_observatory.natural_scenes import NaturalScenes
import pprint
import time
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import scipy.misc as spm
import os
import datetime

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
        
# #############################################################################################
# PROOBLEM with download NWB file -->  https://github.com/AllenInstitute/AllenSDK/issues/22 
# http://api.brain-map.org/api/v2/data/OphysExperiment/501498760.xml?include=well_known_files
# http://api.brain-map.org/api/v2/well_known_file_download/514422179 --> then change file name to 5014987600.nwb
# #############################################################################################
# Setting
#config.data_set_code = 'Area-VISp_Depth-175um_NS-sig_NMall-nonzero_LSN-rfChi2-0.05_allStim-true'
CODE = config.data_set_code
SAVED_PICKLE = True
metadata ={}
metadata['code']=CODE
today_dir=datetime.datetime.now().strftime ("%Y_%m_%d")+'/'
directory = config.data_loc+config.save_folder+today_dir
if not os.path.exists(directory):
    os.makedirs(directory) # Have to run this script as root
config.save_folder = config.save_folder + today_dir
# ==================================================================================================
# Download cells ID list and their corresponding experiment container
# ==================================================================================================
download_time=time.time()

if(config.FILTERS):
    filters= load_object(config.filters_file)
    cells = boc.get_cell_specimens(filters=filters) # alternatively can use ids = [list of specimen ids]
    cells_pd = pd.DataFrame.from_records(cells) # DataFrame of cells of interest(coi)
    metadata['filters_file'] = config.filters_file
else:
    cells = boc.get_cell_specimens() # all cells
    cells_pd = pd.DataFrame.from_records(cells) 

num_total_coi =len(cells)
metadata['cell_specimen_ids'] = cells_pd['cell_specimen_id']
print("total cells of interest(COI): %d" % len(cells))
print("Download complete: Time %s" %(time.time() - download_time))

# find experiment containers for those cells
EC_ids = cells_pd['experiment_container_id'].unique()
specimen_ids=cells_pd['specimen_id'].unique()
print("Total experiment container: %d" % len(EC_ids))
import allensdk.brain_observatory.stimulus_info as stim_info
exps = boc.get_ophys_experiments(experiment_container_ids=EC_ids, stimuli=[stim_info.NATURAL_SCENES]) # Natural Scenes presented in session B only. Thus, only exp with session B was retrieved

# Which mean, in the case of natural_movies_one ---> all session will be retrieved
metadata['experiment_container']=exps
# ==================================================================================================


# ==================================================================================================
# Extract Natural Scenes Input
# ==================================================================================================

#Get natural scenes template
data_set = boc.get_ophys_experiment_data(exps[0]['id']) 
ns_template = data_set.get_stimulus_template('natural_scenes')
num_image, img_h, img_w = ns_template.shape #118, 918, 1174
if(config.RESHAPE_IMG_NS):
    img_h = config.reshape_img_size_h
    img_w= config.reshape_img_size_w

input_fullsize = np.zeros([num_image, img_h, img_w])
inputscenes = np.zeros([num_image,  img_h*img_w]) 

#unique input image
for scene,i in zip(ns_template,np.arange(num_image)):
    tmp=spm.imresize(ns_template[i,:,:],[img_h,img_w]) # down sampling to 31x31
    input_fullsize[i,:,:]=tmp
    inputscenes[i,:]=np.reshape(tmp, [img_h*img_w]) #vectorized the image

if(SAVED_PICKLE):
    save_object(input_fullsize,config.data_loc+config.save_folder+"ns_input_fullsize_"+CODE+".pkl")
    save_object(inputscenes,config.data_loc+config.save_folder+"ns_input_vectorized_"+CODE+".pkl")


#all trials with image repetition (118*50 = 5900)
inputscenes_big = np.repeat(inputscenes,50,axis=0) # sort from scene 1 - 118 



# ==================================================================================================
# Extract output response cells
# ==================================================================================================

#==============
#response per image
#all_output_response = np.zeros([num_image, num_total_coi,3])
extraction_start = time.time()
all_output_response =[]
all_coi_metadata=[]
all_sweep_response =[]
coi_pnter = 0
big_coi_msr=[]
first_exp = True
for exp in exps:
    print("====================================\nExperiment Container ID : %d"%exp['experiment_container_id'])
    start_time = time.time()
    #exp = exps[0] # session B of exp containers xx
    # Find cell specimens that are in this container
    data_set = boc.get_ophys_experiment_data(exp['id']) 
    mask = np.isin(data_set.get_cell_specimen_ids(), cells_pd['cell_specimen_id'], assume_unique=True)
    coi_ids = data_set.get_cell_specimen_ids()[mask]
# index
    coi_indices = data_set.get_cell_specimen_indices(coi_ids)
    #time_axis, dff_traces = data_set.get_dff_traces(cell_specimen_ids=coi_ids) # The df/f traces 
    num_coi =len(coi_indices) #Number of COI in this experiment
    # Analyze natural scene stimulus
    download_time = time.time()
    ns = NaturalScenes(data_set) #  In addition to computing the sweep_response and mean_sweep_response arrays, NaturalScenes reports the cell's preferred scene, running modulation, time to peak response, and other metrics
    print("done analyzing natural scenes Total time : %s"%(time.time() - download_time)) # Take about 5 min  for the first time
    
    download_time = time.time()
    ns_response = ns.get_response()  # Take about 7mins for the 1st run 
    print("done analyzing natural scenes Total time : %s"%(time.time() - download_time)) # Take about 5 min  for the first time
    #Return is a (# scenes +1, # cells+1, 3) np.ndarray
    #The final dimension contains the mean response to the condition (index 0), 
    #standard error of the mean of the response to the condition (index 1), 
    #and the number of trials with a significant (p < 0.05) response to that condition (index 2)
    #The first scene is blank, the cell #0 is dummy data
    ns_response = ns_response[1:,1:,:]
    coi_response = ns_response[:,coi_indices,:] # mean, std, num trials with p<0.05
    all_output_response.append(coi_response)
 
    #response for all trials
    msr = ns.mean_sweep_response; # [ #trial, #cells]
    stim_table = data_set.get_stimulus_table('natural_scenes') # FrameID, start, finish
    msr.insert(0,'frame',stim_table['frame'])
    resort_msr=msr.sort_values('frame',ascending=True)
    remove_blank = resort_msr[resort_msr.frame != -1]
    
    coi_msr = remove_blank[coi_indices+1]
    tmpdict = {str(x):str(y)  for x,y in zip(coi_indices,coi_ids) }
    coi_msr=msr_coi.rename(columns=tmpdict) # columns are cell_specimen_ids
    coi_msr.reset_index(inplace=True,drop=True)
    if (first_exp):
        big_coi_msr=coi_msr
        first_exp=False
    else:
        big_coi_msr = pd.concat([big_coi_msr, coi_msr], axis=1, join_axes=[big_coi_msr.index])
        
    
    
    trial_mask = stim_table.frame != -1 # -1 is black screen , removed 
    stim_table = stim_table[trial_mask]
    n_trial = stim_table.shape[0]
inputscenes_big = np.zeros([n_trial,s_img_size*s_img_size])
for fr,i in zip(stim_table.frame, np.arange(n_trial)):
    inputscenes_big[i,:] = inputscenes[fr,:].copy()
    print("Experiment Container ID %d Done\n Time = %s"%(exp['experiment_container_id'], time.time()-start_time))
    #all_output_response[:,coi_pnter:coi_pnter+num_coi,:] = coi_response
    #coi_pnter=coi_pnter+num_coi
print("Finish the extraction from all experiment container Time = %s"%(extraction_start -time.time()))
save_start = time.time()
all_output_response = np.concatenate(all_output_response,axis=1)

output_response = all_output_response[:,:,0]
output_response.shape
if(SAVED_PICKLE):
    save_object(all_output_response,config.data_loc+config.save_folder+"ns_unique_scene_response_"+CODE+".pkl")
    save_object(output_response,config.data_loc+config.save_folder+"ns_output_response_"+CODE+".pkl")

print("SAVED : Time = %s"%(save_start -time.time()))


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





"""
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






# sweep_response is a DataFrame that contains the DF/F of each cell during each stimulus trial. 
# It shares its index with stim_table. Each cell contains a timeseries that extends from 1 second prior to the start of the trial 
# to 1 second after the end of the trial. The final column of sweep_response, named dx, is the running speed of the mouse during each trial. 
# The data in this DataFrame is used to create another DataFrame called mean_sweep_response that contains the mean DF/F during the trial for 
#each cell (and the mean running speed in the last column).
#In [8]:
subset = dg.sweep_response[(dg.stim_table.orientation==pref_ori)&(dg.stim_table.temporal_frequency==pref_tf)]


get_response()[source]
Computes the mean response for each cell to each stimulus condition. Return is a (# scenes, # cells, 3) np.ndarray
The final dimension contains the mean response to the condition (index 0), 
standard error of the mean of the response to the condition (index 1), 
and the number of trials with a significant (p < 0.05) response to that condition (index 2)

# ==================================================================================================


##################################
# Get Neural Response data 
##################################



# #######################################################################
# Function to cut and find neuron response for each exp
# Summary : 
#	data_set = nwb file 
#	ns = image data
# 	stim_table = time stamp for each natural scene trial
# 	df_f = df/f trace
# choice 1) cut df_f for each df/f trace
# or     2) look at the presprocess data in data_set 
# #######################################################################

# Cell Response 
# Computes the mean response for each cell to each stimulus condition. 
# Return is a (# scenes, # cells, 3) np.ndarray.  ---> note : contain scenes = -1 which = spontaneous activity
# The final dimension contains the mean response to the condition (index 0), 
# standard error of the mean of the response to the condition (index 1), 
# and the number of trials with a significant (p < 0.05) response to that condition (index 2).
# Returns:	Numpy array storing mean responses.

download_time = time.time()
NS_Response =NaturalScenes.get_response(ns) # Take awhile
print "Download complete: Time %s" %(time.time() - download_time)
"""