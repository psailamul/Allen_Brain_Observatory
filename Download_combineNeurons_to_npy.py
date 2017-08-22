# Download Data from Allen Institute's brain_observatory and save to pickle
# Allows filters for cells from different experiments
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
SAVED_PICKLE = False
metadata ={}
metadata['code']=CODE
config.save_folder=config.save_folder+CODE+'/'
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

#if(config.RESHAPE_IMG_NS):
#    img_h = config.reshape_img_size_h
#    img_w= config.reshape_img_size_w
scale_h = config.reshape_img_size_h
scale_w= config.reshape_img_size_w

inputscenes_fullsize = np.zeros([num_image, img_h*img_w])
inputscenes_scaled = np.zeros([num_image,  scale_h*scale_w]) 

#unique input image
for scene,i in zip(ns_template,np.arange(num_image)):
    inputscenes_fullsize[i,:] = np.reshape(scene,[img_h*img_w])
    
    tmp=spm.imresize(ns_template[i,:,:],[scale_h,scale_w]) # down sampling to 31x31
    inputscenes_scaled[i,:]=np.reshape(tmp, [scale_h*scale_w]) #vectorized the image

#all trials with image repetition (118*50 = 5900)
inputscenes_fs_big = np.repeat(inputscenes_fullsize,50,axis=0) # sort from scene 1 - 118 
inputscenes_sc_big = np.repeat(inputscenes_scaled,50,axis=0) # sort from scene 1 - 118 

if(SAVED_PICKLE):
    save_object(inputscenes_fullsize,config.data_loc+config.save_folder+"ns_input_fullsize.pkl")
    save_object(inputscenes_scaled,config.data_loc+config.save_folder+"ns_input_scaled.pkl")
    save_object(inputscenes_fs_big,config.data_loc+config.save_folder+"ns_input_alltrials_fullsize.pkl")
    save_object(inputscenes_sc_big,config.data_loc+config.save_folder+"ns_input_alltrials_scaled.pkl")

    save_object(inputscenes_fullsize,"ns_input_fullsize.pkl")
    save_object(inputscenes_scaled,"ns_input_scaled.pkl")
    save_object(inputscenes_fs_big,"ns_input_alltrials_fullsize.pkl")
    save_object(inputscenes_sc_big,"ns_input_alltrials_scaled.pkl")

    
#import ipdb; ipdb.set_trace()

# ==================================================================================================
# Extract output response cells
# ==================================================================================================
 #backup_first_coimsr
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
# Note : Check that the average response calculate from msr are accurate




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
    ns_response = ns.get_response()  # Take about 7mins 
    print("ns.get_response() Total time : %s"%(time.time() - download_time))
    
    #Return is a (# scenes +1, # cells+1, 3) np.ndarray
    #The final dimension contains the mean response to the condition (index 0), 
    #standard error of the mean of the response to the condition (index 1), 
    #and the number of trials with a significant (p < 0.05) response to that condition (index 2)
    #The first scene is blank, the last cell ('dx') is running speed
    remove_blank_ns_response = ns_response[1:,:-1,:]
    coi_response = remove_blank_ns_response[:,coi_indices,:] # mean, std, num trials with p<0.05
    all_output_response.append(coi_response)
 
    #response for all trials
    msr = ns.mean_sweep_response; # [ #trial, #cells]
    stim_table=ns.stim_table    # FrameID, start, finish
    sort_frame = stim_table.sort_values('frame',ascending=True)
    resort_msr = msr.reindex(sort_frame.index)
    remove_blank_msr = resort_msr[sort_frame.frame!=-1] # [118*50, # cells]
    
    
    col_coi_ind =[str(ii) for ii in coi_indices]
    msr_subset_coi = remove_blank_msr[col_coi_ind]
    tmpdict = {str(x):str(y)  for x,y in zip(coi_indices,coi_ids) }
    coi_msr=msr_subset_coi.rename(columns=tmpdict) # columns are cell_specimen_ids
    coi_msr.reset_index(inplace=True,drop=True)
    # Response data : sorted by frame ID, columns are cell_specimen_ID
    if(first_exp):
        big_coi_msr=coi_msr
        first_exp=False
    else:
        big_coi_msr = pd.concat([big_coi_msr, coi_msr], axis=1, join_axes=[big_coi_msr.index]) 


print("Finish the extraction from all experiment container Time = %s"%(extraction_start -time.time()))

all_output_response = np.concatenate(all_output_response,axis=1) # mean, std, # p<=0.05
output_response = all_output_response[:,:,0]
output_response.shape
if(SAVED_PICKLE):
    save_start = time.time()
    save_object(all_output_response,config.data_loc+config.save_folder+"ns_unique_scene_response.pkl")
    save_object(output_response,config.data_loc+config.save_folder+"ns_output_response.pkl")
    print("SAVED : Time = %s"%(save_start -time.time()))


print("Finish all ooperations\nSaved at %s"%(config.save_folder))
