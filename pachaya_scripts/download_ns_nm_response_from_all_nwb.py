# Extract cells response from all experiment session (call by experiments ID =  nwb ID)
# Stimuli : NS = Natural Scenes, NM# = Natural Movies (#=1,2,3)
# exp session A : NM1, NM3
# exp session B : NM1, NS
# exp session C : NM1, NM2


# Save stimuli : natural scenes and natural movies
# Won't save the input scenes and movies scene. However, the frame ID for each experiment are recorded so the input can be constructed.
# Note :
#    1) NM were presented 10 times in a row
#    2) Each image in NS was presented 50 times (random order) --- record frame ID sequence
# Get all experiments containers

# For each containers



from config import Allen_Brain_Observatory_Config
config=Allen_Brain_Observatory_Config()
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
boc = BrainObservatoryCache(manifest_file=config.data_loc+'boc/manifest.json')
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.brain_observatory.natural_movie import NaturalMovie
import time
import pprint
import pandas as pd
import numpy as np
import cPickle as pickle        
import scipy.misc as spm
from helper_funcs import *


# START 
#Read data info
df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
SAVE_LOC='/media/data_cifs/pachaya/AllenData/DataForTrain/all_nwbs/'

# ####################################################################################################################
# Save stimuli template s
# ####################################################################################################################
SAVE_STIM_TEMPLATE =False #True

if(SAVE_STIM_TEMPLATE):
    #save template 
    exps_tmpl = boc.get_ophys_experiments(experiment_container_ids=[exp_con_ids[0]])
    NS_SCALE =True
    scale_h = 31
    scale_w= 31
    for exp in exps_tmpl:
        data_set = boc.get_ophys_experiment_data(exp['id'])
        if exp['session_type'] == config.session['A']:  #NM1, NM3
            stim_tmplt=data_set.get_stimulus_template('natural_movie_one')
            NM1_Input_fullsize=stim_tmplt
            num_fr, h,w =stim_tmplt.shape
            NM1_Input_fullsize_vectorize=np.reshape(stim_tmplt,[num_fr,h*w])
            
            stim_tmplt=data_set.get_stimulus_template('natural_movie_three')
            NM3_Input_fullsize=stim_tmplt
            num_fr, h,w =stim_tmplt.shape
            NM3_Input_fullsize_vectorize=np.reshape(stim_tmplt,[num_fr,h*w])
            
        elif exp['session_type'] == config.session['B']: #NS
            stim_tmplt=data_set.get_stimulus_template('natural_scenes')
            NS_Input_fullsize=stim_tmplt
            num_img, h,w =stim_tmplt.shape
            NS_Input_fullsize_vectorize=np.reshape(stim_tmplt,[num_img,h*w])
            if(NS_SCALE):
                NS_Input_scale=np.zeros([num_img,scale_h,scale_w])
                NS_Input_scale_vectorize=np.zeros([num_img,scale_h*scale_w])
                for scene,fr in zip(stim_tmplt,range(num_img)):
                    tmp=spm.imresize(scene,[scale_h,scale_w])
                    NS_Input_scale[fr,:,:]=tmp
                    NS_Input_scale_vectorize[fr,:]=np.reshape(tmp,[scale_h*scale_w])
        elif (exp['session_type'] == config.session['C'])or ( exp['session_type'] == config.session['C2']): #NM2
            stim_tmplt=data_set.get_stimulus_template('natural_movie_two')
            NM2_Input_fullsize=stim_tmplt
            num_fr, h,w =stim_tmplt.shape
            NM2_Input_fullsize_vectorize=np.reshape(stim_tmplt,[num_fr,h*w])
        else:
            print("Unknown session type")
            print exp
    save_object(NM1_Input_fullsize,SAVE_LOC+'NM1_Input_fullsize.pkl')
    save_object(NM1_Input_fullsize_vectorize,SAVE_LOC+'NM1_Input_fullsize_vectorize.pkl')
    save_object(NM2_Input_fullsize,SAVE_LOC+'NM2_Input_fullsize.pkl')
    save_object(NM2_Input_fullsize_vectorize,SAVE_LOC+'NM2_Input_fullsize_vectorize.pkl')
    save_object(NM3_Input_fullsize,SAVE_LOC+'NM3_Input_fullsize.pkl')
    save_object(NM3_Input_fullsize_vectorize,SAVE_LOC+'NM3_Input_fullsize_vectorize.pkl')
    save_object(NS_Input_fullsize,SAVE_LOC+'NS_Input_fullsize.pkl')
    save_object(NS_Input_fullsize_vectorize,SAVE_LOC+'NS_Input_fullsize_vectorize.pkl')
    save_object(NS_Input_scale,SAVE_LOC+'NS_Input_scale.pkl')
    save_object(NS_Input_scale_vectorize,SAVE_LOC+'NS_Input_scale_vectorize.pkl')

# ####################################################################################################################
#
# Save Neuron responses for each type of stimulus and experiment session
#
# ####################################################################################################################
SAVE_ANALYSED = False
# ####################################################################################################################
# Natural Scenes
# ####################################################################################################################
exps = boc.get_ophys_experiments(stimuli=[config.stim['NS']])
start = 20; end =199 #g13
#start = 50; end = 100 #
#start = 100; end =150#
#start = 150; end =len(exps) #
LOAD_NS=False
if(LOAD_NS):
    for run_num in np.arange(start,end) :

        exp = exps[run_num]

        start_analyse = time.time()
        sessionID = exp['id']
        print('====================================================================')
        print("Stim Type = NATURAL SCENES\n    Current session ID = %s || #%g of [%g, %g)"%(sessionID,run_num,start,end))
        print('====================================================================')
        data_set=boc.get_ophys_experiment_data(exp['id']) 
        
        download_time = time.time()
        ns = NaturalScenes(data_set) #  In addition to computing the sweep_response and mean_sweep_response arrays, NaturalScenes reports the cell's preferred scene, running modulation, time to peak response, and other metrics
        print("done analyzing natural scenes Total time : %s"%(time.time() - download_time)) # Take about 5 min  for the first time
        download_time = time.time()
        ns_response = ns.get_response()  # Depends on # of cells , Take about 4-7mins 
        print("ns.get_response() Total time : %s"%(time.time() - download_time))
        if(SAVE_ANALYSED): #Note : The current method doesn't work because the NS object can't pickle
            save_object(ns,SAVE_LOC+'NS_'+str(sessionID)+'.pkl') # Allensdk.Brain_observatory.NaturalScenes class object
        save_object(ns_response,SAVE_LOC+'NS_'+str(sessionID)+'_precal_response.pkl')   # Shape: (#scenes + 1,# cells+1,3) | 
                                                                                        # [0,:,:] Blank (frame = -1)
                                                                                        # [:,end,:] 'dx' : mean running speed
                                                                                        # [:,:,idx] 0: mean, 1: std err, 2: # of cells with p<=0.05
        mean_response_per_scene = ns_response[1:,:-1,0]
        save_object(mean_response_per_scene,SAVE_LOC+'NS_'+str(sessionID)+'_mean_response_per_scene.pkl')# Shape: (#scenes,#cells)  
        print("[MSR]")
        #response for all trials
        stim_table=ns.stim_table # FrameID, start, finish
        frame=np.asarray(stim_table['frame'])
        save_object(frame,SAVE_LOC+'NS_'+str(sessionID)+'_frame.pkl') #Frame number(include blank at fr=-1) |Shape: (#trial=(#scenes+1)*50=5950,) 
        frame_rm_blank = frame[stim_table.frame !=-1]
        save_object(frame_rm_blank,SAVE_LOC+'NS_'+str(sessionID)+'_frame_noBlank.pkl') #Frame number |Shape: (#trial=(#scenes)*50=5900,) 
        
        msr = ns.mean_sweep_response; # [#trials, #cells+1] |  DataFrame with columns = '0', '1', ... , 'N','dx'
        # Note  running speed or dx can obtain by data_set.get_running_speed() which return dx and time-stamp. However, some case has no data and return array of nan for dx
        msr_np = np.asarray(msr.drop(msr.columns[len(msr.columns)-1], axis=1)) # shape = 5950, #cells 
        save_object(msr_np,SAVE_LOC+'NS_'+str(sessionID)+'_responses_all_trials.pkl')   # Include blank
        msr_rmBlank_np = msr_np[stim_table.frame!=-1,:]
        save_object(msr_rmBlank_np,SAVE_LOC+'NS_'+str(sessionID)+'_responses_all_trials_noBlank.pkl')   #remove blank response    
        print('-------------------------------------------------------------------')
        print("Finished : Responses were saved at.. \n %s"%(SAVE_LOC))
        print("Total Time: %s"%(time.time() - start_analyse))
        print('-------------------------------------------------------------------')

# ####################################################################################################################
# Natural Movies
# ####################################################################################################################
from allensdk.brain_observatory.natural_movie import NaturalMovie
NM_CODEs =['NM1'];
#NM_CODEs=['NM1','NM2','NM3']

# data_set=boc.get_ophys_experiment_data(exp['id']) #exp['id'] = 527550473
# nm = NaturalMovie(data_set,movie_name=config.stim[CODE]) # 'natural_movie+two'
# sw_res = nm.sweep_response 
# sw_res.shape ---- (10,265) --- repeat ID , num cells  
# sw_res.loc[0] === row 0 = repeat #1
# tmp_rep1=sw_res.loc[0]
# tmp_rep1.shape == (265,) == num cells
# tmp_rep1[0] == first repeat, cell 0
# tmp_rep1[0].shape == 903  === raw response(df/f trace) of cell 0 to frame i of repeat#1

ssh pachaya@x7.clps.brown.edu
source activate tensorflow 
cd Allen_Brain_Observatory/

for CODE in NM_CODEs:
    exps = boc.get_ophys_experiments(stimuli=[config.stim[CODE]])
    start = 0; end =50
    #start = 50; end =100
    #start = 100; end =150
    #start = 150; end =200
    #start = 200; end =250
    #start = 250; end =300
    #start = 300; end =350
    #start = 350; end =400
    #start = 400; end =450
    #start = 450; end =500
    #start = 500; end =550
    #start = 550; end =len(exps)
    for run_num in np.arange(start,end) :
        exp = exps[run_num]
        start_analyse = time.time()
        sessionID = exp['id']
        print('====================================================================')
        print("Stim Type = NATURAL SCENES\n    Current session ID = %s || #%g of [%g, %g)"%(sessionID,run_num,start,end))
        print('====================================================================')        
        data_set=boc.get_ophys_experiment_data(exp['id']) 
        
        download_time = time.time()
        NM=NaturalMovie(data_set,movie_name=config.stim[CODE])
        print("done analyzing %s  -- Total time : %s"%(config.stim[CODE],time.time() - download_time)) # Take about 5 min  for the first time
        download_time = time.time()
        STIM_response = NM.get_sweep_response()  # Depends on # of cells , Take about 4-7mins 
        print("NM.get_sweep_response() Total time : %s"%(time.time() - download_time))        
        save_object(STIM_response,SAVE_LOC+CODE+'_'+str(sessionID)+'_precal_response.pkl')   # Shape: data frame(10,# cells) | 
        print('-------------------------------------------------------------------')
        print("Finished : Responses were saved at.. \n %s"%(SAVE_LOC))
        print("Total Time: %s"%(time.time() - start_analyse))
        print('-------------------------------------------------------------------')


"""
        #response for all trials
        # stim_table=NM.stim_table # FrameID, start, finish
        # frame=np.asarray(stim_table['frame'])
        # save_object(frame,SAVE_LOC+'NS_'+str(sessionID)+'_frame.pkl') #Frame number(include blank at fr=-1) |Shape: (#trial=(#scenes+1)*50=5950,) 
        # frame_rm_blank = frame[stim_table.frame !=-1]
        # save_object(frame_rm_blank,SAVE_LOC+'NS_'+str(sessionID)+'_frame_noBlank.pkl') #Frame number |Shape: (#trial=(#scenes)*50=5900,) 
        
        # msr = NM.mean_sweep_response; # [#trials, #cells+1] |  DataFrame with columns = '0', '1', ... , 'N','dx'
        # # Note  running speed or dx can obtain by data_set.get_running_speed() which return dx and time-stamp. However, some case has no data and return array of nan for dx
        # msr_np = np.asarray(msr.drop(msr.columns[len(msr.columns)-1], axis=1)) # shape = 5950, #cells 
        # save_object(msr_np,SAVE_LOC+'NS_'+str(sessionID)+'_responses_all_trials.pkl')   # Include blank
        # msr_rmBlank_np = msr_np[stim_table.frame!=-1,:]
        # save_object(msr_rmBlank_np,SAVE_LOC+'NS_'+str(sessionID)+'_responses_all_trials_noBlank.pkl')   #remove blank response    
"""

# ####################################################################################################################
# Receptive Field (from locally sparse noise stimulus )
# ####################################################################################################################

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.visualization as rfvis
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf
import matplotlib.pyplot as plt
from config import Allen_Brain_Observatory_Config
config=Allen_Brain_Observatory_Config()


cell_specimen_id = 587377366

boc = BrainObservatoryCache(manifest_file=config.data_loc+'boc/manifest.json')

exps = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id],
                                 stimuli=['locally_sparse_noise'])

data_set = boc.get_ophys_experiment_data(exps[0]['id'])

cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]

print("cell %d has index %d" % (cell_specimen_id, cell_index))

rf_data = rf.compute_receptive_field_with_postprocessing(data_set, 
                                                         cell_index, 
                                                         'locally_sparse_noise', 
                                                         alpha=0.5, 
                                                         number_of_shuffles=10000)

rfvis.plot_chi_square_summary(rf_data)
plt.show()

fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_rts_summary(rf_data, ax1, ax2)

fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_rts_blur_summary(rf_data, ax1, ax2)

fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_p_values(rf_data, ax1, ax2)

fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_mask(rf_data, ax1, ax2)

fig, (ax1, ax2) = plt.subplots(1,2)
rfvis.plot_gaussian_fit(rf_data, ax1, ax2)