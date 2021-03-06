
# ##############################################################################################################
"""
CREATE TABLE cells (_id bigserial primary key, cell_specimen_id int, session varchar, drifting_gratings boolean, locally_sparse_noise boolean, locally_sparse_noise_four_deg boolean, locally_sparse_noise_eight_deg boolean, natural_movie_one boolean, natural_movie_two boolean, natural_movie_three boolean, natural_scenes boolean, spontaneous boolean, static_gratings boolean, specimen_recording_pointer varchar, traces_loc_pointer varchar, ROImask_loc_pointer varchar, stim_table_loc_pointer varchar)

@@ 9,38 +9,39 @@ ALTER TABLE cells ADD CONSTRAINT unique_cells UNIQUE (cell_specimen_id, session
ALTER TABLE rf ADD CONSTRAINT unique_rfs UNIQUE (cell_specimen_id , lsn_name , experiment_container_id , found_on , found_off , alpha , number_of_shuffles , on_distance , on_area , on_overlap , on_height , on_center_x , on_center_y , on_width_x , on_width_y , on_rotation , off_distance , off_area , off_overlap, off_height , off_center_x , off_center_y , off_width_x , off_width_y , off_rotation )

"""
# ##############################################################################################################
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import matplotlib.pyplot as plt
from config import Allen_Brain_Observatory_Config
from tqdm import tqdm
import pandas as pd
import numpy as np
from db_main import db, add_cell_data, initialize_database, get_performance

from helper_funcs import *
import time 
import os 
import sys
sys.stdout.flush()
from tqdm import tqdm
main_config=Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=main_config.manifest_file)
output_data_folder = main_config.output_pointer_loc

df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info =[]
DATE_STAMP = time.strftime('%D').replace('/','_')

#psql allenedatabase h 127.0.0.1 d allendatabase
# Create DB dict and save output_pointers  at main_config.output_pointer_loc
############################# My take
def session_filters(main_config,cells_ID_list):
     """Find cells that present in all sessions"""
    masterkey = {k:v for k, v in main_config.session.iteritems() if v in cells_ID_list.keys()}
    reference_session = masterkey[masterkey.keys()[0]]
    reference_cells = cells_ID_list[reference_session]
    flist = {} #List of cells that found in all three sessions

def add_None_to_rfdict(Ref_dict, FLAG_ON=False,FLAG_OFF=False):
    Ref_dict[key]=None
    return Ref_dict

def get_all_pointers(main_config, cid,sess,stim):
    sess_type = sess['session_type']
    if stim in main_config.available_stims:
        stim_template="%s%s_template.pkl"%(main_config.stimulus_template_loc,stim)
    else:
        stim_template=None
    pointer_list ={
        'fluorescence_trace':"%s%s_%s_traces.npz"%(main_config.fluorescence_traces_loc,cid,sess_type),
        'stim_table':"%s%s_%s_%s_table.npz"%(main_config.stim_table_loc,sess['id'],sess_type,stim),
        'other_recording':"%s%s_related_traces.npz"%(main_config.specimen_recording_loc,sess['id']),
        'ROImask':"%s%s_%s_ROImask.npz"%(main_config.ROIs_mask_loc,cid,sess_type),
        'stim_template':stim_template,
        }
    return pointer_list

def get_sess_key(main_config, sesstxt):
    for code,txt in main_config.session.iteritems():
        if sesstxt == txt :
            return code

def get_stim_list_boolean(boc, main_config, this_stim, output_dict): 
    for stim in boc.get_all_stimuli():
        if stim in main_config.sess_with_number.keys():
            stim=main_config.sess_with_number[stim]
            output_dict[stim]=False
        output_dict[this_stim]=True
    return output_dict


df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info =[]
start = 0
end =len(exp_con_ids)
if len(sys.argv) >1:
    start=int(sys.argv[1])
    end=int(sys.argv[2])
idx_range =np.arange(start,end)

Recorded_all_cells={}
time_everything=time.time()
RFs_info = load_object(config.RF_info_loc+"all_cells_RF_info_08_30_17.pkl") # ---> list of exp / then dict of cell name

Recorded_cells_list={}
cells_ID_list={}
output_data_folder = main_config.output_pointer_loc
OUTPUT_POINTER_SAVED = False

for idx in tqdm(
            idx_range,
            desc="Data from the experiment",
            total=len(idx_range)):
    exps=exp_con_ids[idx]
    print "Start running experiment container ID #%s  index = %g of [%g,%g)"%(exps,idx,start,end)
    runtime=time.time()

    exp_session = boc.get_ophys_experiments(experiment_container_ids=[exps]) 
    RFinfo_this_exp =RFs_info[idx]

    # Get common cells
    for sess in exp_session:
        sess_code = get_sess_key(main_config,sess['session_type'])
        tmp=boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']]=tmp.get_cell_specimen_ids()
    common_cells = session_filters(main_config,cells_ID_list) 

    common_cell_id=[]
    for cell_specimen_id, session_filter in common_cells.iteritems():
        if session_filter:
            common_cell_id.append(cell_specimen_id)
    ###############################################
    # get RF and cell DB for each cell
    for cell_id in common_cell_id:
        this_cell_rf =RFinfo_this_exp[cell_id]
        for rf in this_cell_rf:
            if rf['lsn_name'] in config.pick_main_RF:
                represent_RF =  rf
        cell_rf_dict = represent_RF.copy()
        cell_rf_dict = add_None_to_rfdict(cell_rf_dict, FLAG_ON=cell_rf_dict['found_on'], FLAG_OFF=cell_rf_dict['found_off'])
        list_of_cell_stim_dicts = []
        for session in exp_session:
            data_set= boc.get_ophys_experiment_data(session['id'])
            for stimulus in data_set.list_stimuli():
                output_pointer = os.path.join(output_data_folder, '%s_%s_%s.npy' % (cell_id, session['session_type'], stimulus)) 
                if(OUTPUT_POINTER_SAVED):
                    all_pointers=get_all_pointers(config=config, cid=cell_id,sess=session, stim=stimulus)
                    if stimulus in config.session_name_for_RF:
                        for rf in this_cell_rf:
                            if rf['lsn_name'] == stimulus:
                                rf_from_this_stim = rf.copy()
                        np.savez(
                          output_pointer,
                          neural_trace=all_pointers['fluorescence_trace'],
                          stim_template=all_pointers['stim_template'],
                          stim_table=all_pointers['stim_table'],
                          ROImask=all_pointers['ROImask'],
                          other_recording=all_pointers['other_recording'],
                          RF_info=rf_from_this_stim)
                    else:
                      np.savez(
                          output_pointer,
                          neural_trace=all_pointers['fluorescence_trace'],
                          stim_template=all_pointers['stim_template'],
                          stim_table=all_pointers['stim_table'],
                          ROImask=all_pointers['ROImask'],
                          other_recording=all_pointers['other_recording'])
            it_stim_dict = {
                  'cell_specimen_id':cell_id,
                  'session':session['session_type'],
                  'cell_output_npy':output_pointer}
            it_stim_dict=get_stim_list_boolean(
                                  boc=boc, 
                                  main_config=main_config, 
                                  this_stim=stimulus, 
                                  output_dict=it_stim_dict) 
            list_of_cell_stim_dicts += [it_stim_dict]
      Recorded_cells_list[cell_id]={'cell_rf_dict':cell_rf_dict, 
                                    'list_of_cell_stim_dicts':list_of_cell_stim_dicts}
      add_cell_data(
                cell_rf_dict,
                list_of_cell_stim_dicts)
tmptime=time.strftime('%D_%H_%M_%S')
tmptime=tmptime.replace('/','_')
fname="Recorded_cells_list_for_db_%s.pkl"%tmptime
save_object(Recorded_cells_list,fname)