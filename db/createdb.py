###############################################################################################################
"""
CREATE TABLE cells (_id bigserial primary key, cell_specimen_id int, session varchar, drifting_gratings boolean, locally_sparse_noise boolean, locally_sparse_noise_four_deg boolean, locally_sparse_noise_eight_deg boolean, natural_movie_one boolean, natural_movie_two boolean, natural_movie_three boolean, natural_scenes boolean, spontaneous boolean, static_gratings boolean, specimen_recording_pointer varchar, traces_loc_pointer varchar, ROImask_loc_pointer varchar, stim_table_loc_pointer varchar)

CREATE TABLE rf (_id bigserial primary key, cell_specimen_id int, lsn_name varchar, experiment_container_id int, found_on boolean, found_off boolean, alpha float, number_of_shuffles int, on_distance float, on_area float, on_overlap float, on_height float, on_center_x float, on_center_y float, on_width_x float, on_width_y float, on_rotation float, off_distance float, off_area float, off_overlap float, off_height float, off_center_x float, off_center_y float, off_width_x float, off_width_y float, off_rotation float)

ALTER TABLE cells ADD CONSTRAINT unique_cells UNIQUE (cell_specimen_id, session , drifting_gratings , locally_sparse_noise , locally_sparse_noise_four_deg , locally_sparse_noise_eight_deg , natural_movie_one , natural_movie_two , natural_movie_three , natural_scenes , spontaneous , static_gratings , specimen_recording_pointer , traces_loc_pointer , ROImask_loc_pointer , stim_table_loc_pointer )

ALTER TABLE rf ADD CONSTRAINT unique_rfs UNIQUE (cell_specimen_id , lsn_name , experiment_container_id , found_on , found_off , alpha , number_of_shuffles , on_distance , on_area , on_overlap , on_height , on_center_x , on_center_y , on_width_x , on_width_y , on_rotation , off_distance , off_area , off_overlap, off_height , off_center_x , off_center_y , off_width_x , off_width_y , off_rotation )

"""
###############################################################################################################
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import matplotlib.pyplot as plt
from config import Allen_Brain_Observatory_Config
from tqdm import tqdm
import pandas as pd
import numpy as np
from db import db
from helper_funcs import *
import time 
import sys
sys.stdout.flush()
from tqdm import tqdm
config=Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)

output_data_folder = config.output_pointer_loc

df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info =[]
DATE_STAMP = time.strftime('%D').replace('/','_')

Recorded_specimen_recording_fname={}
Recorded_traces_loc_pointer ={}
Recorded_ROImask_loc_pointer={}
Recorded_stim_table_loc_pointer={}
time_everything=time.time()



# Create DB dict and save output_pointers  at config.output_pointer_loc



for cell in tqdm(cells, desc='Adding cells to DB', total=len(cells)):
    cell_id  # set cell id number here
    cell_rf_dict = {
        cell_id:cell_id,
        rf_stuff:rf_stuff
    }  # specify RF dict here
    list_of_cell_stim_dicts = []
    for sess in sessions:
        for stim in stimuli:

            output_pointer = os.path.join(output_data_folder, '%s_%s_%s' % (cell_id, session, stimulus)) 
            np.savez(
                output_pointer,
                neural_trace, # Neural traces were recorded session-wise --- all stim in this sess point to the same files
                stimuli:numpy_pointer_to_movie_or_images,  # Insert the directory pointer from above (1)
                stim_table:either store the raw stim table or point to it... doesnt matter,
                other_data:such as pupil responses and running) # also
                
                
            it_stim_dict = {
                cell_id:cell_id,
                cell_npy:output_pointer,
                boolean_for_the_stimulus)
            list_of_cell_stim_dicts += [it_stim_dict]
    db.add_cell_data(
        cell_rf_dict,
        list_of_cell_stim_dicts)


list_of_cell_stim_dicts = [{}]
db.add_cell_data()



############################# My take


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
RFs_info = load_object("/home/pachaya/Allen_Brain_Observatory/all_cells_RF_info_08_30_17.pkl") # ---> list of exp / then dict of cell name

Recorded_cells_list={}

def add_None_to_rfdict(FLAG_ON=False,FLAG_OFF=False)

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
        sess_code = get_sess_key(config,sess['session_type'])
        tmp=boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']]=tmp.get_cell_specimen_ids()
    common_cells = session_filters(config,cells_ID_list) 
    common_cell_id=[]
    for cell_specimen_id, session_filter in common_cells.iteritems():
        if session_filter:
            common_cell_id.append(cell_specimen_id)
    print("Start saving neural traces for cells that presented in all sessions")

    for cell_id in common_cell_id:
      this_cell_rf =RFinfo_this_exp[cell_id]
      for rf in this_cell_rf:
        cell_rf_dict={
          cell_specimen_id: , 
          lsn_name , 
          experiment_container_id , 
          found_on , 
          found_off , 
          alpha , 
          number_of_shuffles , 
          on_distance , 
          on_area , 
          on_overlap , 
          on_height , 
          on_center_x , 
          on_center_y , 
          on_width_x , 
          on_width_y , 
          on_rotation , 
          off_distance , 
          off_area , 
          off_overlap, 
          off_height , 
          off_center_x , 
          off_center_y , 
          off_width_x , 
          off_width_y , 
          off_rotation 
          }



      }







         cell_id  # set cell id number here
    cell_rf_dict = {
        cell_id:cell_id,
        rf_stuff:rf_stuff
    }  # specify RF dict here
    list_of_cell_stim_dicts = []
    for sess in sessions:
        for stim in stimuli:

            output_pointer = os.path.join(output_data_folder, '%s_%s_%s' % (cell_id, session, stimulus)) 
            np.savez(
                output_pointer,
                neural_trace, # Neural traces were recorded session-wise --- all stim in this sess point to the same files
                stimuli:numpy_pointer_to_movie_or_images,  # Insert the directory pointer from above (1)
                stim_table:either store the raw stim table or point to it... doesnt matter,
                other_data:such as pupil responses and running) # also
                
                
            it_stim_dict = {
                cell_id:cell_id,
                cell_npy:output_pointer,
                boolean_for_the_stimulus)
            list_of_cell_stim_dicts += [it_stim_dict]
    db.add_cell_data(
        cell_rf_dict,
        list_of_cell_stim_dicts)


list_of_cell_stim_dicts = [{}]
db.add_cell_data()