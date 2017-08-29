"""Compute and save rf data"""
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.receptive_field_analysis.visualization as rfvis
import allensdk.brain_observatory.receptive_field_analysis.receptive_field as rf
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.brain_observatory.natural_movie import NaturalMovie
import matplotlib.pyplot as plt
from config import Allen_Brain_Observatory_Config
from tqdm import tqdm
import pandas as pd
import numpy as np
from db import db
from helper_funcs import *
import time 


config=Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)
import sys
sys.stdout.flush()
from tqdm import tqdm
#python load_all_nwb.py 0 597 2>&1 | tee -a load_log_all.txt

def session_filters(config,cells_ID_list):
    """Find cells that present in all sessions"""
    masterkey = {k:v for k, v in config.session.iteritems() if v in cells_ID_list.keys()}
    reference_session = masterkey[masterkey.keys()[0]]
    reference_cells = cells_ID_list[reference_session]
    flist = {} #List of cells that found in all three sessions
    for cell in reference_cells:
        checks = []
        for k in masterkey.keys()[1:]:
            v = cells_ID_list[masterkey[k]]
            checks += [cell in v]
        flist[cell] = all(checks)
    return flist


def filter_cells_for_rf(
        exp_session,
        flist,
        session_RF_stim,
        rf_session=['three_session_C', 'three_session_C2'],
        alpha=0.5,
        number_of_shuffles=5000):
    """Filters for cells that has RFs """
    from helper_funcs import load_object
    save_loc = config.Allen_analysed_stimulus_loc
    rf_cells_dict ={}
    for sess in exp_session:
        it_session = sess['session_type']
        if it_session in rf_session:
            data_set = boc.get_ophys_experiment_data(sess['id'])
            sparse_noise_type = session_RF_stim[it_session.split('_')[-1]]
            for cell_specimen_id, session_filter in tqdm(
                    flist.iteritems(),
                    desc="Deriving RFs for %s"%it_session,
                    total=len(flist)):
                if session_filter:
                    cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]
                    print("cell %s has index %d" % (cell_specimen_id, cell_index))
                    rf_list =[]
                    for lsn_deg in sparse_noise_type:
                        DATA_EXIST = find_rf_data_files(exp_session=exp_session, save_loc = config.Allen_analysed_stimulus_loc)
                        if DATA_EXIST:
                            fname="%s%s_%s.pkl"%(save_loc,cell_specimen_id,lsn_deg)
                            rf_data = load_object(fname)
                        else:                        
                            rf_data = rf.compute_receptive_field_with_postprocessing(
                                data_set, 
                                cell_index, 
                                lsn_deg, 
                                alpha=alpha, 
                                number_of_shuffles=number_of_shuffles)
                        rf_info ={}
                        rf_info['cell_specimen_id']=cell_specimen_id 
                        rf_info['experiment_container_id']= sess['experiment_container_id']
                        rf_info['lsn_name']=lsn_deg
                        rf_info['alpha']=alpha
                        rf_info['number_of_shuffles']=number_of_shuffles
                        rf_info['found_on']= False
                        rf_info['found_off']= False
                        
                        for key in config.RF_sign:
                            if 'gaussian_fit' in rf_data[key]:
                                rf_info['found_'+key]= True
                                tmp_attr = rf_data[key]['gaussian_fit']['attrs']
                                rf_info[key+'_distance']=tmp_attr['distance']
                                rf_info[key+'_area']=tmp_attr['area']
                                rf_info[key+'_overlap']=tmp_attr['overlap']
                                rf_info[key+'_height']=tmp_attr['height']
                                rf_info[key+'_center_x']=tmp_attr['center_x']
                                rf_info[key+'_center_y']=tmp_attr['center_y']
                                rf_info[key+'_width_x']=tmp_attr['width_x']
                                rf_info[key+'_width_y']=tmp_attr['width_y']
                                rf_info[key+'_rotation']=tmp_attr['rotation']
                        rf_list.append(rf_info)
                    rf_cells_dict[cell_specimen_id] =rf_list
    return rf_cells_dict

# exp session A : NM1, NM3
# exp session B : NM1, NS
# exp session C : NM1, NM2

def get_sess_key(config, sesstxt):
#    from config import Allen_Brain_Observatory_Config
#    config=Allen_Brain_Observatory_Config()
    for code,txt in config.session.iteritems():
        if sesstxt == txt :
            return code

def save_all_rf_data(
        exp_session,
        flist,
        session_RF_stim,
        save_loc,
        rf_session=['three_session_C', 'three_session_C2'],
        alpha=0.5,
        number_of_shuffles=5000):
    """Save the stimulus analysis results from Allensdk"""
    from helper_funcs import save_object
    for sess in exp_session:
        it_session = sess['session_type']
        if it_session in rf_session:
            data_set = boc.get_ophys_experiment_data(sess['id'])
            sparse_noise_type = session_RF_stim[it_session.split('_')[-1]]
            for cell_specimen_id, session_filter in tqdm(
                    flist.iteritems(),
                    desc="Deriving RFs for %s"%it_session,
                    total=len(flist)):
                if session_filter:
                    cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]
                    print("cell %s has index %d\n" % (cell_specimen_id, cell_index))
                    rf_list =[]
                    for lsn_deg in sparse_noise_type:
                        rf_data = rf.compute_receptive_field_with_postprocessing(
                            data_set, 
                            cell_index, 
                            lsn_deg, 
                            alpha=alpha, 
                            number_of_shuffles=number_of_shuffles)
                        #save_loc=config.Allen_analysed_stimulus_loc
                        fname="%s%s_%s.pkl"%(save_loc,cell_specimen_id,lsn_deg)
                        save_object(rf_data,fname)
                        print("Saved : %s" % (fname))
                        
def find_rf_data_files(exp_session, save_loc):
    import glob
    import os 
    from pprint import pprint
    rep_sess = exp_session[0]
    tmp_dat=boc.get_ophys_experiment_data(rep_sess['id'])
    cells_id = tmp_dat.get_cell_specimen_ids()
    for cell in cells_id:
        import ipdb;
        fname="%s%s_locally_sparse_noise*"%(save_loc,cell)
        file_list = glob.glob(fname)
        if file_list:
            return True
    return False
        
        
        
        
        
#python load_all_nwb.py 0 597 2>&1 | tee -a rf_log_.txt
""" Get RFs information"""
df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
all_cells_RF_info =[]
DATE_STAMP = time.strftime('%D').replace('/','_')
start = 50
end =100
if len(sys.argv) >1:
    start=int(sys.argv[1])
    end=int(sys.argv[2])
idx_range =np.arange(start,end)
for idx in tqdm(
            idx_range,
            desc='RF data',
            total=len(idx_range)):
    exps=exp_con_ids[idx]
    runtime=time.time()
    exp_session = boc.get_ophys_experiments(experiment_container_ids=[exps])
    cells_ID_list={}
    #import ipdb; ipdb.set_trace()
    for sess in exp_session:
        sess_code = get_sess_key(config,sess['session_type'])
        tmp=boc.get_ophys_experiment_data(sess['id'])
        cells_ID_list[sess['session_type']]=tmp.get_cell_specimen_ids()
    common_cells = session_filters(config,cells_ID_list) 
    print("start download response")
    #all_cells_RF_info.append(cells_RF_info)
    common_cell_id=[]
    for cell_specimen_id, session_filter in common_cells.iteritems():
        if session_filter:
            common_cell_id.append(cell_specimen_id)
    common_cells_info={}
    common_cells_info['cell_specimen_id'] = common_cell_id
    cci = np.sort(np.asarray(common_cell_id))
    data_set_record = {}
    for sess in exp_session:
        data_set= boc.get_ophys_experiment_data(sess['id'])
        data_set_record[sess['session_type']] = data_set
        _, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=[cci])
        #cell_indices = data_set.get_cell_specimen_indices(common_cell_id)
        #common_cells_info[sess['session_type']+'_indices']=cell_indices
    epoch_table = data_set.get_stimulus_epoch_table()
    
    
    data_set.list_stimuli()
    data_set.get_stimulus_table()

    tmptrace=data_set.get_corrected_fluorescence_traces()
    
_, corrected_traces = data_set.get_corrected_fluorescence_traces(cell_specimen_ids=[csid])
        
      dxcm, dxtime = data_set.get_running_speed()
      
      mc = data_set.get_motion_correction()

    
    
    
    print "Runtime for experiment container ID #%s is %s"%(exps,time.time()-runtime)








for exps in exp_con_ids:
    exp_session = boc.get_ophys_experiments(experiment_container_ids=[exps])
    cells_ID_list={}
    data_set={}
    import ipdb; ipdb.set_trace()

    for sess in exp_session:
        sess_code = get_sess_key(sess['session_type'])
        if sess_code =='C2':
            sess_code='C'
        tmp=boc.get_ophys_experiment_data(sess['id'])
        data_set[sess_code]=tmp
        cells_ID_list[sess['session_type']]=tmp.get_cell_specimen_ids()
    common_cells = session_filters(config,cells_ID_list)

    """Load Precal"""
    download_time = time.time()
    NM1_precal={}
    sess_ID= [k:v.get_metadata()['ophys_experiment_id'] for k,v in data_set.iteritems()]

    # exp session A : NM1, NM3
    fname = "%s%s_%s_precal_response.pkl"%(config.Response_loc,'NM1',sess_ID['A'])
    NM1_precal['A']=load_object(filename=fname) 
    fname = "%s%s_%s_precal_response.pkl"%(config.Response_loc,'NM3',sess_ID['A'])
    NM3_precal = load_object(filename=fname)

    # exp session B : NM1, NS
    data_set_B =data_set_list[config.session['B']
    sess_info = data_set_B.get_metadata()
    fname = "%s%s_%s_precal_response.pkl"%(config.Response_loc,'NM1',sess_ID['B'])
    NM1_precal['B']=load_object(filename=fname) 

    # exp session C : NM1, NM2
    data_set_C =data_set_list[sess_C]
    sess_info = data_set_C.get_metadata()
    fname = "%s%s_%s_precal_response.pkl"%(config.Response_loc,'NM1',sess_ID['C'])
    NM1_precal['C']=load_object(filename=fname) 
    fname = "%s%s_%s_precal_response.pkl"%(config.Response_loc,'NM2',sess_ID['C'])
    NM2_precal = load_object(filename=fname) 
    print("Total time downloading pre_cal NM responses : %s"%(time.time() - download_time))

    for ck,cv in common_cells.iteritems():
        if cv:
            neural_responses ={}
            cell_id = ck
            cell_index=[k:str(v.get_cell_specimen_indices([cell_id])[0]) for k,v in data_set.iteritems()]

            neural_responses['NM1_A'] = np.vstack(np.asarray(NM1_precal['A'][cell_index_A])) #[10 trials x num frames]
            neural_responses['NM1_B'] = np.vstack(np.asarray(NM1_precal['B'][cell_index_B]))
            neural_responses['NM1_C'] = np.vstack(np.asarray(NM1_precal['C'][cell_index_A]))

np.vstack(np.asarray(tmpres))

NM1_precal['A'][cell_index_A]







"""

rf_flist ={}
for sk, sv in cells_ID_list.iteritems():
    cell_rfs = []  # Boolean
    for cell in sv:
        cell_index = data_set.get_cell_specimen_indices([cell])[0]

    rf_flist[sk] = cell_rfs


cell_index = data_set.get_cell_specimen_indices([cell_specimen_id])[0]

print("cell %d has index %d" % (cell_specimen_id, cell_index))


# (1) loop through all stimuli types and save ONCE into separate numpys

# 
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
                neural_trace,
                stimuli:numpy_pointer_to_movie_or_images,  # Insert the directory pointer from above (1)
                stim_table:either store the raw stim table or point to it... doesnt matter,
                other_data:such as pupil responses and running)
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
"""