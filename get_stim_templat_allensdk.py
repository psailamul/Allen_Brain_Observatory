# Get stimulus template as it is in AllenSDK ( not process)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import allensdk.brain_observatory.stimulus_info as si
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from config import Allen_Brain_Observatory_Config
from tqdm import tqdm
from helper_funcs import *
import time 

config=Allen_Brain_Observatory_Config()
boc = BrainObservatoryCache(manifest_file=config.manifest_file)

#boc.get_all_stimuli()
"""stim_list = ['drifting_gratings',
 'locally_sparse_noise',
 'locally_sparse_noise_4deg',
 'locally_sparse_noise_8deg',
 'natural_movie_one',
 'natural_movie_three',
 'natural_movie_two',
 'natural_scenes',
 'spontaneous',
 'static_gratings']"""

available_stims = [ 'locally_sparse_noise',
    'locally_sparse_noise_4deg',
    'locally_sparse_noise_8deg',
    'natural_movie_one',
    'natural_movie_three',
    'natural_movie_two',
    'natural_scenes']
 
df = pd.read_csv('all_exps.csv')
exp_con_ids = np.asarray(df['experiment_container_id'])
SAVE_LOC=config.stimulus_template_loc
exps_tmpl = boc.get_ophys_experiments(experiment_container_ids=[exp_con_ids[0]])
all_templates = {}
for sess in exps_tmpl:
    data_set = boc.get_ophys_experiment_data(sess['id'])
    stim_lists = data_set.list_stimuli()
    for stim in stim_lists:
        if stim in available_stims:
            template = data_set.get_stimulus_template(stim)
            fname ="%s%s_template.pkl"%(SAVE_LOC,stim)
            save_object(template, fname)
            print "Save template for %s at \n \t %s"%(stim,fname)
            all_templates[stim]=template

save_object(all_templates,"%sall_available_templates_lsn_nm_ns.pkl"%SAVE_LOC)
keys = all_templates.keys()
for k in all_templates.keys():
    fname="%s%s_np.npy"%(SAVE_LOC,k)
    np.save(fname, all_templates[k])