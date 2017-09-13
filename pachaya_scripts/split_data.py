# ee
import numpy as np
from Helper_Funcs import *
DATA_LOC = '/media/data_cifs/pachaya/AllenData/DataForTrain/Area-VISp_Depth-175um_NS-sig_NMall-nonzero_LSN-rfChi2-0.05_allStim-true/2017_08_22/'


"""
ns_input_alltrials_scaled.pkl
ns_input_fullsize.pkl
ns_input_scaled.pkl
ns_output_response.pkl
ns_unique_scene_response.pkl
"""
#Input scene
ns_input_fullsize = load_object(DATA_LOC+'ns_input_fullsize.pkl')

ns_input_fullsize_alltrials = np.repeat(ns_input_fullsize,50,axis=0) # sort from scene 1 - 118 

#Responses
ns_output_response = load_object(DATA_LOC+'ns_output_response.pkl')
ns_unique_scene_response = load_object(DATA_LOC+'ns_unique_scene_response.pkl') #  [#scene,#cell,3]
[num_scenes, num_cells] = ns_output_response.shape
#num_trials = np.shape(ns_unique_scene_response)[0]

num_test = 18; 
permu_sceneID = np.random.permutation(num_scenes)
testIDs = permu_sceneID[:num_test]; #(first #num_test ids)
trainIDs=permu_sceneID[num_test:];
train_scenes = ns_input_fullsize[trainIDs,:]   
test_scenes = ns_input_fullsize[testIDs,:]  
train_response= ns_output_response[trainIDs,:]
test_response =ns_output_response[testIDs,:]


framID_alltrials = np.repeat(range(num_scenes),50,axis=0)
save_object(train_scenes,DATA_LOC+'train_scenes.pkl')
save_object(test_scenes,DATA_LOC+'test_scenes.pkl')
save_object(train_response,DATA_LOC+'train_response.pkl')
save_object(test_response,DATA_LOC+'test_response.pkl')

np.save(DATA_LOC+'train_scenes.npy',train_scenes)
np.save(DATA_LOC+'test_scenes.npy',test_scenes)
np.save(DATA_LOC+'train_response.npy',train_response)
np.save(DATA_LOC+'test_response.npy',test_response)
save_object(test_scenes,DATA_LOC+'test_scenes.pkl')
save_object(train_response,DATA_LOC+'train_response.pkl')
save_object(test_response,DATA_LOC+'test_response.pkl')