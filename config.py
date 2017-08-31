# Config
import os

def get_host_path(HOST=False,PATH=True):
    import socket
    if socket.gethostname() =='x7' :
        host='x7'; path='/home/pachaya/Allen_Brain_Observatory/'
    elif socket.gethostname() =='g13':
        host='g13'; path= '/home/pachaya/Allen_Brain_Observatory/'
    elif socket.gethostname() =='x8' :
        host='x8'; path='/home/pachaya/Allen_Brain_Observatory/'
    else:
        raise Exception('Unknown Host : Please add your directory at get_host_path()')
    if(HOST&PATH):
        return host, path
    elif(HOST):
        return host
    else:
        return path

        
class Allen_Brain_Observatory_Config():
    def __init__(self, **kwargs):
        self._id=''
        """ Directories"""
        self.repo_PATH = get_host_path()
        self.host = get_host_path(HOST=True, PATH=False)

        self.data_loc = '/media/data_cifs/AllenData/'
        self.manifest_file=self.data_loc+'boc/manifest.json'
        self.DB_loc = 'DataForTrain/'
        self.stimulus_template_loc = self.data_loc+self.DB_loc+'all_stimulus_template/'
        
        self.RF_info_loc = self.data_loc+self.DB_loc+'all_RFs_info/'
        
        self.imaging_response_loc= self.data_loc+self.DB_loc+'all_imaging_responses/'
        self.fluorescence_traces_loc= self.imaging_response_loc +'fluorescence_traces/'
        self.ROIs_mask_loc= self.imaging_response_loc +'ROIs_mask/'
        self.stim_table_loc= self.imaging_response_loc +'stim_tables/'
        self.specimen_recording_loc= self.imaging_response_loc +'specimen_recording/'
        
        self.output_pointer_loc = self.data_loc+self.DB_loc+'output_pointers/'
        
        self.Allen_analysed_stimulus_loc=self.data_loc+self.DB_loc+'Allen_stimulus_analysis/By_cell_ID/'
        self.precal_matrix_loc=self.data_loc+self.DB_loc+'Allen_stimulus_analysis/By_container_ID/'
        
        """Brain Observatory project information"""
        self.stim={
        'DG':'drifting_gratings',
        'LSN':'locally_sparse_noise',
        'LSN4':'locally_sparse_noise_4deg',
        'LSN8':'locally_sparse_noise_8deg',
        'NM1':'natural_movie_one',
        'NM2':'natural_movie_two',
        'NM3':'natural_movie_three',
        'NS':'natural_scenes',
        'Spon':'spontaneous',
        'SG':'static_gratings'}
        self.session={
        'A':u'three_session_A',
        'B':u'three_session_B',
        'C':u'three_session_C',
        'C2':u'three_session_C2'
        }
        self.session_RF_stim ={
        'C': ['locally_sparse_noise'],
        'C2': ['locally_sparse_noise_4deg','locally_sparse_noise_8deg'],
        }
        self.RF_sign = ['on', 'off']
        """Parameters"""
        self.rf_shuffles = 5000
        self.alpha = 0.5
        self.FILTERS = True
        self.filters_file =  get_host_path()+'filters_VISp_175_sigNS_nonzeroNM_reliableRF.pkl'  # None if no filters
        self.data_set_code = 'Area-VISp_Depth-175um_NS-sig_NMall-nonzero_LSN-rfChi2-0.05_allStim-true'
        # Order for Name Code 
        # Area / depth / Stimuli - SG, DG, NS, NM, LSN / aShow only data with results from all stimuli allStim = True
        self.RESHAPE_IMG_NS = True
        self.reshape_img_size_h = 31
        self.reshape_img_size_w = 31
        self.save_folder ='DataForTrain/'
        


        self.db_ssh_forward = False

        # Find all precomputed cell metrics  here http://alleninstitute.github.io/AllenSDK/brain_observatory.html#precomputed-cell-metrics
        # Natural Scenes response p value	p_ns
        # natural movie 1	    response reliability (session A)	reliability_nm1_a
        #                          response reliability (session B)	reliability_nm1_b
        #                          response reliability (session C)	reliability_nm1_c
        # natural movie 2	response reliability	reliability_nm2
        # natural movie 3	response reliability	reliability_nm3
        # locally sparse noise      RF chi^2	rf_chi2_lsn   [blank/nonblank]
   
