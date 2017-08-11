# Config
import os

def get_host_path(HOST=False,PATH=True):
    import socket
    if socket.gethostname() =='x7' :
        host='x7'; path='/home/pachaya/Allen_Brain_Observatory/'
    elif socket.gethostname() =='g13':
        host='g13'; path= '/home/pachaya/Allen_Brain_Observatory/'
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
        
        # Allen Data location
        #self.data_loc = '/media/data/pachaya/AllenData/'
        self.data_loc = '/media/data_cifs/pachaya/AllenData/'
        self.repo_PATH = get_host_path()
        self.host = get_host_path(HOST=True, PATH=False)
        
        self.FILTERS = True
        self.filters_file =  get_host_path()+'filters_VISp_175_sigNS_nonzeroNM_reliableRF.pkl'  # None if no filters
        self.data_set_code = 'Area-VISp_Depth-175um_NS-sig_NMall-nonzero_LSN-rfChi2-0.05_allStim-true'
        # Order for Name Code 
        # Area / depth / Stimuli - SG, DG, NS, NM, LSN / aShow only data with results from all stimuli allStim = True

        self.RESHAPE_IMG_NS = True
        self.reshape_img_size_h = 31
        self.reshape_img_size_w = 31
        self.save_folder ='DataForTrain/'

        # Find all precomputed cell metrics  here http://alleninstitute.github.io/AllenSDK/brain_observatory.html#precomputed-cell-metrics
        # Natural Scenes response p value	p_ns
        # natural movie 1	    response reliability (session A)	reliability_nm1_a
        #                          response reliability (session B)	reliability_nm1_b
        #                          response reliability (session C)	reliability_nm1_c
        # natural movie 2	response reliability	reliability_nm2
        # natural movie 3	response reliability	reliability_nm3
        # locally sparse noise      RF chi^2	rf_chi2_lsn   [blank/nonblank]
   
"""
        # Image directories
        self.clicktionary_dir = '/media/data_cifs/clicktionary/'
        self.image_base_path = os.path.join(
            self.clicktionary_dir, 'webapp_data')
        self.training_images = os.path.join(
            self.image_base_path, 'lmdb_trains')
        self.tf_train_name = 'train_7.tfrecords'
        self.validation_images = os.path.join(
            self.image_base_path, 'lmdb_validations')
        self.tf_val_name = 'val.tfrecords'
        self.coco_validation_images = os.path.join(
            self.image_base_path, 'coco_overlap_lmdb_validations')
        self.coco_tf_val_name = 'coco_val.tfrecords'
        self.im_ext = '.JPEG'

        # Project file directories
        self.project_base_path = '/media/data_cifs/clicktionary/clickme_experiment'
        # self.project_base_path = '/home/drew/clickme/'
        # self.project_base_path = '/media/cifs_all/charlie/clickme/'            
        self.tf_record_base = os.path.join(
            self.project_base_path, 'tf_records')
            # '/home/drew/clickme_tf_records/')
        self.results = os.path.join(self.project_base_path, 'results')
        self.train_checkpoint = os.path.join(
            self.project_base_path, 'attgrad_vgg_checkpoints')
        self.train_summaries = os.path.join(
            self.project_base_path, 'attgrad_vgg_summaries')

        # Image settings
        self.image_size = [256, 256, 3]
        self.click_box_radius = 7  # For human clickme observers: 7; for CNNs observers: 20
        self.viz_images = [
            '163_0.JPEG', '23_0.JPEG', '31_0.JPEG',
            '403_0.JPEG', '671_0.JPEG', '818_0.JPEG', '838_0.JPEG',
            '209_0.JPEG', '23_1.JPEG', '339_0.JPEG', '404_0.JPEG',
            '815_0.JPEG', '834_0.JPEG', '838_1.JPEG', '209_1.JPEG',
            '308_0.JPEG', '340_0.JPEG', '471_48043.JPEG', '817_0.JPEG',
            '837_0.JPEG', '838_2.JPEG']
        self.viz_images = [os.path.join(
            self.project_base_path, 'test_images', x) for x in self.viz_images]
        self.hm_scoring = 'uniform'  # linear_decrease/uniform/linear_increase
        self.investigate_subjects = ['r1Ieyvtax', 'B1x-611pl']  # relive_this@live.com ; ['rykyZPzTl'], freemoneyq ['rJkI8aOox'], # 'BklslAOgb', HJy2nW9k 'r1Ieyvtax', 'B1x-611pl'
        self.consolidation_type = 'both'  # clicks, consolidated, or both
        self.click_syn_file = os.path.join(self.clicktionary_dir, 'clicktionary_image_categories.txt')
        with open(self.click_syn_file) as f:
            content = f.readlines()
            self.heatmap_image_dict = {
                v.split(' ')[0]: int(
                    v.split(' ')[1].strip('\n')) for v in content}

        # Model settings
        self.vgg16_weight_path = os.path.join(
            self.clicktionary_dir, 'pretrained_weights', 'vgg16.npy')
        self.train_batch = 32
        self.validation_batch = 32
        # validation_batch * num_validation_evals is num of val images to test
        self.num_validation_evals = 1000
        self.validation_iters = 2000  # test validation every this # of steps
        self.epochs = 2 # 400  # Increase since we are augmenting
        self.top_n_validation = 0  # set to 0 to save all
        self.model_image_size = [224, 224, 3]
        self.output_shape = 1000  # how many categories for classification
        # choose from ['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2',
        # 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2',
        # 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
        self.fine_tune_layers = ['fc6', 'fc7', 'fc8']
        self.initialize_layers = ['fc6', 'fc7', 'fc8']  # must be in fine_tune_layers
        self.wd_layers = ['fc6', 'fc7', 'fc8']
        self.batchnorm_layers = ['fc6', 'fc7', 'fc8']  # ['fc6', 'fc7', 'fc8']  # ['fc6', 'fc7', 'fc8']  # ['fc6', 'fc7', 'fc8']
        self.optimizer = 'adam'  # 'adam' 'sgd'
        self.hold_lr = 1e-8 # 1e-8
        self.new_lr = 3e-4  # 1e-6
        self.keep_checkpoints = 60  # max # of checkpoints
        self.grad_clip = False
        self.weight_loss_with_counts = True
        self.reweighting = 'uniform'  # if above is true, 'uniform' or 'counts'

        # Attention settings
        self.attention_layers = ['fc8']  # [
        # self.attention_layers = ['conv1_2', 'conv2_1', 'conv3_1',
        # 'conv4_1', 'conv5_1'] # [
        # 'conv1_2', 'conv2_2', 'conv3_3', 'conv4_3', 'conv5_3']
        self.attention_type = 'activation'  # 'gradient', 'activation', 'lrp'
        self.combine_type = 'sum_abs'  # 'pass' 'sum_abs' 'sum_p' 'max_p'
        self.plot_gradients = True
        self.loss_function = 'l2'  # 'l2' or 'huber' 'log_loss' 'masked_l2'
        self.attention_loss = 'distance'
        self.normalize = 'l2'  # 'l2 or z or 'softmax' or sigmoid or none
        self.reg_penalty = 0.01  # 0.001  # 0.01  #0.05  # 0.01
        self.wd_penalty = None  # 5e-5
        self.loss_type = 'joint'  # 'joint'
        self.targeted_gradient = True  # True  # Requires attention_layers = ['fc8']
        self.blur_maps = 30  # 0 = no, > 0 blur kernel

        # choose from: random_crop
        self.data_augmentations = [
            'random_crop_resize', 'left_right',
            'random_contrast', 'random_brightness'
        ]

        ######
        # Visualization settings
        ######

        # Directory with images for heatmaps
        self.heatmap_dataset_images = os.path.join(
            '/home/andreas/charlie', 'images_for_heatmaps')
        self.restore_model =  None  #  '/media/data_cifs/clicktionary/clickme_experiment/checkpoints/gradient_001_112341_2017_05_16_16_20_26/model_18000.ckpt-18000'
        self.heatmap_image_labels = '/home/andreas/charlie/MIRC_behavior/exp_3_all_images_no_mircs'

        # Images for visualization parameters
        # > 0 = number of images, < 0 = proportion of images
        self.heatmap_image_amount = 1000
        self.heatmap_batch = 1

        # Bubbles parameters
        self.visualization_output = '/home/andreas/charlie/MIRC_behavior/click_comparisons/heatmaps_for_paper/clickme_bubbles_exp_3_baseline'
        self.generate_plots = True
        self.use_true_label = True
        self.block_size = 14
        self.block_stride = 1

        # update attributes
        self.__dict__.update(kwargs)

"""