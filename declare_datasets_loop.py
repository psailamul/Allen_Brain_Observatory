"""Declare new allen datasets to be encoded as TFrecords."""
import os
import sys
import shutil
import numpy as np
import encode_datasets
from config import Allen_Brain_Observatory_Config
from db import db
from glob import glob
from ops import py_utils


class declare_allen_datasets():
    """Class for declaring datasets to be encoded as tfrecords."""

    def __getitem__(self, name):
        """Method for addressing class methods."""
        return getattr(self, name)

    def __contains__(self, name):
        """Method for checking class contents."""
        return hasattr(self, name)

    def globals(self):
        """Global variables for all datasets."""
        return {
            'neural_delay': 5,  # MS delay * 30fps for neural data
            'tf_types': {  # How to store each in tfrecords
                'neural_trace_trimmed': 'float',
                'proc_stimuli': 'string',
                'ROImask': 'string',
                'pupil_size': 'float',
                'running_speed': 'float',
                'eye_locations_spherical': 'float',
                'cell_specimen_id': 'float',
                'on_center_x': 'float',
                'on_center_y': 'float',
                'off_center_x': 'float',
                'off_center_y': 'float',
                'on_width_x': 'float',
                'off_width_y': 'float',
                'event_index': 'float',
                'stimulus_name': 'string',
                'stimulus_iterations': 'float'
            },
            'include_targets': {  # How to store this data in tfrecords
                # 'neural_trace_trimmed': 'split',
                # 'proc_stimuli': 'split',
                'image': 'split',  # Corresponds to reference_image_key
                'stimulus_name': 'repeat',
                'event_index': 'split',
                'label': 'split',  # Corresponds to reference_label_key
                'ROImask': 'repeat',
                'stimulus_iterations': 'split',
                # 'pupil_size': 'split',
                # 'running_speed': 'split', \
                # 'eye_locations_spherical': 'split',
                'cell_specimen_id': 'repeat',
                # 'on_center_x': 'repeat',
                # 'on_center_y': 'repeat',
                # 'off_center_x': 'repeat',
                # 'off_center_y': 'repeat',
                # 'on_width_x': 'repeat',
                # 'off_width_y': 'repeat'
            },
            'deconv_method': None,
            'randomize_selection': False,
            'warp_stimuli': False,
            'slice_frames': 15,  # Sample every N frames
            'process_stimuli': {
                    # 'natural_movie_one': {  # 1080, 1920
                    #     'resize': [304, 608],  # [270, 480]
                    #  },
                    # 'natural_movie_two': {
                    #     'resize': [304, 608],  # [270, 480]
                    # },
                    # 'natural_movie_three': {
                    #     'resize': [304, 608],  # [270, 480]
                    # },
                    'natural_scenes': {
                        'pad': [1080, 1920],  # Pad to full movie size
                        'resize': [304, 608],  # [270, 480]
                    },
                },
            # natural_movie_one
            # natural_movie_two
            # natural_movie_three
            # natural_scenes
            'stimuli': [
                'natural_movie_one',
                'natural_movie_two',
                'natural_movie_three'
            ],
            'sessions': [
                'three_session_A',
                # 'three_session_B',
                # 'three_session_C'
                # 'three_session_C2'
            ],
            'cv_split': {
                'split_on_stim': 'natural_movie_three'  # Specify train set
            },
            'data_type': np.float32,
            'image_type': np.float32,
        }

    def template_dataset(self):
        """Pull data from all neurons."""
        exp_dict = {
            'experiment_name': 'ALLEN_all_neurons',
            'only_process_n': None,  # Set to None to process all
            'randomize_selection': True,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_query': [{
                'rf_coordinate_range': {  # Get all cells
                    'x_min': 40,
                    'x_max': 70,
                    'y_min': 20,
                    'y_max': 50,
                },
                'cre_line': 'Cux2',
                'structure': 'VISp',
                'imaging_depth': 175}
            ],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'store_means': [
                'image',
                'label'
            ],
            'cc_repo_vars': {
                'output_size': [2, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            },
            # 'deconv_method': 'elephant'
        }
        exp_dict = self.add_globals(exp_dict)
        exp_dict['cv_split'] = {
            'split_on_stim': 'natural_movie_three'  # Specify train set
        }
        return exp_dict

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def ALLEN_ss_cells_1_movies(self):
        """1 cell from across the visual field."""
        exp_dict = self.template_dataset()
        exp_dict = self.add_globals(exp_dict)
        exp_dict['experiment_name'] = 'ALLEN_ss_cells_1_movies'
        exp_dict['only_process_n'] = 1
        exp_dict['randomize_selection'] = True
        exp_dict['reference_image_key'] = {'proc_stimuli': 'image'}
        exp_dict['reference_label_key'] = {'neural_trace_trimmed': 'label'}
        exp_dict['rf_query'] = [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': 20,
                'x_max': 30,
                'y_min': 50,
                'y_max': 60,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
        exp_dict['cross_ref'] = 'rf_coordinate_range_and_stimuli'
        exp_dict['store_means'] = [
                'image',
                'label'
            ]
        # exp_dict['deconv_method'] = 'c2s'
        exp_dict['cc_repo_vars'] = {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [354, 608, 1],  # [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            }
        return exp_dict


def build_multiple_datasets(
        template_dataset='ALLEN_ss_cells_1_movies',
        template_experiment='ALLEN_selected_cells_1',
        model_structs='ALLEN_selected_cells_1'):
    """Dictionary with all cell queries to run."""
    main_config = Allen_Brain_Observatory_Config()

    # Append the BP-CC repo to this python path
    sys.path.append(os.path.join(main_config.cc_path))
    import experiments  # from BP-CC
    import prepare_experiments  # from BP-CC
    exps = experiments()

    # Query neuron data
    queries = [
        [{
            'rf_coordinate_range': {  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            },
            'cre_line': 'Cux2',
            'structure': 'VISp',
            'imaging_depth': 175}]
    ]

    filter_by_stim = [
                'natural_movie_one',
                'natural_movie_two',
                'natural_movie_three']
    print 'Pulling cells by their RFs and stimulus: %s.' % filter_by_stim
    all_data_dicts = []
    for q in queries:
        all_data_dicts += [db.get_cells_all_data_by_rf_and_stimuli(
            rfs=q,
            stimuli=filter_by_stim)]

    # Prepare directories
    model_directory = os.path.join(
        main_config.cc_path,
        'models',
        'structs')
    model_templates = glob(
        os.path.join(
            model_directory,
            model_structs,
            '*.py'))

    # Loop through each query and build all possible datasets with template
    da = declare_allen_datasets()
    for q in all_data_dicts:
        for idx, d in enumerate(q[0]):
            # 1. Prepare dataset
            dataset_method = da[template_dataset]()
            rf_query = dataset_method['rf_query'][0]
            rf_query['rf_coordinate_range']['x_min'] = np.floor(
                d['on_center_x'])
            rf_query['rf_coordinate_range']['x_max'] = np.floor(
                d['on_center_x']) + 1
            rf_query['rf_coordinate_range']['y_min'] = np.floor(
                d['on_center_y'])
            rf_query['rf_coordinate_range']['y_max'] = np.floor(
                d['on_center_y']) + 1
            rf_query['rf_coordinate_range']['structure'] = d[
                'structure']
            rf_query['rf_coordinate_range']['cre_line'] = d[
                'cre_line']
            rf_query['rf_coordinate_range']['imaging_depth'] = d[
                'imaging_depth']
            dataset_method['rf_query'][0] = rf_query

            # 2. Encode dataset
            encode_datasets.main(dataset_method)

            # 3. Prepare models in CC-BP
            dataset_name = '%s_%s_%s_%s_%s_%s' % (
                d['structure'],
                d['cre_line'],
                d['imaging_depth'],
                d['on_center_x'],
                d['on_center_y'],
                idx)
            new_model_dir = os.path.join(
                model_directory,
                dataset_name)
            py_utils.make_dir(new_model_dir)
            for f in model_templates:
                dest = os.path.join(
                    new_model_dir,
                    f.split(os.path.sep)[-1])
                shutil.copy(f, dest)

            # 4. Add dataset to CC-BP database
            it_exp = exps[template_experiment]()
            it_exp['experiment_name'] = dataset_name
            it_exp['dataset'] = dataset_name
            prepare_experiments.main(
                reset_process=False,
                initialize_db=False,
                experiment_name=dataset_name)


if __name__ == '__main__':
    build_multiple_datasets()
