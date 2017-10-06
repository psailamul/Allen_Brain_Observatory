"""Declare new allen datasets to be encoded as TFrecords."""
import numpy as np


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
            'neural_delay': 150,  # MS delay for trimming neural data
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
            # natural_movie_one
            # natural_movie_one
            # natural_movie_one
            # natural_scenes
            'stimuli': [
                'natural_movie_one',
                # 'natural_movie_two',
                'natural_movie_three'
            ],
            'sessions': [
                'three_session_A',
                # 'three_session_B',
                'three_session_C2'
            ],
            'cv_split': {
                'random_cv_split': 0.9
            },
            'np_type': np.float32
        }
        # exp_dict['cv_split'] = {
        #     'stimulus_name': 'natural_movie_one'  # Specify train set
        # }

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def ALLEN_selected_cells_1_extended(self):
        """An expanded set of cells, similar RF properties ("on" response)."""
        exp_dict = self.ALLEN_all_neurons()
        exp_dict['rf_coordinate_range'] = [
            {
                'x_min': 20,
                'x_max': 30,
                'y_min': 40,
                'y_max': 50,
            }
        ]
        exp_dict['experiment_name'] = 'ALLEN_selected_cells_1_extended'
        exp_dict['only_process_n'] = None  # Set to None to process all
        return self.add_globals(exp_dict)

    def ALLEN_all_neurons(self):
        """Pull data from all neurons."""
        exp_dict = {
            'experiment_name': 'ALLEN_all_neurons',
            'only_process_n': None,  # Set to None to process all
            'randomize_selection': True,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_coordinate_range': [{  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }],
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
        # exp_dict['cv_split'] = {
        #     'stimulus_name': 'natural_movie_one'  # Specify train set
        # }
        return exp_dict

    # 10/5/17 datasets
    def ALLEN_selected_cells_1(self):
        """A single cell from the dense RF region."""
        exp_dict = self.ALLEN_all_neurons()
        exp_dict = {
            'experiment_name': 'ALLEN_selected_cells_1',
            'only_process_n': 1,  # Set to None to process all
            'randomize_selection': True,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_coordinate_range': [{  # Get all cells
                'x_min': 26,
                'x_max': 28,
                'y_min': 45,
                'y_max': 47,
            }],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'store_means': [
                'image',
                'label'
            ],
            'cc_repo_vars': {
                'output_size': [1, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            },
            # 'deconv_method': 'elephant'
        }
        exp_dict = self.add_globals(exp_dict)
        exp_dict['cv_split'] = {
            'split_on_stim': 'natural_movie_one'  # Specify train set
        }
        return self.add_globals(exp_dict)

    def ALLEN_selected_cells_103(self):
        """23 cells from the dense RF region."""
        exp_dict = self.ALLEN_all_neurons()
        exp_dict = {
            'experiment_name': 'ALLEN_selected_cells_103',
            'only_process_n': 103,  # Set to None to process all
            'randomize_selection': True,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_coordinate_range': [{  # Get all cells
                'x_min': 26,
                'x_max': 28,
                'y_min': 45,
                'y_max': 47,
            }],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'store_means': [
                'image',
                'label'
            ],
            'cc_repo_vars': {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            },
            # 'deconv_method': 'elephant'
        }
        exp_dict = self.add_globals(exp_dict)
        exp_dict['cv_split'] = {
            'split_on_stim': 'natural_movie_one'  # Specify train set
        }
        return self.add_globals(exp_dict)

    def ALLEN_all_neurons_random(self):
        """103 random cells from across the visual field."""
        exp_dict = {
            'experiment_name': 'ALLEN_all_neurons',
            'only_process_n': 103,  # Set to None to process all
            'randomize_selection': True,
            'reference_image_key': {'proc_stimuli': 'image'},
            'reference_label_key': {'neural_trace_trimmed': 'label'},
            'rf_coordinate_range': [{  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'store_means': [
                'image',
                'label'
            ],
            'cc_repo_vars': {
                'output_size': [103, 1],  # target variable -- neural activity,
                'model_im_size': [152, 304, 1],
                'loss_function': 'pearson',
                'score_metric': 'pearson',
                'preprocess': 'resize'
            },
            # 'deconv_method': 'elephant'
        }
        exp_dict = self.add_globals(exp_dict)
        exp_dict['cv_split'] = {
            'split_on_stim': 'natural_movie_one'  # Specify train set
        }
        return exp_dict
