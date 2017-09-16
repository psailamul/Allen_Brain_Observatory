class declare_allen_datasets():
    """Class for declaring datasets to be encoded as tfrecords."""
    def __getitem__(self, name):
        """Method for addressing class methods."""
        return getattr(self, name)

    def __contains__(self, name):
        """Method for checking class contents."""

    def globals(self):
        """Global variables for all datasets."""
        return {
            'neural_delay': 150  # MS delay for trimming neural data
        }

    def add_globals(self, exp):
        """Add attributes to this class."""
        for k, v in self.globals().iteritems():
            exp[k] = v
        return exp

    def all_neurons(self):
        """Pull data from all neurons."""
        exp_dict = {
            'experiment_name': 'all_neurons',
            'only_process_n': 2,  # Set to None to process all
            'reference_data_key': 'proc_stimuli',
            'rf_coordinate_range': [{  # Get all cells
                'x_min': -10000,
                'x_max': 10000,
                'y_min': -10000,
                'y_max': 10000,
            }],
            'stimuli': ['movies'],
            'cross_ref': 'rf_coordinate_range_and_stimuli',
            'include_targets': {  # How to store this data in tfrecords
                'neural_trace_trimmed': 'split',
                'proc_stimuli': 'split',
                'ROImask': 'repeat',
                'pupil_size': 'split',
                'running_speed': 'split', 
                'eye_locations_spherical': 'split',
                'cell_specimen_id': 'repeat',
                'on_center_x': 'repeat',
                'on_center_y': 'repeat',
                'off_center_x': 'repeat',
                'off_center_y': 'repeat',
                'on_width_x': 'repeat',
                'off_width_y': 'repeat'
            }
        }
        return self.add_globals(exp_dict)
