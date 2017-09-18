import os
import argparse
import numpy as np
import tensorflow as tf
from db import db
from config import Allen_Brain_Observatory_Config as Config
from declare_datasets import declare_allen_datasets as DA
from tqdm import tqdm
import cPickle as pickle


def create_data_loader_class(template_file, meta_dict, output_file):
    """Write a data loader python class for this dataset."""
    with open(template_file, 'r') as f:
        text = f.readlines()
    import ipdb;ipdb.set_trace()
    for k, v in meta_dict:
        text.replace(k, v)
    with open(output_file, 'w') as f:
        f.writelines(text)


def fixed_len_feature(length=[], dtype='int64'):
    """Return variables for loading data with tensorflow."""
    if dtype == 'int64':
        return tf.FixedLenFeature(length, tf.int64)
    elif dtype == 'string':
        return tf.FixedLenFeature(length, tf.string)
    elif dtype == 'float':
        return tf.FixedLenFeature(length, tf.float32)
    else:
        raise RuntimeError('Cannot understand the fixed_len_feature dtype.')


def load_data(f, allow_pkls=False):
    """Load data from npy or pickle files."""
    ext = f.split('.')[-1]
    assert ext is not None, 'Cannot find an extension on file: %s' % f
    if not allow_pkls:
        ext = 'npy'
        f = '%s.npy' % f.split('.')[0]
    if ext == 'npy' or ext == 'npz':
        return np.load(f)
    elif ext == 'pkl' or ext == 'p':
        out = pickle.load(open(f, 'rb'))
        return out
    else:
        raise RuntimeError('Cannot understand file extension %s' % ext)


def bytes_feature(values):
    """Encodes an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Encodes an int list into a tf int64 list for a tfrecord."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """TF floating point feature for tfrecords."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fix_malformed_pointers(d, filter_ext='.npy', rep_ext='.npz'):
    """Replace .npy extensions if they were erroniously placed in the DB."""
    if '.npy' in d:
        d = '%s%s' % (d.split(filter_ext)[0], rep_ext)
    return d


def load_npzs(data_dicts, exp_dict):
    """Load cell data from an npz."""
    data_files = []
    key_list = []
    if exp_dict['only_process_n'] is not None or exp_dict['only_process_n'] > 0:
        print 'Trimming query from %s to %s cells.' % (len(data_dicts), exp_dict['only_process_n'])
        data_dicts = data_dicts[:exp_dict['only_process_n']]
        
    output_data = []
    for d in tqdm(data_dicts, total=len(data_dicts), desc='Preparing data'):
        df = {}
        data_pointer = fix_malformed_pointers(d['cell_output_npy'])
        cell_data = load_data(data_pointer, allow_pkls=True)
        cell_id = d['cell_specimen_id']
        df['cell_specimen_id'] = cell_id

        # Stim table
        stim_table = load_data(cell_data['stim_table'].item(), allow_pkls=True)
        stim_table = stim_table['stim_table']

        # Stimuli
        raw_stimuli = load_data(cell_data['stim_template'].item(), allow_pkls=False)
        df['raw_stimuli'] = raw_stimuli
        proc_stimuli = raw_stimuli[stim_table[:, 0]]
        df['proc_stimuli'] = proc_stimuli

        # Neural data
        neural_data = load_data(cell_data['neural_trace'].item(), allow_pkls=True)
        neural_data = neural_data['corrected_trace']
        df['neural_trace'] = neural_data

        # Trim neural data
        stim_table_idx = stim_table[:, 1] + exp_dict['neural_delay']
        neural_data_trimmed = neural_data[stim_table_idx]
        df['neural_trace_trimmed'] = neural_data_trimmed

        # ROI mask
        ROImask = load_data(cell_data['ROImask'].item(), allow_pkls=True)
        ROImask = ROImask['roi_loc_mask']
        df['ROImask'] = ROImask[:, :, None]

        # AUX data
        aux_data = load_data(cell_data['other_recording'].item(), allow_pkls=True)
        pupil_size = aux_data['pupil_size']
        running_speed = aux_data['running_speed']
        eye_locations_spherical = aux_data['eye_locations_spherical']
        eye_locations_cartesian = aux_data['eye_locations_cartesian']
        df['pupil_size'] = pupil_size[stim_table_idx]
        df['running_speed'] = running_speed.item()['dxcm'][stim_table_idx]
        df['eye_locations_spherical'] = eye_locations_spherical[stim_table_idx, :]
        df['eye_locations_cartesian'] = eye_locations_cartesian[stim_table_idx, :]

        # RF data
        df['on_center_x'] = d['on_center_x']
        df['on_center_y'] = d['on_center_y']
        df['off_center_x'] = d['off_center_x']
        df['off_center_y'] = d['off_center_y']
        df['on_height'] = d['on_height']
        df['on_width'] = d['on_width_x']
        df['off_height'] = d['off_height']
        df['off_width'] = d['off_width_x']

        # Package data
        df = {k: v for k, v in df.iteritems() if k in exp_dict['include_targets']}
        output_data += [df]
        it_check = [k for k, v in df.iteritems() if v is not None] 
        key_list += [it_check]
    keep_keys = np.unique(key_list)
    remove_keys = list(set(exp_dict['include_targets'].keys()) - set(keep_keys))
    if remove_keys is not None:
        print 'Removing keys which were not populated across cells: %s' % remove_keys
        for idx, d in enumerate(output_data):
            it_d = {k: v for k, v in d.iteritems() if k in keep_keys}
            output_data[idx] = it_d
    # TODO handle NaNs in output_data here

    #Concatenate data into equal-sized lists
    event_dict = []
    for d in output_data: 
        ref_length = d[exp_dict['reference_data_key']].shape[0]
        for idx in range(ref_length):
            it_event = {}
            for k, v in d.iteritems():
                if exp_dict['include_targets'][k] == 'split':
                    it_event[k] = v[idx]
                elif exp_dict['include_targets'][k] == 'repeat':
                    it_event[k] = v
                else:
                    raise RuntimeError('Fucked up packing data into list of dicts.')
            event_dict += [it_event]
    return event_dict


def create_example(data_dict, feature_types):
    """Create entry in tfrecords."""
    tf_dict = {}
    for k, v in data_dict.iteritems():
        it_feature_type = feature_types[k]
        if it_feature_type == 'float':
            tf_dict[k] = float_feature(v)
        elif it_feature_type == 'int64':
            tf_dict[k] = int64_feature(v)
        elif it_feature_type == 'string':
            tf_dict[k] = bytes_feature(v.tostring())
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=tf_dict
        )
    )


def prepare_tf_dicts(data_dict, feature_types):
    """Prepare tf data types for loading tf variables."""
    tf_dict = {}
    for k, v in data_dict.iteritems():
        it_feature_type = feature_types[k]
        tf_dict[k] = fixed_len_feature(dtype=v)
    return tf_dict


def prepare_data_for_tf_records(
        data_files,
        output_directory,
        set_name,
        cv_split,
        store_means,
        feature_types,
        cc_repo=None,
        stimuli_key='proc_stimuli',
        ext='tfrecords'):
    """Package dict into tfrecords."""
    if isinstance(cv_split, float):
        cv_inds = np.random.permutation(len(data_files))
        val_len = np.round(len(data_files) * cv_split).astype(int)
        val_ind = cv_inds[val_len:]
        train_ind = cv_inds[:val_len]
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
    elif isinstance(cv_split, basestring):
        raise RuntimeError('Split by session...')
    else:
        raise RuntimeError('Selected crossvalidation %s is not yet implemented.' % cv_split)
    means = {k: [] for k in store_means}
    for k, v in cv_data.iteritems():
        it_name = os.path.join(
            output_directory,
            '%s_%s.%s' % (k, set_name, ext))
        with tf.python_io.TFRecordWriter(it_name) as tfrecord_writer:
            for idx, d in tqdm(enumerate(v), total=len(v), desc='Encoding %s' % k):
                for imk, imv in means.iteritems():
                    means[imk] += [d[imk]]
                example = create_example(d, feature_types)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                example = None
        mean_file = os.path.join(
            output_directory,
            '%s_%s_means' % (k, set_name))
        np.savez(mean_file, means)
        print 'Finished encoding: %s' % it_name

    # Save file containing info about the stimuli (i.e. X for X -> Y)
    meta_file = os.path.join(
        output_directory,
        '%s_meta' % (set_name))
    im_size = d[stimuli_key].shape
    tf_load_vars = prepare_tf_dicts(v[0], feature_types)
    meta = {
        'im_size': im_size,
        'folds':  cv_data.keys(),
        'tf_dict': tf_load_vars,
        'tf_reader': {k: {'dtype': tf.float32, 'reshape': None} for k in v[0].keys()}
    }
    np.save(meta_file, meta)

    # Create a dataloader template file for the cc_bp repo
    if cc_repo is not None:
        dl_file = os.path.join(
            cc_repo['path'],
            '%s.py' % (set_name))
        loader_meta = {
            'NAME': set_name,
            'OUTPUT_SIZE': cc_repo['output_size'],
            'IM_SIZE': im_size,
            'MODEL_IM_SIZE': cc_repo['model_im_size'],
            'META_FILE': meta_file,
            'LOSS_FUNCTION': cc_repo['loss_function'],
            'SCORE_METRIC': cc_repo['score_metric'],
            'PREPROCESS': cc_repo['preprocess']
        }
        create_data_loader_class(cc_repo['template_file'], loader_meta, dl_file)


def package_dataset(config, dataset_info, output_directory):
    """Query and package."""
    dataset_instructions = dataset_info['cross_ref']
    if dataset_instructions == 'rf_coordinate_range':
        # TODO fix this API so it doesn't rely on conditionals.
        data_dicts = db.get_cells_all_data_by_rf(dataset_info['rf_coordinate_range'])[0]
    elif dataset_instructions == 'rf_coordinate_range_and_stimuli':
        data_dicts = db.get_cells_all_data_by_rf_and_stimuli(
            rfs=dataset_info['rf_coordinate_range'],
            stimuli=dataset_info['stimuli'])[0]
    else:
        # Incorporate more queryies and eventually allow inner-joining on them.
        raise RuntimeError('Other instructions are not yet implemented.')
    data_files = load_npzs(data_dicts, dataset_info)

    # Prepare meta file to create a dataset specific data loader
    cc_repo = {
        'template_file': config.cc_data_dir,
        'path': config.cc_path
    }
    for k, v in dataset_info['cc_repo_vars'].iteritems():
        cc_repo[k] = v

    # Create tf records and meta files/data loader
    prepped_data = prepare_data_for_tf_records(
        data_files=data_files,
        output_directory=output_directory,
        set_name=dataset_info['experiment_name'],
        cv_split=dataset_info['cv_split'],
        store_means=dataset_info['store_means'],
        stimuli_key=dataset_info['reference_data_key'],
        feature_types=dataset_info['tf_types'],
        cc_repo=cc_repo)


def main(experiment_name, output_directory=None):
    """Pull desired experiment cells and encode as tfrecords."""
    assert experiment_name is not None, 'Name the experiment to process!'
    config = Config()
    da = DA()[experiment_name]()
    if output_directory is None:
        output_directory = config.tf_record_output
    package_dataset(
        config=config,
        dataset_info=da,
        output_directory=output_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        dest="experiment_name",
        type=str,
        default=None,
        help='Encode this Allen brain institute experiment.')
    parser.add_argument(
        "--output",
        dest="output_directory",
        type=str,
        default=None,
        help='Save tfrecords to this directory.')
    main(**vars(parser.parse_args()))

