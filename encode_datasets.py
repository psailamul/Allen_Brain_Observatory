"""Functions for encoding cells as TFrecords for contextual circuit bp."""

import os
import argparse
import numpy as np
import tensorflow as tf
from db import db
from config import Allen_Brain_Observatory_Config as Config
from declare_datasets import declare_allen_datasets as dad
from tqdm import tqdm
import cPickle as pickle
from ops import helper_funcs, deconvolve


def get_field(d, field, alt):
    """Error handling for absent dict keys."""
    if field in d:
        return d[field]
    else:
        return alt


def create_data_loader_class(template_file, meta_dict, output_file):
    """Write a data loader python class for this dataset."""
    with open(template_file, 'r') as f:
        text = f.readlines()
    for k, v in meta_dict.iteritems():
        for idx, t in enumerate(text):
            text[idx] = t.replace(k, str(v))
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
    f = f.strip('"')
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
    """Encode an float matrix into a byte list for a tfrecord."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    """Encode an int list into a tf int64 list for a tfrecord."""
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def float_feature(values):
    """TF floating point feature for tfrecords."""
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def fix_malformed_pointers(d, filter_ext='.npy', rep_ext='.npz'):
    """Replace .npy extensions if they were erroniously placed in the DB."""
    if '.npy' in d:
        d = '%s%s' % (d.split(filter_ext)[0], rep_ext)
    return d


def load_npzs(data_dicts, exp_dict, stimuli_key=None, neural_key=None):
    """Load cell data from an npz."""
    deconv = deconvolve.deconvolve(exp_dict)
    key_list = []

    # Organize data_dicts by cell
    cell_specimen_ids = [d['cell_specimen_id'] for d in data_dicts]
    unique_cells = np.unique(cell_specimen_ids)

    # Trim to specified number of cells if desired.
    if exp_dict[
            'only_process_n'] is not None or exp_dict['only_process_n'] > 0:
        print 'Trimming query from %s to %s cells.' % (
            len(unique_cells), exp_dict['only_process_n'])
        if exp_dict['randomize_selection']:
            unique_cells = unique_cells[np.random.permutation(
                len(unique_cells))]

        if exp_dict['sessions'] is not None:
            # Filter cells that have entries in specified sessions.
            session_dict = {}
            for d in data_dicts:
                cell = d['cell_specimen_id']
                if cell not in session_dict.keys():
                    session_dict[cell] = []
                session_dict[cell] += [d['session']]
            print 'Sorting cells by sessions: %s' % exp_dict[
                'sessions']
            session_counts = {k: len(v) for k, v in session_dict.iteritems()}
            unique_cells = np.asarray(session_counts.keys())[np.argsort(
                session_counts.values())[::-1]]
        keep_cells = unique_cells[:exp_dict['only_process_n']]
        data_dicts = [
            d for d in data_dicts if d['cell_specimen_id'] in keep_cells]

    # Process cell data
    output_data = []
    for d in tqdm(data_dicts, total=len(data_dicts), desc='Preparing data'):
        df = {}
        data_pointer = fix_malformed_pointers(d['cell_output_npy'])
        try:
            cell_data = load_data(data_pointer, allow_pkls=True)
        except:
            print 'WARNING: Fix the npz extensions for %s' % data_pointer
            cell_data = load_data(
                '%s.npy.npz' % data_pointer.strip('.npz'), allow_pkls=True)
        cell_id = d['cell_specimen_id']
        df['cell_specimen_id'] = cell_id

        # Stim table
        stim_table = load_data(
            cell_data['stim_table'].item(),
            allow_pkls=True)
        stim_table = stim_table['stim_table']

        # Stimuli
        raw_stimuli = load_data(
            cell_data['stim_template'].item(),
            allow_pkls=False)
        df['stimulus_name'] = cell_data['stim_template'].item()
        if len(raw_stimuli.shape) < 4:
            # Ensure that stimuli are a 4D tensor.
            raw_stimuli = np.expand_dims(raw_stimuli, axis=-1)
        raw_stimuli = raw_stimuli.astype(np.float32)
        df['raw_stimuli'] = raw_stimuli
        proc_stimuli = raw_stimuli[stim_table[:, 0]]
        df['proc_stimuli'] = proc_stimuli

        # Neural data
        neural_data = load_data(
            cell_data['neural_trace'].item(),
            allow_pkls=True)
        # TODO: Register fields in pachaya's data creation with create_db.py
        # to avoid the below.
        trace_key = [k for k in neural_data.keys() if 'trace' in k]
        if trace_key is None:
            raise RuntimeError(
                'Could not find a \'trace\' key in the neural_data dict.' +
                'Found the following keys: %s' % neural_data.keys())
        neural_data = neural_data[trace_key[0]].astype(np.float32)
        df['neural_trace'] = neural_data

        # Trim neural data
        if exp_dict['deconv_method'] is not None:
            neural_data = deconv.deconvolve(neural_data)

        stim_table_idx = stim_table[:, 1] + exp_dict['neural_delay']
        neural_data_trimmed = neural_data[stim_table_idx]
        df['neural_trace_trimmed'] = neural_data_trimmed

        # ROI mask
        roi_mask = load_data(
            cell_data['ROImask'].item(),
            allow_pkls=True)
        roi_mask = roi_mask['roi_loc_mask']
        df['ROImask'] = np.expand_dims(roi_mask, axis=-1).astype(np.float32)

        # AUX data
        aux_data = load_data(
            cell_data['other_recording'].item(),
            allow_pkls=True)

        pupil_size = get_field(aux_data, 'pupil_size', None)
        running_speed = get_field(aux_data, 'running_speed', None)
        eye_locations_spherical = get_field(
            aux_data,
            'eye_locations_spherical',
            None)
        eye_locations_cartesian = get_field(
            aux_data,
            'eye_locations_cartesian',
            None)
        if pupil_size is not None:
            df['pupil_size'] = pupil_size[stim_table_idx]
        else:
            df['pupil_size'] = None
        if running_speed is not None:
            df['running_speed'] = running_speed.item()['dxcm'][stim_table_idx]
        else:
            df['running_speed'] = None
        if eye_locations_spherical is not None:
            df['eye_locations_spherical'] = eye_locations_spherical[
                stim_table_idx, :]
        else:
            df['eye_locations_spherical'] = None
        if eye_locations_cartesian is not None:
            df['eye_locations_cartesian'] = eye_locations_cartesian[
                stim_table_idx, :]
        else:
            df['eye_locations_cartesian'] = None

        # RF data
        df['on_center_x'] = d['on_center_x']
        df['on_center_y'] = d['on_center_y']
        df['off_center_x'] = d['off_center_x']
        df['off_center_y'] = d['off_center_y']
        df['on_height'] = d['on_height']
        df['on_width'] = d['on_width_x']
        df['off_height'] = d['off_height']
        df['off_width'] = d['off_width_x']

        # Rename images/neural traces for contextual_circuit_bp
        df['image'] = df.pop(stimuli_key.keys()[0])
        df['label'] = df.pop(neural_key.keys()[0])
        df['event_index'] = range(len(df['label']))

        # Package data
        df = {
            k: v for k, v in df.iteritems()
            if k in exp_dict['include_targets']
        }
        output_data += [df]
        it_check = [k for k, v in df.iteritems() if v is not None]
        key_list += [it_check]
    keep_keys = np.unique(key_list)
    remove_keys = list(
        set(exp_dict['include_targets'].keys()) - set(keep_keys))
    if remove_keys is not None:
        print 'Removing keys which were not populated across cells: %s.' %\
            remove_keys
        for idx, d in enumerate(output_data):
            it_d = {k: v for k, v in d.iteritems() if k in keep_keys}
            output_data[idx] = it_d

    # TODO handle NaNs in output_data here.
    if exp_dict['cc_repo_vars']['output_size'][0] > 1:
        # Multi neuron target; consolidate event_dict.
        stimuli = [d['stimulus_name'] for d in output_data]
        unique_stimuli, ri = np.unique(
            stimuli,
            return_index=True)

        # Concatenate image/labels appropriately.
        labels = {}
        ROImasks = {}
        images = {}
        cell_specimen_ids = {}
        for stim in zip(unique_stimuli):
            stim = stim[0]
            labels[stim] = []
            ROImasks[stim] = []
            images[stim] = []
            cell_specimen_ids[stim] = []
            for d in output_data:
                if d['stimulus_name'] == stim:
                    labels[stim] += [d['label'][:, None]]
                    ROImasks[stim] += [np.expand_dims(d['ROImask'], axis=0)]
                    images[stim] += [d['image']]
                    cell_specimen_ids[stim] += [d['cell_specimen_id']]

        cat_labels = {}
        cat_ROImasks = {}
        cat_images = {}
        cat_repeats = {}
        cat_events = {}
        cat_cell_specimen_ids = {}
        for stim in unique_stimuli:
            cells = np.asarray(cell_specimen_ids[stim])
            unique_cells = np.unique(cells)
            cat_cell_specimen_ids[stim] = unique_cells
            for cell_count, cell in enumerate(unique_cells):
                cell_ids = np.where(cells == cell)[0]
                for cell_it, ci in enumerate(cell_ids):
                    if cell_it == 0:
                        cell_labels = labels[stim][ci]
                        cell_images = images[stim][ci]
                        cell_ROImasks = ROImasks[stim][ci]
                        cell_events = np.arange(len(cell_labels))
                        cell_repeats = np.ones(len(cell_labels)) * cell_it
                    else:
                        cell_labels = np.concatenate(
                            (
                                cell_labels,
                                labels[stim][ci]),
                            axis=0)
                        cell_images = np.concatenate(
                            (
                                cell_images,
                                images[stim][ci]),
                            axis=0)
                        cell_events = np.concatenate(
                            (
                                cell_events,
                                np.arange(len(cell_labels))),
                            axis=0)
                        cell_repeats = np.concatenate(
                            (
                                cell_repeats,
                                np.ones(len(cell_labels)) * cell_it),
                            axis=0)
                if cell_count == 0:
                    cat_labels[stim] = cell_labels
                    cat_ROImasks[stim] = cell_ROImasks
                    cat_images[stim] = cell_images
                    cat_events[stim] = cell_events
                    cat_repeats[stim] = cell_repeats
                else:
                    cat_labels[stim] = np.concatenate(
                        (
                            cat_labels[stim],
                            cell_labels
                        ),
                        axis=1)
                    cat_ROImasks[stim] = np.concatenate(
                        (
                            cat_ROImasks[stim],
                            cell_ROImasks
                        ),
                        axis=0)

        # Package into a list of dicts.
        output_data = []
        for (ik, iv), (lk, lv), (rk, rv), (ck, cv), (rk, rv), (ek, ev) in zip(
                cat_images.iteritems(),
                cat_labels.iteritems(),
                cat_ROImasks.iteritems(),
                cat_cell_specimen_ids.iteritems(),
                cat_repeats.iteritems(),
                cat_events.iteritems()):
            assert ik == lk == rk == ck == ek, 'Issue with keys.'
            output_data += [{
                'image': iv,
                'cell_specimen_id': cv,
                'ROImask': rv,
                'label': lv,
                'stimulus_name': ik,
                'stimulus_iterations': rv,
                'event_index': ev,
            }]

    # Concatenate data into equal-sized lists
    event_dict = []
    for d in output_data:
        ref_length = d['image'].shape[0]
        assert ref_length == d['label'].shape[0],\
            'Stimuli and neural data do not match.'
        for idx in range(ref_length):
            it_event = {}
            for k, v in d.iteritems():
                if exp_dict['include_targets'][k] == 'split':
                    try:
                        it_event[k] = v[idx]
                    except:
                        raise RuntimeError(
                            'Did you mean to repeat %s per frame?' % k)
                elif exp_dict['include_targets'][k] == 'repeat':
                    it_event[k] = v
                else:
                    raise RuntimeError(
                        'Fucked up packing data into list of dicts.')
            event_dict += [it_event]
    return event_dict


def create_example(data_dict, feature_types):
    """Create entry in tfrecords."""
    tf_dict = {}
    for k, v in data_dict.iteritems():
        if k not in feature_types.keys():
            raise RuntimeError('Cannot understand specified feature types.')
        else:
            it_feature_type = feature_types[k]
        if it_feature_type == 'float':
            tf_dict[k] = float_feature(v)
        elif it_feature_type == 'int64':
            tf_dict[k] = int64_feature(v)
        elif it_feature_type == 'string':
            if isinstance(v, basestring):
                # Strings
                tf_dict[k] = bytes_feature(str(v))
            else:
                # Images
                tf_dict[k] = bytes_feature(v.tostring())
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature=tf_dict
        )
    )


def prepare_tf_dicts(feature_types, d):
    """Prepare tf data types for loading tf variables."""
    tf_dict = {}
    for k, v in feature_types.iteritems():
        # TODO: Change this interface to more flexibly preallocate shapes.
        if v is 'float' and k in d.keys():
            if isinstance(d[k], list):
                shape = len(d[k])
            elif isinstance(d[k], np.ndarray):
                shape = d[k].shape[0]
            else:
                print 'WARNING: cannot understand the type of feature %s.' % k
                shape = []
        else:
            shape = []
        tf_dict[k] = fixed_len_feature(dtype=v, length=shape)
    return tf_dict


def prepare_data_for_tf_records(
        data_files,
        output_directory,
        set_name,
        cv_split,
        store_means,
        feature_types,
        cc_repo=None,
        stimuli_key=None,
        ext='tfrecords'):
    """Package dict into tfrecords."""
    if cv_split.keys()[0] == 'random_cv_split':
        cv_inds = np.random.permutation(len(data_files))
        val_len = np.round(
            len(data_files) * cv_split.values()[0]).astype(int)
        val_ind = cv_inds[val_len:]
        train_ind = cv_inds[:val_len]
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
    elif cv_split.keys()[0] == 'cv_split':
        cv_inds = np.arannge(len(data_files))
        val_len = np.round(
            len(data_files) * cv_split.values()[0]).astype(int)
        val_ind = cv_inds[val_len:]
        train_ind = cv_inds[:val_len]
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
    elif cv_split.keys()[0] == 'split_on_stim':
        train_ind = np.asarray(
            [True if cv_split.values()[0] in d['stimulus_name'] else False
                for d in data_files])
        val_ind = train_ind == False
        cv_data = {
            'train': np.asarray(data_files)[train_ind],
            'val': np.asarray(data_files)[val_ind]
        }
        print 'Split data into Train: %s, Validation: %s.' % (
            np.sum(train_ind),
            np.sum(val_ind))
    elif cv_split.keys()[0] == 'split_on_repeat':
        raise RuntimeError('CV type: split_on_repeat is not yet implemented.')
        # stim_names = [d['stimulus_name'] for d in data_files]
        cv_data = {
            'train': [],
            'val': []
        }

    else:
        raise RuntimeError(
            'Selected crossvalidation %s is not yet implemented.' % cv_split)
    means = {k: [] for k in store_means}
    maxs = {k: [] for k in store_means}
    for k, v in cv_data.iteritems():
        it_name = os.path.join(
            output_directory,
            '%s_%s.%s' % (set_name, k, ext))
        idx = 0
        with tf.python_io.TFRecordWriter(it_name) as tfrecord_writer:
            for idx, d in tqdm(
                    enumerate(v),
                    total=len(v),
                    desc='Encoding %s' % k):
                for imk, imv in means.iteritems():
                    if idx == 0:
                        means[imk] = d[imk]
                    else:
                        means[imk] += d[imk]
                    maxs[imk] = np.max(d[imk])
                example = create_example(d, feature_types)
                serialized = example.SerializeToString()
                tfrecord_writer.write(serialized)
                example = None
                idx += 1
        mean_file = os.path.join(
            output_directory,
            '%s_%s_means' % (set_name, k))
        num_its = float(len(v))
        means = {k: {
            'mean': v / num_its,
            'max': maxs[k]
            } for k, v in means.iteritems()}
        np.savez(mean_file, means)
        print 'Finished encoding: %s' % it_name

    # Save file containing info about the stimuli (i.e. X for X -> Y)
    meta_file = os.path.join(
        output_directory,
        '%s_meta' % (set_name))
    im_size = d[stimuli_key.values()[0]].shape
    tf_load_vars = prepare_tf_dicts(feature_types, d)
    tf_reader = {}
    for ik, iv in v[0].iteritems():
        if isinstance(iv, float) or\
            isinstance(iv, int) or\
                isinstance(iv, basestring):
            it_shape = []
        else:
            it_shape = iv.shape
        tf_reader[ik] = {'dtype': tf.float32, 'reshape': it_shape}
    # tf_reader['image']['reshape'] = cc_repo['model_im_size']
    meta = {
        'im_size': im_size,
        'folds': {k: k for k in cv_data.keys()},
        'tf_dict': tf_load_vars,
        'tf_reader': tf_reader
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
            'PREPROCESS': cc_repo['preprocess'],
        }
        create_data_loader_class(
            cc_repo['template_file'], loader_meta, dl_file)


def package_dataset(config, dataset_info, output_directory):
    """Query and package."""
    dataset_instructions = dataset_info['cross_ref']
    if dataset_instructions == 'rf_coordinate_range':
        # TODO fix this API so it doesn't rely on conditionals.
        data_dicts = db.get_cells_all_data_by_rf(
            dataset_info['rf_coordinate_range'])[0]
    elif dataset_instructions == 'rf_coordinate_range_and_stimuli':
        data_dicts = db.get_cells_all_data_by_rf_and_stimuli(
            rfs=dataset_info['rf_coordinate_range'],
            stimuli=dataset_info['stimuli'],
            sessions=dataset_info['sessions']
            )[0]
    else:
        # Incorporate more queryies and eventually allow inner-joining on them.
        raise RuntimeError('Other instructions are not yet implemented.')
    if len(data_dicts) == 0:
        raise RuntimeError('Empty cell query.')
    data_files = load_npzs(
        data_dicts,
        dataset_info,
        stimuli_key=dataset_info['reference_image_key'],
        neural_key=dataset_info['reference_label_key'],
    )

    # Prepare meta file to create a dataset specific data loader
    cc_repo = {
        'template_file': config.cc_template,
        'path': config.cc_data_dir
    }
    for k, v in dataset_info['cc_repo_vars'].iteritems():
        cc_repo[k] = v

    # Add tf_type entries for reference keys
    cat_dict = dict(
        dataset_info['reference_image_key'].items() +
        dataset_info['reference_label_key'].items())
    for k, v in cat_dict.iteritems():
        dataset_info['tf_types'][v] = dataset_info['tf_types'][k]

    # Create tf records and meta files/data loader
    prepare_data_for_tf_records(
        data_files=data_files,
        output_directory=output_directory,
        set_name=dataset_info['experiment_name'],
        cv_split=dataset_info['cv_split'],
        store_means=dataset_info['store_means'],
        stimuli_key=dataset_info['reference_image_key'],
        feature_types=dataset_info['tf_types'],
        cc_repo=cc_repo)


def main(dataset, output_directory=None):
    """Pull desired experiment cells and encode as tfrecords."""
    assert dataset is not None, 'Name the experiment to process!'
    config = Config()
    da = dad()[dataset]()
    if output_directory is None:
        output_directory = os.path.join(
            config.tf_record_output)
    da['deconv_dir'] = config.deconv_model_dir
    helper_funcs.make_dir(output_directory)
    package_dataset(
        config=config,
        dataset_info=da,
        output_directory=output_directory)
    # TODO: Incorporate logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="dataset",
        type=str,
        default=None,
        help='Encode an Allen brain institute dataset.')
    parser.add_argument(
        "--output",
        dest="output_directory",
        type=str,
        default=None,
        help='Save tfrecords to this directory.')
    main(**vars(parser.parse_args()))
