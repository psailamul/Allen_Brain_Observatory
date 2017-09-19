"""Helper functions for handling Allen brain data."""

import cPickle as pickle
from config import Allen_Brain_Observatory_Config


def save_object(obj, filename):
    """Pkl object."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Un-pkl object."""
    with open(filename, 'rb') as inputfile:
        return pickle.load(inputfile)


def get_sess_key(sesstxt, config=None):
    """Get session key from Allen brain."""
    if config is None:
        config = Allen_Brain_Observatory_Config()
    for code, txt in config.session.iteritems():
        if sesstxt == txt:
            return code
