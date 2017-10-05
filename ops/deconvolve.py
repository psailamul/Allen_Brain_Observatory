from deconv_methods import elephant_preprocess, elephant_deconv


class deconvolve(object):
    """Wrapper class for deconvolving spikes from Ca2+ data."""

    def __getitem__(self, name):
        """Get attribute from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains attribute."""
        return hasattr(self, name)

    def __init__(self, kwargs=None):
        """Class global variable init."""
        self.data_fps = 30.  # Ca2+ FPS for Allen.
        self.batch_size = 4096
        self.update_params(kwargs)
        self.check_params()

    def check_params(self):
        if not hasattr(self, 'deconv_method'):
            raise RuntimeError(
                'You must pass a deconv_method.')
        if not hasattr(self, 'batch_size'):
            raise RuntimeError(
                'You must pass a batch_size.')
        if not hasattr(self, 'deconv_dir'):
            raise RuntimeError(
                'You must pass a deconv_dir.')
        if not hasattr(self, 'data_fps'):
            raise RuntimeError(
                'You must pass a data_fps.')

    def update_params(self, kwargs):
        """Update the class attributes with kwargs."""
        if kwargs is not None:
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def deconvolve(self, neural_data):
        """Wrapper for deconvolution operations."""
        preproc_op, deconv_op = self.interpret_deconv(self.deconv_method)
        print 'Preprocessing neural data.'
        preproc_data = preproc_op(neural_data, self)
        print 'Deconvolving neural data.'
        return deconv_op(neural_data, preproc_data, self)

    def interpret_deconv(self, method):
        """Wrapper for returning the preprocessing and main operations."""
        if method == 'elephant':
            return (
                elephant_preprocess.preprocess,
                elephant_deconv.deconv)
