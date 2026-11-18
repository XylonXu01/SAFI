"""Trusted-file pickle compatibility, without restoring the old public package."""
import pickle


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        old = 'rst_diff_async'
        if module == old or module.startswith(old + '.'):
            module = 'safi_net' + module[len(old):]
            if name == 'AsyncFusionModel':
                name = 'SAFINet'
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def loads(data, **kwargs):
    import io
    return load(io.BytesIO(data), **kwargs)


dump = pickle.dump
dumps = pickle.dumps


def load_trusted(path, **kwargs):
    """Load the same trusted training payload with explicit historical aliases."""
    import sys
    import torch
    return torch.load(path, map_location=kwargs.pop('map_location', 'cpu'),
                      weights_only=False, pickle_module=sys.modules[__name__], **kwargs)
