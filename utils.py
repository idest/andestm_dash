#Utils
import numpy as np

class DotDict(dict):
    # dot.notation access to dictionary attributes"
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print_data(name, data, boolean=False):
    if boolean is True:
        np.savetxt(name, data, fmt="%1d", delimiter=" ")
    else:
        np.savetxt(name, data, fmt="%9.2e", delimiter="   ")

def cmap_to_cscale(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

class MagicMethodWrapper(type):

    def __init__(cls, name, bases, dct):
        def make_proxy(name):
            def proxy(self, *args):
                return getattr(self._obj, name)
            return proxy
        type.__init__(cls, name, bases, dct)
        if cls.__wraps__:
            ignore = set("__%s__" % n for n in cls.__ignore__.split())
            for name in dir(cls.__wraps__):
                if name.startswith("__"):
                    if name not in ignore and name not in dct:
                        #attr = getattr(cls.__wraps__, name)
                        setattr(cls, name, property(make_proxy(name)))


class Wrapper(object, metaclass = MagicMethodWrapper):

    __wraps__ = None
    __ignore__ = "class mro new init setattr getattr getattribute dir"

    def __init__(self, obj):
        if self.__wraps__ is None:
            raise TypeError("base class Wrapper may not be instansiated")
        elif isinstance(obj, self.__wraps__):
            self._obj = obj
        else:
            raise ValueError("wrapped object must be of {}".format(self.__wraps__))

    def __getattr__(self, name):
        return getattr(self._obj, name)

"""
# class Wrapper use
from utils import Wrapper
import numpy as np

class ArrayWrapper(Wrapper):
    __wraps__ = np.ndarray
    def salute(self):
        print('hola')

numpy_array = np.random.rand(5,6)
wa = ArrayWrapper(numpy_array)
"""
