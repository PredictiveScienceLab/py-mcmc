"""
Various utility functions.

Author:
    Ilias Bilionis
"""


import tables as pt
import numpy as np


__all__ = ['UnknownTypeException', 'state_to_table_dtype']


class UnknownTypeException(Exception):

    """
    This exception is raise when we have to deal with an unknown type.
    """

    pass


DTYPE_STR_BUFFER_SAFETY_FACTOR = 2


def state_to_table_dtype(state,
                         str_buffer_safety_factor=DTYPE_STR_BUFFER_SAFETY_FACTOR):
    """
    Get a state of an object represented as a dictionary and derive the
    appropriate type of a tables.Table.

    :param state:       The state of an object.
    :type state:        dict
    :raises:            :class:`pymc.UnknownTypeException`
    """
    dtype_dict = {}
    for name in state.keys():
        if isinstance(state[name], int):
            dtype = pt.UInt32Col()
        elif isinstance(state[name], float):
            dtype = pt.Float64Col()
        elif isinstance(state[name], str):
            dtype = pt.StringCol(itemsize=len(state[name]) *
                                          str_buffer_safety_factor)
        elif isinstance(state[name], np.ndarray):
            dtype = pt.Float64Col(shape=state[name].shape)
        else:
            raise UnknownTypeException('I cannot deal with the type of %s (%s)'
                                       %(name, type(state[name])))
        dtype_dict[name] = dtype
    return dtype_dict
