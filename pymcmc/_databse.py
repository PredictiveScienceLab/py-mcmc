"""
A database to store the MCMC chains.

Author:
    Ilias Bilionis
"""

import tables as pt
import numpy as np


class DataBase(object):

    """
    A database to store MCMC chains.
    """


    def __init__(self, filename):
        """
        Initialize the object.
        """
        self.filename = filename
        self.db = pt.open_file(filename, mode='a')
