"""
A database to store the MCMC chains.

Author:
    Ilias Bilionis
"""

import tables as pt
import numpy as np
import os


CHAIN_TYPE = {'id': pt.UInt16Col(),
              'date': pt.StringCol(itemsize=16),


class DataBase(object):

    """
    A database to store MCMC chains.

    :param filename:     The filename of the database.
    :type filename:      str
    """

    def __init__(self, filename):
        """
        Initialize the object.
        """
        self.filename = filename
        if os.path.exists(filename) and pt.isPyTablesFile(filename):
            self.db = pt.open_file(filename, mode='a')
        else:
            self.db = pt.open_file(filename, mode='w')
            fd.create_group('/', 'mcmc', 'Metropolis-Hastings Algorithm Data')
