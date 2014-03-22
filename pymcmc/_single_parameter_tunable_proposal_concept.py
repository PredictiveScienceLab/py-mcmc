"""
A tunable proposal concept that tunes just one parameter.

Author:
    Ilias Bilionis

Date:
    3/22/2014

"""


__all__ = ['SingleParameterTunableProposalConcept']


from . import TunableProposalConcept


class SingleParameterTunableProposalConcept(TunableProposalConcept):

    """
    A tunable proposal concept that tunes a single parameter based on the
    acceptance ratio only.

    :param param_name:  The name of the parameter we are actually tuning. Note
                        that we do test if a param_name parameter really
                        exists in the object that inherits this concept.
                        Therefore, the constructor to this class should be
                        called **after** the parameter has been initialized
                        by the proposal.
    :type param_name:   str
    :param lowest_ac:   The lowest allowed acceptance rate
                        (``0 <= lowest_ac < highest_ac``).
    :type lowest_ac:    float
    :param highest_ac:  The highest allowed acceptance rate
                        (``lowest_ac < highest_ac <= 1``).
    :type highest_ac:   float
    :param inc_f:       The factor by which we multiply the parameter if the
                        acceptance rate is too big (``inc_f > 1``).
    :type inc_f:        float
    :param dec_f:       The factor by which we multiply the parameter if the
                        acceptance rate is too low (``dec_f < 1``).
    """

    # The name of the parameter we are tuning
    _param_name = None

    # Lowest allowed acceptance rate
    _lowest_ac = None

    # The highest allowed acceptance rate
    _highest_ac = None

    # The increase factor
    _inc_f = None

    # The decrease factor
    _dec_f = None

    @property
    def param_name(self):
        """
        Set/Get the name of the parameter we are tuning.
        """
        return self._param_name

    @param_name.setter
    def param_name(self, value):
        """
        Set the name of the parameter we are tuning.
        """
        value = str(value)
        if not hasattr(self, value):
            raise RuntimeError('The proposal does not contain any parameter named'
                               ' `' + value + '`. This can probably be fixed by'
                               ' calling the constructor of this object after'
                               ' the parameter has been initializd in the'
                               ' proposal. Of course, you might have the name'
                               ' of the parameter wrong...')
        self._param_name = value

    @property
    def lowest_ac(self):
        """
        Set/Get the lowest allowed acceptance rate.
        """
        return self._lowest_ac

    @lowest_ac.setter
    def lowest_ac(self, value):
        """
        Set the lowest ac.
        """
        value = float(value)
        assert value >= 0.
        h_ac = self.highest_ac if self.highest_ac is not None else 1.
        assert value < h_ac
        self._lowest_ac = value

    @property
    def highest_ac(self):
        """
        Set/Get the highest allowed acceptance rate.
        """
        return self._highest_ac

    @highest_ac.setter
    def highest_ac(self, value):
        """
        Set the highest ac.
        """
        value = float(value)
        l_ac = self.lowest_ac if self.lowest_ac is not None else 0.
        assert value > l_ac
        assert value <= 1.
        self._highest_ac = value

    @property
    def inc_f(self):
        """
        Set/Get the increase factor.
        """
        return self._inc_f

    @inc_f.setter
    def inc_f(self, value):
        """
        Set the increase factor.
        """
        value = float(value)
        assert value > 1.
        self._inc_f = value

    @property
    def dec_f(self):
        """
        Set/Get the decrease factor.
        """
        return self._dec_f

    @dec_f.setter
    def dec_f(self, value):
        """
        Set the decrease factor.
        """
        value = float(value)
        assert value < 1.
        assert value > 0.
        self._dec_f = value

    def __init__(self, param_name, lowest_ac=0.2, highest_ac=0.6,
                 inc_f=1.2, dec_f=0.7, **kwargs):
        """
        Initialize the object.
        """
        self.param_name = param_name
        self.lowest_ac = lowest_ac
        self.highest_ac = highest_ac
        self.inc_f = inc_f
        self.dec_f = dec_f
        super(SingleParameterTunableProposalConcept, self).__init__(**kwargs)

    def tune(self, ac, verbose=False, **kwargs):
        """
        Tune the proposal.

        This really accepts just one parameter the ``ac`` and ignores any other
        parameter passed as ``kwargs``.
        """
        m_f = 1.
        if ac < self.lowest_ac:
            m_f = self.dec_f
        elif ac > self.highest_ac:
            m_f = self.inc_f
        else:
            return
        old_param = getattr(self, self.param_name)
        setattr(self, self.param_name, m_f * old_param)
        if verbose:
            s = ('\nTuning parameter `' + self.param_name +
                 '`: %2.6f -> %2.6f' % (old_param, getattr(self, self.param_name)))
            print s

    def __getstate__(self):
        """
        Get the state of the object.
        """
        state = {}
        state['lowest_ac'] = self.lowest_ac
        state['highest_ac'] = self.highest_ac
        state['inc_f'] = self.inc_f
        state['dec_f'] = self.dec_f
        state['param_name'] = self.param_name
        return state
    
    def __setstate__(self, state):
        """
        Set the state of the object.
        """
        self.lowest_ac = state['lowest_ac']
        self.highest_ac = state['highest_ac']
        self.inc_f = state['inc_f']
        self.dec_f = state['dec_f']
        self.param_name = state['param_name']
