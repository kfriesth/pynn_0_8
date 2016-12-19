"""
Connection method classes for PyNN SpiNNaker

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
# Import modules
import numpy as np
import scipy
from spinnaker import lazy_param_map
import lazyarray as la

from pyNN.random import RandomDistribution

# Import classes
from pyNN.connectors import (AllToAllConnector,
                             FixedProbabilityConnector,
                             FixedTotalNumberConnector,
                             OneToOneConnector,
                             FixedNumberPreConnector,
                             FixedNumberPostConnector,
                             DistanceDependentProbabilityConnector,
                             DisplacementDependentProbabilityConnector,
                             IndexBasedProbabilityConnector,
                             SmallWorldConnector,
                             FromListConnector,
                             FromFileConnector,
                             CloneConnector,
                             ArrayConnector)

# TODO for handling allow_self_connections=False we need the allow_self
# connections flag, to test if post_slice and pre_slice overlap,
# and modify context['N']
def _draw_num_connections(context, post_slice, pre_slice, **kwargs):
    nsample = len(post_slice) * len(pre_slice)

    if context['n'] == 0:
        sample = 0
    elif nsample == context['N']:
        sample = context['n']
    elif context['with_replacement']:
        sample = np.random.binomial(n = context['n'],
                                    p = float(nsample) / context['N'])
    else:
        sample = np.random.hypergeometric(ngood = context['n'],
                                          nbad = context['N'] - context['n'],
                                          nsample = nsample)
    context['n'] -= sample
    context['N'] -= nsample

    return la.larray(sample, shape=(1,))

# TODO for handling allow_self_connections=False, modify value here?
def _submat_size(context, post_slice, pre_slice, **kwargs):
    return la.larray(len(post_slice) * len(pre_slice), shape=(1,))

# ----------------------------------------------------------------------------
# AllToAllConnector
# ----------------------------------------------------------------------------
class AllToAllConnector(AllToAllConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # If this connector can be generated on chip, parameter map to use
    _on_chip_param_map = [("allow_self_connections", "u4", lazy_param_map.integer)]

    # Are the statistics of this connector the same across all slices
    _cachable = True

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # We know the number of synapses per sub-row. Use a distribution that
        # can only return that value
        num_synapses = len(post_slice)

        return scipy.stats.randint(num_synapses, num_synapses + 1)

    def _get_projection_initial_state(self, pre_size, post_size):
        return None

# ----------------------------------------------------------------------------
# FixedProbabilityConnector
# ----------------------------------------------------------------------------
class FixedProbabilityConnector(FixedProbabilityConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # If this connector can be generated on chip, parameter map to use
    _on_chip_param_map = [("allow_self_connections", "u4", lazy_param_map.integer),
                          ("p_connect", "u4", lazy_param_map.u032)]

    # Are the statistics of this connector the same across all slices
    _cachable = True

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # There are len(post_slice) possible connections in the sub-row, each
        # formed with probability self.p_connect
        n = len(post_slice)

        return scipy.stats.binom(n=n, p=self.p_connect)

    def _get_projection_initial_state(self, pre_size, post_size):
        return None

# ----------------------------------------------------------------------------
# OneToOneConnector
# ----------------------------------------------------------------------------
class OneToOneConnector(OneToOneConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = True

    # Are the statistics of this connector the same across all slices
    _cachable = False

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # We know the number of synapses per sub-row. Use a distribution that
        # can only return that value
        p = pre_slice.intersection(post_slice) / float(len(pre_slice))
        return scipy.stats.bernoulli(p)

    def _get_projection_initial_state(self, pre_size, post_size):
        return None

# ----------------------------------------------------------------------------
# FromListConnector
# ----------------------------------------------------------------------------
class FromListConnector(FromListConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # Are the statistics of this connector the same across all slices
    _cachable = False

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _get_slice_row_length_histogram(self, pre_slice, post_slice):
        # Extract columns of pre and post indices from connection list
        pre_indices = self.conn_list[:, 0]
        post_indices = self.conn_list[:, 1]

        # Build mask to select list entries in slice
        mask = ((pre_indices >= pre_slice.start) &
                (pre_indices < pre_slice.stop) &
                (post_indices >= post_slice.start) &
                (post_indices < post_slice.stop))

        # Return histogram of masked pre-indices
        return np.bincount(pre_indices[mask].astype(int))

    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Use the histogram of per-row synapse number within this row
        # to construct a discrete distribution with the computed probabilities
        hist = self._get_slice_row_length_histogram(pre_slice, post_slice)
        vs = range(len(hist))
        ps = hist / float(hist.sum())
        return scipy.stats.rv_discrete(values = (vs, ps))

    def _get_projection_initial_state(self, pre_size, post_size):
        return None


# ----------------------------------------------------------------------------
# FixedNumberPostConnector
# ----------------------------------------------------------------------------
class FixedNumberPostConnector(FixedNumberPostConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    _on_chip_param_map = [("allow_self_connections", "u4", lazy_param_map.integer)]

    # Are the statistics of this connector the same across all slices
    _cachable = True

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):

        N = len(post_slice)
        M = post_size

        # There are n connections amongst the M=post_size possible
        # connections per row

        # If the connections are made with replacement, then each of the n
        # connections has an independent p=float(N)/M probability of being
        # selected, and the number of synapses in the sub-row is binomially
        # distributed

        # If the connections are made without replacement, then
        # n of the M possible connections are made, and we take a sample
        # of size N from those M. The number of synapses in the sub-row
        # is hypergeometrically distributed

        # scipy.stats.binom does not handle the n=0 case
        if self.n == 0:
            return scipy.stats.randint(0, 1)
        elif self.with_replacement:
            return scipy.stats.binom(n=self.n, p=float(N)/M)
        else:
            return scipy.stats.hypergeom(M=M, N=N, n=self.n)

    def _get_projection_initial_state(self, pre_size, post_size):
        return None


# ----------------------------------------------------------------------------
# FixedNumberPreConnector
# ----------------------------------------------------------------------------
class FixedNumberPreConnector(FixedNumberPreConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    _on_chip_param_map = [("allow_self_connections", "u4", lazy_param_map.integer)]

    # Are the statistics of this connector the same across all slices
    _cachable = True

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):

        N = len(post_slice)

        # There are n connections amongst the M=pre_size possible
        # connections per column

        # In the with replacement case, in each column, the number of connections
        # that end up in this row are distributed as Binom(n, 1.0/pre_size).
        # The number that end up in the sub-row are a sum of N such binomials,
        # which equals Binom(n*N, 1.0/pre_size)

        # In the without replacement case, the number of connections that end up
        # in a single column of this row is distributed as Bernoulli(n/pre_size).
        # The number that end up in this sub-row are distributed as
        # Binom(N, n/float(pre_size))

        # scipy.stats.binom does not handle the n=0 case
        if self.n == 0:
            return scipy.stats.randint(0, 1)
        elif self.with_replacement:
            return scipy.stats.binom(n=self.n * N, p=1.0/pre_size)
        else:
            return scipy.stats.binom(n=N, p=self.n/float(pre_size))

    def _get_projection_initial_state(self, pre_size, post_size):
        return None


# ----------------------------------------------------------------------------
# FixedTotalNumberConnector
# ----------------------------------------------------------------------------
class FixedTotalNumberConnector(FixedTotalNumberConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # Are the statistics of this connector the same across all slices
    _cachable = True
    
    _on_chip_param_map = [("allow_self_connections", "u4", lazy_param_map.integer),
                          ("with_replacement", "u4", lazy_param_map.integer),
                          (_draw_num_connections, "u4"),
                          (_submat_size, "u4")]

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _row_synapses_distribution(self, pre_slice, post_slice,
                                   pre_size, post_size):

        M = pre_size * post_size
        N = len(post_slice)

        # There are n connections amongst the M=pre_size*post_size possible
        # connections

        # If the connections are made with replacement, then each of the n
        # connections has an independent p=float(N)/M probability of being
        # selected, and the number of synapses in the sub-row is binomially
        # distributed

        # If the connections are made without replacement, then
        # n of the M possible connections are made, and we take a sample
        # of size N from those M. The number of synapses in the sub-row
        # is hypergeometrically distributed

        # scipy.stats.binom does not handle n=0
        if self.n == 0:
            return scipy.stats.randint(0, 1)
        elif self.with_replacement:
            return scipy.stats.binom(n=self.n, p=float(N)/M)
        else:
            return scipy.stats.hypergeom(M=M, N=N, n=self.n)

    def _get_projection_initial_state(self, pre_size, post_size):
        return {'n': self.n, 'N': pre_size * post_size,
                'with_replacement':self.with_replacement}
