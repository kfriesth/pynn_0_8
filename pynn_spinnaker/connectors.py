"""
Connection method classes for PyNN SpiNNaker

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""
# Import modules
import numpy as np
import scipy
from spinnaker import lazy_param_map

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


# ----------------------------------------------------------------------------
# AllToAllConnector
# ----------------------------------------------------------------------------
class AllToAllConnector(AllToAllConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # If this connector can be generated on chip, parameter map to use
    _on_chip_param_map = []

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        return len(post_slice)

    def _estimate_num_synapses(self, pre_slice, post_slice,
                               pre_size, post_size):
        return len(pre_slice) * len(post_slice)


# ----------------------------------------------------------------------------
# FixedProbabilityConnector
# ----------------------------------------------------------------------------
class FixedProbabilityConnector(FixedProbabilityConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # If this connector can be generated on chip, parameter map to use
    _on_chip_param_map = [("p_connect", "u4", lazy_param_map.u032)]

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Create array of possible row lengths
        x = np.arange(len(post_slice))

        # Calculate CDF of binomial distribution representing len(post_slice)
        # number of events with p_connect probability
        cdf = scipy.stats.binom.cdf(x, len(post_slice), self.p_connect)

        # Return row-length corresponding to 99.99% of rows
        return np.searchsorted(cdf, 0.9999)

    def _estimate_num_synapses(self, pre_slice, post_slice,
                               pre_size, post_size):
        return int(round(self.p_connect * float(len(pre_slice)) *
                         float(len(post_slice))))


# ----------------------------------------------------------------------------
# OneToOneConnector
# ----------------------------------------------------------------------------
class OneToOneConnector(OneToOneConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = True

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        return 1 if pre_slice.overlaps(post_slice) else 0

    def _estimate_num_synapses(self, pre_slice, post_slice,
                               pre_size, post_size):
        return min(len(pre_slice), len(post_slice))


# ----------------------------------------------------------------------------
# FromListConnector
# ----------------------------------------------------------------------------
class FromListConnector(FromListConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Extract columns of pre and post indices from connection list
        pre_indices = self.conn_list[:, 0]
        post_indices = self.conn_list[:, 1]

        # Build mask to select list entries in slice
        mask = ((pre_indices >= pre_slice.start) &
                (pre_indices < pre_slice.stop) &
                (post_indices >= post_slice.start) &
                (post_indices < post_slice.stop))

        # Use mask to select slice pre-indices
        slice_pre_indices = pre_indices[mask].astype(int)

        # Return maximum number of list entries in each bin
        return np.amax(np.bincount(slice_pre_indices));

    def _estimate_num_synapses(self, pre_slice, post_slice,
                              pre_size, post_size):
        # Extract columns of pre and post indices from connection list
        pre_indices = self.conn_list[:, 0]
        post_indices = self.conn_list[:, 1]

        # Return number of list entries which contain
        # connections in both pre and post slices
        # http://stackoverflow.com/questions/9560207/how-to-count-values-in-a-certain-range-in-a-numpy-array
        return ((pre_indices >= pre_slice.start) &
                (pre_indices < pre_slice.stop) &
                (post_indices >= post_slice.start) &
                (post_indices < post_slice.stop)).sum()

# ----------------------------------------------------------------------------
# FixedNumberPostConnector
# ----------------------------------------------------------------------------
class FixedNumberPostConnector(FixedNumberPostConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Create array of possible row lengths - there can't be more than n!
        x = np.arange(self.n)

        # Calculate the probability that any of the
        # n synapses in the row is within post-slice
        prob_in_row = float(len(post_slice)) / float(post_size)

        # Calculate CDF of binomial distribution representing n
        # events with prob_in_slice probability of being in post_slice
        cdf = scipy.stats.binom.cdf(x, self.n, prob_in_row)

        # Return row-length corresponding to 99.99% of rows
        return min(len(post_slice), np.searchsorted(cdf, 0.9999))

    def _estimate_num_synapses(self, pre_slice, post_slice,
                               pre_size, post_size):
        # How large a fraction of the full post populations is this
        post_fraction = float(len(post_slice)) / float(post_size)

        return int(len(pre_slice) * self.n * post_fraction)

# ----------------------------------------------------------------------------
# FixedNumberPreConnector
# ----------------------------------------------------------------------------
class FixedNumberPreConnector(FixedNumberPreConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Create array of possible row lengths
        x = np.arange(len(post_slice))

        # Calculate the probability that any of the
        # n synapses in the column will be within this row
        prob_in_row = float(self.n) / pre_size

        # Calculate CDF of binomial distribution representing len(post_slice)
        # events with prob_in_row probability of being connected to post neuron
        cdf = scipy.stats.binom.cdf(x, len(post_slice), prob_in_row)

        # Return row-length corresponding to 99.99% of rows
        return np.searchsorted(cdf, 0.9999)

    def _estimate_num_synapses(self, pre_slice, post_slice,
                               pre_size, post_size):
        # How large a fraction of the full pre populations is this
        pre_fraction = float(len(pre_slice)) / float(pre_size)

        return int(len(post_slice) * self.n * pre_fraction)

# ----------------------------------------------------------------------------
# FixedTotalNumberConnector
# ----------------------------------------------------------------------------
class FixedTotalNumberConnector(FixedTotalNumberConnector):
    # Can suitable populations connected with this connector be connected
    # using an in-memory buffer rather than by sending multicast packets
    _directly_connectable = False

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Create array of possible row lengths - there can't be more than n!
        x = np.arange(self.n)

        # Calculate the probability that any of the n synapses
        # in the full matrix are within post-slice of a single row
        prob_in_row = (1.0 / float(pre_size)) *\
            (float(len(post_slice)) / float(post_size))

        # Calculate CDF of binomial distribution representing n events with
        # prob_in_row probability of being in slice
        cdf = scipy.stats.binom.cdf(x, self.n, prob_in_row)

        # Return row-length corresponding to 99.9% of rows
        return min(len(post_slice), np.searchsorted(cdf, 0.999))

    def _estimate_num_synapses(self, pre_slice, post_slice,
                               pre_size, post_size):
        # How large a fraction of the full pre and post populations is this
        pre_fraction = float(len(pre_slice)) / float(pre_size)
        post_fraction = float(len(post_slice)) / float(post_size)

        # Multiply these by the total number of synapses
        return int(pre_fraction * post_fraction * float(self.n))
