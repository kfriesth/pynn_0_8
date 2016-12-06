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

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        return len(post_slice)

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
        # Each connection is made with probability p_connect
        # Return the row-length that 99.99% of rows will be shorter than
        return int(scipy.stats.binom.ppf(
            0.9999, len(post_slice), self.p_connect))

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        return int(round(self.p_connect * float(len(post_slice))))

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

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        return 1 if pre_slice.overlaps(post_slice) else 0

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

    def _estimate_max_row_synapses(self, pre_slice, post_slice,
                                   pre_size, post_size):
        # Get the row length histogram of slice
        hist = self._get_slice_row_length_histogram(pre_slice, post_slice)

        # Return maximum row length
        return np.amax(hist);

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        # Get the row length histogram of slice
        hist = self._get_slice_row_length_histogram(pre_slice, post_slice)

        # Return average row length
        return np.average(hist)

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
        # Each pre-synaptic neuron connects to n of the M=post_size
        # post-synaptic neurons.
        # Determining which of those n connections are within this post_slice is
        # a matter of sampling N=len(post_slice) times without replacement.
        # The number within the row and post_slice are
        # hypergeometrically distributed.

        # Return the row-length that 99.99% of rows will be shorter than
        return int(scipy.stats.hypergeom.ppf(
            0.9999, M=post_size, n=self.n, N=len(post_slice)))

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        # How large a fraction of the full post populations is this
        post_fraction = float(len(post_slice)) / float(post_size)

        return int(self.n * post_fraction)

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
        # Calculate the probability that any of the
        # n synapses in the column will be within this row
        prob_in_row = float(self.n) / pre_size

        # Return the row-length that 99.99% of rows will be shorter than
        return int(scipy.stats.binom.ppf(
            0.9999, len(post_slice), prob_in_row))

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        return int(len(post_slice) * float(self.n) / float(pre_size))

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
        # There are n connections amongst the M=pre_size*post_size possible
        # connections.
        # Determining which of those n connections are within this row and
        # post_slice is a matter of sampling N=len(post_slice) times without
        # replacement. The number within the row and post_slice are
        # hypergeometrically distributed.

        M = pre_size * post_size
        N = len(post_slice)

        return int(scipy.stats.hypergeom.ppf(0.9999, M=M, N=N, n=self.n))

    def _estimate_mean_row_synapses(self, pre_slice, post_slice,
                                    pre_size, post_size):
        # How large a fraction of the full post populations is this
        pre_fraction = float(len(pre_slice)) / float(pre_size)
        post_fraction = float(len(post_slice)) / float(post_size)

        # Multiply these by the total number of synapses
        return int(pre_fraction * post_fraction * float(self.n) / float(pre_size))
