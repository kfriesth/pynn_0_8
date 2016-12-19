try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
from pyNN import common
from pyNN.random import RandomDistribution
from pyNN.space import Space
from pyNN.standardmodels import StandardCellType
from . import simulator
import itertools
import logging
import math
import numpy as np
import scipy
from bisect import bisect_left
import lazyarray as la
from rig import machine

# Import classes
from collections import namedtuple
from rig.utils.contexts import ContextMixin
from spinnaker.current_input_cluster import CurrentInputCluster
from .standardmodels.synapses import StaticSynapse
from .random import NativeRNG

# Import functions
from spinnaker.utils import get_model_comparable, is_scalar

logger = logging.getLogger("pynn_spinnaker")

distribution = {
    "normal":
        (scipy.stats.norm,
         lambda mu, sigma: {"loc": mu, "scale": sigma}),
    "normal_clipped":
        (scipy.stats.truncnorm,
         lambda mu, sigma, low, high: {"loc": mu, "scale": sigma,
                                       "a": (low - mu) / sigma,
                                       "b": (high - mu) / sigma}),
}

# The cdf of the mixture distribution which has component distributions
# dists weighted as ps
def eval_mixture_cdf(ps, dists, k):
    return sum(p * dist.cdf(k) for p, dist in zip(ps, dists))

# The cdf of a mixture distribution
# Vals is an array of integer values
# The component distributions are the distributions of the maximum of
# a vector of length val (for each val in vals) of variates from distribution
# dist. The component distributions are weighted by the probabilities ps.
def eval_mixture_of_maxes_cdf(ps, vals, dist, k):
    cdf_k = dist.cdf(k)
    return sum(p * cdf_k**val for p, val in zip(ps, vals))

# The cdf of a mixture distribution
# Vals is an array of integer values
# The component distributions are the distributions of the minimum of
# a vector of length val (for each val in vals) of variates from distribution
# dist. The component distributions are weighted by the probabilities ps.
def eval_mixture_of_mins_cdf(ps, vals, dist, k):
    cdf_k = dist.cdf(k)
    return sum(p * 1-(1-cdf_k)**val for p, val in zip(ps, vals))

# Do binary search over continuous val_range for the boundary
# between f(k) <= v and f(k) > v
def continuous_bisect_fun_left(f, v, val_lower, val_upper):
    val_range = [val_lower, val_upper]
    k = 0.5 * sum(val_range)
    for i in xrange(32):
        val_range[int(f(k) > v)] = k
        next_k = 0.5 * sum(val_range)
        if next_k == k:
            break
        k = next_k
    return k

# --------------------------------------------------------------------------
# SynapseClusterType
# --------------------------------------------------------------------------
class SynapseClusterType(namedtuple("SynapseClusterType",
                                    ["model", "receptor"])):
    # Override hash and equality magic methods so synapse
    # cluster types are compared based on compatibility
    def __hash__(self):
        return hash(self._comparable)

    def __eq__(self, other):
        return self._comparable == other._comparable

    def __ne__(self, other):
        return not(self == other)

    @property
    # Concatenate together the receptor type and
    # the comparable tuple of the model
    def _comparable(self):
        return (self.receptor,) + get_model_comparable(self.model)


# --------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------
class Projection(common.Projection, ContextMixin):
    __doc__ = common.Projection.__doc__
    _simulator = simulator
    _static_synapse_class = StaticSynapse

    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population,
                                   postsynaptic_population, connector,
                                   synapse_type, source, receptor_type,
                                   space, label)

        # Initialise the context stack
        ContextMixin.__init__(self, {})

        # Add projection to simulator
        self._simulator.state.projections.append(self)

        # In many cases, the maximum dimensions of each equally-sized
        # slice is going to be the same so we can cache them
        self._max_dims_estimate_cache = {}

        # If pre-synaptic population in an assembly
        if isinstance(self.pre, common.Assembly):
            # Add this projection to each pre-population in
            # assembly's list of outgoing connections
            for p in self.pre.populations:
                p.outgoing_projections.append(self)
        # Otherwise add it to the pre-synaptic population's list
        # **THINK** what about population-views? add to their parent?
        else:
            self.pre.outgoing_projections.append(self)

        # If post-synaptic population in an assembly
        if isinstance(self.post, common.Assembly):
            assert self.post._homogeneous_synapses, (
                "Inhomogeneous assemblies not yet supported")

            # Add this projection to each post-population in
            # assembly's list of incoming connections
            for p in self.post.populations:
                p.incoming_projections[self._synapse_cluster_type][self.pre].append(self)
        # Otherwise add it to the post-synaptic population's list
        # **THINK** what about population-views? add to their parent?
        else:
            self.post.incoming_projections[self._synapse_cluster_type][self.pre].append(self)

    def __len__(self):
        raise NotImplementedError

    def set(self, **attributes):
        raise NotImplementedError

    # --------------------------------------------------------------------------
    # Internal PyNN methods
    # --------------------------------------------------------------------------
    def _get_attributes_as_list(self, *names):
        logger.info("Downloading synaptic matrices for projection %s",
                    self.label)

        # Read synaptic matrices from the post-synaptic population
        synaptic_matrices = self.post._read_synaptic_matrices(
            self.pre, self._synapse_cluster_type, names)

        # Loop through all the rows of all the matrices and convert to a list
        return list(itertools.chain.from_iterable(
            row for matrix in synaptic_matrices for row in matrix))

    def _get_attributes_as_arrays(self, *names):
        logger.info("Downloading synaptic matrices for projection %s",
                    self.label)

        # **YUCK** As it's rather hard to build an array without
        # indices, add index parameters if not specified
        if "presynaptic_index" not in names:
            names += ("presynaptic_index",)
        if "postsynaptic_index" not in names:
            names += ("postsynaptic_index",)

        # Read synaptic matrices from the post-synaptic population
        synaptic_matrices = self.post._read_synaptic_matrices(
            self.pre, self._synapse_cluster_type, names)

        # If there are no synaptic matrices i.e. this
        # projection got entirely optimised out
        if len(synaptic_matrices) == 0:
            # Initialize all attribute arrays to NaN indicating no connections
            attribute_arrays = tuple(
                np.empty((self.pre.size, self.post.size)) * np.nan
                for name in names
                if name != "presynaptic_index" and name != "postsynaptic_index")
        else:
            # Stack all rows together into single mega-row
            all_rows = np.hstack(
                row for matrix in synaptic_matrices for row in matrix)

            # Count connections and build mask array of the pairs
            # of neurons between which there are no connections
            connection_bins = (self.pre.size, self.post.size)
            no_connection_mask = np.histogram2d(all_rows["presynaptic_index"],
                                                all_rows["postsynaptic_index"],
                                                connection_bins)[0] == 0

            # Build a tuple containing the sum of each connection
            # property (skipping over the pre and postsynaptic indices)
            attribute_arrays = tuple(
                scipy.stats.binned_statistic_2d(all_rows["presynaptic_index"],
                                                all_rows["postsynaptic_index"],
                                                all_rows[name],
                                                "sum", connection_bins)[0]
                for name in names
                if name != "presynaptic_index" and name != "postsynaptic_index")

            # Loop through each attribute array and set the
            # value to NaN wherever there is no connection
            for a in attribute_arrays:
                a[no_connection_mask] = np.nan

        # Return the tuple of attribute arrays
        return attribute_arrays

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _build(self, **context_kwargs):
        # **TODO** this may already have been connected
        # by another assembled post population
        # Build each projection, adding the matrix rows to the context
        with self.get_new_context(**context_kwargs):
            self._connector.connect(self)

    def _create_current_input_cluster(self, timer_period_us, simulation_ticks,
                                      frontend):
        # If this projection is directory connectable
        if self._directly_connectable:
            logger.debug("\t\tProjection:%s", self.label)

            # Find index of receptor type
            receptor_index =\
                self.post.celltype.receptor_types.index(self.receptor_type)

            # Create current input cluster
            self._current_input_cluster = CurrentInputCluster(
                self.pre.celltype, self.pre._parameters, self.pre.initial_values,
                self._simulator.state.dt, timer_period_us, simulation_ticks,
                self.pre.recorder.indices_to_record, self.pre.spinnaker_config,
                receptor_index, frontend, self._current_input_j_constraint,
                self.pre.size)
        # Otherwise, null current input cluster
        else:
            self._current_input_cluster = None

        return self._current_input_cluster

    @ContextMixin.use_contextual_arguments()
    def _direct_convergent_connect(self, presynaptic_indices,
                                   postsynaptic_index, direct_weights,
                                   **connection_parameters):
        # **TODO** one-to-one connections that
        # reshuffle cells COULD be supported
        assert len(presynaptic_indices) == 1
        assert presynaptic_indices[0] == postsynaptic_index

        # Warn if delay doesn't match simulation timestep
        # if connection_parameters["delay"] != self._simulator.state.dt:
        #    logger.warn("Direct connections are treated "
        #                "as having delay of one timestep")

        # Set weight in direct weights array
        direct_weights[postsynaptic_index] =\
            abs(connection_parameters["weight"])

    @ContextMixin.use_contextual_arguments()
    def _synaptic_convergent_connect(self, presynaptic_indices,
                                     postsynaptic_index, matrix_rows,
                                     weight_range, **connection_parameters):
        self.post._convergent_connect(presynaptic_indices, postsynaptic_index,
                                      matrix_rows, weight_range,
                                      **connection_parameters)

    @ContextMixin.use_contextual_arguments()
    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            directly_connect, **connection_parameters):
        # If post-synaptic population in an assembly
        if isinstance(self.post, common.Assembly):
            assert False
        # **TODO** figure out which population within assembly post index
        # relates to Otherwise add it to the post-synaptic population's list
        # **TODO** what about population-views? add to their parent?
        else:
            if directly_connect:
                self._direct_convergent_connect(presynaptic_indices,
                                                postsynaptic_index,
                                                **connection_parameters)
            else:
                self._synaptic_convergent_connect(presynaptic_indices,
                                                  postsynaptic_index,
                                                  **connection_parameters)

    def _build_direct_connection(self):
        # Assert that the connection is directly connectable
        assert self._directly_connectable

        # Create, initially zeroed away of direct connection weights
        direct_weights = np.zeros(self.post.size)

        # Build
        self._build(directly_connect=True, direct_weights=direct_weights)

        return direct_weights

    def _estimate_max_dims(self, pre_slice, post_slice):
        # The key to look it up in the cache of maximum
        # dimension estimates is a tuple of the lengths of each slice
        cache_key =  (len(pre_slice), len(post_slice))

        # If the connector is cachable return the cached values
        if (self._connector._cachable and cache_key in self._max_dims_estimate_cache):
            return self._max_dims_estimate_cache[cache_key]

        # The number of synapses per row within this post_slice are
        # distributed as follows
        row_synapses_dist = self._connector._row_synapses_distribution(
            pre_slice, post_slice, self.pre.size, self.post.size)

        # dist.ppf(0.9999) gives a value such that the distribution yields a value less than that
        # value with probability 0.9999
        # dist.ppf(0.9999 ** k) gives a value such that if we sample from the distribution k
        # times, the value will be less than that value with probability 0.9999
        quantile = 0.9999 ** (float(len(post_slice)) / (self.pre.size * self.post.size))

        # Calculate maximum synapses per row
        max_row_synapses = int(row_synapses_dist.ppf(quantile))

        # Calculate maximum row delay
        max_row_delay = (float(self.synapse_type._max_dtcm_delay_slots) *
                         self._simulator.state.dt)

        # Get delay parameter from synapse type
        delay = self.synapse_type.native_parameters["delay"]

        # If this projection has no synapses, so will all its sub-rows
        if max_row_synapses == 0:
            max_cols = 0
            max_sub_row_synapses = 0
            max_sub_rows = 0
        # If parameter is randomly distributed
        elif isinstance(delay.base_value, RandomDistribution):
            dist_name = delay.base_value.name
            pynn_params = delay.base_value.parameters

            # If we have a means of sampling from this distribution using scipy
            if dist_name in distribution:
                # Get scipy distribution object and convert PyNN
                # params into suitable form to pass to it
                delay_dist = distribution[dist_name][0](**distribution[dist_name][1](**pynn_params))

                # Calculate the probability of a given
                # synapse being in the first sub-row
                prob_first_sub_row = delay_dist.cdf(max_row_delay)

                # The number of synapses ending up in the first delay sub-row
                # is distributed as Binomial(n, p=prob_first_sub_row), where n
                # is distributed as row_synapses_dist
                # I.e., this is a mixture distribution

                # Calculate the range of plausible values for the number of synapses in the row
                # and the probability of getting that value
                p_limit = 1e-9
                row_synapses_range = np.arange(int(row_synapses_dist.ppf(p_limit)), int(row_synapses_dist.ppf(1-p_limit)) + 1)
                row_synapses_ps = row_synapses_dist.pmf(row_synapses_range)

                # Create a list of component distributions for the mixture distribution, one for each
                # plausible value for the number of synapses per row
                row_dists = [scipy.stats.binom(n=n, p=prob_first_sub_row) if n > 0 else scipy.stats.randint(0, 1) for n in row_synapses_range]
                # We want to compute dist.ppf(quantile) where dist is the mixture distribution.
                # Compute upper and lower limits on this value.
                row_range = [int(row_dists[0].ppf(quantile)), int(row_dists[-1].ppf(1-p_limit)) + 1]
                # Create a lazily-evaluated array of cdf(k) for each k in the range. Note that the cdf
                # of a mixture distribution is a weighted sum of the cdfs of the component distributions
                cdfs = la.larray(lambda k: eval_mixture_cdf(row_synapses_ps, row_dists, k),
                                 shape=(row_range[1],))
                # Use binary search to find the first value k such that dist.cdf(k) is greater than
                # or equal to quantile. I.e., compute dist.ppf(quantile)
                # Max cols is an upper bound on the number of synapses in the first delay row,
                # such that with probability 0.9999 this value will not be exceeded within
                # the whole projection
                max_cols = bisect_left(cdfs, quantile, row_range[0], row_range[1])

                # As above, but for an upper bound on the number of synapses not in the first delay row
                row_dists = [scipy.stats.binom(n=n, p=1-prob_first_sub_row) if n > 0 else scipy.stats.randint(0, 1) for n in row_synapses_range]
                row_range = [int(row_dists[0].ppf(quantile)), int(row_dists[-1].ppf(1-p_limit)) + 1]
                cdfs = la.larray(lambda k: eval_mixture_cdf(row_synapses_ps, row_dists, k),
                                 shape=(row_range[1],))
                max_sub_row_synapses = bisect_left(cdfs, quantile, row_range[0], row_range[1])

                # If there are no synapses outside of first delay sub-row
                if max_sub_row_synapses == 0:
                    assert max_cols == max_row_synapses
                    max_sub_rows = 0
                else:

                    # The distribution over the number of synapses (row_synapses_dist)
                    # tells us how many delays there are.
                    # We want the max of **that many** delays
                    # use row_synapses_range and row_synapses_ps
                    lower_bound = delay_dist.ppf(p_limit)
                    upper_bound = delay_dist.ppf(1-p_limit)
                    upper_delay_bound = continuous_bisect_fun_left(
                        lambda k: eval_mixture_of_maxes_cdf(
                            row_synapses_ps, row_synapses_range,
                            delay_dist, k),
                        quantile, lower_bound, upper_bound)
                    lower_delay_bound = continuous_bisect_fun_left(
                        lambda k: eval_mixture_of_mins_cdf(
                            row_synapses_ps, row_synapses_range,
                            delay_dist, k),
                        quantile, lower_bound, upper_bound)
                    lower_delay_bound = max(max_row_delay, lower_delay_bound)

                    max_extension_delay_range = upper_delay_bound - lower_delay_bound
                    # Convert this to a maximum number of sub-rows
                    max_sub_rows = max(1, int(math.ceil(max_extension_delay_range /
                                                        max_row_delay)))
            else:
                logger.warn("Cannot estimate delay sub-row distribution with %s",
                            dist_name)
                max_cols = max_row_synapses
                max_sub_rows = 0
                max_sub_row_synapses = 0
        # If parameter is a scalar
        elif is_scalar(delay.base_value):
            # If the delay is within the maximum row delay, then all
            # the synapses in the row can be represented in a single sub-row
            if delay.base_value <= max_row_delay:
                max_cols = max_row_synapses
                max_sub_rows = 0
                max_sub_row_synapses = 0
            # Otherwise, the first sub-row will contain no synapses,
            # just a pointer forwards to the delay sub-row
            else:
                max_cols = 0
                max_sub_rows = 1
                max_sub_row_synapses = max_row_synapses
        else:
            raise NotImplementedError()

        # If the projection's connector is cachable,
        # add the estimated dimensions to the cache
        if self._connector._cachable:
            self._max_dims_estimate_cache[cache_key] = (max_cols, max_sub_rows,
                                                        max_sub_row_synapses)
        return max_cols, max_sub_rows, max_sub_row_synapses

    def _estimate_spike_processing_cpu_cycles(self, pre_slice, post_slice):

        # The number of synapses per row within this post_slice are
        # distributed as follows
        row_synapses_dist = self._connector._row_synapses_distribution(
            pre_slice, post_slice, self.pre.size, self.post.size)

        # Use distribution to estimate mean number of synapses in each row
        mean_row_synapses = row_synapses_dist.mean()

         # Calculate maximum row delay
        max_row_delay = (float(self.synapse_type._max_dtcm_delay_slots) *
                         self._simulator.state.dt)

        # Get delay parameter from synapse type
        delay = self.synapse_type.native_parameters["delay"]

        # If this projection has no synapses, so will all its sub-rows
        if mean_row_synapses == 0.0:
            num_sub_rows = 1.0
            mean_sub_row_synapses = 0.0
        # If parameter is randomly distributed
        elif isinstance(delay.base_value, RandomDistribution):
            dist_name = delay.base_value.name
            pynn_params = delay.base_value.parameters

            # If we have a means of sampling from this distribution using scipy
            if dist_name in distribution:
                # Get scipy distribution object and convert PyNN
                # params into suitable form to pass to it
                dist = distribution[dist_name][0]
                params = distribution[dist_name][1](**pynn_params)

                # Get the mean upper and lower bounds of the row's delays
                mean_probability = 0.5 ** (1.0 / np.ceil(mean_row_synapses))
                mean_row_upper = dist.ppf(mean_probability, **params)
                mean_row_lower = dist.ppf(1.0 - mean_probability, **params)

                # If lower bound is smaller than simulation timestep it
                # cannot be simulated, give a warning and increase
                # it to simulation timestep
                if mean_row_lower < self._simulator.state.dt:
                    logger.warn("Delay distribution likely to result "
                                "in delays below simulation timestep of %f",
                                self._simulator.state.dt)
                    mean_row_lower = self._simulator.state.dt

                # If the mean range of delays overlaps
                # the obligatory first sub row
                if mean_row_lower <= max_row_delay:
                    # Move the mean lower delay down to the
                    # minimum delay of the first sub-row
                    mean_row_lower = self._simulator.state.dt

                    # Determine the number of sub-rows required for this range
                    delay_range = mean_row_upper - mean_row_lower
                    num_sub_rows = math.ceil(
                        (delay_range + self._simulator.state.dt) / max_row_delay)
                # Otherwise
                else:
                    # Determine the number of sub-rows required for this range
                    delay_range = mean_row_upper - mean_row_lower
                    num_sub_rows = math.ceil(
                        (delay_range + self._simulator.state.dt) / max_row_delay)

                    # Add an extra sub-row to take into account the first sub-row
                    num_sub_rows += 1

                assert num_sub_rows > 0

                # Divide mean number of synapses in row evenly between sub-rows
                mean_sub_row_synapses = mean_row_synapses / num_sub_rows
            else:
                logger.warn("Cannot estimate delay sub-row distribution with %s",
                            dist_name)

                mean_sub_row_synapses = mean_row_synapses
                num_sub_rows = 1.0
        # If parameter is a scalar
        elif is_scalar(delay.base_value):
            # If the delay is within the maximum row delay, then all
            # the synapses in the row can be represented in a single sub-row
            if delay.base_value <= max_row_delay:
                num_sub_rows = 1.0
                mean_sub_row_synapses = mean_row_synapses
            # Otherwise, the first sub-row will contain no synapses,
            # just a pointer forwards to the delay sub-row
            else:
                num_sub_rows = 2.0
                mean_sub_row_synapses = 0.5 * mean_row_synapses
        else:
            raise NotImplementedError()

        # Use synapse type to estimate CPU cost of processing sub row
        pre_rate = self.pre.spinnaker_config.mean_firing_rate
        post_rate = self.post.spinnaker_config.mean_firing_rate
        row_cpu_cost = self.synapse_type._get_row_cpu_cost(mean_sub_row_synapses,
                                                           pre_rate=pre_rate,
                                                           post_rate=post_rate)
        # Multiply this by the number of required subrows
        row_cpu_cost *= num_sub_rows

        # Scale row CPU cycles by number of presynaptic
        # neurons and their firing rate
        return (row_cpu_cost * pre_rate * len(pre_slice))

    def _allocate_out_buffers(self, placements, transceiver, app_id):
         # If projection has no current input cluster, skip
        if self._current_input_cluster is None:
            return

        logger.info("\tProjection label:%s from population label:%s",
                    self.label, self.pre.label)

        self._current_input_cluster.allocate_out_buffers(placements,
                                                         transceiver, app_id)

    def _load_verts(self, placements, transceiver, app_id):
        # If projection has no current input cluster, skip
        if self._current_input_cluster is None:
            return

        logger.info("\tProjection label:%s from population label:%s",
                    self.label, self.pre.label)

        # Build direct connection for projection
        direct_weights = self._build_direct_connection()

        # Load
        self._current_input_cluster.load(placements, transceiver, app_id,
                                         direct_weights)

    def _get_native_rngs(self, synapse_param_name):
        # Get named parameter
        param = self.synapse_type.native_parameters[synapse_param_name]

        # If parameter is randomly distributed
        if isinstance(param.base_value, RandomDistribution):
            # Assert that it uses our native RNG
            assert isinstance(param.base_value.rng, NativeRNG)

            # Return list containing RNG used to generate parameter
            return (param.base_value.rng,)
        # Otherwise return empty list
        else:
            return ()

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    @property
    def _synapse_cluster_type(self):
        return SynapseClusterType(self.synapse_type, self.receptor_type)

    @property
    def _directly_connectable(self):
        # If conversion of direct connections is disabled, return false
        if not self._simulator.state.convert_direct_connections:
            return False

        # If the pre-synaptic celltype can be directly connectable,
        # the connector can be reduced to a direct connector and
        # the synapse type is static
        return (self.pre.celltype._directly_connectable and
                self._connector._directly_connectable and
                type(self.synapse_type) is self._static_synapse_class)

    @property
    def _weight_range_estimate(self):
        # Extract weight parameters
        weights = self.synapse_type.native_parameters["weight"]

        # If weights are randomly distributed
        if isinstance(weights.base_value, RandomDistribution):
            # Get RNG and distribution
            rng = weights.base_value.rng
            distribution = weights.base_value.name
            parameters = weights.base_value.parameters

            # Assert that it uses our native RNG
            assert isinstance(rng, NativeRNG)

            # Return estimated maximum value of distribution
            return rng._estimate_dist_range(distribution, parameters)
        # Otherwise, if it's a scalar, return it
        elif is_scalar(weights.base_value):
            return (weights.base_value, weights.base_value)
        # Otherwise assert
        else:
            assert False

    @property
    def _can_generate_on_chip(self):
        # If generation of connections on chip is disabled, return false
        if not self._simulator.state.generate_connections_on_chip:
            return False

        # If the projection can be optimised out
        # into a direct connection, return false
        if self._directly_connectable:
            return False

        # If connector doesn't have a parameter map
        # for generating on-chip data, return false
        if not hasattr(self._connector, "_on_chip_param_map"):
            return False

        # If connector has an RNG and it is not a native RNG, return false
        # **YUCK** this more by convention than anything else
        if (hasattr(self._connector, "rng") and
            not isinstance(self._connector.rng, NativeRNG)):
            return False

        # If synaptic matrix type doesn't have a parameters
        # map for generating on chip data, return false
        if not hasattr(self.synapse_type._synaptic_matrix_region_class,
                       "OnChipParamMap"):
            return False

        # Get synapse native parameters
        s_params = self.synapse_type.native_parameters._parameters
        for p in s_params.values():
            # If parameter is specified using a random distribution
            if isinstance(p.base_value, RandomDistribution):
                # If it doesn't use the native RNG, return false
                if not isinstance(p.base_value.rng, NativeRNG):
                    return False

                # If the distribution isn't supported, return false
                if not p.base_value.rng._supports_dist(p.base_value.name):
                    return False
            # Otherwise, if parameter isn't a scalar, return false
            # **NOTE** Intuition is that parameters specified using arrays are
            # a)Not well-defined by PyNN
            # b)Probably wasteful to transfer to board
            elif not is_scalar(p.base_value):
                return False

        # All checks passed
        return True

    @property
    def _native_rngs(self):
        # If connector has an RNG
        # **YUCK** this more by convention than anything else
        rngs = set()
        if hasattr(self._connector, "rng"):
            # Assert that it uses our native RNG
            assert isinstance(self._connector.rng, NativeRNG)

            # Add RNG to set
            rngs.add(self._connector.rng)

        # Add any RNGs required to generate delay and weight parameters
        rngs.update(self._get_native_rngs("delay"))
        rngs.update(self._get_native_rngs("weight"))

        # Return list of unique RNGs required
        return list(rngs)
