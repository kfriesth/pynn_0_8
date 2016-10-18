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
import numpy as np
import scipy
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
                                      vertex_load_applications, vertex_run_applications,
                                      vertex_resources):
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
                receptor_index, vertex_load_applications, vertex_run_applications,
                vertex_resources, self.current_input_j_constraint, self.pre.size)
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

    def _estimate_max_row_synapses(self, pre_slice, post_slice):
        return self._connector._estimate_max_row_synapses(
            pre_slice, post_slice, self.pre.size, self.post.size)

    def _estimate_num_synapses(self, pre_slice, post_slice):
        return self._connector._estimate_num_synapses(
            pre_slice, post_slice, self.pre.size, self.post.size)

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
    def _max_weight_estimate(self):
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
            return rng._estimate_dist_max_value(distribution, parameters)
        # Otherwise, if it's a scalar, return it
        elif is_scalar(weights.base_value):
            return weights.base_value
        # Otherwise assert
        else:
            assert False

    @property
    def _can_generate_on_chip(self):
        # If generation of connections on chip is disabled, return false
        if not self._simulator.state.generate_connections_on_chip:
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

        # Calculate maximum delay that is supported using ring-buffer
        # **TODO** support on-chip generation of rowlets
        max_delay_slots = self.synapse_type._max_dtcm_delay_slots
        max_delay = float(max_delay_slots) * self._simulator.state.dt

        # If delay is random and its maximum value is
        # larger than the maximum, return false
        delay = s_params["delay"].base_value
        if (isinstance(delay, RandomDistribution)
            and delay.rng._estimate_dist_max_value(delay.name,
                                                   delay.parameters) > max_delay):
            return False

        # If delay is a constant larger than the maximum, return false
        if is_scalar(delay) and delay > max_delay:
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
