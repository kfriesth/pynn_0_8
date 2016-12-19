# Import modules
import itertools
import logging
import math
import numpy as np
from rig import machine
from pyNN import common

# Import classes
from collections import defaultdict, Iterable, namedtuple
from operator import itemgetter
from pacman.model.graphs.machine.impl.machine_edge import MachineEdge
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace
from . import simulator
from .recording import Recorder
from rig.netlist import Net
from spinnaker.neural_cluster import NeuralCluster
from spinnaker.synapse_cluster import SynapseCluster
from spinnaker.spinnaker_population_config import SpinnakerPopulationConfig
from spinnaker.utils import UnitStrideSlice

# Import functions
from copy import deepcopy
from pyNN.parameters import simplify
from six import iteritems, iterkeys, itervalues

logger = logging.getLogger("pynn_spinnaker")

Synapse = namedtuple("Synapse", ["weight", "delay", "index"])


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------
def _calc_clusters_per_core(cluster_width, constraint):
    return int(math.ceil(2.0 ** math.floor(np.log2(float(constraint) /
                                                   float(cluster_width)))))

def _calc_cores_per_cluster(cluster_width, constraint):
    return int(math.ceil(2.0 ** math.ceil(np.log2(float(cluster_width) /
                                                 float(constraint)))))

# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
class Assembly(common.Assembly):
    _simulator = simulator

# --------------------------------------------------------------------------
# PopulationView
# --------------------------------------------------------------------------
class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    # --------------------------------------------------------------------------
    # Internal PyNN methods
    # --------------------------------------------------------------------------
    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = self.parent._parameters[name]
            if isinstance(value, np.ndarray):
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, shape=(self.size,))

    def _set_parameters(self, parameter_space):
        # Loop through parameters we're setting, evaluate the value we're
        # Setting and assign it to the masked section of
        # parent's parameters this view represents
        for name, value in parameter_space.items():
            evaluated_value = value.evaluate(simplify=True)
            self.parent._parameters[name][self.mask] = evaluated_value

    def _set_initial_value_array(self, variable, initial_values):
        # Initial values are handled by common.Population
        # so we can evaluate them at build-time
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)


# --------------------------------------------------------------------------
# Population
# --------------------------------------------------------------------------
class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly

    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        super(Population, self).__init__(size, cellclass, cellparams,
                                         structure, initial_values, label)

        # Create a spinnaker config
        self.spinnaker_config = SpinnakerPopulationConfig()
        
        # Dictionary mapping pre-synaptic populations to
        # incoming projections, subdivided by synapse type
        # {synapse_cluster_type: {pynn_population: [pynn_projection]}}
        self.incoming_projections = defaultdict(lambda: defaultdict(list))

        # List of outgoing projections from this population
        # [pynn_projection]
        self.outgoing_projections = list()

        # Add population to simulator
        self._simulator.state.populations.append(self)

    # --------------------------------------------------------------------------
    # Public SpiNNaker methods
    # --------------------------------------------------------------------------
    def get_neural_profile_data(self):
        logger.info("Downloading neural profile for population %s",
                    self.label)

        # Assert that profiling is enabled
        assert self.spinnaker_config.num_profile_samples is not None

        # Read profile from neuron cluster
        return self._neural_cluster.read_profile()

    def get_synapse_profile_data(self):
        logger.info("Downloading synapse profile for population %s",
                    self.label)

        # Assert that profiling is enabled
        assert self.spinnaker_config.num_profile_samples is not None

        # Read profile from each synapse cluster
        return {t: c.read_profile()
                for t, c in iteritems(self._synapse_clusters)}

    def get_current_input_profile_data(self):
        logger.info("Downloading current input profile for population %s",
                    self.label)

        # Assert that profiling is enabled
        assert self.spinnaker_config.num_profile_samples is not None

        # Read profile from each current input cluster
        c_clusters = self._simulator.state.proj_current_input_clusters
        return {p: c_clusters[p].read_profile()
                for p in self.outgoing_projections
                if p._directly_connectable}

    def get_synapse_statistics(self):
        logger.info("\tDownloading synapse statistics for population %s",
                    self.label)

        # Read statistics from each synapse cluster
        return {t: c.read_statistics()
                for t, c in iteritems(self._synapse_clusters)}

    def get_neural_statistics(self):
        logger.info("\tDownloading neural statistics for population %s",
                    self.label)

        # Read statistics from neuron cluster
        return self._neural_cluster.read_statistics()

    # --------------------------------------------------------------------------
    # Internal PyNN methods
    # --------------------------------------------------------------------------
    def _create_cells(self):
        id_range = np.arange(simulator.state.id_counter,
                             simulator.state.id_counter + self.size)
        self.all_cells = np.array([simulator.ID(id) for id in id_range],
                                  dtype=simulator.ID)

        # In terms of MPI, all SpiNNaker neurons are local
        self._mask_local = np.ones((self.size,), bool)

        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size

        # Take a deep copy of cell type parameters
        if isinstance(self.celltype, StandardCellType):
            self._parameters = deepcopy(self.celltype.native_parameters)
        else:
            self._parameters = deepcopy(self.celltype.parameter_space)

        # Set shape
        self._parameters.shape = (self.size,)

    def _set_initial_value_array(self, variable, initial_values):
        # Initial values are handled by common.Population
        # so we can evaluate them at build-time
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            parameter_dict[name] = simplify(self._parameters[name])
        return ParameterSpace(parameter_dict, shape=(self.local_size,))

    def _set_parameters(self, parameter_space):
        # Loop through values we're setting and
        # deep copy into our parameter space
        for name, value in parameter_space.items():
            self._parameters[name] = deepcopy(value)

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    def _read_synaptic_matrices(self, pre_pop, synapse_type, names):
        # Is the receptor type inhibitory
        # **YUCK** this is not a great test
        is_inhibitory = (synapse_type.receptor == "inhibitory")

        # Return synaptic weights from correct synapse cluster
        synapse_cluster = self._synapse_clusters[synapse_type]
        return synapse_cluster.read_synaptic_matrices(
            pre_pop, names, float(self._simulator.state.dt),
            is_inhibitory,
            self._simulator.state.frontend.routing_infos)

    def _read_recorded_vars(self, vars_to_read):
        spike_times = {}
        signals = {}

        # If we have a neuron clusters
        if self._neural_cluster is not None:
            # Loop through all variables to read
            for var in vars_to_read:
                # If this variable is a spike recording, update the
                # spike times dictionary with spikes from this vertex
                if var == "spikes":
                    spike_times = self._neural_cluster.read_recorded_spikes()
                # Otherwise
                else:
                    # Convert variable name to channel number
                    # **HACK** subtract one assuming first entry is spikes
                    channel = self.celltype.recordable.index(var) - 1

                    # Read signal from this channel and add to dictionary
                    sig = self._neural_cluster.read_recorded_signal(channel)
                    signals[var] = sig
        # Otherwise, if we're recording spikes
        elif "spikes" in vars_to_read:
            # Loop through outgoing connections
            for o in self.outgoing_projections:
                # If this connection isn't directly connectable skip
                if not o._directly_connectable:
                    continue

                # Read spike times from the current input
                # cluster associated with this projection
                spike_times = o._current_input_cluster.read_recorded_spikes()
                break


        return spike_times, signals

    def _estimate_synaptic_constraints(self, hardware_timestep_us,
                                       dc_projections, dc_j_constraints):
        # Iterate to find cluster configuration
        while True:
            max_constraint = max(itervalues(self._synapse_j_constraints))
            logger.debug("\t\t\tMax synapse j constraint:%u",
                         max_constraint)

            # Loop again through incoming synapse types to estimate i_constraints
            total_syn_processors = 0
            total_i_cores = 0
            for s_type, pre_pop_projections in iteritems(self.incoming_projections):
                # Get list of synaptic connections
                projections = itertools.chain.from_iterable(
                    itervalues(pre_pop_projections))
                synaptic_projections = [p for p in projections
                                        if not p._directly_connectable]

                # If there are any
                if len(synaptic_projections) > 0:
                    s_type_constraint = self._synapse_j_constraints[s_type]

                    # Build suitable post-slice to estimate CPU usage over
                    post_slice = UnitStrideSlice(0, s_type_constraint)

                    # Loop through list of projections
                    total_cpu_cycles = 0.0
                    for proj in synaptic_projections:
                        # Estimate CPU cycles per second required to process sub-matrix
                        cpu_cycles = proj._estimate_spike_processing_cpu_cycles(
                            UnitStrideSlice(0, proj.pre.size), post_slice)

                        total_cpu_cycles += cpu_cycles

                    # Calculate the constant overhead for each
                    # simulation timestep and thus the number
                    # of cycles available for row processing
                    constant_overhead = (s_type.model._constant_cpu_overhead *
                                         (1000.0 / self._simulator.state.dt))
                    available_core_cpu_cycles = 200E6 - constant_overhead

                    # Scale CPU cycles by realtime proportion
                    available_core_cpu_cycles /= self._simulator.state.realtime_proportion

                    # Calculate presynaptic (i) 'height' of synapse processors
                    # required to handle the synaptic processing of post_slice
                    num_i_cores = int(math.ceil(float(total_cpu_cycles) / float(available_core_cpu_cycles)))

                    # Calculate the postsynaptic (j) 'width' of synapse
                    # processors required to fill max_constraint width
                    num_j_cores = int(math.ceil(float(max_constraint) /
                                                float(s_type_constraint)))
                    num_cores = num_i_cores * num_j_cores
                    logger.debug("\t\t\t\tSynapse type:%s, receptor:%s - Total CPU cycles:%f, num cores:%u",
                                s_type.model.__class__.__name__, s_type.receptor,
                                total_cpu_cycles, num_cores)

                    # Add number of i cores to total
                    total_syn_processors += num_cores
                    total_i_cores += num_i_cores

            logger.debug("\t\t\t\tTotal synapse processors:%u",
                         total_syn_processors)

            # Calculate maximum number of neurons
            # each neuron processor can handle
            neuron_j_constraint =\
                self.celltype._calc_max_neurons_per_core(
                    hardware_timestep_us=hardware_timestep_us,
                    num_input_processors=total_i_cores + len(dc_projections))

            num_neuron_j = _calc_clusters_per_core(max_constraint, neuron_j_constraint)
            max_clusters = num_neuron_j
            logger.debug("\t\t\t\tNeuron J constraint:%u, num synapse clusters per neuron core:%u",
                         neuron_j_constraint, num_neuron_j)

            # If there are any direct connections
            if len(dc_j_constraints) > 0:
                # Convert into a number required to fill max_constraint
                max_current_input_j = max(
                    _calc_clusters_per_core(max_constraint, c)
                    for c in dc_j_constraints)
                logger.debug("\t\t\t\tMax synapse clusters per current input core:%u",
                            max_current_input_j)
                max_clusters = max(max_clusters, max_current_input_j)

            # Search downwards through possible power-of-two cluster widths
            logger.debug("\t\t\t\tMax synapse clusters:%u",
                         max_clusters)
            max_cluster_power = int(np.log2(max_clusters))
            for s in (2 ** p for p in range(max_cluster_power, -1, -1)):
                cluster_width = s * max_constraint
                logger.debug("\t\t\t\tCluster width:%u", cluster_width)

                cluster_synapse_processors = total_syn_processors * s
                logger.debug("\t\t\t\t\t%u synapse processors",
                             cluster_synapse_processors)

                cluster_neuron_processors =\
                    _calc_cores_per_cluster(cluster_width, neuron_j_constraint)
                logger.debug("\t\t\t\t\t%u neuron processors",
                             cluster_neuron_processors)

                cluster_current_input_processors = [
                    _calc_cores_per_cluster(cluster_width, c)
                    for c in dc_j_constraints]
                logger.debug("\t\t\t\t\t%u current input processors",
                             sum(cluster_current_input_processors))

                # If this configuration can fit on a chip
                if((cluster_synapse_processors + cluster_neuron_processors +
                    sum(cluster_current_input_processors)) <= 17):
                    logger.debug("\t\t\t\t\tCluster fits on chip!")

                    # Calculate final neuron J constraint
                    self._neuron_j_constraint =\
                        cluster_width // cluster_neuron_processors
                    logger.debug("\t\t\t\t\t%u neurons per neuron processor",
                                 self._neuron_j_constraint)

                    # Calculate final current input J
                    # constraint for each projection
                    for p, c in zip(dc_projections, cluster_current_input_processors):
                        p._current_input_j_constraint = cluster_width // c
                        logger.debug("\t\t\t\t\t%s - %u neurons per current input processor",
                                 p.label, p._current_input_j_constraint)
                    return

            # Divide the constraint on any synapse processors
            # which currently have maximum constraint by 2
            new_max_constraint = max_constraint // 2
            self._synapse_j_constraints = {
                s_type: (new_max_constraint
                         if constraint == max_constraint
                         else constraint)
                for s_type, constraint in iteritems(self._synapse_j_constraints)}

    def _estimate_non_synaptic_constraints(self, hardware_timestep_us,
                                           dc_projections, dc_j_constraints):
        logger.debug("\t\t\tNo synapse processors in cluster")

        # Calculate maximum number of neurons each neuron processor
        # can handle without any synaptic inputs
        self._neuron_j_constraint =\
            self.celltype._calc_max_neurons_per_core(
                hardware_timestep_us=hardware_timestep_us,
                num_input_processors=len(dc_projections))
        logger.debug("\t\t\t%u neurons per neuron processor",
                        self._neuron_j_constraint)

        # Use maximum current input processor constraint directly
        for c, p in zip(dc_j_constraints, dc_projections):
            p._current_input_j_constraint = c
            logger.debug("\t\t\t%s - %u neurons per current input processor",
                        p.label, p._current_input_j_constraint)

    def _estimate_constraints(self, hardware_timestep_us):
        logger.debug("\t\tFinding maximum synapse J constraints")

         # Loop through synapse types
        self._synapse_j_constraints = {}
        dc_projections = []
        dc_j_constraints = []
        for s_type, pre_pop_projections in iteritems(self.incoming_projections):
            logger.debug("\t\tSynapse type:%s, receptor:%s",
                             s_type.model.__class__.__name__,
                             s_type.receptor)

            # Get list of incoming directly connectable projections
            projections = list(itertools.chain.from_iterable(
                itervalues(pre_pop_projections)))

            # Filter this to just those that can be directly connected
            s_type_dc_projections = [p for p in projections
                                     if p._directly_connectable]

            # If there's any non-directly connectable projections of this type
            if len(projections) != len(s_type_dc_projections):
                synapse_constraint = s_type.model._max_post_neurons_per_core
                logger.debug("\t\t\tSynapse processor j constraint:%u",
                             synapse_constraint)
                self._synapse_j_constraints[s_type] = synapse_constraint

            # Calculate maximum number of current inputs
            # each current input processor can handle
            s_type_dc_j_constraints =\
                [p.pre.celltype._calc_max_current_inputs_per_core(hardware_timestep_us)
                for p in s_type_dc_projections]
            for c, p in zip(s_type_dc_j_constraints, s_type_dc_projections):
                logger.debug("\t\t\tProjection %s - current input J constraint:%u",
                             p.label, c)

            # Add directly connectable projections and
            # their corresponding constraint to list
            dc_projections.extend(s_type_dc_projections)
            dc_j_constraints.extend(s_type_dc_j_constraints)

        logger.debug("\t\tFinding cluster configuration")

        # If we have no synaptic inputs, call
        # simple algorithm to estimate constraints
        if len(self._synapse_j_constraints) == 0:
            self._estimate_non_synaptic_constraints(hardware_timestep_us,
                                                    dc_projections,
                                                    dc_j_constraints)
        # Otherwise, call full algorithm to estimate constraints
        # based primarily on synaptic requirements
        else:
            self._estimate_synaptic_constraints(hardware_timestep_us,
                                                dc_projections,
                                                dc_j_constraints)

    def _create_neural_cluster(self, pop_id, timer_period_us, simulation_ticks,
                               frontend):
        # Create neural cluster
        if not self._entirely_directly_connectable:
            # Determine if any of the incoming projections
            # to this population require back-propagation
            requires_back_prop = any(
                s_type.model._requires_back_propagation
                for s_type in iterkeys(self.incoming_projections))

            self._neural_cluster = NeuralCluster(
                pop_id, self.celltype, self._parameters, self.initial_values,
                self._simulator.state.dt, timer_period_us,
                simulation_ticks, self.recorder.sampling_interval,
                self.recorder.indices_to_record, self.spinnaker_config,
                frontend, self._neuron_j_constraint, requires_back_prop,
                self.size)
        else:
            self._neural_cluster = None

    def _create_synapse_clusters(self, timer_period_us, simulation_ticks,
                                 frontend):
        # Loop through newly partioned incoming projections
        self._synapse_clusters = {}
        for s_type, pre_pop_projs in iteritems(self.incoming_projections):
            # Chain together incoming projections from all populations
            projs = list(itertools.chain.from_iterable(
                itervalues(pre_pop_projs)))
            synaptic_projs = [p for p in projs if not p._directly_connectable]

            # If there are any synaptic projections
            if len(synaptic_projs) > 0:
                # Find index of receptor type
                receptor_index = self.celltype.receptor_types.index(s_type.receptor)

                # Create synapse cluster
                c = SynapseCluster(self._simulator.state.dt, timer_period_us,
                                   self._simulator.state.realtime_proportion,
                                   simulation_ticks,
                                   self._simulator.state.max_delay,
                                   self.spinnaker_config, self.size,
                                   s_type.model, receptor_index,
                                   synaptic_projs, frontend,
                                   self._synapse_j_constraints[s_type])

                # Add cluster to dictionary
                self._synapse_clusters[s_type] = c

    def _build_nets(self, frontend):
        # If population has no  neural cluster, skip
        if self._neural_cluster is None:
            return

        logger.debug("\tPopulation label:%s", self.label)

        # Get synapse vertices associated
        # with post-synaptic population
        post_s_verts = list(itertools.chain.from_iterable(
            [o.post._synapse_clusters[o._synapse_cluster_type].verts
            for o in self.outgoing_projections]))

        logger.debug("\t\t%u post-synaptic vertices",
                        len(post_s_verts))

        # Loop through each neuron vertex that makes up population
        for n_vert in self._neural_cluster.verts:
            # Loop through post-synaptic synapse vertices
            for s_vert in post_s_verts:
                # If this synapse vertex doesn't need to be
                # connected to this neuron vertex, skip
                if n_vert not in s_vert.incoming_connections[self]:
                    continue

                # Calculate traffic weight based on
                # mean firing rate of this population
                mean_firing_rate = self.spinnaker_config.mean_firing_rate
                traffic_weight = mean_firing_rate * len(n_vert.neuron_slice)

                # Add an edge to the front end representing this connection
                # **THINK** could we use a second 'partition_id' for flushing
                frontend.add_machine_edge(
                    MachineEdge(n_vert, s_vert,
                                traffic_weight=traffic_weight), 0)

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            matrix_rows, weight_range,
                            **connection_parameters):
        # Convert delay into timesteps and round
        delay_timesteps = np.around(
            connection_parameters["delay"] / float(self._simulator.state.dt))
        delay_timesteps = delay_timesteps.astype(int)

        # Check that delays are greater than zero after converting to timesteps
        assert np.all(delay_timesteps > 0)

        # If delay is not iterable, make it so using repeat
        if not isinstance(delay_timesteps, Iterable):
            delay_timesteps = itertools.repeat(delay_timesteps)

        # If weight is an iterable, update weight range
        weight = connection_parameters["weight"]
        if isinstance(weight, Iterable):
            weight_range.update_iter(weight)
        # Otherwise
        else:
            # Update weight range
            weight_range.update(weight)

            # Make weight iterable using repeat
            weight = itertools.repeat(weight)

        # Add synapse to each row
        for i, w, d in zip(presynaptic_indices, weight, delay_timesteps):
            matrix_rows[i].append(Synapse(w, d, postsynaptic_index))

    def _allocate_out_buffers(self, placements, transceiver, app_id):
        logger.info("\tPopulation label:%s", self.label)

         # Loop through synapse types and associated cluster
        for s_type, s_cluster in iteritems(self._synapse_clusters):
            logger.debug("\t\tSynapse type:%s, receptor:%s",
                        s_type.model.__class__.__name__, s_type.receptor)

            # Allocate out buffers
            s_cluster.allocate_out_buffers(placements, transceiver, app_id)

        # If population has a neuron cluster,
        # allow it to allocate any output buffers
        if self._neural_cluster is not None:
            self._neural_cluster.allocate_out_buffers(placements, transceiver,
                                                      app_id)

    def _load_verts(self, routing_info, placements, transceiver, app_id):
        logger.info("\tPopulation label:%s", self.label)

        # Loop through synapse types and associated cluster
        for s_type, s_cluster in iteritems(self._synapse_clusters):
            logger.info("\t\tSynapse type:%s, receptor:%s",
                        s_type.model.__class__.__name__, s_type.receptor)

            # Load vertices that make up cluster
            s_cluster.load(routing_info, placements, transceiver, app_id,
                           self.incoming_projections[s_type])

        # If population has a neuron cluster, load it
        if self._neural_cluster is not None:
            logger.info("\t\tNeurons")
            self._neural_cluster.load(routing_info, placements, transceiver,
                                      app_id)

    def _add_loader_executable_targets(self, placements,
                                       loader_executable_targets):
        # Loop through synapse types and associated clusters and allow
        # them to add any executable targets required for loading their data
        for s_type, s_cluster in iteritems(self._synapse_clusters):
            s_cluster.add_loader_executable_targets(placements,
                                                    loader_executable_targets)

        # **TODO** allow neuron clusters to do the same

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    @property
    def _entirely_directly_connectable(self):
        # If conversion of direct connections is disabled, return false
        if not self._simulator.state.convert_direct_connections:
            return False

        # If cell type isn't directly connectable, the population can't be
        if not self.celltype._directly_connectable:
            return False

        # If none of the outgoing projections aren't directly connectable!
        return not any([not o._connector._directly_connectable
                        for o in self.outgoing_projections])
