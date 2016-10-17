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

# Round a j constraint to the lowest power-of-two
# multiple of the minium j constraint
def round_j_constraint(j_constraint, min_j_constraint):
    return min_j_constraint * int(2 ** math.floor(
        math.log(j_constraint / min_j_constraint, 2)))


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
        # Return synaptic weights from correct synapse cluster
        synapse_cluster = self._synapse_clusters[synapse_type]
        return synapse_cluster.read_synaptic_matrices(
            pre_pop, names, float(self._simulator.state.dt))

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

    def _clamp_j_constraint(self, j_constraint):
        if self.spinnaker_config.max_cluster_width is None:
            return min(self.size, j_constraint)
        else:
            return min(self.spinnaker_config.max_cluster_width,
                       self.size,
                       j_constraint)

    def _estimate_constraints(self, hardware_timestep_us):
        # Determine the fraction of 1ms that the hardware timestep is.
        # This is used to scale all time-driven estimates
        timestep_mul = min(1.0, float(hardware_timestep_us) / 1000.0)
        logger.debug("\t\tTimestep multiplier:%f", timestep_mul)

        # Apply timestep multipliers to the hard maximum specified in celltype
        max_neurons_per_core = (
            self.spinnaker_config.max_neurons_per_core
            if self.spinnaker_config.max_neurons_per_core is not None
            else int(self.celltype._max_neurons_per_core * timestep_mul))

        # Clamp constraint to actual size of population
        self.neuron_j_constraint = self._clamp_j_constraint(
            max_neurons_per_core)
        logger.debug("\t\tNeuron j constraint:%u",
                     self.neuron_j_constraint)

        # Loop through synapse types
        self.synapse_j_constraints = {}
        current_input_j_constraints = {}
        for s_type, pre_pop_projections in iteritems(self.incoming_projections):
            # Get list of incoming directly connectable projections
            projections = list(itertools.chain.from_iterable(
                itervalues(pre_pop_projections)))
            directly_connectable_projections = [p for p in projections
                                                if p._directly_connectable]

            # If there's any non-directly connectable projections of this type
            if len(projections) != len(directly_connectable_projections):
                # Get hard maximum from synapse type
                synapse_constraint = s_type.model._max_post_neurons_per_core

                # Clamp constraint to actual size of
                # population and add to dictionary
                self.synapse_j_constraints[s_type] =\
                    self._clamp_j_constraint(synapse_constraint)

                logger.debug("\t\tSynapse type:%s, receptor:%s - j constraint:%u",
                             s_type.model.__class__.__name__, s_type.receptor,
                             synapse_constraint)

            # Loop through directly connectable projections
            for p in directly_connectable_projections:
                # Apply timestep multipliers to the
                # hard maximum specified in celltype
                current_input_constraint = (
                    p.pre.spinnaker_config.max_neurons_per_core
                    if p.pre.spinnaker_config.max_neurons_per_core is not None
                    else int(p.pre.celltype._max_current_inputs_per_core *
                             timestep_mul))
                
                # Clamp constraint to actual size of
                # population and add to dictionary
                current_input_constraint =\
                    self._clamp_j_constraint(current_input_constraint)
                current_input_j_constraints[p] = current_input_constraint
                logger.debug("\t\tDirectly connectable projection:%s "
                             "- Current input contraint:%u", p.label,
                             current_input_constraint)

        # Find the minimum constraint in j
        min_j_constraint = self.neuron_j_constraint
        if len(self.synapse_j_constraints) > 0 or len(current_input_j_constraints) > 0:
            min_j_constraint = min(
                min_j_constraint,
                *itertools.chain(itervalues(self.synapse_j_constraints),
                                itervalues(current_input_j_constraints)))

        logger.debug("\t\tMin j constraint:%u", min_j_constraint)

        # Round j constraints to multiples of minimum
        self.neuron_j_constraint = round_j_constraint(
            self.neuron_j_constraint, min_j_constraint)

        self.synapse_j_constraints = {
            t: round_j_constraint(c, min_j_constraint)
            for t, c in iteritems(self.synapse_j_constraints)}

        current_input_j_constraints = {
            t: round_j_constraint(c, min_j_constraint)
            for t, c in iteritems(current_input_j_constraints)}

        # Loop again through incoming synapse types to estimate i_constraints
        synapse_num_i_cores = {}
        for s_type, pre_pop_projections in iteritems(self.incoming_projections):
            # Get list of synaptic connections
            projections = itertools.chain.from_iterable(
                itervalues(pre_pop_projections))
            synaptic_projections = [p for p in projections
                                    if not p._directly_connectable]

            # If there are any
            if len(synaptic_projections) > 0:
                 # Build suitable post-slice for
                post_slice = UnitStrideSlice(
                    0, self.synapse_j_constraints[s_type])

                # Loop through list of projections
                total_synaptic_event_rate = 0.0
                for proj in synaptic_projections:
                    # If projection is directly connectable, skip
                    if proj._directly_connectable:
                        continue

                    # Estimate number of synapses the connection between
                    # The pre and the post-slice of neurons will contain
                    total_synapses = proj._estimate_num_synapses(
                        UnitStrideSlice(0, proj.pre.size), post_slice)

                    # Use this to calculate event rate
                    pre_mean_rate = proj.pre.spinnaker_config.mean_firing_rate
                    total_synaptic_event_rate += total_synapses * pre_mean_rate

                num_i_cores = int(math.ceil(total_synaptic_event_rate / float(s_type.model._max_synaptic_event_rate)))
                logger.debug("\t\tSynapse type:%s, receptor:%s - Total synaptic event rate:%f, num cores:%u",
                            s_type.model.__class__.__name__, s_type.receptor,
                            total_synaptic_event_rate, num_i_cores)

                # Add number of i cores to dictionary
                synapse_num_i_cores[s_type] = num_i_cores

        # Now determin the maximum constraint i.e. the 'width'
        # that will be constrained together
        max_j_constraint = self.neuron_j_constraint
        if len(self.synapse_j_constraints) > 0 or len(current_input_j_constraints) > 0:
            max_j_constraint = max(
                max_j_constraint,
                *itertools.chain(itervalues(self.synapse_j_constraints),
                                itervalues(current_input_j_constraints)))

        # Calculate how many cores this means will be required
        num_neuron_cores = max_j_constraint / self.neuron_j_constraint
        num_synapse_cores = sum(
            synapse_num_i_cores[t] * (max_j_constraint / c)
            for t, c in iteritems(self.synapse_j_constraints))
        num_current_input_cores = sum(
            max_j_constraint / c
            for c in itervalues(current_input_j_constraints))
        num_cores = num_neuron_cores + num_synapse_cores + num_current_input_cores

        # Check that this will fit on a chip
        # **TODO** iterate, dividing maximum constraint by 2
        assert num_cores <= 16

        logger.debug("\t\tConstraints will contain up to %u neuron cores, %u synapse cores and %u current input cores (%u in total)",
                     num_neuron_cores, num_synapse_cores,
                     num_current_input_cores, num_cores)

        logger.debug("\t\tNeuron j constraint:%u", self.neuron_j_constraint)
        for s_type, constraint in iteritems(self.synapse_j_constraints):
            logger.debug("\t\tSynapse type:%s, receptor:%s - J constraint:%u",
                         s_type.model.__class__.__name__,
                         s_type.receptor, constraint)
        for proj, constraint in iteritems(current_input_j_constraints):
            logger.debug("\t\tDirect input projection:%s - J constraint:%u",
                         proj.label, constraint)

            # Also store constraint in projection
            proj.current_input_j_constraint = constraint

    def _create_neural_cluster(self, pop_id, timer_period_us, simulation_ticks,
                               vertex_load_applications, vertex_run_applications,
                               vertex_resources, keyspace):
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
                vertex_load_applications, vertex_run_applications,
                vertex_resources, keyspace, self.neuron_j_constraint,
                requires_back_prop, self.size)
        else:
            self._neural_cluster = None

    def _create_synapse_clusters(self, timer_period_us, simulation_ticks,
                                 vertex_load_applications, vertex_run_applications,
                                 vertex_resources):
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
                                   simulation_ticks,
                                   self._simulator.state.max_delay,
                                   self.spinnaker_config, self.size,
                                   s_type.model, receptor_index,
                                   synaptic_projs, vertex_load_applications,
                                   vertex_run_applications, vertex_resources,
                                   self.synapse_j_constraints[s_type])

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
                front_end.add_machine_edge(
                    MachineEdge(n_vert, s_vert,
                                traffic_weight=traffic_weight))

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

    def _allocate_out_buffers(self, placements, allocations, machine_controller):
        logger.info("\tPopulation label:%s", self.label)

         # Loop through synapse types and associated cluster
        for s_type, s_cluster in iteritems(self._synapse_clusters):
            logger.debug("\t\tSynapse type:%s, receptor:%s",
                        s_type.model.__class__.__name__, s_type.receptor)

            # Allocate out buffers
            s_cluster.allocate_out_buffers(placements, allocations,
                                           machine_controller)

        # If population has a neuron cluster,
        # allow it to allocate any output buffers
        if self._neural_cluster is not None:
            self._neural_cluster.allocate_out_buffers(placements, allocations,
                                                      machine_controller)

    def _load_verts(self, placements, allocations,
                    machine_controller, flush_mask):
        logger.info("\tPopulation label:%s", self.label)

        # Loop through synapse types and associated cluster
        for s_type, s_cluster in iteritems(self._synapse_clusters):
            logger.info("\t\tSynapse type:%s, receptor:%s",
                        s_type.model.__class__.__name__, s_type.receptor)

            # Load vertices that make up cluster
            s_cluster.load(placements, allocations, machine_controller,
                           self.incoming_projections[s_type],
                           flush_mask)

        # If population has a neuron cluster, load it
        if self._neural_cluster is not None:
            logger.info("\t\tNeurons")
            self._neural_cluster.load(placements, allocations, machine_controller)

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
