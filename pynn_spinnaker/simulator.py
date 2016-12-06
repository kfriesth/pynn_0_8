# Import modules
import config
import itertools
import logging
import math
import numpy as np
from os import path, stat
import time

# Import classes
from collections import defaultdict
from pacman.executor.algorithm_decorators.algorithm_decorator import algorithm
from pacman.model.constraints.placer_constraints\
    .placer_same_chip_as_constraint import PlacerSameChipAsConstraint
from pyNN import common
from spinn_front_end_common.interface.spinnaker_main_interface import \
    SpinnakerMainInterface
from spinn_front_end_common.utilities.utility_objs.executable_finder \
    import ExecutableFinder
from spinn_front_end_common.utilities.utility_objs.executable_targets \
    import ExecutableTargets
from spinnman.model.cpu_state import CPUState
from spinn_storage_handlers.file_data_reader import FileDataReader
from spinn_machine.utilities.progress_bar import ProgressBar

# Import functions
from six import iteritems, itervalues
from spinn_front_end_common.utilities.helpful_functions import \
    wait_for_cores_to_be_ready

# Import globals
from pynn_spinnaker.standardmodels import __file__ as standard_models

logger = logging.getLogger("pynn_spinnaker")

name = "SpiNNaker"

 # ----------------------------------------------------------------------------
# PACMAN algorithms
# ----------------------------------------------------------------------------
@algorithm(input_definitions={
    "placements": "MemoryPlacements",
    "transceiver": "MemoryTransceiver",
    "app_id": "APPID"},
    outputs=[])
def _allocate_algorithm(placements, transceiver, app_id):
    # Allocate buffers for SDRAM-based communication between vertices
    logger.info("Allocating population output buffers")
    for pop in state.populations:
        pop._allocate_out_buffers(placements, transceiver, app_id)
    logger.info("Allocating projection output buffers")
    for proj in state.projections:
        proj._allocate_out_buffers(placements, transceiver, app_id)

@algorithm(input_definitions={
    "placements": "MemoryPlacements",
    "transceiver": "MemoryTransceiver",
    "app_id": "APPID"},
    outputs=["LoadedApplicationDataToken"])
def _load_data_algorithm(placements, transceiver, app_id):
    # Load vertices
    # **NOTE** projection vertices need to be loaded
    # first as weight-fixed point is only calculated at
    # load time and this is required by neuron vertices
    logger.info("Loading projection vertices")
    for proj in state.projections:
        proj._load_verts(placements, transceiver, app_id)

    logger.info("Loading population vertices")
    for pop in state.populations:
        pop._load_verts(state.frontend.routing_infos, placements,
                        transceiver, app_id)
    return True

@algorithm(input_definitions={
    "placements": "MemoryPlacements",
    "transceiver": "MemoryTransceiver",
    "executable_finder": "ExecutableFinder",
    "app_id": "APPID",
    "loaded_application_data_token": "LoadedApplicationDataToken"},
    outputs=[])
def _build_data_on_chip_algorithm(placements, transceiver, executable_finder,
                                  app_id, loaded_application_data_token):
    # Check host-generated application data has been loaded
    if not loaded_application_data_token:
            raise exceptions.ConfigurationException(
                "The token for having loaded the application data token is set"
                " to false and therefore I cannot run. Please fix and try "
                "again")

    # Create executable targets
    executable_targets = ExecutableTargets()

    # Loop through each population and add any executable
    # targets required to build application data
    for pop in state.populations:
        pop._add_loader_executable_targets(placements,
                                           executable_targets)

    if len(executable_targets.binaries) > 0:
        # Loop through loader binaries in targets
        progress_bar = ProgressBar(executable_targets.total_processors,
                                "Loading loader executables onto the machine")
        for binary_name in executable_targets.binaries:
            # Attempt to find this within search paths
            binary_path = executable_finder.get_executable_path(binary_name)
            if binary_path is None:
                raise exceptions.ExecutableNotFoundException(binary_name)

            # Get file reader to read binary and subset of cores to load it to;
            # and flood fill
            file_reader = FileDataReader(binary_path)
            core_subset = executable_targets.get_cores_for_binary(binary_name)

            transceiver.execute_flood(core_subset, file_reader, app_id,
                                      stat(binary_path).st_size)

            # Count the number of cores loaded and
            # update the progress bar accordingly
            actual_cores_loaded = sum(len(list(chip.processor_ids))
                                      for chip in core_subset.core_subsets)
            progress_bar.update(amount_to_add=actual_cores_loaded)
        progress_bar.end()

        # Wait for loader executables to finish
        progress_bar = ProgressBar(executable_targets.total_processors,
                                "Building data on-chip")
        wait_for_cores_to_be_ready(executable_targets, app_id,
                                   transceiver, CPUState.FINISHED)
        progress_bar.end()

# ----------------------------------------------------------------------------
# ID
# ----------------------------------------------------------------------------
class ID(int, common.IDMixin):
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
class State(common.control.BaseState):
    # These are required to be present for various
    # bits of PyNN, but not really relevant for SpiNNaker
    mpi_rank = 0
    num_processes = 1

    def __init__(self):
        # Superclass
        common.control.BaseState.__init__(self)

        self.frontend = None
        self.dt = 0.1

        self.clear()

    # ----------------------------------------------------------------------------
    # PyNN State methods
    # ----------------------------------------------------------------------------
    def run(self, simtime):
        # Build data
        try:
            self._build(simtime)
        except:
            self.end()

        self.t += simtime
        self.running = True

    def run_until(self, tstop):
        # Build data
        self._build(tstop - self.t)

        self.t = tstop
        self.running = True

    def clear(self):
        self.recorders = set([])
        self.id_counter = 42
        self.segment_counter = -1
        self.reset()

        # Mapping from post-synaptic PyNN population (i.e. the
        # one the current input cluster is injecting current INTO)
        # to list of current input clusters
        # {pynn_population: [current_input_cluster]}
        self.post_pop_current_input_clusters = defaultdict(list)

        # List of populations
        self.populations = []

        # List of projections
        self.projections = []

        # Stop any currently running SpiNNaker application
        self.stop()

    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

    def stop(self):
        if self.frontend is not None and self.stop_on_spinnaker:
            logger.info("Stopping SpiNNaker application")
            self.frontend.stop()
            self.frontend = None

    # ----------------------------------------------------------------------------
    # PyNN SpiNNaker internal methods
    # ----------------------------------------------------------------------------
    def _estimate_constraints(self, hardware_timestep_us):
        logger.info("Estimating constraints")

        # Loop through populations whose output can't be
        # entirely be replaced by direct connections
        populations = [p for p in self.populations
                       if not p._entirely_directly_connectable]
        for pop_id, pop in enumerate(populations):
            logger.debug("\tPopulation:%s", pop.label)
            pop._estimate_constraints(hardware_timestep_us)

    def _constrain_clusters(self):
        logger.info("Constraining vertex clusters to same chip")

        # Loop through populations
        constraints = []
        for pop in self.populations:
            # If population has no neuron cluster, skip
            if pop._neural_cluster is None:
                continue

            # Get lists of synapse, neuron and current input
            # vertices associated with this PyNN population
            s_verts = list(itertools.chain.from_iterable(
                c.verts for c in itervalues(pop._synapse_clusters)))
            c_verts = list(itertools.chain.from_iterable(
                c.verts for c in self.post_pop_current_input_clusters[pop]))

            # If there are any synapse vertices
            if len(s_verts) > 0 or len(c_verts) > 0:
                logger.debug("\tPopulation:%s", pop.label)

                # Loop through neuron vertices
                for n in pop._neural_cluster.verts:
                    # Find synapse and current vertices
                    # with overlapping slices
                    n.input_verts = [
                        i for i in itertools.chain(s_verts, c_verts)
                        if i.post_neuron_slice.overlaps(n.neuron_slice)]

                    logger.debug("\t\tConstraining neuron vert and %u input "
                                 "verts to same chip", len(n.input_verts))

                    # Build same chip constraint and add to neuron vertex
                    # **YUCK** should be able to use a list of vertices here
                    for i in n.input_verts:
                        n.add_constraint(PlacerSameChipAsConstraint(i))

            # Loop through synapse clusters
            for s_type, s_cluster in iteritems(pop._synapse_clusters):
                # If synapse cluster doesn't require back propagation, skip
                if not s_type.model._requires_back_propagation:
                    continue

                logger.debug("\t\tSynapse type:%s, receptor:%s",
                             s_type.model.__class__.__name__, s_type.receptor)

                # Loop through synapse vertices
                for s_vert in s_cluster.verts:
                    # Set synapse vetices list of back propagation
                    # input vertices to all neural cluster vertices
                    # whose neuron slices overlap
                    s_vert.back_prop_in_verts = [
                        n_vert for n_vert in pop._neural_cluster.verts
                        if s_vert.post_neuron_slice.overlaps(n_vert.neuron_slice)]

                    logger.debug("\t\t\tVertex %s has %u back propagation vertices",
                                 s_vert, len(s_vert.back_prop_in_verts))

        return constraints

    def _read_stats(self, duration_ms):
        logger.info("Reading stats")

        # Loop through populations
        duration_s = float(duration_ms) / 1000.0
        for pop in self.populations:
            for s_type, stats in iteritems(pop.get_synapse_statistics()):
                logger.info("\t\tSynapse type:%s receptor:%s",
                            s_type.model.__class__.__name__, s_type.receptor)
                logger.info("\t\t\tRows requested - Average per vertex per second:%f, Total per second:%f",
                            np.mean(stats["row_requested"]) / duration_s,
                            np.sum(stats["row_requested"]) / duration_s)
                logger.info("\t\t\tDelay rows requested - Average per vertex per second:%f, Total per second:%f",
                            np.mean(stats["delay_row_requested"]) / duration_s,
                            np.sum(stats["delay_row_requested"]) / duration_s)
                logger.info("\t\t\tDelay buffers not processed:%u",
                            np.sum(stats["delay_buffers_not_processed"]))
                logger.info("\t\t\tInput buffer overflows:%u",
                            np.sum(stats["input_buffer_overflows"]))
                logger.info("\t\t\tKey lookup failures:%u",
                            np.sum(stats["key_lookup_fails"]))

    def _build(self, duration_ms):
        # Convert dt into microseconds and divide by
        # realtime proportion to get hardware timestep
        hardware_timestep_us = int(round((1000.0 * float(self.dt)) /
                                         float(self.realtime_proportion)))

        # Determine how long simulation is in timesteps
        duration_timesteps =\
            int(math.ceil(float(duration_ms) / float(self.dt)))

        logger.info("Simulating for %u %fms timesteps "
                    "using a hardware timestep of %uus",
                    duration_timesteps, self.dt, hardware_timestep_us)

        # Estimate constraints
        self._estimate_constraints(hardware_timestep_us)

        # If there isn't already a frontend
        # **TODO** this probably doesn't belong here
        if self.frontend is None:
            logger.info("Creating frontend")

            # Rad config file
            config_parser = config.read_config()

            # Create executable finder and add standard models path
            # **TODO** move executable finder somewhere globally instantiated
            executable_finder = ExecutableFinder()
            executable_finder.add_path(path.join(path.dirname(standard_models),
                                                 "binaries"))

            # Create frontend
            self.frontend = SpinnakerMainInterface(
                config_parser, executable_finder,
                extra_algorithm_xml_paths=(),
                extra_load_algorithms=["_allocate_algorithm",
                                       "_load_data_algorithm",
                                       "_build_data_on_chip_algorithm"])

            # Pass hostname to frontend
            self.frontend.set_up_machine_specifics(self.spinnaker_hostname)

            # **YUCK** I can't find any way to set this,
            # presumably from its name, internal property
            self.frontend._machine_time_step = hardware_timestep_us
            self.frontend._time_scale_factor = 1.0 / self.realtime_proportion

        # Allocate clusters
        # **NOTE** neuron clusters and hence vertices need to be allocated
        # first as synapse cluster allocateion is dependant on neuron vertices
        logger.info("Allocating neuron clusters")
        for pop_id, pop in enumerate(self.populations):
            logger.debug("\tPopulation:%s", pop.label)
            pop._create_neural_cluster(hardware_timestep_us,
                                       duration_timesteps, self.frontend)

        logger.info("Allocating synapse clusters")
        for pop in self.populations:
            logger.debug("\tPopulation:%s", pop.label)
            pop._create_synapse_clusters(hardware_timestep_us,
                                         duration_timesteps, self.frontend)

        logger.info("Allocating current input clusters")
        for proj in self.projections:
            # Create cluster
            c = proj._create_current_input_cluster(
                hardware_timestep_us, duration_timesteps, self.frontend)

            # Add cluster to data structures
            if c is not None:
                self.post_pop_current_input_clusters[proj.post].append(c)

        # Constrain all vertices in clusters to same chip
        constraints = self._constrain_clusters()

        # Build nets
        logger.info("Building nets")

        # Loop through all populations and build nets
        for pop in self.populations:
            pop._build_nets(self.frontend)

        # Run
        # **NOTE** allocation and loading will be performed using algorithms
        self.frontend.run(duration_ms)

        self._read_stats(duration_ms)
state = State()
