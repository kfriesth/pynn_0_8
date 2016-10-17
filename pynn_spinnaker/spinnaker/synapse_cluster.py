# Import modules
import enum
import inspect
import itertools
import logging
import math
import numpy as np
from os import path
import regions
import sys

# Import classes
from collections import defaultdict
from utils import Args, InputVertex

# Import functions
from pkg_resources import resource_filename
from six import iteritems, iterkeys
from utils import (get_model_executable_filename, load_regions, split_slice)

logger = logging.getLogger("pynn_spinnaker")

# Numpy data type used for synaptic matrix rows
# **NOTE** delay is signed to make catching the 
# condition where zero delays are inserted easier
row_dtype = [("weight", np.float32), ("delay", np.int32),
             ("index", np.uint32)]


# ------------------------------------------------------------------------------
# WeightRange
# ------------------------------------------------------------------------------
class WeightRange(object):
    def __init__(self, signed_weight):
        # Based on signedness, determine how many
        # bits we need to fit range of weights within
        self.weight_val_bits = 15 if signed_weight else 16

        self.min = sys.float_info.max
        self.max = sys.float_info.min

    def update(self, weight):
        abs_weight = abs(weight)

        self.min = min(self.min, abs_weight)
        self.max = max(self.max, abs_weight)

    def update_iter(self, weight):
        abs_weight = np.abs(weight)

        self.min = min(self.min, np.amin(abs_weight))
        self.max = max(self.max, np.amax(abs_weight))

    @property
    def fixed_point(self):
        # Get MSB for maximum weight
        max_msb = math.floor(math.log(self.max, 2)) + 1

        # If minimum weight isn't zero
        if self.min != 0.0:
            # Get MSB of minimum weight
            min_msb = math.floor(math.log(self.min, 2)) + 1

            # Check there's enough bits to represent this range
            if (max_msb - min_msb) >= self.weight_val_bits:
                logger.warn("Insufficient range in %u-bit weight to represent "
                            "minimum weight:%f and maximum weight:%f",
                            self.weight_val_bits, self.min, self.max)

        # Calculate where the weight format fixed-point lies
        # **NOTE** we clamp so that there is at least a 1-bit overlap with
        # The bottom of the S16.15 format used by the neuron processors
        max_shift = self.weight_val_bits + 14
        return min(max_shift, (self.weight_val_bits - int(max_msb)))

# ----------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------
class Regions(enum.IntEnum):
    """Region names, corresponding to those defined in `synapse_processor.h`"""
    system = 0
    key_lookup = 1
    synaptic_matrix = 2
    plasticity = 3
    output_buffer = 4
    delay_buffer = 5
    back_prop_input = 6
    connection_builder = 7
    profiler = 8
    statistics = 9

# ----------------------------------------------------------------------------
# Vertex
# ----------------------------------------------------------------------------
class Vertex(InputVertex):
    def __init__(self, post_neuron_slice, receptor_index,
                 incoming_connections, sdram, app_name):
        # Superclass
        super(Vertex, self).__init__(post_neuron_slice, receptor_index,
                                     sdram, app_name)

        self.back_prop_in_verts = []
        self.incoming_connections = incoming_connections

# ------------------------------------------------------------------------------
# SynapseCluster
# ------------------------------------------------------------------------------
class SynapseCluster(object):
    # Tag names, corresponding to those defined in synapse_processor.h
    profiler_tag_names = {
        0:  "Timer tick",
        1:  "Multicast packet received",
        2:  "Setup next DMA row read",
        3:  "Process row",
        4:  "Process back propagation",
    }

    # Names of statistics
    statistic_names = (
        "row_requested",
        "delay_row_requested",
        "delay_buffers_not_processed",
        "input_buffer_overflows",
        "key_lookup_fails",
    )

    def __init__(self, sim_timestep_ms, timer_period_us, sim_ticks,
                 max_delay_ms, config, post_pop_size, synapse_model,
                 receptor_index, synaptic_projections,
                 frontend, post_synaptic_width):
        # Dictionary of regions
        self.regions = {}
        self.regions[Regions.system] = regions.System(timer_period_us,
                                                      sim_ticks)
        self.regions[Regions.key_lookup] = regions.KeyLookupBinarySearch()
        self.regions[Regions.output_buffer] = regions.OutputBuffer()
        self.regions[Regions.delay_buffer] = regions.DelayBuffer(
            synapse_model._max_synaptic_event_rate,
            sim_timestep_ms, max_delay_ms)
        self.regions[Regions.back_prop_input] = regions.SDRAMBackPropInput()
        self.regions[Regions.connection_builder] = regions.ConnectionBuilder(
            sim_timestep_ms)
        self.regions[Regions.statistics] = regions.Statistics(
            len(self.statistic_names))

        # Create correct type of synaptic matrix region
        self.regions[Regions.synaptic_matrix] =\
            synapse_model._synaptic_matrix_region_class(synapse_model)

        # If synapse mode has a plasticity parameter map
        if hasattr(synapse_model, "_plasticity_param_map"):
            self.regions[Regions.plasticity] =\
                regions.HomogeneousParameterSpace(
                    synapse_model._plasticity_param_map,
                    synapse_model.native_parameters,
                    sim_timestep_ms)

        # Add profiler region if required
        if config.num_profile_samples is not None:
            self.regions[Regions.profiler] =\
                regions.Profiler(config.num_profile_samples)

        # Split population slice
        self.post_slices = split_slice(post_pop_size, post_synaptic_width)

        logger.debug("\t\tSynapse model:%s, Receptor index:%u",
                     synapse_model.__class__.__name__, receptor_index)

        # Get synapse application name
        synapse_app = get_model_executable_filename(
            "synapse_", synapse_model, config.num_profile_samples is not None)

        logger.debug("\t\t\tSynapse application:%s", synapse_app)

        # Cache synapse model
        self.synapse_model = synapse_model

        # Loop through the post-slices
        generate_matrix_on_chip = False
        self.verts = []
        for post_slice in self.post_slices:
            logger.debug("\t\t\tPost slice:%s", str(post_slice))

            # Loop through all non-directly connectable
            # projections of this type
            vert_event_rate = 0.0
            vert_sdram_bytes = 0
            vert_incoming_connections = defaultdict(list)
            for proj in synaptic_projections:
                logger.debug("\t\t\t\tProjection:%s", proj.label)

                # If this projection can be generated on chip, set flag
                if proj._can_generate_on_chip:
                    generate_matrix_on_chip = True

                # Loop through the vertices which the pre-synaptic
                # population has been partitioned into
                for pre_vertex in proj.pre._neural_cluster.verts:
                    logger.debug("\t\t\t\t\tPre slice:%s",
                                 str(pre_vertex.neuron_slice))

                    # Estimate number of synapses the connection between
                    # The pre and the post-slice of neurons will contain
                    total_synapses = proj._estimate_num_synapses(
                        pre_vertex.neuron_slice, post_slice)

                    # If this projection doesn't result in any
                    # synapses don't add connection
                    if total_synapses == 0:
                        logger.debug("\t\t\t\t\t\tNo synapses")
                        continue

                    # Use this to calculate event rate
                    pre_mean_rate = proj.pre.spinnaker_config.mean_firing_rate
                    pre_rate = total_synapses * pre_mean_rate

                    # Estimate MAXIMUM number of synapses that may be in a row
                    max_row_synapses = proj._estimate_max_row_synapses(
                        pre_vertex.neuron_slice, post_slice)

                    # Estimate size of matrix
                    synaptic_matrix = self.regions[Regions.synaptic_matrix]
                    sdram_bytes = synaptic_matrix.estimate_matrix_bytes(
                        pre_vertex.neuron_slice, max_row_synapses)

                    logger.debug("\t\t\t\t\t\tTotal synapses:%d, "
                                 "synaptic event rate:%f Hz, SDRAM:%u bytes",
                                 total_synapses, pre_rate, sdram_bytes)

                    # Add this connection to the synapse vertex
                    vert_incoming_connections[proj.pre].append(pre_vertex)

                    # Add event rate and SDRAM to totals
                    # for current synapse processor
                    vert_event_rate += pre_rate
                    vert_sdram_bytes += sdram_bytes

                    # If the event rate is more than this type of synapse
                    # processor can handle or the matrix requires more
                    # than the 16mb the key lookup data structure can address
                    if (vert_event_rate > synapse_model._max_synaptic_event_rate
                        or vert_sdram_bytes > (16 * 1024 * 1024)):
                        # Create vertex
                        vert = Vertex(post_slice, receptor_index,
                                      vert_incoming_connections,
                                      vert_sdram_bytes, synapse_app)

                        # Add to list and frontend
                        self.verts.append(vert)
                        frontend.add_machine_vertex(vert)

                        logger.debug("\t\t\t\t\t\tVertex: total event rate:%f Hz, SDRAM:%u bytes",
                                     vert_event_rate, vert_sdram_bytes)

                        # Create new list of incoming connections
                        # for next vert and reset counters
                        vert_incoming_connections = defaultdict(list)
                        vert_event_rate = 0.0
                        vert_sdram_bytes = 0

            # If the last synapse vertex created had any incoming connections
            if len(vert_incoming_connections) > 0:
                # Create vertex
                vert = Vertex(post_slice, receptor_index,
                              vert_incoming_connections,
                              vert_sdram_bytes, synapse_app)

                # Add to list and frontend
                self.verts.append(vert)
                frontend.add_machine_vertex(vert)
                logger.debug("\t\t\t\t\t\tVertex: total event rate:%f Hz, SDRAM:%u bytes",
                             vert_event_rate, vert_sdram_bytes)

        logger.debug("\t\t\t%u synapse vertices", len(self.verts))

        # If any matrices should be generated on chip, show message
        '''
        if generate_matrix_on_chip:
            # Find path to connection builder aplx
            connection_builder_app = resource_filename(
                "pynn_spinnaker",
                "standardmodels/binaries/connection_builder.aplx")
            logger.debug("\t\t\tConnection builder application:%s",
                         connection_builder_app)
        '''
        
    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------
    def allocate_out_buffers(self, placements, allocations,
                             machine_controller):
        # Loop through synapse verts
        for v in self.verts:
            # Get placement and allocation
            vertex_placement = placements[v]
            vertex_allocation = allocations[v]

            # Get core this vertex should be run on
            core = vertex_allocation[machine.Cores]
            assert (core.stop - core.start) == 1

            logger.debug("\t\tVertex %s (%u, %u, %u)",
                         v, vertex_placement[0], vertex_placement[1],
                         core.start)

            # Select placed chip
            with machine_controller(x=vertex_placement[0],
                                    y=vertex_placement[1]):
                # Allocate two output buffers
                # for this synapse population
                out_buffer_bytes = len(v.post_neuron_slice) * 4
                v.out_buffers = [
                    machine_controller.sdram_alloc(out_buffer_bytes,
                                                   clear=True)
                    for _ in range(2)]

    def load(self, placements, allocations, machine_controller,
             incoming_projections, flush_mask):

        # Loop through all the postsynaptic slices in this synapse cluster
        for post_slice in self.post_slices:
            logger.debug("\t\t\tPost slice:%s", str(post_slice))

            # Get 'column' of vertices in this postsynaptic slice
            post_slice_verts = [v for v in self.verts
                                if v.post_neuron_slice == post_slice]

            # Create weight range
            weight_range = WeightRange(self.synapse_model._signed_weight)

            # Loop through unique presynaptic populations with connections
            # terminating in any of the vertices in this postsynaptic slice
            pre_pop_sub_rows = {}
            pre_pop_on_chip_proj = {}
            for pre_pop in set(itertools.chain.from_iterable(
                iterkeys(v.incoming_connections)
                for v in post_slice_verts)):

                # If all incoming projections from this population
                # are generatable on chip and there aren't multiple
                # projections that need merging
                incoming_from_pre = incoming_projections[pre_pop]
                if (all(i._can_generate_on_chip for i in incoming_from_pre) and
                    len(incoming_from_pre) == 1):

                    # Mark list of projections for generating on chip
                    pre_pop_on_chip_proj[pre_pop] = incoming_from_pre

                    # Loop through projections to generate on chip and update
                    # weight range based on maximum weight estimates
                    for proj in incoming_from_pre:
                        weight_range.update(proj._max_weight_estimate)
                # Otherwise
                else:
                    # Create list of lists to contain matrix rows
                    sub_rows = [[] for _ in range(pre_pop.size)]

                    # Loop through projections leading from pre_pop
                    for proj in incoming_from_pre:
                        # Check local mask isn't currently in use
                        assert np.all(proj.post._mask_local)

                        # Cache original post mask (due to above
                        # this is slightly pointless but still)
                        old_post_mask = proj.post._mask_local

                        # Create new local mask to select only the columns
                        # corresponding to neurons in postsynaptic vertex
                        proj.post._mask_local = np.zeros((proj.post.size,),
                                                         dtype=bool)
                        proj.post._mask_local[post_slice.python_slice] = True

                        # Cache original connector callback
                        old_connector_callback = proj._connector.callback
                        proj._connector.callback = None

                        # Add synapses from projection to rows
                        proj._build(matrix_rows=sub_rows,
                                    weight_range=weight_range,
                                    directly_connect=False)

                        # Restore old mask and connector callback
                        proj.post._mask_local = old_post_mask
                        proj._connector.callback = old_connector_callback

                    # Convert rows to numpy and add to dictionary
                    pre_pop_sub_rows[pre_pop] = [np.asarray(r, dtype=row_dtype)
                                                for r in sub_rows]

            logger.debug("\t\t\t\t%u generated on host, %u to generate on chip",
                         len(pre_pop_sub_rows), len(pre_pop_on_chip_proj))
            # If the synapse model has a function to update weight range
            if hasattr(self.synapse_model, "_update_weight_range"):
                self.synapse_model._update_weight_range(weight_range)

            # Calculate where the weight format fixed-point lies
            weight_fixed_point = weight_range.fixed_point
            logger.debug("\t\t\t\tWeight fixed point:%u", weight_fixed_point)

            # Loop through synapse verts in this postsynaptic slice
            for v in post_slice_verts:
                # Get placement and allocation
                vertex_placement = placements[v]
                vertex_allocation = allocations[v]

                # Get core this vertex should be run on
                core = vertex_allocation[machine.Cores]
                assert (core.stop - core.start) == 1

                logger.debug("\t\t\t\tVertex %s (%u, %u, %u)",
                            v, vertex_placement[0], vertex_placement[1],
                            core.start)

                # Partition matrices that have been generated on host
                host_sub_matrix_props, host_sub_matrix_rows =\
                    self.regions[Regions.synaptic_matrix].partition_matrices(
                        post_slice, pre_pop_sub_rows, v.incoming_connections)

                # Partition matrices that should be generated on chip
                chip_sub_matrix_props, chip_sub_matrix_projs =\
                    self.regions[Regions.synaptic_matrix].partition_on_chip_matrix(
                        post_slice, pre_pop_on_chip_proj, v.incoming_connections)

                # Build combined list of matrix properties
                sub_matrix_props = host_sub_matrix_props + chip_sub_matrix_props

                # Cache weight fixed-point for
                # this synapse point in vertex
                v.weight_fixed_point = weight_fixed_point

                # Place them in memory
                matrix_placements =\
                    self.regions[Regions.key_lookup].place_matrices(
                        sub_matrix_props)

                # Select placed chip
                with machine_controller(x=vertex_placement[0],
                                        y=vertex_placement[1]):
                    # Get the back propagation buffers from
                    # each back-propagating neuron vertex
                    back_prop_in_buffers = [
                        b.get_back_prop_in_buffer(v.post_neuron_slice)
                        for b in v.back_prop_in_verts]

                    # Get region arguments required to
                    # calculate size and write
                    region_arguments = self._get_region_arguments(
                        v.post_neuron_slice, sub_matrix_props,
                        host_sub_matrix_rows, chip_sub_matrix_projs,
                        matrix_placements, weight_fixed_point, v.out_buffers,
                        back_prop_in_buffers, flush_mask)

                    # Load regions
                    v.region_memory = load_regions(
                        self.regions, region_arguments,
                        machine_controller, core)

                    # Store sub matrix properties and placements in vertex
                    # so they can be used to subsequently read weights back
                    v.sub_matrix_props = sub_matrix_props
                    v.matrix_placements = matrix_placements

    def read_profile(self):
        # Get the profile recording region
        region = self.regions[Regions.profiler]

        # Return profile data for each vertex that makes up population
        return [(v.post_neuron_slice.python_slice,
                 region.read_profile(v.region_memory[Regions.profiler],
                                     self.profiler_tag_names))
                for v in self.verts]

    def read_statistics(self):
        # Get the statistics recording region
        region = self.regions[Regions.statistics]

        # Convert stats to numpy array
        np_stats = np.asarray([region.read_stats(v.region_memory[Regions.statistics])
                            for v in self.verts])
        # Convert stats into record array
        stat_names = ",".join(self.statistic_names)
        stat_format = ",".join(
            itertools.repeat("u4", len(self.statistic_names)))
        return np.core.records.fromarrays(np_stats.T, names=stat_names,
                                          formats=stat_format)

    def read_synaptic_matrices(self, pre_pop, names, sim_timestep_ms):
        # Get the synaptic matrix region
        region = self.regions[Regions.synaptic_matrix]

        # Loop through synapse vertices (post-synaptic)
        sub_matrices = []
        for post_s_vert in self.verts:
            # If this synapse vertex has no incoming connections
            # from pre-synaptic population, skip
            if pre_pop not in post_s_vert.incoming_connections:
                continue

            # Get region memory for synaptic matrix
            region_mem = post_s_vert.region_memory[Regions.synaptic_matrix]

            # Loop through list of pre-synaptic vertices
            # this synapse vertex is connected to
            for pre_n_vert in post_s_vert.incoming_connections[pre_pop]:
                # Read associated sub-matrix
                sub_matrices.append(
                    region.read_sub_matrix(pre_n_vert, post_s_vert,
                                           names, region_mem, sim_timestep_ms))

        return sub_matrices

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------
    def _get_region_arguments(self, post_vertex_slice, sub_matrix_props,
                              host_sub_matrix_rows, chip_sub_matrix_projs,
                              matrix_placements,
                              weight_fixed_point, out_buffers,
                              back_prop_in_buffers, flush_mask):
        region_arguments = defaultdict(Args)

        # Add kwargs for regions that require them
        region_arguments[Regions.system].kwargs["application_words"] =\
            [weight_fixed_point, len(post_vertex_slice), flush_mask]

        region_arguments[Regions.key_lookup].kwargs["sub_matrix_props"] =\
            sub_matrix_props
        region_arguments[Regions.key_lookup].kwargs["matrix_placements"] =\
            matrix_placements

        region_arguments[Regions.synaptic_matrix].kwargs["sub_matrix_props"] =\
            sub_matrix_props
        region_arguments[Regions.synaptic_matrix].kwargs["host_sub_matrix_rows"] =\
            host_sub_matrix_rows
        region_arguments[Regions.synaptic_matrix].kwargs["matrix_placements"] =\
            matrix_placements
        region_arguments[Regions.synaptic_matrix].kwargs["weight_fixed_point"] =\
            weight_fixed_point

        region_arguments[Regions.output_buffer].kwargs["out_buffers"] =\
            out_buffers

        region_arguments[Regions.delay_buffer].kwargs["sub_matrix_props"] =\
            sub_matrix_propsmachine_controlle

        region_arguments[Regions.plasticity].kwargs["fixed_point"] =\
            weight_fixed_point

        region_arguments[Regions.back_prop_input].kwargs["back_prop_in_buffers"] =\
            back_prop_in_buffers

        region_arguments[Regions.connection_builder].kwargs["sub_matrix_props"] =\
            sub_matrix_props
        region_arguments[Regions.connection_builder].kwargs["chip_sub_matrix_projs"] =\
            chip_sub_matrix_projs
        region_arguments[Regions.connection_builder].kwargs["post_vertex_slice"] =\
            post_vertex_slice
        region_arguments[Regions.connection_builder].kwargs["weight_fixed_point"] =\
            weight_fixed_point

        return region_arguments
