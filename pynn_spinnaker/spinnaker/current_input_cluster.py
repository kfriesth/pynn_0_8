# Import modules
import enum
import logging
import regions
from rig import machine

# Import classes
from collections import defaultdict
from utils import (Args, InputVertex)

# Import functions
from six import iteritems
from utils import (get_model_executable_filename, load_regions, split_slice)

logger = logging.getLogger("pynn_spinnaker")


# ------------------------------------------------------------------------------
# Regions
# ------------------------------------------------------------------------------
class Regions(enum.IntEnum):
    """Region names, corresponding to those defined in `current_input.h`"""
    system = 0
    neuron = 1
    output_buffer = 2
    output_weight = 3
    spike_recording = 4
    profiler = 5


# ------------------------------------------------------------------------------
# CurrentInputCluster
# ------------------------------------------------------------------------------
class CurrentInputCluster(object):
    # Tag names, corresponding to those defined in current_input.h
    profiler_tag_names = {
        0:  "Timer tick",
    }

    def __init__(self, cell_type, parameters, initial_values, sim_timestep_ms,
                 timer_period_us, sim_ticks, indices_to_record, config,
                 receptor_index, frontend,
                 post_synaptic_width, pop_size):
        # Create standard regions
        self.regions = {}
        self.regions[Regions.system] = regions.System(
            timer_period_us, sim_ticks)
        self.regions[Regions.neuron] = cell_type._neuron_region_class(
            cell_type, parameters, initial_values, sim_timestep_ms, pop_size)
        self.regions[Regions.output_buffer] = regions.OutputBuffer()
        self.regions[Regions.output_weight] = regions.OutputWeight()
        self.regions[Regions.spike_recording] = regions.SpikeRecording(
            indices_to_record, sim_timestep_ms, sim_ticks)

        # Add profiler region if required
        if config.num_profile_samples is not None:
            self.regions[Regions.profiler] =\
                regions.Profiler(config.num_profile_samples)

        # Slice current input
        post_slices = split_slice(pop_size, post_synaptic_width)

        current_input_app = get_model_executable_filename(
            "current_input_", cell_type, config.num_profile_samples is not None)
        logger.debug("\t\t\tCurrent input application:%s",
                     current_input_app)

        # Loop through slice
        self.verts = []
        for post_slice in post_slices:
            # Estimate SDRAM usage
            sdram = self._estimate_sdram(post_slice)
            logger.debug("\t\t\tPost slice %s: %u bytes SDRAM",
                         str(post_slice), sdram)

            # Build input vert, add to list and add to frontend
            input_vert = InputVertex(post_slice, receptor_index, sdram,
                                     current_input_app)
            self.verts.append(input_vert)
            frontend.add_machine_vertex(input_vert)

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
                # Allocate two output buffers for this synapse population
                out_buffer_bytes = len(v.post_neuron_slice) * 4
                v.out_buffers = [
                    machine_controller.sdram_alloc(out_buffer_bytes,
                                                   clear=True)
                    for _ in range(2)]

    def load(self, placements, allocations, machine_controller,
             direct_weights):
        # Loop through synapse verts
        for v in self.verts:
            # Use native S16.15 format
            v.weight_fixed_point = 15

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
                # Get region arguments required to calculate size and write
                region_arguments = self._get_region_arguments(
                    v.post_neuron_slice, direct_weights, v.out_buffers)

                # Load regions
                v.region_memory = load_regions(self.regions, region_arguments,
                                               machine_controller, core)

    def read_recorded_spikes(self):
        # Loop through all current input vertices
        # and read spike times into dictionary
        spike_times = {}
        region = self.regions[Regions.spike_recording]
        for v in self.verts:
            region_mem = v.region_memory[Regions.spike_recording]
            spike_times.update(region.read_spike_times(v.post_neuron_slice,
                                                       region_mem))
        return spike_times

    def read_profile(self):
        # Get the profile recording region and
        region = self.regions[Regions.profiler]

        # Return profile data for each vertex that makes up population
        return [(v.post_neuron_slice.python_slice,
                 region.read_profile(v.region_memory[Regions.profiler],
                                     self.profiler_tag_names))
                for v in self.verts]

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------
    def _estimate_sdram(self, vertex_slice):
        # Begin with size of spike recording region
        sdram = self.regions[Regions.spike_recording].sizeof(vertex_slice);

        # Add on size of neuron region
        sdram += self.regions[Regions.neuron].sizeof(vertex_slice)

        # If profiler region exists, add its size
        if Regions.profiler in self.regions:
            sdram += self.regions[Regions.profiler].sizeof()

        return sdram

    def _get_region_arguments(self, post_vertex_slice, weights, out_buffers):
        region_arguments = defaultdict(Args)

        # Add vertex slice to regions that require it
        for r in (Regions.neuron,
                  Regions.output_weight,
                  Regions.spike_recording):
            region_arguments[r] = Args(post_vertex_slice)

        # Add kwargs for regions that require them
        region_arguments[Regions.system].kwargs["application_words"] =\
            [len(post_vertex_slice)]

        region_arguments[Regions.output_buffer].kwargs["out_buffers"] =\
            out_buffers

        region_arguments[Regions.output_weight].kwargs["weights"] = weights

        return region_arguments
