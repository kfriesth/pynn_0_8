# Import modules
import enum
import itertools
import logging
import regions
from rig import machine

# Import classes
from collections import defaultdict
from utils import Args

# Import functions
from six import iteritems
from utils import (calc_bitfield_words, calc_slice_bitfield_words,
                   get_model_executable_filename, load_regions, split_slice)

logger = logging.getLogger("pynn_spinnaker")


# ----------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------
class Regions(enum.IntEnum):
    """Region names, corresponding to those defined in `ensemble.h`"""
    system = 0
    neuron = 1
    synapse = 2
    input_buffer = 3
    back_prop_output = 4
    spike_recording = 5
    analogue_recording_0 = 6
    analogue_recording_1 = 7
    analogue_recording_2 = 8
    analogue_recording_3 = 9
    analogue_recording_start = analogue_recording_0
    analogue_recording_end = analogue_recording_3 + 1
    profiler = analogue_recording_end


# ----------------------------------------------------------------------------
# Vertex
# ----------------------------------------------------------------------------
class Vertex(object):
    def __init__(self, parent_keyspace, neuron_slice,
                 population_index, vertex_index):
        self.neuron_slice = neuron_slice
        self.keyspace = parent_keyspace(population_index=population_index,
                                        vertex_index=vertex_index)
        self.input_verts = []
        self.back_prop_out_buffers = None

        self.region_memory = None

    # ------------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------------
    def __str__(self):
        return "<neuron slice:%s>" % (str(self.neuron_slice))

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def get_back_prop_in_buffer(self, post_slice):
        # Check the slices involved overlap and that this
        # neuron vertex actually has back propagation buffers
        assert post_slice.overlaps(self.neuron_slice)
        assert self.back_prop_out_buffers is not None

        # Calculate start and end bit in neuron id-space
        neuron_start_bit = max(post_slice.start, self.neuron_slice.start)
        neuron_end_bit = min(post_slice.stop, self.neuron_slice.stop)
        logger.debug("\t\t\tNeuron start bit:%u, Neuron end bit:%u",
                     neuron_start_bit, neuron_end_bit)

        # Calculate where in the buffer post_slice starts
        buffer_start_bit = neuron_start_bit - self.neuron_slice.start
        assert buffer_start_bit >= 0

        # Seperate where the buffer starts in words and bits
        buffer_start_word = buffer_start_bit // 32
        buffer_start_bit -= (buffer_start_word * 32)
        buffer_end_bit = (neuron_end_bit - neuron_start_bit) + buffer_start_bit
        buffer_num_words = calc_bitfield_words(buffer_end_bit)
        logger.debug("\t\t\tBuffer start word:%u, Buffer start bit:%u, Buffer end bit:%u, Buffer num words:%u",
                     buffer_start_word, buffer_start_word,
                     buffer_end_bit, buffer_num_words)

        # Return offset pointers into out buffers
        return (
            [b + (buffer_start_word * 4) for b in self.back_prop_out_buffers],
            buffer_num_words, buffer_start_bit, buffer_end_bit)

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def key(self):
        return self.keyspace.get_value(tag="routing")

    @property
    def mask(self):
        return self.keyspace.get_mask(tag="routing")


# -----------------------------------------------------------------------------
# NeuralCluster
# -----------------------------------------------------------------------------
class NeuralCluster(object):
    # Tag names, corresponding to those defined in neuron_processor.h
    profiler_tag_names = {
        0:  "Synapse shape",
        1:  "Update neurons",
        2:  "Apply buffer",
    }

    def __init__(self, pop_id, cell_type, parameters, initial_values,
                 sim_timestep_ms, timer_period_us, sim_ticks,
                 indices_to_record, config, vertex_applications,
                 vertex_resources, keyspace, post_synaptic_width,
                 requires_back_prop):
        # Create standard regions
        self.regions = {}
        self.regions[Regions.system] = regions.System(
            timer_period_us, sim_ticks)
        self.regions[Regions.neuron] = cell_type.neuron_region_class(
            cell_type, parameters, initial_values, sim_timestep_ms)
        self.regions[Regions.spike_recording] = regions.SpikeRecording(
            indices_to_record, sim_timestep_ms, sim_ticks)
        self.regions[Regions.back_prop_output] = regions.SDRAMBackPropOutput(
            requires_back_prop)

        # If cell type has any receptors i.e. any need for synaptic input
        if len(cell_type.receptor_types) > 0:
            # Add a synapse region and an input buffer
            self.regions[Regions.synapse] = cell_type.synapse_region_class(
                cell_type, parameters, initial_values, sim_timestep_ms)

            self.regions[Regions.input_buffer] = regions.InputBuffer()

        # Assert that there are sufficient analogue
        # recording regions for this celltype's needs
        num_analogue_rec_regions = Regions.analogue_recording_end -\
            Regions.analogue_recording_start
        assert num_analogue_rec_regions >= (len(cell_type.recordable) - 1)

        # Loop through cell's non-spike recordables
        # and create analogue recording regions
        # **HACK** this assumes the first entry is spike
        for i, v in enumerate(cell_type.recordable[1:]):
            self.regions[Regions(Regions.analogue_recording_start + i)] =\
                regions.AnalogueRecording(indices_to_record, v,
                                          sim_timestep_ms, sim_ticks)

        # Add profiler region if required
        if config.num_profile_samples is not None:
            self.regions[Regions.profiler] =\
                regions.Profiler(config.num_profile_samples)

        # Split population slice
        neuron_slices = split_slice(parameters.shape[0], post_synaptic_width)

        # Build neuron vertices for each slice,
        # allocating a keyspace for each vertex
        self.verts = [Vertex(keyspace, neuron_slice, pop_id, vert_id)
                      for vert_id, neuron_slice in enumerate(neuron_slices)]

        # Get neuron executable name
        neuron_app = get_model_executable_filename(
            "neuron_", cell_type, config.num_profile_samples is not None)

        logger.debug("\t\tNeuron application:%s", neuron_app)
        logger.debug("\t\t%u neuron vertices", len(self.verts))

        # Loop through neuron vertices and their corresponding resources
        for v in self.verts:
            # Add application to dictionary
            vertex_applications[v] = neuron_app

            # Add resources to dictionary
            # **TODO** add SDRAM
            vertex_resources[v] = {machine.Cores: 1}

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------
    def allocate_out_buffers(self, placements, allocations,
                             machine_controller):
        # Loop through vertices
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
                # If back propagation is enabled, allocate two back
                # propagation out buffers for this neuron vertex
                if self.regions[Regions.back_prop_output].enabled:
                    back_prop_buffer_bytes =\
                        calc_slice_bitfield_words(v.neuron_slice) * 4
                    v.back_prop_out_buffers = [
                        machine_controller.sdram_alloc(back_prop_buffer_bytes,
                                                       clear=True)
                        for _ in range(2)]

    def load(self, placements, allocations, machine_controller):
        # Loop through vertices
        for v in self.verts:
            # Get placement and allocation
            vertex_placement = placements[v]
            vertex_allocation = allocations[v]

            # Get core this vertex should be run on
            core = vertex_allocation[machine.Cores]
            assert (core.stop - core.start) == 1

            logger.debug("\t\tVertex %s (%u, %u, %u): Key:%08x",
                            v, vertex_placement[0], vertex_placement[1],
                            core.start, v.key)

            # Select placed chip
            with machine_controller(x=vertex_placement[0],
                                    y=vertex_placement[1]):
                # Get the input buffers from each synapse vertex
                in_buffers = [
                    (s.get_in_buffer(v.neuron_slice), s.receptor_index,
                        s.weight_fixed_point)
                    for s in v.input_verts]

                # Get regiona arguments
                region_arguments = self._get_region_arguments(
                    v.key, v.neuron_slice, in_buffers,
                    v.back_prop_out_buffers)

                # Load regions
                v.region_memory = load_regions(self.regions, region_arguments,
                                               machine_controller, core)

    def read_recorded_spikes(self):
        # Loop through all neuron vertices and read spike times into dictionary
        spike_times = {}
        region = self.regions[Regions.spike_recording]
        for v in self.verts:
            region_mem = v.region_memory[Regions.spike_recording]
            spike_times.update(region.read_spike_times(v.neuron_slice,
                                                       region_mem))
        return spike_times

    def read_recorded_signal(self, channel):
        # Get index of channelread_profile
        region_index = Regions(Regions.analogue_recording_start + channel)
        region = self.regions[region_index]

        # Loop through all neuron vertices and read signal
        signal = {}
        for v in self.verts:
            region_mem = v.region_memory[region_index]
            signal.update(region.read_signal(v.neuron_slice, region_mem))

        return signal

    def read_profile(self):
        # Get the profile recording region and
        region = self.regions[Regions.profiler]

        # Return profile data for each vertex that makes up population
        return [(v.neuron_slice.python_slice,
                 region.read_profile(v.region_memory[Regions.profiler],
                                     self.profiler_tag_names))
                for v in self.verts]

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------
    def _get_region_arguments(self, key, vertex_slice, in_buffers,
                              back_prop_out_buffers):
        region_arguments = defaultdict(Args)

        analogue_recording_regions = range(Regions.analogue_recording_start,
                                           Regions.analogue_recording_end)
        # Add vertex slice to regions that require it
        for r in itertools.chain((Regions.neuron,
                                  Regions.synapse,
                                  Regions.spike_recording),
                                 analogue_recording_regions):
            region_arguments[r] = Args(vertex_slice)

        # Add kwargs for regions that require them
        region_arguments[Regions.system].kwargs["application_words"] =\
            [key, len(vertex_slice)]
        region_arguments[Regions.input_buffer].kwargs["in_buffers"] =\
            in_buffers
        region_arguments[Regions.back_prop_output].kwargs["out_buffers"] =\
            back_prop_out_buffers
        return region_arguments
