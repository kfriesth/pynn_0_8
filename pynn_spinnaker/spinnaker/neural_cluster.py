# Import modules
import enum
import itertools
import logging
import regions
from rig import machine

# Import classes
from collections import defaultdict
from pacman.model.graphs.machine.impl.machine_vertex \
    import MachineVertex
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
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
    flush = 5
    intrinsic_plasticity = 6
    spike_recording = 7
    analogue_recording_0 = 8
    analogue_recording_1 = 9
    analogue_recording_2 = 10
    analogue_recording_3 = 11
    analogue_recording_start = analogue_recording_0
    analogue_recording_end = analogue_recording_3 + 1
    profiler = analogue_recording_end


# ----------------------------------------------------------------------------
# Vertex
# ----------------------------------------------------------------------------
class Vertex(MachineVertex, AbstractHasAssociatedBinary):
    def __init__(self, parent_keyspace, neuron_slice, pop_index, vert_index,
                 sdram, app_name):
        self.neuron_slice = neuron_slice

        # Build child keyspaces for spike and
        # flush packets coming from this vertex
        self.spike_keyspace = parent_keyspace(pop_index=pop_index,
                                              vert_index=vert_index,
                                              flush=0)
        self.flush_keyspace = parent_keyspace(pop_index=pop_index,
                                              vert_index=vert_index,
                                              flush=1)

        self.input_verts = []
        self.back_prop_out_buffers = None
        self.region_memory = None
        self.app_name = app_name

        # Superclass
        # **NOTE** as vertex partitioning is already done,
        # only SDRAM is required for subsequent placing decisions
        MachineVertex.__init__(
            self, label="<neuron slice:%s>" % (str(self.neuron_slice)
            resources_required=ResourceContainer(sdram=SDRAMResource(sdram)))

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
    def spike_tx_key(self):
        return self.spike_keyspace.get_value(tag="transmission")

    @property
    def flush_tx_key(self):
        return self.flush_keyspace.get_value(tag="transmission")

    @property
    def routing_key(self):
        # Check that routing key for the spike and flush keyspace are the same
        spike_key = self.spike_keyspace.get_value(tag="routing")
        flush_key = self.flush_keyspace.get_value(tag="routing")
        assert spike_key == flush_key

        # Return the spike key (arbitarily)
        return spike_key

    @property
    def routing_mask(self):
        # Check that routing mask for the spike and flush keyspace are the same
        spike_mask = self.spike_keyspace.get_mask(tag="routing")
        flush_mask = self.flush_keyspace.get_mask(tag="routing")
        assert spike_mask == flush_mask

        # Return the spike mask (arbitarily)
        return spike_mask


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
                 record_sample_interval, indices_to_record, config,
                 frontend, keyspace, post_synaptic_width,
                 requires_back_prop, pop_size):
        # Create standard regions
        self.regions = {}
        self.regions[Regions.system] = regions.System(
            timer_period_us, sim_ticks)
        self.regions[Regions.neuron] = cell_type._neuron_region_class(
            cell_type, parameters, initial_values, sim_timestep_ms, pop_size)
        self.regions[Regions.back_prop_output] = regions.SDRAMBackPropOutput(
            requires_back_prop)
        self.regions[Regions.flush] = regions.Flush(config.flush_time,
                                                    sim_timestep_ms)

        self.regions[Regions.spike_recording] = regions.SpikeRecording(
            indices_to_record, sim_timestep_ms, sim_ticks)

        # If cell type has any receptors i.e. any need for synaptic input
        if len(cell_type.receptor_types) > 0:
            # Add a synapse region and an input buffer
            self.regions[Regions.synapse] = regions.ParameterSpace(
                cell_type._synapse_mutable_param_map,
                cell_type._synapse_immutable_param_map,
                parameters, initial_values, pop_size,
                sim_timestep_ms=sim_timestep_ms)

            self.regions[Regions.input_buffer] = regions.InputBuffer()

         # If cell type has an intrinsic plasticity parameter map
        if hasattr(cell_type, "intrinsic_plasticity_param_map"):
            self.regions[Regions.intrinsic_plasticity] =\
                regions.HomogeneousParameterSpace(
                    cell_type._intrinsic_plasticity_param_map,
                    parameters,
                    sim_timestep_ms)

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
                                          record_sample_interval,
                                          sim_timestep_ms, sim_ticks)

        # Add profiler region if required
        if config.num_profile_samples is not None:
            self.regions[Regions.profiler] =\
                regions.Profiler(config.num_profile_samples)

        # Split population slice
        neuron_slices = split_slice(pop_size, post_synaptic_width)

        # Get neuron executable name
        neuron_app = get_model_executable_filename(
            "neuron_", cell_type, config.num_profile_samples is not None)

        logger.debug("\t\tNeuron application:%s", neuron_app)
        logger.debug("\t\t%u neuron vertices", len(neuron_slices))


        # Build neuron vertices for each slice,
        # allocating a keyspace for each vertex
        self.verts = []
        for vert_id, neuron_slice in enumerate(neuron_slices):
            # Create vertex
            vert = Vertex(keyspace, neuron_slice, pop_id, vert_id,
                          self._estimate_sdram(v.neuron_slice), neuron_app)

            # Add to frontend and verts list
            frontend.add_machine_vertex(vert)
            self.verts.append(vert)

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

            logger.debug("\t\t\tVertex %s (%u, %u, %u): Spike key:%08x, Flush key:%08x",
                            v, vertex_placement[0], vertex_placement[1],
                            core.start, v.spike_tx_key, v.flush_tx_key)

            # Select placed chip
            with machine_controller(x=vertex_placement[0],
                                    y=vertex_placement[1]):
                # Get the input buffers from each synapse vertex
                in_buffers = [
                    s.get_in_buffer(v.neuron_slice)
                    for s in v.input_verts]

                # Get regiona arguments
                region_arguments = self._get_region_arguments(
                    v.spike_tx_key, v.flush_tx_key, v.neuron_slice,
                    in_buffers, v.back_prop_out_buffers)

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
    def _estimate_sdram(self, vertex_slice):
        # Begin with size of spike recording region
        sdram = self.regions[Regions.spike_recording].sizeof(vertex_slice);

        # Add on size of neuron region
        sdram += self.regions[Regions.neuron].sizeof(vertex_slice)
        
        # If profiler region exists, add its size
        if Regions.profiler in self.regions:
            sdram += self.regions[Regions.profiler].sizeof()

        # Loop through possible analogue recording regions
        for t in range(Regions.analogue_recording_start,
                       Regions.analogue_recording_end):
            # If region exists, add its size to total
            if Regions(t) in self.regions:
                sdram += self.regions[Regions(t)].sizeof(vertex_slice)

        return sdram

    def _get_region_arguments(self, spike_tx_key, flush_tx_key, vertex_slice,
                              in_buffers, back_prop_out_buffers):
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
            [spike_tx_key, flush_tx_key, len(vertex_slice)]
        region_arguments[Regions.input_buffer].kwargs["in_buffers"] =\
            in_buffers
        region_arguments[Regions.back_prop_output].kwargs["out_buffers"] =\
            back_prop_out_buffers
        return region_arguments
