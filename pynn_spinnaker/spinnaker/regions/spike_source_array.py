# Import modules
import logging
import numpy as np
import struct

# Import classes
from region import Region

# Import functions
from ..utils import calc_slice_bitfield_words

logger = logging.getLogger("pynn_spinnaker")


# ----------------------------------------------------------------------------
# SpikeSourceArray
# ----------------------------------------------------------------------------
class SpikeSourceArray(Region):
    def __init__(self, cell_type, parameters, initial_values,
                 sim_timestep_ms, pop_size):
        # Cache spike times and timestep
        self.spike_times = parameters["spike_times"]
        self.sim_timestep_ms = sim_timestep_ms
        
        # Set shape of spike times
        self.spike_times.shape = (pop_size,)

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, vertex_slice):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------
        vertex_slice : :py:func:`slice`
            A slice object which indicates which rows, columns or other
            elements of the region should be included.

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # Calculate the size of each spike block
        # (1 extra word for next spike time)
        spike_block_bytes = (calc_slice_bitfield_words(vertex_slice) + 1) * 4

        # Find the largest number of spikes in any of the neurons in the slice
        slice_spike_times = self.spike_times[vertex_slice.python_slice]
        max_spike_blocks = max(s.max() if len(s.value) > 0 else 0
                               for s in slice_spike_times)

        # Total size is a single word to specify time of first spike
        # block and maximum number of spike blocks required
        return 4 + int(max_spike_blocks * spike_block_bytes)

    def write_subregion_to_file(self, fp, vertex_slice):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        vertex_slice : :py:func:`slice`
            A slice object which indicates which rows, columns or other
            elements of the region should be included.
        """
        # Get slice of spike times for this vertex
        slice_spike_times = self.spike_times[vertex_slice.python_slice]

        # Determine time of last spike and convert to ms
        max_time_ms = max(s.max() if len(s.value) > 0 else 0
                          for s in slice_spike_times)
        max_timestep = int(round(max_time_ms / self.sim_timestep_ms))

        # Build mask array to hold spikes
        vertex_words = calc_slice_bitfield_words(vertex_slice)
        spike_vector = np.zeros((max_timestep + 1,
                                 vertex_words * 32), dtype=np.uint8)

        # Loop through neurons
        num_multiple_spikes = 0
        for i, s in enumerate(slice_spike_times):
            # Round spiketimes to nearest timestep and
            # bin to create spike vector
            n_spike_vector = np.bincount(
                np.rint(s.value / self.sim_timestep_ms).astype(int))

            # Add count of timesteps in which multiple
            # spikes are specified to total
            num_multiple_spikes += (n_spike_vector > 1).sum()

            # Copy bit vector into mask array
            spike_vector[0:len(n_spike_vector), i] = (n_spike_vector > 0)

        if num_multiple_spikes > 0:
            logger.warn("Spike source array spike data contains %u "
                        "%f ms timesteps, in which there are multiple spikes. "
                        "These will each be treated as single spikes",
                        num_multiple_spikes, self.sim_timestep_ms)

        # Determine which columns have any spike blocks associated
        timestep_mask = np.any(spike_vector, axis=1)

        # Get timesteps in which there are spike blocks
        timesteps = np.where(timestep_mask)[0]

        # If there are no spike blocks, write magic number to signify
        if len(timesteps) == 0:
            fp.write(struct.pack("I", 0xFFFFFFFF))
        # Otherwise
        else:
            spike_vector = spike_vector[timestep_mask, :]

            # Seperate out first timestep from the next timesteps
            # to be  written with each spike block
            first_timestep = timesteps[0]
            next_timesteps = np.append(timesteps[1:], 0xFFFFFFFF)
            logger.debug("\t\t\tFirst timestep %u", first_timestep)

            # Reverse bit order within each word
            spike_vector = np.fliplr(spike_vector.reshape(-1, 32))

            # Pack into bytes and swap word endianess
            spike_vector = np.packbits(spike_vector, axis=1)
            spike_vector = spike_vector.view(dtype=np.uint32).byteswap()

            # Reshape spike vectors into rows for each timestep
            spike_vector = spike_vector.reshape(-1, vertex_words)

            # Stack spike vector next to next timesteps to form spike blocks
            # **NOTE** cast to uint32 as np.where dumps out int64
            spike_blocks = np.hstack((next_timesteps[:, np.newaxis],
                                    spike_vector)).astype(np.uint32)

            # Write first timestep followed by spike blocks
            fp.write(struct.pack("I", first_timestep))
            fp.write(spike_blocks.tostring())
