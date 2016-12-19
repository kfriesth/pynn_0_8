# Import modules
import logging
import math
import struct

# Import classes
from region import Region

logger = logging.getLogger("pynn_spinnaker")

# ------------------------------------------------------------------------------
# DelayBuffer
# ------------------------------------------------------------------------------
class DelayBuffer(Region):
    def __init__(self, sim_timestep_ms, max_delay_ms):
        # Calculate maximum delay in ticks and
        # round up to next largest power-of-two
        self.max_delay_ticks = int(math.ceil(max_delay_ms / sim_timestep_ms))
        self.max_delay_ticks = 2 ** (self.max_delay_ticks - 1).bit_length()

        # Convert simulation timestep to seconds and cache
        self.sim_timestep_s = float(sim_timestep_ms) / 1000.0


    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, sub_matrix_props):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------
        sub_matrix_props : list of :py:class:`._SubMatrix`
            Properties of the sub matrices to be written
            to synaptic matrix region

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # Calculate buffer size
        buffer_size = self._calc_buffer_size(sub_matrix_props)

        # One word for number of delay slots and buffer size,
        # followed by delay matrix
        return 8 + (4 * int(self.max_delay_ticks * buffer_size))

    def write_subregion_to_file(self, fp, sub_matrix_props):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        sub_matrix_props : list of :py:class:`._SubMatrix`
            Properties of the sub matrices to be written
            to synaptic matrix region
        """
        # Calculate buffer size
        buffer_size = self._calc_buffer_size(sub_matrix_props)

        # Write maximum delay ticks and buffer size
        fp.write(struct.pack("II", self.max_delay_ticks, buffer_size))

    def _calc_buffer_size(self, sub_matrix_props):
        # Calculate the total number of delay rows per second
        # this synapse processor is going to have to handle
        delay_rows_per_second = sum(s.max_delay_rows_per_second
                                    for s in sub_matrix_props)

        # Scale this to obtain the number of delay rows per second
        delay_rows_per_timestep = int(math.ceil(delay_rows_per_second *
                                                self.sim_timestep_s))
        logger.debug("\t\tDelay rows per-second:%f, per timestep:%u",
                     delay_rows_per_second, delay_rows_per_timestep)

        # Check that delay rows per timestep
        # can be indexed with an 8-bit counter
        if delay_rows_per_timestep >= 256:
            logger.warn("Maximum of 255 delay rows supported by timestep - %u estimated",
                        delay_rows_per_timestep)
            delay_rows_per_timestep = 255

        # Clamp this above 1 to ensure that memory is always allocated
        return max(1, delay_rows_per_timestep)