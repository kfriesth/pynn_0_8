import struct
from region import Region

RECORD_SPIKE_HISTORY  = (1 << 0)
RECORD_VOLTAGE        = (1 << 1)
RECORD_GSYN           = (1 << 2)

#------------------------------------------------------------------------------
# SystemRegion
#------------------------------------------------------------------------------
class SystemRegion(Region):
    def __init__(self, timer_period_us, simulation_ticks):
        """Create a new system region.

        Parameters
        ----------
        timer_period_us : int
            period of hardware timer in microseconds
        duration_timestep : int
            Length of simulation in terms of timer ticks
        """
        self.timer_period_us = timer_period_us
        self.simulation_ticks = simulation_ticks
    
    #--------------------------------------------------------------------------
    # Region methods
    #--------------------------------------------------------------------------
    def sizeof(self, vertex_slice, **kwargs):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------
        vertex_slice : :py:func:`slice`
            A slice object which indicates which rows, columns or other
            elements of the region should be included.
        kwargs : optional
            Arguments which will be passed to the (optional) formatter along
            with each value that is being written.
            
        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # Extract application words from formatter
        application_words = kwargs["application_words"]

        return 4 * (2 + len(application_words))

    def write_subregion_to_file(self, fp, vertex_slice, **kwargs):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        vertex_slice : :py:func:`slice`
            A slice object which indicates which rows, columns or other
            elements of the region should be included.
        kwargs : optional
            Arguments which will be passed to the (optional) formatter along
            with each value that is being written.
        """
        # Extract application words from formatter
        application_words = kwargs["application_words"]

        # Write structure
        fp.write(struct.pack("%uI" % (2 + len(application_words)),
            self.timer_period_us,
            self.simulation_ticks,
            *application_words))