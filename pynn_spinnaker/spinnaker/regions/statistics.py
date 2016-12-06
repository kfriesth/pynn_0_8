# Import modules
import itertools
import logging
import numpy as np
import struct


# Import classes
from region import Region

logger = logging.getLogger("pynn_spinnaker")

# ------------------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------------------
class Statistics(Region):
    def __init__(self, n_statistics):
        self.n_statistics = n_statistics

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self):
        """Get the size requirements of the region in bytes.

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # 1 word per sample
        return (4 * self.n_statistics)

    def write_subregion_to_file(self, fp):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        """
        pass

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def read_stats(self, vertex_region_memory, statistic_names):
        # Loop through list of statistic recording memory views
        stats = []
        for m in vertex_region_memory:
            # Seek to start
            m.seek(0)

            # Read statistics and add to list
            stats.append(struct.unpack("%uI" % self.n_statistics,
                                       m.read(self.n_statistics * 4)))
        # Convert stats to numpy array
        np_stats = np.asarray(stats)

        # Convert stats into record array
        stat_names = ",".join(statistic_names)
        stat_format = ",".join(
            itertools.repeat("u4", len(statistic_names)))
        return np.core.records.fromarrays(np_stats.T, names=stat_names,
                                          formats=stat_format)
