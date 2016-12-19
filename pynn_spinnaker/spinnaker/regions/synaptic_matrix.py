# Import modules
import itertools
import logging
import numpy as np

# Import classes
from collections import namedtuple
from region import Region
from rig.type_casts import NumpyFloatToFixConverter, NumpyFixToFloatConverter

# Import functions
from six import iteritems
from ..utils import combine_row_offset_length, extract_row_offset_length

logger = logging.getLogger("pynn_spinnaker")

SubMatrix = namedtuple("SubMatrix", ["key", "mask", "pre_n_slice", "pre_slice_index",
                                     "size_words", "max_cols",
                                     "max_delay_rows_per_second"])


# ------------------------------------------------------------------------------
# SynapticMatrix
# ------------------------------------------------------------------------------
class SynapticMatrix(Region):
    # Number of bits for various synapse components
    IndexBits = 10
    DelayBits = 3

    # How many bits are used to represent (extension) row length
    LengthBits = 10

    # 3 header words :
    # > num synapses
    # > next delay row time
    # > next delay offset-length
    NumHeaderWords = 3

    def __init__(self, synapse_type):
        self.signed_weight = synapse_type._signed_weight
        self.max_dtcm_delay_slots = synapse_type._max_dtcm_delay_slots

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, sub_matrix_props, host_sub_matrix_rows, matrix_placements,
               weight_fixed_point):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------haha when
        sub_matrix_props : list of :py:class:`._SubMatrix`
            Properties of the sub matrices to be written
            to synaptic matrix region
        host_sub_matrix_rows : list of list of numpy arrays
            Partitioned matrix rows generated on host to be written to SpiNNaker
        matrix_placements : list of integers
            Offsets in words at which sub_matrices will be
            written into synaptic matrix region

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        # Get the offset of last matrix, add its size and convert to bytes
        # **NOTE** assumes placement is monotonic
        if len(matrix_placements) == 0:
            return 0
        else:
            return 4 * (matrix_placements[-1] + sub_matrix_props[-1].size_words)

    def write_subregion_to_file(self, fp, sub_matrix_props, host_sub_matrix_rows,
                                matrix_placements, weight_fixed_point):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        sub_matrix_props : list of :py:class:`._SubMatrix`
            Properties of the sub matrices to be written
            to synaptic matrix region
        host_sub_matrix_rows : list of list of numpy arrays
            Partitioned matrix rows generated on host to be written to SpiNNaker
        matrix_placements : list of integers
            Offsets in words at which sub_matrices will be
            written into synaptic matrix region
        """
        # Create a converter to convert floating
        # point weights to correct format
        float_to_weight = NumpyFloatToFixConverter(
            self.signed_weight, self.FixedPointWeightBits, weight_fixed_point)

        # Loop through sub matrices
        # **NOTE** because only the matrices generated
        # on host are included, in sub_matrix_rows, this
        # loop will not loop over the matrices to generate on chip
        for matrix, matrix_rows, placement in zip(sub_matrix_props,
                                                  host_sub_matrix_rows,
                                                  matrix_placements):
            # Seek to the absolute offset for this matrix
            # **NOTE** placement is in WORDS
            fp.seek(placement * 4, 0)

            # Build matrix large enough for entire ragged matrix
            num_row_words = self._get_num_row_words(matrix.max_cols)
            num_matrix_words = len(matrix_rows) * num_row_words
            matrix_words = np.empty((len(matrix_rows), num_row_words),
                                    dtype=np.uint32)

            # Calculate the number of extension words required and build
            # Second numpy array to contain concatenated extension rows
            num_ext_words = matrix.size_words - num_matrix_words
            ext_words = np.empty(num_ext_words, dtype=np.uint32)

            logger.debug("\t\t\t\t\tWriting matrix placement:%u, max cols:%u, "
                         "matrix words:%u, num extension words:%u, num rows:%u",
                         placement, matrix.max_cols, num_matrix_words,
                         matrix.size_words - num_matrix_words, len(matrix_rows))

            # Loop through matrix rows
            next_row_offset = 0
            for i, row in enumerate(matrix_rows):
                # Write base row to matrix
                next_row = None if len(row) == 1 else row[1]
                self._write_row(row[0], next_row,
                                placement + next_row_offset + num_matrix_words,
                                float_to_weight, matrix_words[i])

                # Loop through extension rows
                for r, ext_row in enumerate(row[1:], start=1):
                    num_ext_row_words = self._get_num_row_words(len(ext_row[1]))
                    next_row = None if len(row) == (r + 1) else row[r + 1]
                    self._write_row(
                        ext_row, next_row,
                        placement + next_row_offset + num_ext_row_words + num_matrix_words,
                        float_to_weight, ext_words[next_row_offset:])

                    next_row_offset += num_ext_row_words

            # Write matrix followed by extension words
            fp.write(matrix_words.tostring())
            fp.write(ext_words.tostring())

    # --------------------------------------------------------------------------
    # Public methods
    # --------------------------------------------------------------------------
    def estimate_matrix_words(self, num_pre, max_cols, max_sub_rows,
                              max_total_sub_row_length):
        # Because of the number of bits available for storing
        # max_cols, rows with 0 synapses cannot be represented
        max_cols = max(1, max_cols)

        # Calculate the 'width' in words of
        # the ragged portion of the synaptic matrix
        ragged_words = self._get_num_row_words(max_cols)

        # Estimate the number of words requires for the sub-rows
        ext_words = self._estimate_num_ext_words(max_sub_rows,
                                                 max_total_sub_row_length)

        # Calculate final size
        return num_pre * (ragged_words + ext_words)

    def partition_matrices(self, routing_info, post_vertex_slice,
                           pre_pop_sub_rows, incoming_connections):
        # Loop through all incoming connections
        sub_matrix_props = []
        sub_matrix_rows = []

        # Loop through presynaptic population
        for pre_pop, pre_n_verts in iteritems(incoming_connections):
            # If no rows have been generated on the
            # host for this connection, skip
            if not pre_pop in pre_pop_sub_rows:
                continue

            # Get the list of sub-rows associated
            # with this presynaptic population
            sub_rows = pre_pop_sub_rows[pre_pop]

            # Loop through presynaptic vertices
            for pre_n_vert in pre_n_verts:
                # Slice out the sub-rows asssociated
                # with this presynaptic neuron vert
                vert_sub_rows = sub_rows[pre_n_vert.neuron_slice.python_slice]

                max_sub_rows = 0
                max_cols = 1
                num_ext_words = 0
                any_connections = False
                for i, sub_row in enumerate(vert_sub_rows):
                    # If sub-row has any elements
                    if len(sub_row) != 0:
                        # Make indices relative to vertex start
                        sub_row["index"] -= post_vertex_slice.start

                        # There is at least one connection so
                        # this matrix needs to be placed
                        any_connections = True

                        # Determine which delay slot each sub-rob entry is in
                        sub_row_delay_slot = (sub_row["delay"] - 1) / self.max_dtcm_delay_slots

                        # Sort sub-row by delay slot
                        sub_row_order = np.argsort(sub_row_delay_slot)
                        sub_row = sub_row[sub_row_order]
                        sub_row_delay_slot = sub_row_delay_slot[sub_row_order]

                        # Check that no zero delays were inserted
                        # This would result in a negative sub-row delay slot
                        assert sub_row_delay_slot[0] >= 0

                        # Take cumulative sum of the number of synapses
                        # in each delay slot to obtain sections of
                        # sub_row which belong in each delay slot
                        sub_row_lengths = np.bincount(sub_row_delay_slot)
                        sub_row_sections = np.cumsum(sub_row_lengths)

                        # Split sub-row into delay rows based
                        # on these sections, filtering out empty
                        # rows if they aren't the first row
                        vert_sub_rows[i] = [(e * self.max_dtcm_delay_slots, r)
                                    for e, r in enumerate(
                                        np.split(sub_row, sub_row_sections))
                                    if e == 0 or len(r) > 0]

                        # Update maximum number of sub-rows
                        max_sub_rows = max(max_sub_rows,
                                           len(vert_sub_rows[i]) - 1)

                        # Calculate number of extension words thos
                        num_ext_words += self._get_num_ext_words(
                            len(vert_sub_rows[i]), sub_row_lengths,
                            sub_row_sections)

                        # Update maximum number of columns based
                        # on length of first delay slot
                        max_cols = max(max_cols, sub_row_sections[0])
                    # Otherwise, add empty row
                    else:
                        vert_sub_rows[i] = [(0, ())]

                if any_connections:
                    # Calculate matrix size in words - size of square
                    # matrix added to number of extension words
                    size_words = num_ext_words +\
                        (len(vert_sub_rows) * self._get_num_row_words(max_cols))

                    # Estimate the maximum number of delay rows the
                    # synapse processor handling this sub-matrix
                    # will be required to process each second
                    max_delay_rows_per_second =\
                        (max_sub_rows * len(pre_n_vert.neuron_slice) *
                         pre_pop.spinnaker_config.mean_firing_rate)

                    # Add sub matrix to list
                    sub_matrix_props.append(
                        SubMatrix(pre_n_vert.get_routing_key(routing_info),
                                  pre_n_vert.get_routing_mask(routing_info),
                                  pre_n_vert.neuron_slice,
                                  pre_n_vert.vert_index,
                                  size_words, max_cols,
                                  max_delay_rows_per_second))

                    sub_matrix_rows.append(vert_sub_rows)

        return sub_matrix_props, sub_matrix_rows

    def partition_on_chip_matrix(self, routing_info, post_vertex_slice,
                                 pre_pop_on_chip_projection,
                                 incoming_connections):
        # Loop through all incoming connections
        sub_matrix_props = []
        sub_matrix_projs = []

        for pre_pop, pre_n_verts in iteritems(incoming_connections):
            # If connections from this populations
            # should be generated on host, skip
            if not pre_pop in pre_pop_on_chip_projection:
                continue

            # Get list of projections coming from
            # pre_pop which should be expanded on chip
            assert len(pre_pop_on_chip_projection[pre_pop]) == 1
            proj = pre_pop_on_chip_projection[pre_pop][0]

            # Loop through presynaptic vertices
            for pre_n_vert in pre_n_verts:
                # Estimate max dimensions of sub-matrix
                max_cols, max_sub_rows, max_total_sub_row_length =\
                    proj._estimate_max_dims(pre_n_vert.neuron_slice,
                                            post_vertex_slice)

                # If sub-matrix has any synapses
                if max_cols > 0 or max_sub_rows > 0:
                    # Estimate the maximum size of this in SDRAM
                    size_words = self.estimate_matrix_words(
                        len(pre_n_vert.neuron_slice), max_cols,
                        max_sub_rows, max_total_sub_row_length)

                    # Estimate the maximum number of delay rows the
                    # synapse processor handling this sub-matrix
                    # will be required to process each second
                    max_delay_rows_per_second =\
                        (max_sub_rows * len(pre_n_vert.neuron_slice) *
                         proj.pre.spinnaker_config.mean_firing_rate)

                    # Add sub matrix to list
                    sub_matrix_props.append(
                        SubMatrix(pre_n_vert.get_routing_key(routing_info),
                                  pre_n_vert.get_routing_mask(routing_info),
                                  pre_n_vert.neuron_slice,
                                  pre_n_vert.vert_index,
                                  size_words, max(1, max_cols),
                                  max_delay_rows_per_second))
                                  
                    sub_matrix_projs.append((proj, len(pre_n_vert.neuron_slice)))

        return sub_matrix_props, sub_matrix_projs

    def read_sub_matrix(self, pre_n_vert, post_s_vert, names, region_mem,
                        sim_timestep_ms, is_inhibitory, routing_info):
        # Find the matrix properties and placement of sub-matrix
        # associated with pre-synaptic neuron vertex
        vert_matrix_prop, vert_matrix_placement = next((
            (s, p) for s, p in zip(post_s_vert.sub_matrix_props,
                                   post_s_vert.matrix_placements)
            if s.key == pre_n_vert.get_routing_key(routing_info)),
            (None, None))
        assert vert_matrix_prop is not None
        assert vert_matrix_placement is not None

        # Calculate the size of the ragged matrix
        num_rows = len(pre_n_vert.neuron_slice)
        num_row_words = self._get_num_row_words(vert_matrix_prop.max_cols)
        num_matrix_words = (num_row_words * num_rows)

        logger.debug("\tReading matrix - max cols:%u, size words:%u, "
                     "num row words:%u, num matrix words:%u, num rows:%u",
                     vert_matrix_prop.max_cols, vert_matrix_prop.size_words,
                     num_row_words, num_matrix_words, num_rows)

        # Seek to the absolute offset for this matrix
        # **NOTE** placement is in WORDS
        region_mem.seek(vert_matrix_placement * 4, 0)

        # Read matrix from memory
        data = region_mem.read(vert_matrix_prop.size_words * 4)

        # Load into numpy
        data = np.frombuffer(data, dtype=np.uint32)

        # On this basis, create two views
        matrix_words = data[:num_matrix_words]
        ext_words = data[num_matrix_words:]

        # Reshape matrix words
        matrix_words = matrix_words.reshape((num_rows, -1))

        # Create a converter to convert fixed
        # point weights back to floating point
        weight_to_float = NumpyFixToFloatConverter(
            post_s_vert.weight_fixed_point)

        # Build data type for rows
        dtype = np.dtype(
            [(n, np.float64 if n == "weight" or n == "delay" else np.uint32)
             for n in names])
        logger.debug("\tUsing row dtype:%s, weight fixed point:%u",
                     dtype, post_s_vert.weight_fixed_point)

        # Downloaded weights should be negated
        # if they are unsigned and inhibitory
        negate_weights = (not self.signed_weight and is_inhibitory)

        # Loop through matrix rows
        synapses = []
        for i, r in enumerate(matrix_words):
            # Read row
            row = self._read_row(i, r, pre_n_vert.neuron_slice,
                                 post_s_vert.post_neuron_slice,
                                 weight_to_float, dtype, negate_weights)

            # If delays are required, scale into simulation timesteps
            if "delay" in dtype.names:
                row[3]["delay"] *= sim_timestep_ms

            # Extract synapses from row
            row_synapses = row[3]

            # While this row has more extension
            total_ext_delay = 0
            while row[0] != 0:
                # Add next row's delay to total
                total_ext_delay += row[0]

                # Make offset relative to start of extension words
                ext_row_start = row[1] - num_matrix_words - vert_matrix_placement
                assert ext_row_start >= 0

                # Convert extension row length to words
                ext_row_end = ext_row_start + self._get_num_row_words(row[2])

                # Create view of new extension row data
                ext_row_data = ext_words[ext_row_start:ext_row_end]

                # Read next extension row
                row = self._read_row(i, ext_row_data, pre_n_vert.neuron_slice,
                                    post_s_vert.post_neuron_slice,
                                    weight_to_float, dtype, negate_weights)

                # If delays are required, add total extended delay
                # and scale into simulation timesteps
                if "delay" in dtype.names:
                    row[3]["delay"] =\
                        (row[3]["delay"] + total_ext_delay) * sim_timestep_ms

                # Stack extension row synapses onto row
                row_synapses = np.hstack((row_synapses, row[3]))

            # Add complete row of synapses to list
            synapses.append(row_synapses)

        return synapses

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------
    def _read_row(self, pre_idx, row_words, pre_slice, post_slice,
                  weight_to_float, dtype, negate_weights):
        num_synapses = row_words[0]

        # Create empty array to hold synapses
        synapses = np.empty(num_synapses, dtype=dtype)

        # Extract the delay extension fields from row
        next_row_delay = row_words[1]
        next_row_offset, next_row_length = extract_row_offset_length(
            row_words[2], self.LengthBits)

        # If pre-synaptic indices are required, fill them in
        if "presynaptic_index" in dtype.names:
            synapses["presynaptic_index"][:] = pre_idx + pre_slice.start

        # Read synapses
        self._read_synapses(row_words[3:], weight_to_float, dtype, synapses)

        # If weights are required and we should negate them, do so
        if "weight" in dtype.names and negate_weights:
            synapses["weight"] = -synapses["weight"]

        # If post-synaptic indices are required,
        # add post-synaptic slice start to them
        if "postsynaptic_index" in dtype.names:
            synapses["postsynaptic_index"] += post_slice.start

        return next_row_delay, next_row_offset, next_row_length, synapses

    def _write_row(self, row, next_row, next_row_offset,
                   float_to_weight, destination):
        # Write actual length of row (in synapses)
        num_synapses = len(row[1])
        destination[0] = num_synapses

        # If there is no next row, write zeros to next two words
        if next_row is None:
            destination[1] = 0
            destination[2] = 0
        # Otherwise
        else:
            # Write relative delay of next_row from row
            destination[1] = (next_row[0] - row[0])

            # Write word containing the offset to the
            # next row and its length (in synapses)
            destination[2] = combine_row_offset_length(next_row_offset,
                                                       len(next_row[1]),
                                                       self.LengthBits)
        # If there are any synapses in row
        # **NOTE** empty rows may be tuples or empty
        # lists both of which break the following code
        if num_synapses > 0:
            # Extract the DTCM component of delay
            # **NOTE** subtract one so there is a minimum of 1 slot of delay
            dtcm_delay = 1 + ((row[1]["delay"] - 1) % self.max_dtcm_delay_slots)

            # Convert weight to fixed point, taking
            # absolute if weight is unsigned
            if self.signed_weight:
                weight_fixed = float_to_weight(row[1]["weight"])
            else:
                weight_fixed = float_to_weight(np.abs(row[1]["weight"]))

            # Write synapses
            num_row_words = self._get_num_row_words(num_synapses)
            self._write_synapses(dtcm_delay, weight_fixed,
                                 row[1]["index"],
                                 destination[3:num_row_words])
