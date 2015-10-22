# Import modules
import numpy as np
import struct

# Import classes
from collections import namedtuple
from region import Region

# Import functions
from bisect import bisect_left

# **YUCK** duplication
ROW_HEADER_BYTES = 3 * 4

# Master population table type-specific context generated by calc_sub_matrix_placement
# And passed via formatter to write_subregion_to_file to determine sub-matrix placement
MatrixPlacement = namedtuple("MatrixPlacement", ["padding_bytes", "offset_kb", "row_size_index"])

#------------------------------------------------------------------------------
# MasterPopulationArrayRegion
#------------------------------------------------------------------------------
class MasterPopulationArrayRegion(Region):
    def __init__(self, num_entries = 1152):
        self.num_entries = num_entries
        
    #--------------------------------------------------------------------------
    # Region methods
    #--------------------------------------------------------------------------
    def sizeof(self, vertex_slice, **formatter_args):
        """Get the size requirements of the region in bytes.

        Parameters
        ----------
        vertex_slice : :py:func:`slice`
            A slice object which indicates which rows, columns or other
            elements of the region should be included.
        formatter_args : optional
            Arguments which will be passed to the (optional) formatter along
            with each value that is being written.

        Returns
        -------
        int
            The number of bytes required to store the data in the given slice
            of the region.
        """
        
        # Each entry is a halfword
        return (self.num_entries * 2)

    def write_subregion_to_file(self, vertex_slice, fp, **formatter_args):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        vertex_slice : :py:func:`slice`
            A slice object which indicates which rows, columns or other
            elements of the region should be included.
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        formatter_args : optional
            Arguments which will be passed to the (optional) formatter along
            with each value that is being written.
        """
        weight_scale = formatter_args["weight_scale"]
    
    #--------------------------------------------------------------------------
    # Public methods
    #--------------------------------------------------------------------------
    def calc_sub_matrix_placement(self, matrices, vertex_slice, row_sizes):
        # Loop through matrices
        matrix_placement = {}
        next_offset_kb = 0
        for p, m in matrices.iteritems():
            # Get slice of matrices for vertex
            sub_mask = m[:,vertex_slice]["mask"]
            
            # Get maximum row length and convert to size
            # **YUCK** duplication
            max_row_length = max(sub_mask.sum(1))
            max_row_size = ROW_HEADER_BYTES + (4 * max_row_length)
            
            # Find suitable row size from table
            row_size_index = bisect_left(row_sizes, max_row_size)
            table_row_size = row_sizes[row_size_index]
            
            # Multiply by number of rows to get matrix size
            sub_matrix_bytes = table_row_size * sub_mask.shape[1]
            
            # If number of bytes is not kilobyte aligned, calculate padding
            padding_bytes = 0
            if (sub_matrix_bytes & 1023) != 0:
                padding_bytes = 1024 - (sub_matrix_bytes & 1023)
            
            # Add placement to dictionary
            matrix_placement[p] = MatrixPlacement(padding_bytes=padding_bytes, 
                offset_kb=next_offset_kb, row_size_index=row_size_index)
            
            # Update next offset
            padded_submatrix_bytes = padding_bytes + sub_matrix_bytes
            assert (padded_submatrix_bytes & 1023) == 0, "Padding calculation failed"
            next_offset_kb += (padded_submatrix_bytes / 1024)
        
        return matrix_placement;
                