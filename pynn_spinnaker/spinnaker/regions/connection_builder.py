# Import modules
from .. import lazy_param_map
import itertools
import logging
import numpy as np
import struct

# Import classes
from ...random import NativeRNG
from pyNN.random import RandomDistribution
from region import Region

# Import functions
from rig.type_casts import float_to_fp
from zlib import crc32
from ..utils import is_scalar

logger = logging.getLogger("pynn_spinnaker")

def _crc_u32(value):
    return crc32(value) & 0xFFFFFFFF

def _get_param_type_name(param):
    # If parameter is a random distribution
    if isinstance(param.base_value, RandomDistribution):
        # Assert that it uses our native RNG
        assert isinstance(param.base_value.rng, NativeRNG)

        # Return distribution name
        return param.base_value.name
    # Otherwise if it's a scalar, return the magic string constant
    elif is_scalar(param.base_value):
        return "constant"
    # Otherwise assert
    else:
        assert False

def _get_param_size(param):
    # If parameter is a random distribution
    if isinstance(param.base_value, RandomDistribution):
        # Get RNG and distribution
        rng = param.base_value.rng
        distribution = param.base_value.name

        # Assert that it uses our native RNG
        assert isinstance(rng, NativeRNG)

        # Return distribution size
        return rng._get_dist_size(distribution)
    # Otherwise if it's a scalar, return 4 bytes
    elif is_scalar(param.base_value):
        return 4
    # Otherwise assert
    else:
        assert False

def _write_param(fp, param, fixed_point):
    # If parameter is randomly distributed
    if isinstance(param.base_value, RandomDistribution):
        # Get RNG and distribution
        rng = param.base_value.rng
        distribution = param.base_value.name
        parameters = param.base_value.parameters

        # Assert that it uses our native RNG
        assert isinstance(rng, NativeRNG)

        # Return distribution size
        rng._write_dist(fp, distribution, parameters, fixed_point)
    # Otherwise if it's a scalar, convert to fixed point and write
    elif is_scalar(param.base_value):
        convert = float_to_fp(signed=True, n_bits=32, n_frac=fixed_point)
        fixed_point = convert(param.base_value)
        fp.write(struct.pack("i", fixed_point))
    # Otherwise assert
    else:
        assert False

def _get_native_rngs(chip_sub_matrix_projs):
    # Chain together the native RNGs required for each projection
    rngs = itertools.chain.from_iterable(proj._native_rngs
                                         for proj, _ in chip_sub_matrix_projs)
    # Make RNG list unique and return
    return list(set(rngs))

# ------------------------------------------------------------------------------
# ConnectionBuilder
# ------------------------------------------------------------------------------
class ConnectionBuilder(Region):
    SeedWords = 4

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, post_vertex_slice, sub_matrix_props,
               chip_sub_matrix_projs, weight_fixed_point):
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

        Parameters
        ----------
        post_vertex_slice : :py:func:`slice`
            A slice object which indicates the slice of postsynaptic neurons.
        sub_matrix_props :
            A list of SubMatrix structures
            specifying matrices to generate on chip
        chip_sub_matrix_projs :
            A list of Projections from which to extract
            properties required to build matrices on chip
        """
        # Slice sub matrices to generate on chip out from end of sub matrix properties
        chip_sub_matrix_props = sub_matrix_props[-len(chip_sub_matrix_projs):]

        # Count number of RNGs
        num_rngs = len(_get_native_rngs(chip_sub_matrix_projs))
        assert num_rngs <= 1

        # Fixed size consists of seed for each RNG and connection count
        size = 4 + (self.SeedWords * 4)

        # Loop through projections
        for prop, proj in zip(chip_sub_matrix_props, chip_sub_matrix_projs):
            # Extract required properties from projections
            synapse_type = proj[0].synapse_type
            synaptic_matrix = synapse_type._synaptic_matrix_region_class
            connector = proj[0]._connector

            # Add words for key and type hashes to size
            size += (6 * 4)

            # Add size required for any synaptic matrix parameters
            size += lazy_param_map.size(synaptic_matrix.OnChipParamMap, 1)

            # Add size required to specify connector
            size += lazy_param_map.size(connector._on_chip_param_map, 1)

            # Add size required to specify delay and weight parameters
            size += _get_param_size(synapse_type.native_parameters["delay"])
            size += _get_param_size(synapse_type.native_parameters["weight"])

        # Return complete size
        return size

    def write_subregion_to_file(self, fp, post_vertex_slice, sub_matrix_props,
                                chip_sub_matrix_projs, weight_fixed_point):
        """Write a portion of the region to a file applying the formatter.

        Parameters
        ----------
        fp : file-like object
            The file-like object to which data from the region will be written.
            This must support a `write` method.
        post_vertex_slice : :py:func:`slice`
            A slice object which indicates the slice of postsynaptic neurons.
        sub_matrix_props :
            A list of SubMatrix structures
        chip_sub_matrix_projs :
            A list of Projections from which to extract
            properties required to build matrices on chip
        """
        # Count number of sub matrices to generate on chip
        num_chip_matrices = len(chip_sub_matrix_projs)

        # Slice these out from end of sub matrix properties
        chip_sub_matrix_props = sub_matrix_props[-num_chip_matrices:]

        # Get list of RNGs
        rngs = _get_native_rngs(chip_sub_matrix_projs)
        assert len(rngs) <= 1

        # Write seed
        seed = np.random.randint(0x7FFFFFFF,
                                 size=self.SeedWords).astype(np.uint32)
        fp.write(seed.tostring())

        # Write number of matrices
        fp.write(struct.pack("I", num_chip_matrices))

        # Loop through projections
        for prop, proj in zip(chip_sub_matrix_props, chip_sub_matrix_projs):
            # Extract required properties from projections
            synapse_type = proj[0].synapse_type
            synaptic_matrix = synapse_type._synaptic_matrix_region_class
            connector = proj[0]._connector

            delay = synapse_type.native_parameters["delay"]
            weight = synapse_type.native_parameters["weight"]

            logger.debug("\t\t\t\t\tWriting connection builder data for "
                "projection key:%08x, num rows:%u, matrix type:%s, "
                "connector type:%s, delay type:%s, weight type:%s, ",
                prop.key, proj[1], synaptic_matrix.__name__,
                connector.__class__.__name__, _get_param_type_name(delay),
                _get_param_type_name(weight))

            # Write header
            fp.write(struct.pack("6I", prop.key, proj[1],
                                 _crc_u32(synaptic_matrix.__name__),
                                 _crc_u32(connector.__class__.__name__),
                                 _crc_u32(_get_param_type_name(delay)),
                                 _crc_u32(_get_param_type_name(weight))))


            # Apply parameter map to synapse type parameters and write to region
            fp.write(lazy_param_map.apply_attributes(
                synapse_type, synaptic_matrix.OnChipParamMap).tostring())

            # Apply parameter map to connector parameters and write to region
            fp.write(lazy_param_map.apply_attributes(
                connector, connector._on_chip_param_map).tostring())

            # Write delay parameter with fixed point of zero to round to nearest timestep
            _write_param(fp, delay, 0)

            # Write weights using weight fixed point
            _write_param(fp, weight, weight_fixed_point)