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

def _write_param(fp, param, fixed_point, scale, absolute):
    # If parameter is randomly distributed
    if isinstance(param.base_value, RandomDistribution):
        # Get RNG and distribution
        rng = param.base_value.rng
        distribution = param.base_value.name
        parameters = param.base_value.parameters

        # Assert that it uses our native RNG
        assert isinstance(rng, NativeRNG)

        # Return distribution size
        rng._write_dist(fp, distribution, parameters,
                        fixed_point, scale, absolute)
    # Otherwise if it's a scalar, convert to fixed point and write
    elif is_scalar(param.base_value):
        # Scale value and take absolute if required
        scaled_value = param.base_value * scale
        if absolute:
            scaled_value = abs(scaled_value)

        # Convert scaled value to fixed-point
        convert = float_to_fp(signed=True, n_bits=32, n_frac=fixed_point)
        fixed_point = convert(scaled_value)

        # Write to fp
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

    def __init__(self, sim_timestep_ms):
        self.sim_timestep_ms = sim_timestep_ms

    # --------------------------------------------------------------------------
    # Region methods
    # --------------------------------------------------------------------------
    def sizeof(self, post_vertex_slice, sub_matrix_props,
               chip_sub_matrix_projs, weight_fixed_point,
               post_slice_index, projection_state_dict):
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

        # Get list of RNGs
        native_rngs = _get_native_rngs(chip_sub_matrix_projs)
        assert len(native_rngs) <= 1

        # Fixed size consists of connection count, post slice index
        size = 8

        # Loop through projections
        for prop, proj in zip(chip_sub_matrix_props, chip_sub_matrix_projs):
            # Extract required properties from projections
            synapse_type = proj[0].synapse_type
            synaptic_matrix = synapse_type._synaptic_matrix_region_class
            connector = proj[0]._connector

            size += 4 # pre slice index

            # Add words for seed
            size += self.SeedWords * 4

            # Add words for key and type hashes to size
            size += (7 * 4)

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
                                chip_sub_matrix_projs, weight_fixed_point,
                                post_slice_index, projection_state_dict):
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

        # Write number of matrices, index of start of post_vertex_slice
        fp.write(struct.pack("II", num_chip_matrices, post_vertex_slice.start))

        # Loop through projections
        for prop, proj in zip(chip_sub_matrix_props, chip_sub_matrix_projs):

            # Write index of pre_slice
            fp.write(struct.pack("I", prop.pre_slice_index))

            # Generate four word base seed for SpiNNaker KISS RNG
            # Note, if we instantiate more than 54,000 KISS RNGs
            # with random seeds, it is probable that two will share
            # the same state for one of their sub-RNGs.
            if len(rngs):
                seed = rngs[0]._seed_generator.randint(
                2**32, dtype=np.uint32, size=self.SeedWords)
            else:
                seed = np.zeros(4, dtype=np.uint32)

            fp.write(seed.tostring())

            # Extract required properties from projections
            synapse_type = proj[0].synapse_type
            synaptic_matrix = synapse_type._synaptic_matrix_region_class
            connector = proj[0]._connector

            delay = synapse_type.native_parameters["delay"]
            weight = synapse_type.native_parameters["weight"]

            logger.debug("\t\t\t\t\tWriting connection builder data for "
                "projection key:%08x, num words:%u, num rows:%u, "
                "matrix type:%s, connector type:%s, delay type:%s, weight type:%s, ",
                prop.key, prop.size_words, proj[1], synaptic_matrix.__name__,
                connector.__class__.__name__, _get_param_type_name(delay),
                _get_param_type_name(weight))

            # Write header
            fp.write(struct.pack("7I", prop.key, prop.size_words, proj[1],
                                 _crc_u32(synaptic_matrix.__name__),
                                 _crc_u32(connector.__class__.__name__),
                                 _crc_u32(_get_param_type_name(delay)),
                                 _crc_u32(_get_param_type_name(weight))))


            # Apply parameter map to synapse type parameters and write to region
            fp.write(lazy_param_map.apply_attributes(
                synapse_type, synaptic_matrix.OnChipParamMap).tostring())

            # Apply parameter map to connector parameters and write to region
            fp.write(lazy_param_map.apply_attributes(
                connector, connector._on_chip_param_map,
                context=projection_state_dict[proj[0]], post_slice=post_vertex_slice,
                pre_slice=prop.pre_n_slice).tostring())

            # Write delay parameter scaled to convert to timesteps and
            # with fixed point of zero to round to nearest timestep
            delay_scale = 1.0 / self.sim_timestep_ms
            _write_param(fp, delay, 0, delay_scale, True)

            # Write weights using weight fixed point and taking
            # The absolute value if the weight is unsigned
            _write_param(fp, weight, fixed_point=weight_fixed_point, scale=1.0,
                         absolute=not synapse_type._signed_weight)
