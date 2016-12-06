# Import modules
import inspect
import itertools
import lazyarray as la
import logging
import math
import numpy as np
import struct
from os import path

# Import classes
from collections import namedtuple, Iterable
from pacman.model.graphs.machine.impl.machine_vertex \
    import MachineVertex
from pacman.model.resources.resource_container import ResourceContainer
from pacman.model.resources.sdram_resource import SDRAMResource
from spinn_front_end_common.abstract_models.abstract_has_associated_binary \
    import AbstractHasAssociatedBinary
from spinn_front_end_common.abstract_models.abstract_provides_n_keys_for_partition \
    import AbstractProvidesNKeysForPartition
from spinn_front_end_common.abstract_models.abstract_starts_synchronized \
    import AbstractStartsSynchronized
from spinnman.utilities.io.memory_io import MemoryIO

# Import functions
from copy import copy, deepcopy
from six import iteritems, iterkeys

logger = logging.getLogger("pynn_spinnaker")

# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
class Args(namedtuple("Args", "args, kwargs")):
    def __new__(cls, *args, **kwargs):
        return super(Args, cls).__new__(cls, args, kwargs)


# ----------------------------------------------------------------------------
# InputBuffer
# ----------------------------------------------------------------------------
InputBuffer = namedtuple("InputBuffer",
                         ["pointers", "start_neuron", "num_neurons",
                          "receptor_index", "weight_fixed_point"])


# ----------------------------------------------------------------------------
# InputVertex
# ----------------------------------------------------------------------------
class InputVertex(MachineVertex, AbstractHasAssociatedBinary,
                  AbstractStartsSynchronized):
    def __init__(self, post_neuron_slice, receptor_index, sdram, app_name):
        self.post_neuron_slice = post_neuron_slice
        self.weight_fixed_point = None
        self.receptor_index = receptor_index
        self.out_buffers = None
        self.region_memory = None
        self.app_name = app_name

        # Superclass
        # **NOTE** as vertex partitioning is already done,
        # only SDRAM is required for subsequent placing decisions
        label = ("<post neuron slice:%s, receptor index:%u>" %
                 (str(self.post_neuron_slice), self.receptor_index))
        MachineVertex.__init__(
            self, label=label,
            resources_required=ResourceContainer(sdram=SDRAMResource(sdram)))

    # ------------------------------------------------------------------------
    # AbstractHasAssociatedBinary methods
    # ------------------------------------------------------------------------
    def get_binary_file_name(self):
        return self.app_name

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def get_in_buffer(self, post_slice):
        # Check the slices involved overlap and that this
        # input vertex actually has output buffers
        assert post_slice.overlaps(self.post_neuron_slice)
        assert self.out_buffers is not None
        
        # The number of neurons-orth of input
        # transferred should match smallest slice
        num_neurons = min(len(post_slice), len(self.post_neuron_slice))

        # Buffer should be applied to neuron at start of
        # synapse start relative to start of neuron slice
        start_neuron = max(0, self.post_neuron_slice.start - post_slice.start)

        # If neuron slice starts after input slice then it should be offset
        offset_bytes = max(0, (post_slice.start - self.post_neuron_slice.start) * 4)

        # Return offset pointers into out buffers
        return InputBuffer([b + offset_bytes for b in self.out_buffers],
                           start_neuron, num_neurons,
                           self.receptor_index, self.weight_fixed_point)


# ----------------------------------------------------------------------------
# UnitStrideSlice
# ----------------------------------------------------------------------------
class UnitStrideSlice(namedtuple("UnitStrideSlice", ["start", "stop"])):
    # ------------------------------------------------------------------------
    # Magic methods
    # ------------------------------------------------------------------------
    def __len__(self):
        return self.stop - self.start

    def __str__(self):
        return "[%u, %u)" % (self.start, self.stop)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    def overlaps(self, other):
        return (self.start < other.stop) and (self.stop > other.start)

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def python_slice(self):
        return slice(self.start, self.stop)


# ----------------------------------------------------------------------------
# LazyArrayFloatToFixConverter
# ----------------------------------------------------------------------------
class LazyArrayFloatToFixConverter(object):
    """A callable which converts a lazy array of floats to fixed point

    General usage is to create a new converter and then call this on arrays of
    values.  The `dtype` of the returned array is determined from the
    parameters passed.  For example::
        >>> f = LazyArrayFloatToFixConverter(signed=True, n_bits=8, n_frac=4)
    """
    dtypes = {
        (False, 8): np.uint8,
        (True, 8): np.int8,
        (False, 16): np.uint16,
        (True, 16): np.int16,
        (False, 32): np.uint32,
        (True, 32): np.int32,
        (False, 64): np.uint64,
        (True, 64): np.int64,
    }

    def __init__(self, signed, n_bits, n_frac, copy):
        """Create a new converter from floats into ints.

        Parameters
        ----------
        signed : bool
            Indicates that the converted values are to be signed or otherwise.
        n_bits : int
            The number of bits each value will use overall (must be 8, 16, 32,
            or 64).
        n_frac : int
            The number of fractional bits.

        copy : bool
            Should array being converted to fixed-point be deepcopied
        """
        # Check the number of bits is sane
        if n_bits not in [8, 16, 32, 64]:
            raise ValueError(
                "n_bits: {}: Must be 8, 16, 32 or 64.".format(n_bits))

        # Determine the maximum and minimum values after conversion
        if signed:
            self.max_value = 2**(n_bits - 1) - 1
            self.min_value = -self.max_value - 1
        else:
            self.max_value = 2**n_bits - 1
            self.min_value = 0

        # Store the settings
        self.bytes_per_element = n_bits / 8
        self.dtype = self.dtypes[(signed, n_bits)]
        self.n_frac = n_frac
        self.copy = copy

    def __call__(self, values):
        """Convert the given lazy array of values into fixed point format."""
        # Make a copy of the original lazy array
        # **YUCK** deep copy here as lazy array constructor doesn't give the
        # option to copy and hence doesn't deep copy operations
        vals = deepcopy(values) if self.copy else copy(values)

        # Saturate the values
        # JK: Think more here
        # **TODO** this needs implementing in terms of less and more
        # vals = np.clip(values, self.min_value, self.max_value)
        # vals = vals.apply(partial(np.clip, a_min=self.min_value,
        #                          m_max=self.max_value))
        # Scale and round
        vals *= (2.0 ** self.n_frac)
        vals = la.rint(vals)

        # Set data type to fixed-point type and return
        vals.dtype = self.dtype
        return vals

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------
def split_slice(quantity, maximum_slice_size):
    # Build lists of start and end indices of slices
    slice_starts = range(0, quantity, maximum_slice_size)
    slice_ends = [min(s + maximum_slice_size, quantity) for s in slice_starts]

    # Zip starts and ends together into list
    # of slices and pair these with resources
    return [UnitStrideSlice(s, e) for s, e in zip(slice_starts, slice_ends)]


def calc_bitfield_words(bits):
    return int(math.ceil(float(bits) / 32.0))


def calc_slice_bitfield_words(vertex_slice):
    return calc_bitfield_words(len(vertex_slice))


def calc_timestep_mul(hardware_timestep_us):
    return min(1.0, float(hardware_timestep_us) / 1000.0)

def combine_row_offset_length(offset, length, num_length_bits):
    assert length >= 1 and length <= (2 ** num_length_bits)
    assert offset >= 0 and offset < (2 ** (32 - num_length_bits))

    return (length - 1) | (offset << num_length_bits)

def extract_row_offset_length(offset_length, num_length_bits):
    # Build mask to extract length
    length_mask = (2 ** num_length_bits) - 1

    # Extract fields
    offset = offset_length >> num_length_bits
    length = (offset_length & length_mask) + 1

    return offset, length

def is_scalar(value):
    return isinstance(value, (int, long, np.integer, float, bool))

def get_model_executable_filename(prefix, model, profiled):
    # Start filename with prefix
    filename = prefix

    # If executable filename is specified use it,
    # otherwise use lowercase classname
    filename += getattr(model, "_executable_filename",
                        model.__class__.__name__.lower())

    # If profiling is enabled, add prefix
    if profiled:
        filename += "_profiled"

    # Join filename to path and add extension
    return filename + ".aplx"

def get_homogeneous_param(param_space, param_name):
    # Extract named parameter lazy array from parameter
    # space and check that it's homogeneous
    param_array = param_space[param_name]
    assert param_array.is_homogeneous

    # Set it's shape to 1
    # **NOTE** for homogeneous arrays this is a)free and b)works
    param_array.shape = 1

    # Evaluate param array, simplifying it to a scalar
    return param_array.evaluate(simplify=True)

# Recursively build a tuple containing basic python types allowing an
# (annotated) PyNN StandardModelType to compared/hashed for compatibility)
def get_model_comparable(value):
    # **YUCK** if this isn't a class object - model classes will have
    # have the same attributes, they'll just be property object
    if not inspect.isclass(value):
        # If model type has a list of param names to use for hash
        if hasattr(value, "_comparable_param_names"):
            # Start tuple with class type - various STDP components
            # are likely to have similarly named parameters
            # with simular values so this is important 1st check
            comp = (value.__class__,)

            # Loop through names of parameters which much match for objects
            # to be equal and read them from parameter space into tuple
            for p in value._comparable_param_names:
                comp += (get_homogeneous_param(value.parameter_space, p),)
            return comp
        # Otherwise, if model type has a collection of comparable properties,
        # Loop through the properties and recursively call this function
        elif hasattr(value, "_comparable_properties"):
            return tuple(itertools.chain.from_iterable(
                get_model_comparable(p) for p in value._comparable_properties))
    # Otherwise, return value itself
    return (value,)

def load_regions(regions, region_arguments,
                 placement, transceiver, app_id):
    # Calculate region size
    size = sizeof_regions_named(regions, region_arguments)

    logger.debug("\t\t\t\t\tRegion size = %u bytes", size)

    # Allocate a suitable memory block
    # for this vertex and get memory io
    # **NOTE** this is tagged by core
    start_address = transceiver.malloc_sdram(placement.x, placement.y,
                                             size, app_id, placement.p)
    end_address = start_address + size
    memory_io = MemoryIO(transceiver, placement.x, placement.y,
                         start_address, end_address)

    logger.debug("\t\t\t\t\tMemory with tag:%u begins at:%08x",
                    placement.p, start_address)

    # Layout the slice of SDRAM we have been given
    region_memory = create_app_ptr_and_region_files_named(
        memory_io, regions, region_arguments)

    # Write each region into memory
    for key, region in iteritems(regions):
        # Get memory
        mem = region_memory[key]

        # Get the arguments
        args, kwargs = region_arguments[key]

        # Perform the write
        region.write_subregion_to_file(mem, *args, **kwargs)

        # **HACK** force a flush of memory region
        mem.flush()

    # Close memory IO?
    memory_io.close()

    # Return region memory
    return region_memory

# **FUTUREFRONTEND** with a bit of word to add magic number
# to the start, this is common with Nengo SpiNNaker
def create_app_ptr_and_region_files_named(fp, regions, region_args):
    """Split up a file-like view of memory into smaller views, one per region,
    and write into the first region of memory the offsets to these later
    regions.

    Parameters
    ----------
    regions : {name: Region, ...}
        Map from keys to region objects.  The keys MUST support `int`, items
        from :py:class:`enum.IntEnum` are recommended.
    region_args : {name: (*args, **kwargs)}
        Map from keys to the arguments and keyword-arguments that should be
        used when determining the size of a region.

    Returns
    -------
    {name: file-like}
        Map from region name to file-like view of memory.
    """
    # Determine the number of entries needed in the application pointer table
    ptr_len = max(int(k) for k in iterkeys(regions)) + 1

    # Construct an empty pointer table of the correct length
    ptrs = [0] * ptr_len

    # Update the offset and then begin to allocate memory
    region_memory = dict()
    offset = (ptr_len * 4) + 4  # 1 word per region and magic number
    for k, region in iteritems(regions):
        # Get the size of this region
        args, kwargs = region_args[k]
        region_size = region.sizeof_padded(*args, **kwargs)

        # Store the current offset as the pointer for this region
        ptrs[int(k)] = offset

        # Get the memory region and update the offset
        next_offset = offset + region_size
        region_memory[k] = fp[offset:next_offset]
        offset = next_offset

    # Write the pointer table into memory
    fp.seek(0)
    fp.write(struct.pack("<{}I".format(1 + ptr_len), 0xAD130AD6, *ptrs))
    fp.seek(0)

    # Return the region memories
    return region_memory


def sizeof_regions_named(regions, region_args, include_app_ptr=True):
    """Return the total amount of memory required to represent
    all regions when padded to a whole number of words each
    and dictionary of any any extra allocations required

    Parameters
    ----------
    regions : {name: Region, ...}
        Map from keys to region objects.  The keys MUST support `int`, items
        from :py:class:`enum.IntEnum` are recommended.
    region_args : {name: (*args, **kwargs)}
        Map from keys to the arguments and keyword-arguments that should be
        used when determining the size of a region.
    """
    if include_app_ptr:
        # Get the size of the application pointer
        size = 4 + ((max(int(k) for k in iterkeys(regions)) + 1) * 4)
    else:
        # Don't include the application pointer
        size = 0

    # Get the size of all the regions
    allocations = {}
    for key, region in iteritems(regions):
        # Get the arguments for the region
        args, kwargs = region_args[key]

        # Add size to total and include allocations in dictionary
        size += region.sizeof_padded(*args, **kwargs)

    return size
