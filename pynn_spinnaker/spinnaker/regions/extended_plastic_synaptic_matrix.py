# Import modules
import logging
import math
import numpy as np
from .. import lazy_param_map

# Import classes
from synaptic_matrix import SynapticMatrix

logger = logging.getLogger("pynn_spinnaker")


# ------------------------------------------------------------------------------
# ExtendedPlasticSynapticMatrix
# ------------------------------------------------------------------------------
class ExtendedPlasticSynapticMatrix(SynapticMatrix):
    # How many bits should fixed point weights be converted into
    FixedPointWeightBits = 16

    # Parameters required from synapse type for on-chip generation
    OnChipParamMap = [("_signed_weight", "u4", lazy_param_map.integer),
                      ("_pre_state_bytes", "u4", lazy_param_map.integer),
                      ("_synapse_trace_bytes", "u4", lazy_param_map.integer)]

    def __init__(self, synapse_type):
        # Superclass
        super(ExtendedPlasticSynapticMatrix, self).__init__(synapse_type)

        # Round up number of bytes required by presynaptic state to words
        self.pre_state_words = int(math.ceil(
            float(synapse_type._pre_state_bytes) / 4.0))

        # Add number of extra bytes associated
        # with each synapse to 2 bytes weight
        self.synapse_bytes = 2 + synapse_type._synapse_trace_bytes

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------
    def _estimate_num_ext_words(self, max_sub_rows, max_total_sub_row_length):
        # Standard header and presynaptic state are required for each sub-row
        header = (self.NumHeaderWords + self.pre_state_words) * max_sub_rows

        # In the worst case each array of plastic BYTES will require
        # 3 bytes of padding to align to a word boundary and each array
        # of control HALF-WORDS will require 1 half-word = 2 bytes.
        synapse_padding_bytes = (3 + 2) * max_sub_rows

        # Each synapse in the row will require 2 bytes for its control
        # half-word and whatever the synapse type required
        synapse_data_bytes = max_total_sub_row_length * (self.synapse_bytes + 2)

        # Convert to words and add header
        synapse_bytes = synapse_padding_bytes + synapse_data_bytes
        return header + int(math.ceil(float(synapse_bytes) / 4.0))

    def _get_num_row_words(self, num_synapses):
        # Calculate size of control and plastic word arrays
        num_control_words, num_plastic_words =\
            self._get_num_array_words(num_synapses)

        # Complete row consists of standard header, presynaptic state
        #  and arrays of control words and plastic weights
        return self.NumHeaderWords + self.pre_state_words +\
            num_control_words + num_plastic_words

    def _get_num_ext_words(self, num_sub_rows, sub_row_lengths,
                           sub_row_sections):
        # Round up each extension sub-row's length to keep
        # control arrays word aligned and take sum
        num_control_words = np.sum(np.ceil(sub_row_lengths[1:] / 2.0),
                                   dtype=int)

        # Round up each extension sub-row's length to keep
        # plastic arrays word aligned and take sum
        num_plastic_words = np.sum(np.ceil(
            sub_row_lengths[1:] * self.synapse_bytes / 4.0), dtype=int)

        # Number of synapses in all but 1st delay
        # slot and header for each extension row to total
        return num_control_words + num_plastic_words +\
            ((self.NumHeaderWords + self.pre_state_words) * (num_sub_rows - 1))

    def _write_synapses(self, dtcm_delay, weight_fixed, indices, destination):
        # Zero time of last pre-synaptic spike and pre-synaptic trace
        destination[0: self.pre_state_words] = 0

        # Re-calculate size of control and plastic word arrays
        num_control_words, num_plastic_words =\
            self._get_num_array_words(len(indices))

        # Based on this get index of where control words begin
        control_start_idx = self.pre_state_words + num_plastic_words

        # Create 8-bit view of plastic section of row
        plastic_view = destination[self.pre_state_words: control_start_idx]
        plastic_view = plastic_view.view(dtype=np.uint8)

        # Restrict this view to non-padding synapses
        plastic_view = plastic_view[:len(weight_fixed) * self.synapse_bytes]

        # Reshape into a 2D array where each synapse is a row
        plastic_view.shape = (-1, self.synapse_bytes)

        # View fixed point weights as bytes and reshape into matching shape
        weight_bytes = weight_fixed.view(dtype=np.uint8)
        weight_bytes.shape = (-1, 2)

        # Copy weights into first two byte columns of plastic view
        plastic_view[:,0:2] = weight_bytes

        # Zero remaining columns
        plastic_view[:,2:] = 0

        # Create 16-bit view of control word
        # section of destination and copy them in
        control_view = destination[control_start_idx:]
        control_view = control_view.view(dtype=np.uint16)[:len(indices)]
        control_view[:] = (indices
                           | (dtcm_delay << self.IndexBits)).astype(np.uint16)

    def _read_synapses(self, synapse_words, weight_to_float, dtype, synapses):
        # Re-calculate size of control and plastic word arrays
        num_control_words, num_plastic_words =\
            self._get_num_array_words(len(synapses))

        # Based on this get index of where control words begin
        control_start_idx = self.pre_state_words + num_plastic_words

        # Create 8-bit view of plastic section of row
        plastic_view = synapse_words[self.pre_state_words: control_start_idx]
        plastic_view = plastic_view.view(dtype=np.uint8)

        # Restrict this view to non-padding synapses
        plastic_view = plastic_view[:len(synapses) * self.synapse_bytes]

        # Reshape into a 2D array where each synapse is a row
        plastic_view = plastic_view.reshape((-1, self.synapse_bytes))

        # If weights are required
        if "weight" in dtype.names:
            # Determine type for weight
            weight_type = np.int16 if self.signed_weight else np.uint16

            # Create 16-bit weight view of first two bytes in each synapse,
            # reshape this back into a 1D view and convert weights to float
            weight_view = plastic_view[:,0:2].reshape(-1)
            weight_view = weight_view.view(dtype=weight_type)

            # Convert the weight view to floating point
            synapses["weight"] = weight_to_float(weight_view)

        # Create 16-bit view of control word section of synapse words
        control_view = synapse_words[control_start_idx:]
        control_view = control_view.view(dtype=np.uint16)[:len(synapses)]

        # Extract the delays if required
        if "delay" in dtype.names:
            delay_mask = (1 << self.DelayBits) - 1
            synapses["delay"] = (control_view >> self.IndexBits) & delay_mask

        # Extract the post-synaptic index if required
        if "postsynaptic_index" in dtype.names:
            index_mask = (1 << self.IndexBits) - 1
            synapses["postsynaptic_index"] = control_view & index_mask

    def _get_num_array_words(self, num_synapses):
        # Control words are stored as an array of
        # 16-bit elements, padded to a word boudary
        num_control_words = int(math.ceil(float(num_synapses) / 2.0))

        # Plastic words are stored as an array of synapse_bytes
        # long structures, padded to a word boundary
        num_plastic_words = int(math.ceil(
            float(num_synapses * self.synapse_bytes) / 4.0))

        return num_control_words, num_plastic_words
