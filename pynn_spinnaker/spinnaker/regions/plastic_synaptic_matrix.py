# Import modules
import logging
import math
import numpy as np
from .. import lazy_param_map

# Import classes
from synaptic_matrix import SynapticMatrix

logger = logging.getLogger("pynn_spinnaker")


# ------------------------------------------------------------------------------
# PlasticSynapticMatrix
# ------------------------------------------------------------------------------
class PlasticSynapticMatrix(SynapticMatrix):
    # How many bits should fixed point weights be converted into
    FixedPointWeightBits = 16

    # Parameters required from synapse type for on-chip generation
    OnChipParamMap = [("_signed_weight", "u4", lazy_param_map.integer),
                      ("_pre_state_bytes", "u4", lazy_param_map.integer),
                      (0, "u4")]

    def __init__(self, synapse_type):
        # Superclass
        super(PlasticSynapticMatrix, self).__init__(synapse_type)

        # Round up number of bytes required by presynaptic state to words
        self.pre_state_words = int(math.ceil(
            float(synapse_type._pre_state_bytes) / 4.0))

    # --------------------------------------------------------------------------
    # Private methods
    # --------------------------------------------------------------------------
    def _estimate_num_ext_words(self, max_sub_rows, max_total_sub_row_length):
        # Standard header and presynaptic state are required for each sub-row
        header = (self.NumHeaderWords + self.pre_state_words) * max_sub_rows

        # In the worst case each half-word array will have half a
        # word of padding and there are two arrays in each sub-row.
        # Therefore maximum size is one word per synapse
        # and one padding word per sub-row
        return header + max_total_sub_row_length + max_sub_rows

    def _get_num_row_words(self, num_synapses):
        # Both control and plastic words are stored as seperate
        # arrays of 16-bit elements so numbers of words
        # should be rounded up to keep them word aligned
        num_array_words = int(math.ceil(float(num_synapses) / 2.0))

        # Complete row consists of standard header, presynaptic state
        # and arrays of control words and plastic weights
        return self.NumHeaderWords + self.pre_state_words +\
            (2 * num_array_words)

    def _get_num_ext_words(self, num_sub_rows, sub_row_lengths,
                           sub_row_sections):
        # Round up each extension sub-row's length
        # to keep word aligned and take sum
        num_array_words = np.sum(np.ceil(sub_row_lengths[1:] / 2.0), dtype=int)

        # Number of synapses in all but 1st delay
        # slot and header for each extension row to total
        return (2 * num_array_words) +\
            ((self.NumHeaderWords + self.pre_state_words) * (num_sub_rows - 1))

    def _write_synapses(self, dtcm_delay, weight_fixed, indices, destination):
        # Zero presynaptic state
        destination[0: self.pre_state_words] = 0

        # Re-calculate size of control and plastic arrays in words
        num_array_words = int(math.ceil(float(len(indices)) / 2.0))

        # Based on this get index of where control words begin
        control_start_idx = self.pre_state_words + num_array_words

        # Create 16-bit view of plastic weight
        # section of destination and copy them in
        weight_view = destination[self.pre_state_words: control_start_idx]
        weight_view = weight_view.view(dtype=np.uint16)[:len(weight_fixed)]
        weight_view[:] = weight_fixed

        # Create 16-bit view of control word
        # section of destination and copy them in
        control_view = destination[control_start_idx:]
        control_view = control_view.view(dtype=np.uint16)[:len(indices)]
        control_view[:] = (indices
                           | (dtcm_delay << self.IndexBits)).astype(np.uint16)

    def _read_synapses(self, synapse_words, weight_to_float, dtype, synapses):
        # Re-calculate size of control and plastic arrays in words
        num_array_words = int(math.ceil(float(len(synapses)) / 2.0))

        # Based on this get index of where control words begin
        control_start_idx = self.pre_state_words + num_array_words

        # If weights are required
        if "weight" in dtype.names:
            # Determine type for weight
            weight_type = np.int16 if self.signed_weight else np.uint16

            # Create 16-bit view of plastic weight section of synapse words
            weight_view = synapse_words[self.pre_state_words: control_start_idx]
            weight_view = weight_view.view(dtype=weight_type)[:len(synapses)]

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
