# Import modules
import lazyarray as la
import logging
from pyNN.standardmodels import cells
from ..spinnaker import lazy_param_map
from ..spinnaker import regions

# Import functions
from copy import deepcopy
from functools import partial
from pyNN.standardmodels import build_translations
from ..spinnaker.utils import calc_timestep_mul

logger = logging.getLogger("PyNN")

# Function to use with lazy_param_map.choose to determine whether
# a Poisson source should be modelled using the fast or slow model
def _poisson_slow_model(values, sim_timestep_ms, **kwargs):
    # Convert rates into spikes per time step
    rate_vals = deepcopy(values)
    spikes_per_timestep = (rate_vals * sim_timestep_ms) / 1000.0

    # Based on this return mask specifying which spikes sources
    # should be simulated using the slow rather than fast model
    return la.larray(spikes_per_timestep <= 0.25)

def calc_max_neurons_per_core(hardware_timestep_us,
                              num_input_processors,
                              neuron_update_cpu_cycles,
                              synapse_shape_cpu_cycles,
                              apply_input_cpu_cycles=10):
    # Calculate the number of timesteps we have available
    total_cycles = int(200000 * calc_timestep_mul(hardware_timestep_us))

    # Calculate the number of cycles per neuron
    cycles_per_neuron = (neuron_update_cpu_cycles + synapse_shape_cpu_cycles
                         + (num_input_processors * apply_input_cpu_cycles))

    # Divide the total by this
    return min(1024, total_cycles // cycles_per_neuron)

# ----------------------------------------------------------------------------
# Neuron type translations
# ----------------------------------------------------------------------------
# Build translations from PyNN to SpiNNaker neuron model parameters
if_curr_neuron_translations = build_translations(
    ("tau_m",       "tau_m"),
    ("cm",          "r_membrane", "tau_m / cm", ""),
    ("v_rest",      "v_rest"),
    ("v_thresh",    "v_thresh"),
    ("v_reset",     "v_reset"),
    ("tau_refrac",  "tau_refrac"),
    ("i_offset",    "i_offset"),
)

if_cond_neuron_translations = build_translations(
    ("tau_m",       "tau_m"),
    ("cm",          "r_membrane", "tau_m / cm", ""),
    ("v_rest",      "v_rest"),
    ("e_rev_E",     "e_rev_e"),
    ("e_rev_I",     "e_rev_i"),
    ("v_thresh",    "v_thresh"),
    ("v_reset",     "v_reset"),
    ("tau_refrac",  "tau_refrac"),
    ("i_offset",    "i_offset"),
)

izhikevich_neuron_translations = build_translations(
    ("a",         "a"),
    ("b",         "b"),
    ("c",         "c"),
    ("d",         "d"),
    ("i_offset",  "i_offset"),
)

# ----------------------------------------------------------------------------
# Synapse type translations
# ----------------------------------------------------------------------------
# Build translations from PyNN to SpiNNaker synapse model parameters
exp_synapse_translations = build_translations(
    ("tau_syn_E",   "tau_syn_e"),
    ("tau_syn_I",   "tau_syn_i"),
)

# ----------------------------------------------------------------------------
# Neuron region maps
# ----------------------------------------------------------------------------
# Build maps of where and how parameters need to be written into neuron regions
if_curr_neuron_immutable_param_map = [
    ("v_thresh",    "i4", lazy_param_map.s1615),
    ("v_reset",     "i4", lazy_param_map.s1615),
    ("v_rest",      "i4", lazy_param_map.s1615),
    ("i_offset",    "i4", lazy_param_map.s1615),
    ("r_membrane",  "i4", lazy_param_map.s1615),
    ("tau_m",       "i4", lazy_param_map.s1615_exp_decay),
    ("tau_refrac",  "u4", lazy_param_map.integer_time_divide),
]

if_curr_neuron_mutable_param_map = [
    ("v", "i4", lazy_param_map.s1615),
    (0,   "i4"),
]

if_cond_neuron_immutable_param_map = [
    ("v_thresh",    "i4", lazy_param_map.s1615),
    ("v_reset",     "i4", lazy_param_map.s1615),
    ("v_rest",      "i4", lazy_param_map.s1615),
    ("e_rev_e",     "i4", lazy_param_map.s1615),
    ("e_rev_i",     "i4", lazy_param_map.s1615),
    ("i_offset",    "i4", lazy_param_map.s1615),
    ("r_membrane",  "i4", lazy_param_map.s1615),
    ("tau_m",       "i4", lazy_param_map.s1615_exp_decay),
    ("tau_refrac",  "u4", lazy_param_map.integer_time_divide),
]

if_cond_neuron_mutable_param_map = [
    ("v", "i4", lazy_param_map.s1615),
    (0,   "i4"),
]

# ----------------------------------------------------------------------------
# Synapse shaping region maps
# ----------------------------------------------------------------------------
exp_synapse_immutable_param_map = [
    ("tau_syn_e", "u4", lazy_param_map.u032_exp_decay),
    ("tau_syn_e", "i4", lazy_param_map.s1615_exp_init),
    ("tau_syn_i", "u4", lazy_param_map.u032_exp_decay),
    ("tau_syn_i", "i4", lazy_param_map.s1615_exp_init),
]

exp_synapse_curr_mutable_param_map = [
    ("isyn_exc", "i4", lazy_param_map.s1615),
    ("isyn_inh", "i4", lazy_param_map.s1615),
]

exp_synapse_cond_mutable_param_map = [
    ("gsyn_exc", "i4", lazy_param_map.s1615),
    ("gsyn_inh", "i4", lazy_param_map.s1615),
]

# ----------------------------------------------------------------------------
# Cell types
# ----------------------------------------------------------------------------
class IF_curr_exp(cells.IF_curr_exp):
    __doc__ = cells.IF_curr_exp.__doc__

    translations = deepcopy(if_curr_neuron_translations)
    translations.update(exp_synapse_translations)

    # SpiNNaker additionally supports recording synaptic input currents
    recordable = cells.IF_curr_exp.recordable + ["isyn_exc", "isyn_inh"]

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    # JK: not necessary
    _neuron_region_class = regions.Neuron
    
    _directly_connectable = False

    _neuron_immutable_param_map = if_curr_neuron_immutable_param_map
    _neuron_mutable_param_map = if_curr_neuron_mutable_param_map

    _synapse_immutable_param_map = exp_synapse_immutable_param_map
    _synapse_mutable_param_map = exp_synapse_curr_mutable_param_map

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    # How many of these neurons per core can
    # a SpiNNaker neuron processor handle
    _calc_max_neurons_per_core = partial(calc_max_neurons_per_core,
                                         neuron_update_cpu_cycles=143,
                                         synapse_shape_cpu_cycles=28)

class IF_cond_exp(cells.IF_cond_exp):
    __doc__ = cells.IF_cond_exp.__doc__

    translations = deepcopy(if_cond_neuron_translations)
    translations.update(exp_synapse_translations)

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    _neuron_region_class = regions.Neuron

    _directly_connectable = False

    _neuron_immutable_param_map = if_cond_neuron_immutable_param_map
    _neuron_mutable_param_map = if_cond_neuron_mutable_param_map

    _synapse_immutable_param_map = exp_synapse_immutable_param_map
    _synapse_mutable_param_map = exp_synapse_cond_mutable_param_map

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    # How many of these neurons per core can
    # a SpiNNaker neuron processor handle
    _calc_max_neurons_per_core = partial(calc_max_neurons_per_core,
                                         neuron_update_cpu_cycles=167,
                                         synapse_shape_cpu_cycles=28)

'''
class Izhikevich(cells.Izhikevich):
    __doc__ = cells.Izhikevich.__doc__

    translations = deepcopy(izhikevich_neuron_translations)
    translations.update(current_synapse_translations)
'''


class SpikeSourcePoisson(cells.SpikeSourcePoisson):
    __doc__ = cells.SpikeSourcePoisson.__doc__

    translations = build_translations(
        ("start",    "start_time"),
        ("rate",     "rate"),
        ("duration", "end_time",  "start + duration", "end_time - start_time"),
    )

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    _directly_connectable = True
    _neuron_region_class = regions.SpikeSourcePoisson
    _current_input_region_class = regions.SpikeSourcePoisson

    _immutable_param_map = [
        ("rate", "u4", _poisson_slow_model),
        ("start_time", "u4", lazy_param_map.integer_time_divide),
        ("end_time", "u4", lazy_param_map.integer_time_divide),
        ("rate", "u4", partial(lazy_param_map.choose,
                               mask_function=_poisson_slow_model,
                               a_function=lazy_param_map.s1615_rate_isi,
                               b_function=lazy_param_map.u032_rate_exp_minus_lambda)),
    ]

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    # How many of these neurons per core can
    # a SpiNNaker neuron processor handle
    _calc_max_neurons_per_core = partial(calc_max_neurons_per_core,
                                         neuron_update_cpu_cycles=58,
                                         synapse_shape_cpu_cycles=0,
                                         apply_input_cpu_cycles=0)

    def _calc_max_current_inputs_per_core(self, hardware_timestep_us):
        # Calculate timestep multiplier
        timestep_mul = calc_timestep_mul(hardware_timestep_us)

        # Scale by timestep mul
        return int(2048 * timestep_mul)

class SpikeSourceArray(cells.SpikeSourceArray):
    __doc__ = cells.SpikeSourceArray.__doc__

    translations = build_translations(
        ("spike_times", "spike_times"),
    )

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    # How many of these neurons per core can
    # a SpiNNaker neuron processor handle
    _max_current_inputs_per_core = 2048

    _directly_connectable = True
    _neuron_region_class = regions.SpikeSourceArray
    _current_input_region_class = regions.SpikeSourceArray

    # --------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # --------------------------------------------------------------------------
    # How many of these neurons per core can
    # a SpiNNaker neuron processor handle
    def _calc_max_neurons_per_core(self, hardware_timestep_us,
                                   num_input_processors):
        assert num_input_processors == 0

        # Calculate timestep multiplier
        timestep_mul = calc_timestep_mul(hardware_timestep_us)

        # Scale by timestep mul
        return int(1024 * timestep_mul)

    def _calc_max_current_inputs_per_core(self, hardware_timestep_us):
        # Calculate timestep multiplier
        timestep_mul = calc_timestep_mul(hardware_timestep_us)

        # Scale by timestep mul
        return int(2048 * timestep_mul)