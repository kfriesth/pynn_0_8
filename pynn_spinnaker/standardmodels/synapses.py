from pyNN.standardmodels import synapses, build_translations
from ..spinnaker import regions
from ..simulator import state
from ..spinnaker import lazy_param_map
from ..spinnaker.utils import get_homogeneous_param
import logging

# Import functions
from functools import partial

logger = logging.getLogger("PyNN")

# ------------------------------------------------------------------------------
# StaticSynapse
# ------------------------------------------------------------------------------
class StaticSynapse(synapses.StaticSynapse):
    __doc__ = synapses.StaticSynapse.__doc__
    translations = build_translations(
        ("weight", "weight"),
        ("delay", "delay"),
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == "auto":
            d = state.dt
        return d

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    # How many post-synaptic neurons per core can a
    # SpiNNaker synapse processor of this type handle
    _max_post_neurons_per_core = 1024

    # How many CPU cycles are spent doing non-row processing things
    _constant_cpu_overhead = 3.85E6

    # What format of synaptic matrix does this synapse type require
    _synaptic_matrix_region_class = regions.StaticSynapticMatrix

    # How many timesteps of delay can DTCM ring-buffer handle
    # **NOTE** only 7 timesteps worth of delay can be handled by
    # 8 element delay buffer - The last element is purely for output
    _max_dtcm_delay_slots = 7

    # Static weights are unsigned
    _signed_weight = False

    # Static synapses don't require post-synaptic
    # spikes back-propagated to them
    _requires_back_propagation = False

    # Static synapses are always compatible with each other
    @property
    def _comparable_properties(self):
        return (self.__class__,)

    # How many CPU cycles does it take to process a row
    def _get_row_cpu_cost(self, row_length, **kwargs):
        # How many CPU cycles does it take to fetch a row
        # and initialize the synapse processing loop
        constant_cost = 486 + 53

        # How many CPU cycles does it take to process a synapse
        synapse_cost = 15

        return constant_cost + (synapse_cost * row_length)

# ------------------------------------------------------------------------------
# STDPMechanism
# ------------------------------------------------------------------------------
class STDPMechanism(synapses.STDPMechanism):
    __doc__ = synapses.STDPMechanism.__doc__

    base_translations = build_translations(
        ("weight", "weight"),
        ("delay", "delay"),
        ("dendritic_delay_fraction", "dendritic_delay_fraction")
    )

    def _get_minimum_delay(self):
        d = state.min_delay
        if d == 'auto':
            d = state.dt
        return d

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    # How many post-synaptic neurons per core can a
    # SpiNNaker synapse_processor of this type handle
    _max_post_neurons_per_core = 512

    # How many CPU cycles are spent doing non-row processing things
    _constant_cpu_overhead = 11.15E6

    # What format of synaptic matrix does this synapse type require
    _synaptic_matrix_region_class = regions.PlasticSynapticMatrix

    # How many timesteps of delay can DTCM ring-buffer handle
    # **NOTE** only 7 timesteps worth of delay can be handled by
    # 8 element delay buffer - The last element is purely for output
    _max_dtcm_delay_slots = 7

    # Static weights are unsigned
    _signed_weight = False

    # STDP synapses require post-synaptic
    # spikes back-propagated to them
    _requires_back_propagation = True

    def _update_weight_range(self, weight_range):
        self.weight_dependence._update_weight_range(weight_range)

    # The presynaptic state for STDP synapses consists of a uint32 containing
    # time of last update a uint32 containing time of last presynaptic spike
    # and the presynaptic trace required by the timing dependence
    @property
    def _pre_state_bytes(self):
        return 8 + self.timing_dependence._pre_trace_bytes

    # STDP mechanisms should be compared based on their class, timing
    # dependence (parameters) and weight dependence (parameters)
    @property
    def _comparable_properties(self):
        return (self.__class__, self.timing_dependence,
                self.weight_dependence)

    # The parameter map used to create the plasticity region is just the
    # timing and weight dependence's parameter maps concatenated together
    @property
    def _plasticity_param_map(self):
        return (self.timing_dependence._plasticity_param_map +
                self.weight_dependence._plasticity_param_map)

    @property
    def _executable_filename(self):
        return (self.__class__.__name__.lower() + "_" +
                self.weight_dependence.__class__.__name__.lower() + "_" +
                self.timing_dependence.__class__.__name__.lower())

    # How many CPU cycles does it take to process a row
    def _get_row_cpu_cost(self, row_length, pre_rate, post_rate, **kwargs):
        # How many CPU cycles does it take to fetch a row
        # and initialize the synapse processing loop
        constant_cost = 1143 + 226

        # How many CPU cycles does it take to process a synapse
        synapse_cost = 107 + (30 * (float(post_rate) / float(pre_rate)))

        return constant_cost + (synapse_cost * row_length)

# ------------------------------------------------------------------------------
# AdditiveWeightDependence
# ------------------------------------------------------------------------------
class AdditiveWeightDependence(synapses.AdditiveWeightDependence):
    __doc__ = synapses.AdditiveWeightDependence.__doc__

    translations = build_translations(
        ("w_max",     "w_max"),
        ("w_min",     "w_min"),
    )

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    _plasticity_param_map = [
        ("w_min", "i4", lazy_param_map.s32_fixed_point),
        ("w_max", "i4", lazy_param_map.s32_fixed_point),
    ]

    _comparable_param_names =  ("w_max", "w_min")

    def _update_weight_range(self, weight_range):
        weight_range.update(get_homogeneous_param(self.parameter_space, "w_max"))
        weight_range.update(get_homogeneous_param(self.parameter_space, "w_min"))

# ------------------------------------------------------------------------------
# MultiplicativeWeightDependence
# ------------------------------------------------------------------------------
class MultiplicativeWeightDependence(synapses.MultiplicativeWeightDependence):
    __doc__ = synapses.MultiplicativeWeightDependence.__doc__

    translations = build_translations(
        ("w_max",     "w_max"),
        ("w_min",     "w_min"),
    )

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    _plasticity_param_map = [
        ("w_min", "i4", lazy_param_map.s32_fixed_point),
        ("w_max", "i4", lazy_param_map.s32_fixed_point),
    ]

    _comparable_param_names =  ("w_max", "w_min")

    def _update_weight_range(self, weight_range):
        # **HACK** So we can use 16-bit DSP instructions on weights we need to be
        # able to treat them as SIGNED 16-bit numbers so require an extra bit of headroom
        weight_range.update(2.0 * get_homogeneous_param(self.parameter_space, "w_max"))
        weight_range.update(2.0 * get_homogeneous_param(self.parameter_space, "w_min"))


# ------------------------------------------------------------------------------
# SpikePairRule
# ------------------------------------------------------------------------------
class SpikePairRule(synapses.SpikePairRule):
    __doc__ = synapses.SpikePairRule.__doc__

    translations = build_translations(
        ("A_plus",    "a_plus"),
        ("A_minus",   "a_minus"),
        ("tau_plus",  "tau_plus"),
        ("tau_minus", "tau_minus"),
    )

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    _plasticity_param_map = [
        ("tau_plus", "256i2", partial(lazy_param_map.s411_exp_decay_lut,
                                      num_entries=256, time_shift=0)),
        ("tau_minus", "256i2", partial(lazy_param_map.s411_exp_decay_lut,
                                       num_entries=256, time_shift=0)),
        ("a_plus", "i4", lazy_param_map.s32_fixed_point),
        ("a_minus", "i4", lazy_param_map.s32_fixed_point),
    ]

    _comparable_param_names = ("tau_plus", "tau_minus", "A_plus", "A_minus")

    # Single int16 to contain trace
    _pre_trace_bytes = 2

# ------------------------------------------------------------------------------
# Vogels2011Rule
# ------------------------------------------------------------------------------
class Vogels2011Rule(synapses.Vogels2011Rule):
    __doc__ = synapses.Vogels2011Rule.__doc__

    translations = build_translations(
        ("tau", "tau"),
        ("eta", "eta"),
        ("rho", "rho"),
    )

    # --------------------------------------------------------------------------
    # Internal SpiNNaker properties
    # --------------------------------------------------------------------------
    _plasticity_param_map = [
        ("rho", "i4", lazy_param_map.s2011),
        ("tau", "256i2", partial(lazy_param_map.s411_exp_decay_lut,
                                 num_entries=256, time_shift=0)),
        ("eta", "i4", lazy_param_map.s32_fixed_point),
        ("eta", "i4", lazy_param_map.s32_fixed_point),
    ]

    _comparable_param_names = ("tau", "eta", "rho")

    # Single int16 to contain trace
    _pre_trace_bytes = 2
