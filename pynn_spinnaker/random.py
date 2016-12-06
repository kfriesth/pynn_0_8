# Import modules
import lazyarray as la
from spinnaker import lazy_param_map
from scipy.stats import norm, expon

# Import classes
from pyNN.random import NativeRNG
from pyNN.errors import InvalidParameterValueError

# Import functions
from six import iteritems


def _estimate_max_value_normal(parameters):
    estimated_max = norm.ppf(1-1e-6)
    return parameters["mu"] + parameters["sigma"] * estimated_max

def _estimate_max_value_normal_clipped(parameters):
    return min(_estimate_max_value_normal(parameters), parameters["high"])

def _estimate_max_value_exponential(parameters):
    return parameters["beta"] * expon.ppf(1-1e-6)

def _check_parameters_normal(parameters):
    msg = "Expected positive sigma"
    
    return parameters["sigma"] > 0, msg

def _check_parameters_normal_clipped(parameters):
    msg = ("Expected positive sigma and greater than "
          "1e-4 probability of sampling between 'low' and 'high'")
    
    low = (parameters["low"] - parameters["mu"]) / parameters["sigma"]
    high = (parameters["high"] - parameters["mu"]) / parameters["sigma"]
    return (parameters["sigma"] > 0 and norm.cdf(high)
            - norm.cdf(low) > 1e-4, msg)

def _check_parameters_normal_clipped_to_boundary(parameters):
    msg = "Expected positive simga and 'low' <= 'high'"
    
    return (parameters["sigma"] > 0
            and parameters["high"] >= parameters["low"], msg)


# ----------------------------------------------------------------------------
# NativeRNG
# ----------------------------------------------------------------------------
# Signals that the random numbers will be supplied by RNG running on SpiNNaker
class NativeRNG(NativeRNG):
    # Maps specifying how each distribution type's
    # parameters will be written to SpiNNaker
    _dist_param_maps = {
        "uniform":      [("low",    "i4", lazy_param_map.s32_fixed_point_scale_abs),
                         ("high",   "i4", lazy_param_map.s32_fixed_point_scale_abs)],
        "uniform_int":  [("low",    "i4", lazy_param_map.s32_fixed_point_scale_abs),
                         ("high",   "i4", lazy_param_map.s32_fixed_point_scale_abs)],
        "normal":       [("mu",     "i4", lazy_param_map.s32_fixed_point_scale_abs),
                         ("sigma",  "i4", lazy_param_map.s32_fixed_point_scale_abs)],
        "exponential":  [("beta",   "i4", lazy_param_map.s32_fixed_point_scale_abs)],
        "normal_clipped": [("mu",    "i4", lazy_param_map.s32_fixed_point_scale_abs),
                           ("sigma", "i4", lazy_param_map.s32_fixed_point_scale_abs),
                           ("low",   "i4", lazy_param_map.s32_fixed_point_scale_abs),
                           ("high",  "i4", lazy_param_map.s32_fixed_point_scale_abs)],
        "normal_clipped_to_boundary": [("mu",    "i4", lazy_param_map.s32_fixed_point_scale_abs),
                                       ("sigma", "i4", lazy_param_map.s32_fixed_point_scale_abs),
                                       ("low",   "i4", lazy_param_map.s32_fixed_point_scale_abs),
                                       ("high",  "i4", lazy_param_map.s32_fixed_point_scale_abs)]
    }

    # Functions to estimate the maximum value a distribution will result in
    # **THINK** should this be moved out of NativeRNG
    # for more general estimation of max delays etc

    _dist_estimate_max_value = {
        "uniform":        lambda parameters: parameters["high"],
        "uniform_int":    lambda parameters: parameters["high"],
        "normal":         _estimate_max_value_normal,
        "normal_clipped": _estimate_max_value_normal_clipped,
        "normal_clipped_to_boundary": _estimate_max_value_normal_clipped,
        "exponential":    _estimate_max_value_exponential
    }

    # Functions to check that the distribution parameters are valid.
    # For normal_clipped we also check that the probability of sampling
    # within the specified region is greater than 1e-4. Distributions
    # not in this dictionary are assumed to have valid parameters.

    _dist_check_parameters = {
        "normal": _check_parameters_normal,
        "normal_clipped": _check_parameters_normal_clipped,
        "normal_clipped_to_boundary": _check_parameters_normal_clipped_to_boundary
    }

    def __init__(self, host_rng, seed=None):
        # Superclass
        super(NativeRNG, self).__init__(seed)

        # Cache RNG to use on the host
        assert host_rng is not None
        self._host_rng = host_rng

    # ------------------------------------------------------------------------
    # AbstractRNG methods
    # ------------------------------------------------------------------------
    def next(self, n=None, distribution=None, parameters=None, mask_local=None):
        # Draw from host RNG
        return self._host_rng.next(n, distribution, parameters, mask_local)

    # ------------------------------------------------------------------------
    # Internal SpiNNaker methods
    # ------------------------------------------------------------------------
    def _supports_dist(self, distribution):
        return distribution in self._dist_param_maps

    def _estimate_dist_max_value(self, distribution, parameters):
         # Check translation and parameter map exists for this distribution
        if not self._supports_dist(distribution):
            raise NotImplementedError("SpiNNaker native RNG does not support"
                                      "%s distributions" % distribution)
        else:
            return self._dist_estimate_max_value[distribution](parameters)

    def _check_dist_parameters(self, distribution, parameters):
        if not self._supports_dist(distribution):
            raise NotImplementedError("SpiNNaker native RNG does not support"
                                      "%s distributions" % distribution)
        else:
            if distribution in self._dist_check_parameters:
                valid_parameters, msg = \
                        self._dist_check_parameters[distribution](parameters)

                if not valid_parameters:
                    raise InvalidParameterValueError(msg)

    def _get_dist_param_map(self, distribution):
        # Check translation and parameter map exists for this distribution
        if not self._supports_dist(distribution):
            raise NotImplementedError("SpiNNaker native RNG does not support"
                                      "%s distributions" % distribution)
        else:
            return self._dist_param_maps[distribution]

    def _get_dist_size(self, distribution):
        # Check translation and parameter map exists for this distribution
        if not self._supports_dist(distribution):
            raise NotImplementedError("SpiNNaker native RNG does not support"
                                      "%s distributions" % distribution)
        else:
            return lazy_param_map.size(self._get_dist_param_map(distribution),
                                       1)

    def _write_dist(self, fp, distribution, parameters, fixed_point, scale, absolute):
        self._check_dist_parameters(distribution, parameters)

        # Wrap parameters in lazy arrays
        parameters = {name: la.larray(value)
                      for name, value in iteritems(parameters)}

        # Evaluate parameters and write to file
        data = lazy_param_map.apply(
            parameters, self._get_dist_param_map(distribution),
            1, fixed_point=fixed_point, scale=scale, absolute=absolute)

        fp.write(data.tostring())

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------
    @property
    def parallel_safe(self):
        return self._host_rng.parallel_safe

