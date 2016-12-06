"""
Mock implementation of the PyNN API, for testing and documentation purposes.

This simulator implements the PyNN API, but generates random data rather than
really running simulations.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import atexit
import logging
import warnings

from pyNN import common
from pyNN.common.control import (DEFAULT_MAX_DELAY, DEFAULT_TIMESTEP,
                                 DEFAULT_MIN_DELAY)
from pyNN.recording import *

from rig_cpp_common import profiling
import simulator

from .standardmodels.cells import *
from .standardmodels.synapses import *
from .connectors import *
from .random import NativeRNG
from .populations import Population, PopulationView, Assembly
from .projections import Projection

from rig.machine_control.machine_controller import TruncationWarning

logger = logging.getLogger("PyNN")

# Convert Rig TruncationWarnings to errors
warnings.simplefilter("error", TruncationWarning)

@atexit.register
def _stop_on_spinnaker():
    # Stop SpiNNaker simulation
    simulator.state.stop()


def list_standard_models():
    """Return a list of all the StandardCellType
    classes available for this simulator."""
    return [obj.__name__
            for obj in globals().values()
            if isinstance(obj, type) and issubclass(obj, StandardCellType)]


def setup(timestep=DEFAULT_TIMESTEP, min_delay=DEFAULT_MIN_DELAY,
          max_delay=DEFAULT_MAX_DELAY, **extra_params):
    common.setup(timestep, min_delay, max_delay, **extra_params)
    simulator.state.clear()
    simulator.state.dt = timestep
    simulator.state.min_delay = min_delay
    simulator.state.max_delay = max_delay
    simulator.state.spinnaker_hostname = extra_params.get("spinnaker_hostname")
    simulator.state.realtime_proportion =\
        extra_params.get("realtime_proportion", 1.0)
    simulator.state.convert_direct_connections =\
        extra_params.get("convert_direct_connections", True)
    simulator.state.generate_connections_on_chip =\
        extra_params.get("generate_connections_on_chip", True)
    simulator.state.stop_on_spinnaker =\
        extra_params.get("stop_on_spinnaker", True)
    simulator.state.stop_after_loader =\
        extra_params.get("stop_after_loader", False)
    simulator.state.disable_software_watchdog =\
        extra_params.get("disable_software_watchdog", False)
    simulator.state.allocation_fudge_factor =\
        extra_params.get("allocation_fudge_factor", 1.6)

    return rank()


def end(compatible_output=True):
    """Do any necessary cleaning up before exiting."""
    for (population, variables, filename) in simulator.state.write_on_end:
        io = get_io(filename)
        population.write_data(io, variables)
    simulator.state.write_on_end = []

    # Stop SpiNNaker simulation
    _stop_on_spinnaker()

run, run_until = common.build_run(simulator)
run_for = run

reset = common.build_reset(simulator)

initialize = common.initialize

get_current_time, get_time_step, get_min_delay, get_max_delay, \
    num_processes, rank = common.build_state_queries(simulator)


create = common.build_create(Population)

connect = common.build_connect(Projection, FixedProbabilityConnector,
                               StaticSynapse)

record = common.build_record(simulator)


def record_v(source, filename):
    record(['v'], source, filename)


def record_gsyn(source, filename):
    record(['gsyn_exc', 'gsyn_inh'], source, filename)
