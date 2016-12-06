# Import modules
import numpy as np
import pytest
import pynn_spinnaker as sim

# Import classes
from pyNN.random import NumpyRNG, RandomDistribution
from pynn_spinnaker.spinnaker.utils import UnitStrideSlice

# Import functions
from scipy.stats import binned_statistic, ks_2samp


native_rng = sim.NativeRNG(NumpyRNG())

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _run_network(pop_size, connector, synapse, required_params, dt=1.0):
    # Setup simulator
    sim.setup(timestep=dt, stop_after_loader=True)

    # Create two populations
    pre = sim.Population(pop_size, sim.IF_curr_exp())
    post = sim.Population(pop_size, sim.IF_curr_exp())

    # As we're not actually simulating the network, give a low
    # mean firing rate estimate so partitioning will be optimistic
    pre.spinnaker_config.mean_firing_rate = 0.1

    # Connect the populations together
    proj = sim.Projection(pre, post, connector,
                          synapse)

    # Check that projection will be generated on chip
    assert proj._can_generate_on_chip

    # Run for a token period (it's ignored as we're stopping after load)
    sim.run(1)

    # Read data from projection
    proj_data = proj.get(required_params, format="list")

    # Unzip and return
    return proj, zip(*proj_data)

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("pop_size", [200, 2000])
@pytest.mark.parametrize("synapse",
                         [sim.StaticSynapse(weight=0.0, delay=4.0),
                          sim.StaticSynapse(weight=0.0, delay=10.0),
                          sim.STDPMechanism(
                              timing_dependence=sim.SpikePairRule(A_plus=0.0, A_minus=0.0),
                              weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
                              weight=0.0, delay=4.0),
                          sim.STDPMechanism(
                              timing_dependence=sim.SpikePairRule(A_plus=0.0, A_minus=0.0),
                              weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0),
                              weight=0.0, delay=10.0)])
def test_static_delay(pop_size, synapse):
    # Run network
    proj, proj_data = _run_network(pop_size, sim.AllToAllConnector(),
                                   synapse, ["delay"])

    # Check they all match
    delay = synapse.native_parameters["delay"].base_value
    assert all(d == delay for d in proj_data[2])

@pytest.mark.parametrize("pop_size", [200, 2000])
@pytest.mark.parametrize("connector",
                         [sim.AllToAllConnector(),
                          sim.FixedProbabilityConnector(0.1, rng=native_rng)])
def test_connector_metrics(pop_size, connector):
    # Run network
    proj, proj_data = _run_network(pop_size, connector,
                                   sim.StaticSynapse(weight=0.0, delay=1.0),
                                   [])

    # Estimate maximum and mean row length
    pop_slice = UnitStrideSlice(0, pop_size)
    estimated_max_row_synapses = connector._estimate_max_row_synapses(
        pop_slice, pop_slice, pop_size, pop_size)
    estimated_mean_row_synapses = connector._estimate_mean_row_synapses(
        pop_slice, pop_slice, pop_size, pop_size)

    # Build row-length histogram
    row_length_histogram = binned_statistic(proj_data[0], proj_data[1],
                                            statistic="count", bins=range(pop_size + 1))[0]

    # Computer max and mean
    actual_max_row_synapses = np.amax(row_length_histogram)
    actual_mean_row_synapses = np.average(row_length_histogram)

    # Check estimated maximum is greater or equal than actual maximum
    assert estimated_max_row_synapses >= actual_max_row_synapses

    # Check estimated maximum isn't TOO big
    assert estimated_max_row_synapses <= actual_max_row_synapses * 1.25

    # Check estimated mean is approximately correct
    assert estimated_mean_row_synapses <= (1.25 * actual_mean_row_synapses)
    assert estimated_mean_row_synapses >= (0.75 * actual_mean_row_synapses)

@pytest.mark.parametrize("delay_dist", [RandomDistribution("normal_clipped",
                                                           [1.5, 0.75, 0.1, 1000.0],
                                                           rng=native_rng)])
def test_connector_delay_dist(delay_dist):
    # Run network
    proj, proj_data = _run_network(1000, sim.AllToAllConnector(),
                                   sim.StaticSynapse(weight=0.0, delay=delay_dist),
                                   ["delay"], 0.1)

    # Generate some samples from original distribution
    samples = delay_dist.next(len(proj_data))
    samples = np.around(samples, decimals=1)

    # Run Kolmogorov-Smirnov test on data
    test = ks_2samp(proj_data[2], samples)
    assert test[1] > 0.05

@pytest.mark.parametrize("weight_dist",
                         [RandomDistribution("normal_clipped", [87.8e-3, 87.8e-3 * 0.1, 0.0, 1000.0], rng=native_rng),
                          RandomDistribution("uniform", low=0.0, high=0.01, rng=native_rng)])
def test_connector_weight_dist(weight_dist):
    # Run network
    proj, proj_data = _run_network(1000, sim.AllToAllConnector(),
                                   sim.StaticSynapse(weight=weight_dist, delay=1.0),
                                   ["weight"], 1.0)

    # Generate some samples from original distribution
    samples = weight_dist.next(len(proj_data))

    # Run Kolmogorov-Smirnov test on data
    test = ks_2samp(proj_data[2], samples)
    assert test[1] > 0.05
