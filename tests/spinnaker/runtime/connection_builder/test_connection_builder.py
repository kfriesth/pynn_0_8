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
def _run_network(pre_size, post_size, connector, synapse, required_params, dt=1.0):
    # Setup simulator
    sim.setup(timestep=dt, stop_after_loader=True)

    # Create two populations
    pre = sim.Population(pre_size, sim.IF_curr_exp())
    post = sim.Population(post_size, sim.IF_curr_exp())

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
    proj, proj_data = _run_network(pop_size, pop_size, sim.AllToAllConnector(),
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
    proj, proj_data = _run_network(pop_size, pop_size, connector,
                                   sim.StaticSynapse(weight=0.0, delay=1.0),
                                   [])

    # Estimate maximum and mean row length
    pop_slice = UnitStrideSlice(0, pop_size)

    # Distribution over synapses per row per post slice
    row_synapses_distribution = connector._row_synapses_distribution(
        pop_slice, pop_slice, pop_size, pop_size)

    estimated_max_row_synapses = row_synapses_distribution.ppf(0.9999 ** (1.0 / pop_size))
    estimated_mean_row_synapses = row_synapses_distribution.mean()

    # Build row-length histogram
    row_length_histogram = binned_statistic(proj_data[0], proj_data[1],
                                            statistic="count", bins=range(pop_size + 1))[0]

    # Computer max and mean
    actual_max_row_synapses = np.amax(row_length_histogram)
    actual_mean_row_synapses = np.average(row_length_histogram)

    # Check estimated maximum is greater or equal than actual maximum
    assert estimated_max_row_synapses >= actual_max_row_synapses

    # Check estimated maximum isn't TOO big
    assert estimated_max_row_synapses <= actual_max_row_synapses * 3

    # Check estimated mean is approximately correct
    assert estimated_mean_row_synapses <= (1.25 * actual_mean_row_synapses)
    assert estimated_mean_row_synapses >= (0.75 * actual_mean_row_synapses)

# Due to issue #65 large population sizes, without replacement stick in an infinite loop
#@pytest.mark.parametrize("pop_size", [200, 2000])
#@pytest.mark.parametrize("connection_proportion", [0.05, 0.2])
#@pytest.mark.parametrize("with_replacement", [True, False])
@pytest.mark.parametrize("pop_size, connection_proportion, with_replacement",
                         [(200, 0.05, True),
                          (200, 0.05, False),
                          (200, 0.2, True),
                          (200, 0.2, False),
                          (2000, 0.05, True),
                          pytest.mark.xfail(reason="Issue #65")((2000, 0.05, False)),
                          (2000, 0.2, True),
                          pytest.mark.xfail(reason="Issue #65")((2000, 0.2, False))])
def test_fixed_number_connector(pop_size, connection_proportion, with_replacement):

    num_connections = int(connection_proportion * pop_size**2)
    connector = sim.FixedTotalNumberConnector(num_connections,
                                              with_replacement=with_replacement,
                                              rng=native_rng)

    proj, proj_data = _run_network(pop_size, pop_size, connector,
                                   sim.StaticSynapse(weight=0.0, delay=1.0),
                                   [])

    # Estimate maximum and mean row length
    pop_slice = UnitStrideSlice(0, pop_size)

    # Distribution over synapses per row per post slice
    row_synapses_distribution = connector._row_synapses_distribution(
        pop_slice, pop_slice, pop_size, pop_size)

    estimated_max_row_synapses = row_synapses_distribution.ppf(0.9999 ** (1.0 / pop_size))
    estimated_mean_row_synapses = row_synapses_distribution.mean()

    # Build row-length histogram
    row_length_histogram = binned_statistic(proj_data[0], proj_data[1],
                                            statistic="count", bins=range(pop_size + 1))[0]

    assert int(row_length_histogram.sum()) == num_connections

    # Computer max and mean
    actual_max_row_synapses = np.amax(row_length_histogram)
    actual_mean_row_synapses = np.average(row_length_histogram)

    # Check estimated maximum is greater or equal than actual maximum
    assert estimated_max_row_synapses >= actual_max_row_synapses

    # Check estimated maximum isn't TOO big
    assert estimated_max_row_synapses <= actual_max_row_synapses * 3.0

    # Check estimated mean is approximately correct
    assert estimated_mean_row_synapses <= (1.25 * actual_mean_row_synapses)
    assert estimated_mean_row_synapses >= (0.75 * actual_mean_row_synapses)


@pytest.mark.parametrize("delay_dist_name_params", [("normal_clipped", [1.5, 0.75, 0.1, 1000.0])])
def test_connector_delay_dist(delay_dist_name_params):
    # Build PyNN distribution object
    delay_dist = RandomDistribution(delay_dist_name_params[0],
                                    delay_dist_name_params[1],
                                    rng=native_rng)
    # Run network
    proj, proj_data = _run_network(1000, 1000, sim.AllToAllConnector(),
                                   sim.StaticSynapse(weight=0.0, delay=delay_dist),
                                   ["delay"], 0.1)

    # Generate some samples from original distribution
    samples = delay_dist.next(len(proj_data))
    samples = np.around(samples, decimals=1)

    # Run Kolmogorov-Smirnov test on data
    test = ks_2samp(proj_data[2], samples)
    assert test[1] > 0.05

@pytest.mark.parametrize("weight_dist_name_params",
                         [("normal_clipped", [0.277647978563, 0.0277647978563, 0.0, 1000.0], False),
                          ("normal_clipped", [-1.11059191425, 0.111059191425, -1000.0, 0.0], True),
                          ("uniform", [0.0, 0.01], False)])
def test_connector_weight_dist(weight_dist_name_params):
    # Build PyNN distribution object
    weight_dist = RandomDistribution(weight_dist_name_params[0],
                                     weight_dist_name_params[1],
                                     rng=native_rng)

    # Run network
    proj, proj_data = _run_network(1000, 100, sim.AllToAllConnector(),
                                   sim.StaticSynapse(weight=weight_dist, delay=1.0),
                                   ["weight"], 1.0)

    # **HACK** because of issue #60 
    if weight_dist_name_params[2]:
        proj_data[2] = np.multiply(proj_data[2], -1)

    # Generate some samples from original distribution
    samples = weight_dist.next(len(proj_data))

    # Run Kolmogorov-Smirnov test on data
    test = ks_2samp(proj_data[2], samples)
    assert test[1] > 0.05

@pytest.mark.parametrize("pop_size_1,pop_size_2,num_connections",
                         [(547,547,52147),(106,1439,3049),(294,583,0),(294,294,13506),(485,547,877),
                          (294,2068,0),(1439,2068,22711),(485,106,3181),(485,485,20407),(106,294,251),
                          (547,1439,13198),(2191,583,41025),(1439,106,1318),(583,485,18171),(583,2068,223203),
                          (583,106,1686),(583,583,50153),(294,106,0),(294,2191,0),(294,1439,108239),
                          (2191,2068,202507),(2068,2191,35029),(547,294,81),(2068,547,81009),(2191,2191,244773),
                          (2068,294,22547),(106,106,4284),(106,2068,0),(2068,1439,46796),(2191,1439,67252),
                          (106,485,24079),(106,583,0),(2191,547,99172),(547,106,128),(583,294,172),
                          (2068,2068,454932),(1439,485,14390),(1439,547,87964),(1439,1439,83697),(2068,485,106136),
                          (1439,2191,146211),(1439,294,28806),(485,2191,7144),(2068,583,174317),(547,485,1519),
                          (2191,294,2194),(547,2068,96695),(485,2068,32931),(547,2191,174096),(583,2191,7564),
                          (485,1439,41108),(583,547,927),(485,294,4005),(547,583,16889),(1439,583,3532),
                          (294,547,0),(294,485,0),(583,1439,5559),(2068,106,12356),(485,583,22197),
                          (2191,485,55078),(106,547,0),(106,2191,70),(2191,106,6048)])
def test_microcircuit_projections(pop_size_1, pop_size_2, num_connections):

    connector = sim.FixedTotalNumberConnector(num_connections,
                                              with_replacement=True,
                                              rng=native_rng)

    # Estimate maximum and mean row length
    pre_slice = UnitStrideSlice(0, pop_size_1)
    post_slice = UnitStrideSlice(0, pop_size_2)

    # Distribution over synapses per row per post slice
    row_synapses_distribution = connector._row_synapses_distribution(
        pre_slice, post_slice, pop_size_1, pop_size_2)

    estimated_max_row_synapses = row_synapses_distribution.ppf(0.9999 ** (1.0 / pop_size_1))
    estimated_mean_row_synapses = row_synapses_distribution.mean()

    proj, proj_data = _run_network(pop_size_1, pop_size_2, connector,
                                   sim.StaticSynapse(weight=0.0, delay=0.1),
                                   [], 0.1)
    
    if num_connections == 0:
        assert len(proj_data) == 0
    else:
        # Build row-length histogram
        row_length_histogram = binned_statistic(proj_data[0], proj_data[1],
                                                statistic="count", bins=range(pop_size_1 + 1))[0]

        # Test that the number of connections is correct
        assert int(row_length_histogram.sum()) == num_connections

        # Computer max and mean
        actual_max_row_synapses = np.amax(row_length_histogram)
        actual_mean_row_synapses = np.average(row_length_histogram)

        # Check estimated maximum is greater or equal than actual maximum
        assert estimated_max_row_synapses >= actual_max_row_synapses

        # Check estimated maximum isn't TOO big
        assert estimated_max_row_synapses <= actual_max_row_synapses * 3.0

        # Check estimated mean is approximately correct
        assert estimated_mean_row_synapses <= (1.25 * actual_mean_row_synapses)
        assert estimated_mean_row_synapses >= (0.75 * actual_mean_row_synapses)