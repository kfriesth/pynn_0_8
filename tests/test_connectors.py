# Import modules
import numpy as np
import scipy.stats as sp
import pytest
import pynn_spinnaker as sim

# Import classes
from pynn_spinnaker.spinnaker.synapse_cluster import WeightRange
from pynn_spinnaker.spinnaker.utils import UnitStrideSlice

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
# **NOTE** on host implementation of FixedTotalNumberConnector doesn't
# handle slices that aren't an exact fraction of post_size correctly
# hence the post_size of 1024*5 = 5120 - see issue #67
@pytest.mark.parametrize("pre_size", [1000, 64])
@pytest.mark.parametrize("post_size, post_slice",
                         [(5120, UnitStrideSlice(0, 1024)),
                          (5120, UnitStrideSlice(0, 256)),
                          (5120, UnitStrideSlice(1024, 2048)),
                          (5120, UnitStrideSlice(4096, 5120))])
@pytest.mark.parametrize("connector",
                         [sim.AllToAllConnector(),
                          sim.OneToOneConnector(),
                          sim.FixedProbabilityConnector(0.1),
                          sim.FixedProbabilityConnector(0.5),
                          sim.FixedProbabilityConnector(1.0),
                          sim.FixedNumberPreConnector(10),
                          sim.FixedNumberPostConnector(100),
                          sim.FixedNumberPostConnector(1000),
                          sim.FixedTotalNumberConnector(300000)])
def test_row_synapses_distribution(pre_size, post_size, post_slice, connector):
    # Setup simulator
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=8.0, spinnaker_hostname="")

    # Create two populations
    pre = sim.Population(pre_size, sim.IF_curr_exp())
    post = sim.Population(post_size, sim.IF_curr_exp())

    # Connect the populations together
    proj = sim.Projection(pre, post, connector, sim.StaticSynapse())

    # Distribution over synapses per row per post slice
    row_synapses_distribution = proj._connector._row_synapses_distribution(
        UnitStrideSlice(0, pre_size), post_slice, pre_size, post_size)

    # Create list of lists to contain matrix rows
    sub_rows = [[] for _ in range(pre_size)]

    # Create weight range
    weight_range = WeightRange(sim.StaticSynapse._signed_weight)

    # Local mask to select only the columns
    # corresponding to neurons in postsynaptic vertex
    proj.post._mask_local = np.zeros((post_size,), dtype=bool)
    proj.post._mask_local[post_slice.python_slice] = True

    # Some connectors also use num_processes for partial connector building
    proj._simulator.state.num_processes = int(np.ceil(float(post_size) /
                                                      float(len(post_slice))))

    # Build projection
    proj._build(matrix_rows=sub_rows,
                weight_range=weight_range,
                directly_connect=False)

    # Reset number of processes
    proj._simulator.state.num_processes = 1

    # Calculate actual maximum and mean row length
    synapse_counts = [len(s) for s in sub_rows]
    actual_frequencies = np.bincount(synapse_counts) / float(pre_size)
    actual_max_row_synapses = max(synapse_counts)
    actual_range = np.arange(actual_max_row_synapses + 1)
    expected_frequencies = row_synapses_distribution.pmf(actual_range)
    actual_frequencies = actual_frequencies[expected_frequencies > 0]
    expected_frequencies = expected_frequencies[expected_frequencies > 0]

    if expected_frequencies.size == 1:
        assert expected_frequencies[0] == 1.0
        assert actual_frequencies[0] == 1.0
    else:
        # Check that the synapses-per-sub-row histogram matches the distribution
        chisquare_statistic, p = sp.chisquare(actual_frequencies, expected_frequencies)

        assert p > 0.05

    estimated_max_row_synapses = row_synapses_distribution.ppf(0.9999**(1.0/pre_size))
    estimated_mean_row_synapses = row_synapses_distribution.mean()
    actual_mean_row_synapses = sum(synapse_counts) / float(pre_size)

    # Check estimated maximum is greater or equal than actual maximum
    assert estimated_max_row_synapses >= actual_max_row_synapses
    # Check estimated maximum isn't TOO big
    assert estimated_max_row_synapses <= actual_max_row_synapses * 3

    # Check estimated mean is approximately correct
    assert estimated_mean_row_synapses <= (1.25 * actual_mean_row_synapses)
    assert estimated_mean_row_synapses >= (0.75 * actual_mean_row_synapses)
