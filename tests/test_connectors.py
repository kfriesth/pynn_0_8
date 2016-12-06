# Import modules
import numpy as np
import pytest
import pynn_spinnaker as sim

# Import classes
from pynn_spinnaker.spinnaker.synapse_cluster import WeightRange
from pynn_spinnaker.spinnaker.utils import UnitStrideSlice

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("pre_size", [1000])
@pytest.mark.parametrize("post_size, post_slice",
                         [(5000, UnitStrideSlice(0, 1024)),
                          (5000, UnitStrideSlice(1024, 2048)),
                          (5000, UnitStrideSlice(3976, 5000))])
@pytest.mark.parametrize("connector",
                         [sim.AllToAllConnector(),
                          sim.OneToOneConnector(),
                          sim.FixedProbabilityConnector(0.1),
                          sim.FixedProbabilityConnector(0.5),
                          sim.FixedProbabilityConnector(1.0),
                          sim.FixedNumberPreConnector(10),
                          sim.FixedNumberPostConnector(100),
                          sim.FixedNumberPostConnector(1000),
                          sim.FixedTotalNumberConnector(500000)])
def test_estimate_max_row_synapses(pre_size, post_size, post_slice, connector):
    # Setup simulator
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=8.0, spinnaker_hostname="")

    # Create two populations
    pre = sim.Population(pre_size, sim.IF_curr_exp())
    post = sim.Population(post_size, sim.IF_curr_exp())

    # Connect the populations together
    proj = sim.Projection(pre, post, connector, sim.StaticSynapse())

    # Estimate maximum and mean row length
    estimated_max_row_synapses = proj._connector._estimate_max_row_synapses(
        UnitStrideSlice(0, pre_size), post_slice, pre_size, post_size)
    estimated_mean_row_synapses = proj._connector._estimate_mean_row_synapses(
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
    proj._simulator.state.num_processes = 5

    # Build projection
    proj._build(matrix_rows=sub_rows,
                weight_range=weight_range,
                directly_connect=False)

    # Reset number of processes
    proj._simulator.state.num_processes = 1

    # Calculate actual maximum and mean row length
    actual_max_row_synapses = max(len(s) for s in sub_rows)
    actual_mean_row_synapses = sum(len(s) for s in sub_rows) / float(len(sub_rows))

    # Check estimated maximum is greater or equal than actual maximum
    assert estimated_max_row_synapses >= actual_max_row_synapses

    # Check estimated maximum isn't TOO big
    assert estimated_max_row_synapses <= actual_max_row_synapses * 1.25

    # Check estimated mean is approximately correct
    assert estimated_mean_row_synapses <= (1.25 * actual_mean_row_synapses)
    assert estimated_mean_row_synapses >= (0.75 * actual_mean_row_synapses)
