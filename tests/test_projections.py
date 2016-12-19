# Import modules
import numpy as np
import pytest
import pynn_spinnaker as sim

# Import classes
from collections import defaultdict
from pyNN.random import RandomDistribution
from pynn_spinnaker.spinnaker.neural_cluster import Vertex
from pynn_spinnaker.spinnaker.regions import (KeyLookupBinarySearch,
                                              StaticSynapticMatrix)
from pynn_spinnaker.spinnaker.synapse_cluster import row_dtype
from pynn_spinnaker.spinnaker.synapse_cluster import WeightRange
from pynn_spinnaker.spinnaker.utils import UnitStrideSlice
from rig.bitfield import BitField

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("pre_size", [1000, 64])
@pytest.mark.parametrize("post_size, post_slice",
                         [(5000, UnitStrideSlice(0, 1024)),
                          (5000, UnitStrideSlice(0, 256)),
                          (5000, UnitStrideSlice(1024, 2048))])
@pytest.mark.parametrize("connector",
                         [sim.AllToAllConnector(),
                          sim.FixedProbabilityConnector(0.1),
                          sim.FixedTotalNumberConnector(300000)])
@pytest.mark.parametrize("delay", [0.1, 1.0,
                                   RandomDistribution("normal_clipped",
                                                      mu=1.5, sigma=0.75,
                                                      low=0.1, high=np.inf),
                                   RandomDistribution("normal_clipped",
                                                      mu=0.5, sigma=0.2,
                                                      low=0.1, high=0.7)])
def test_estimate_max_dims(pre_size, post_size, post_slice, delay, connector):
    # Setup simulator
    sim.setup(timestep=0.1, min_delay=1.0, max_delay=8.0)

    # Create two populations
    pre = sim.Population(pre_size, sim.IF_curr_exp())
    post = sim.Population(post_size, sim.IF_curr_exp())

    # Connect the populations together
    proj = sim.Projection(pre, post, connector, sim.StaticSynapse(delay=delay))

    # Create a 32-bit keyspace
    keyspace = BitField(32)
    keyspace.add_field("pop_index", tags=("routing", "transmission"))
    keyspace.add_field("vert_index", tags=("routing", "transmission"))
    keyspace.add_field("flush", length=1, start_at=10, tags="transmission")
    keyspace.add_field("neuron_id", length=10, start_at=0)

    # Split pre population amongst multiple incoming connections
    incoming_connections = defaultdict(list)
    incoming_connections[pre] = [Vertex(keyspace, UnitStrideSlice(0, pre_size), 0, 0)]

    # Finalise keyspace fields
    keyspace.assign_fields()

    # Create regions
    key_lookup_region = KeyLookupBinarySearch()
    synaptic_matrix_region = StaticSynapticMatrix(proj.synapse_type)

    # Estimate matrix dimensions
    max_cols, max_sub_rows, max_sub_row_synapses =\
        proj._estimate_max_dims(UnitStrideSlice(0, pre_size), post_slice)

    # From this calculate max words
    max_size_words = synaptic_matrix_region.estimate_matrix_words(
        pre_size, max_cols, max_sub_rows, max_sub_row_synapses)

    # Create list of lists to contain matrix rows
    sub_rows = [[] for _ in range(pre_size)]

    # Create weight range
    weight_range = WeightRange(sim.StaticSynapse._signed_weight)

    # Local mask to select only the columns
    # corresponding to neurons in postsynaptic vertex
    proj.post._mask_local = np.zeros((post_size,), dtype=bool)
    proj.post._mask_local[post_slice.python_slice] = True

    proj._simulator.state.num_processes = int(np.ceil(post_size / float(len(post_slice))))

    # Build projection
    proj._build(matrix_rows=sub_rows,
                weight_range=weight_range,
                directly_connect=False)

    proj._simulator.state.num_processes = 1

    # Convert rows to numpy and add to dictionary
    pre_pop_sub_rows = {pre: [np.asarray(r, dtype=row_dtype)
                              for r in sub_rows]}


    # Partition matrices
    sub_matrix_props, sub_matrix_rows =\
        synaptic_matrix_region.partition_matrices(post_slice, pre_pop_sub_rows,
                                                  incoming_connections)

    assert len(sub_matrix_props) == 1
    assert (sub_matrix_props[0].max_cols - 1) <= max_cols
    assert (sub_matrix_props[0].max_cols - 1) * 3 >= max_cols
    assert sub_matrix_props[0].size_words <= max_size_words
    assert sub_matrix_props[0].size_words * 3 >= max_size_words
