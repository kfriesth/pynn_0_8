# Import modules
import mock
import numpy as np
import pytest
from pynn_spinnaker.spinnaker import utils
import tempfile

# Import classes
from collections import defaultdict
from pynn_spinnaker.spinnaker.regions import (ExtendedPlasticSynapticMatrix,
                                              KeyLookupBinarySearch,
                                              PlasticSynapticMatrix,
                                              StaticSynapticMatrix)
from pynn_spinnaker.spinnaker.spinnaker_population_config import SpinnakerPopulationConfig
from rig.bitfield import BitField

# Import globals
from pynn_spinnaker.spinnaker.neural_cluster import Vertex
from pynn_spinnaker.spinnaker.synapse_cluster import row_dtype

def _generate_random_matrix(pre_size, post_slice, row_length, multapse):
    assert row_length <= len(post_slice)

    # Loop through rows
    rows = []
    for i in range(pre_size):
        # Create numpy row
        row = np.empty(shape=row_length, dtype=row_dtype)

        # Fill fields with random data
        row["index"] = np.random.choice(np.arange(post_slice.start, post_slice.stop),
                                        row_length, replace=multapse)
        row["weight"] = np.random.random(row_length)
        row["delay"] = np.random.randint(1, 100, row_length)

        rows.append(row)
    return rows

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("pre_size, post_slice, row_length",
                         [(1000, utils.UnitStrideSlice(0, 1000), 100)])
@pytest.mark.parametrize("pre_vert_size", [500, 1000])
@pytest.mark.parametrize("synaptic_matrix_region", [
    StaticSynapticMatrix(mock.Mock(_max_dtcm_delay_slots=7,
                                   _signed_weight=False)),
    ExtendedPlasticSynapticMatrix(mock.Mock(_max_dtcm_delay_slots=7,
                                            _signed_weight=False,
                                            _pre_state_bytes=10,
                                            _synapse_trace_bytes=2)),
    PlasticSynapticMatrix(mock.Mock(_max_dtcm_delay_slots=7,
                                    _signed_weight=False,
                                    _pre_state_bytes=10))])
@pytest.mark.parametrize("sim_timestep_ms", [0.1, 1.0])
def test_matrix_process(pre_size, pre_vert_size, post_slice, row_length,
                        synaptic_matrix_region, sim_timestep_ms):
    # Fix the seed so the test is consistent
    np.random.seed(123456)

    # Create a mock pre_population connected with a random weight matrix
    pre_pop = mock.Mock(spinnaker_config=SpinnakerPopulationConfig())
    pre_pop_sub_rows = {pre_pop: _generate_random_matrix(pre_size, post_slice,
                                                         row_length, False)}

    # Create a 32-bit keyspace
    keyspace = BitField(32)
    keyspace.add_field("pop_index", tags=("routing", "transmission"))
    keyspace.add_field("vert_index", tags=("routing", "transmission"))
    keyspace.add_field("flush", length=1, start_at=10, tags="transmission")
    keyspace.add_field("neuron_id", length=10, start_at=0)

    # Split pre population amongst multiple incoming connections
    incoming_connections = defaultdict(list)
    incoming_connections[pre_pop] = [Vertex(keyspace, s, 0, i)
                                     for i, s in enumerate(utils.split_slice(pre_size, pre_vert_size))]

    # Finalise keyspace fields
    keyspace.assign_fields()

    # Create regions
    key_lookup_region = KeyLookupBinarySearch()

    # Partition matrices
    sub_matrix_props, sub_matrix_rows =\
        synaptic_matrix_region.partition_matrices(post_slice, pre_pop_sub_rows,
                                                  incoming_connections)

    # Check correct number of matrices have been generated
    assert len(sub_matrix_props) == len(incoming_connections[pre_pop])

    # Place matrices in memory
    matrix_placements = key_lookup_region.place_matrices(sub_matrix_props)

    # Calculate size of matrix region
    matrix_size = synaptic_matrix_region.sizeof(
        sub_matrix_props, sub_matrix_rows, matrix_placements, 16)

    # Mock up out postsynaptic synapse vertex
    post_s_vert = mock.Mock()
    post_s_vert.weight_fixed_point = 16
    post_s_vert.sub_matrix_props = sub_matrix_props
    post_s_vert.matrix_placements = matrix_placements
    post_s_vert.post_neuron_slice = post_slice

    # PyNN properties to read from weight matrix
    names = ["presynaptic_index", "postsynaptic_index", "weight", "delay"]

    # Write the items to file
    with tempfile.TemporaryFile() as fp:
        # Write synaptic matrix to file
        synaptic_matrix_region.write_subregion_to_file(
            fp, sub_matrix_props, sub_matrix_rows, matrix_placements, 16)

        # Check correct amount of data has been written back
        assert fp.tell() == matrix_size

        # Loop through our presynaptic vertices
        for pre_n_vert in incoming_connections[pre_pop]:
            # Read sub-matrix back
            synapses = np.hstack(synaptic_matrix_region.read_sub_matrix(
                pre_n_vert, post_s_vert, names, fp, sim_timestep_ms, False))

            # Loop through original rows we wrote
            orig_rows =\
                pre_pop_sub_rows[pre_pop][pre_n_vert.neuron_slice.python_slice]
            for i, orig_row in enumerate(orig_rows):
                # Get presynaptic index of row
                pre_index = i + pre_n_vert.neuron_slice.start

                # Select read back synapses that are in this row
                row_mask = (synapses["presynaptic_index"] == pre_index)
                row = synapses[row_mask]

                # Check lengths match
                assert len(row) == len(orig_row)

                # Sort both rows
                # **TODO** unstable on multapses
                orig_row_order = np.argsort(orig_row["index"])
                row_order = np.argsort(row["postsynaptic_index"])

                # Check rows match
                scaled_delay = np.multiply(orig_row[orig_row_order]["delay"],
                                           sim_timestep_ms, dtype=float)
                assert np.array_equal(orig_row[orig_row_order]["index"],
                                      row[row_order]["postsynaptic_index"])
                assert np.allclose(orig_row[orig_row_order]["weight"],
                                      row[row_order]["weight"], atol=0.0001)
                assert np.allclose(scaled_delay,
                                   row[row_order]["delay"], atol=0.0001)
