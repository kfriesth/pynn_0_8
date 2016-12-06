# --------------------------------------------------------------------------
# Song, S., Miller, K. D., & Abbott, L. F. (2000).
# Competitive Hebbian learning through spike-timing-dependent synaptic plasticity.
# Nature Neuroscience, 3(9), 919-26. http://doi.org/10.1038/78829
# --------------------------------------------------------------------------

import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
from six import iteritems, iterkeys, itervalues

dt = 1.0
num_ex_synapses = 1000
num_neurons = 1
g_max = 0.01
duration = 300000

def simulate(sim, rng, setup_kwargs, scale_a):
    sim.setup(timestep=dt, **setup_kwargs)

    # Brian was performing synaptic input with ge*(Ee-vr)
    # Ee = 0 so infact Vr is being treated as an input resistance and therefore C = tau_m / v_rest = 10*10^-3 / 60*10^6 = 0.17*10^-9

    neuron = sim.Population(num_neurons,
                            sim.IF_curr_exp(v_rest=-74.0, v_reset=-60.0, v_thresh=-54.0,
                                            tau_syn_E=5.0, tau_syn_I=5.0, tau_m=10.0, cm=0.17))
    ex_poisson = sim.Population(num_ex_synapses, sim.SpikeSourcePoisson(rate=15.0))

    neuron.record("spikes")

    a_plus = 0.01
    a_minus = 1.05 * a_plus
    #if scale_a:
    #    a_plus *= g_max
    #    a_minus *= g_max

    # Create weight distribution
    weight_dist = RandomDistribution("uniform", low=0, high=g_max, rng=rng)

    # Plastic Connection between pre_pop and post_pop
    stdp_model = sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(tau_plus=20.0, tau_minus=20.00,
                                            A_plus=a_plus, A_minus=a_minus),
        weight_dependence=sim.MultiplicativeWeightDependence(w_min=0.0, w_max=g_max),
        weight=weight_dist, delay=dt, dendritic_delay_fraction=1.0,
    )

    proj = sim.Projection(ex_poisson, neuron, sim.AllToAllConnector(),
                          stdp_model, receptor_type="excitatory")

    sim.run(duration)

    weights = np.asarray(proj.get("weight", format="list", with_address=False))
    data = neuron.get_data()

    # End simulation
    sim.end()

    # Return learned weights and data
    return weights, data

def simulate_spinnaker():
    import pynn_spinnaker as sim

    logger = logging.getLogger("pynn_spinnaker")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    rng = sim.NativeRNG(host_rng=NumpyRNG())

    return simulate(sim, rng, {"max_delay": dt}, True)

def simulate_nest():
    import pyNN.nest as sim
    rng = NumpyRNG()
    return simulate(sim, rng, {"spike_precision": "on_grid"}, False)

# Simulate network
weights, data = simulate_spinnaker()

print("Firing rate %fHz" % (float(len(data.segments[0].spiketrains[0])) / float(duration / 1000)))
figure, axis = plt.subplots()
axis.hist(weights / g_max, 20)
axis.set_xlabel("Normalised weight")
axis.set_ylabel("Number of synapses")
plt.show()