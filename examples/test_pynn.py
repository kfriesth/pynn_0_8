import itertools
import logging
import math
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from six import iteritems, iterkeys, itervalues

record_voltages = True
num_neurons = 10
duration = 100
dt = 1.0
nest = False
spinnaker = True
current = True
record_gsyn = not current
spinnaker_config = defaultdict(dict)

def simulate(sim, setup_kwargs):
    sim.setup(timestep=dt, **setup_kwargs)
    
    # Define neuron models
    neuron_a = (sim.IF_curr_exp(i_offset=2.0, tau_refrac=2.0) if current
                else sim.IF_cond_exp(i_offset=2.0, tau_refrac=2.0))
    neuron_b = (sim.IF_curr_exp(tau_refrac=2.0) if current
                else sim.IF_cond_exp(tau_refrac=2.0))

    pop_a = sim.Population(num_neurons, neuron_a, label="pop_a")
    pop_a.record("spikes")
    #spinnaker_config[pop_a] = { "profile_samples": 2000 }

    if record_voltages:
        pop_a.sample(1).record("v")

    if record_gsyn:
        pop_a.sample(1).record("gsyn_exc")

    pop_b = sim.Population(num_neurons, neuron_b,
                           label="pop_b")
    pop_b.record("spikes")
    #spinnaker_config[pop_b] = { "profile_samples": 2000 }
    
    if record_voltages:
        pop_b.sample(1).record("v")

    if record_gsyn:
        pop_b.sample(1).record("gsyn_exc")

    # Build list connector that sweeps delay space
    proj = sim.Projection(pop_a, pop_b, sim.OneToOneConnector(),
                          sim.StaticSynapse(weight=2.0 if current else 0.2, delay=dt),
                          receptor_type="excitatory")

    sim.run(duration)

    pop_a_data = pop_a.get_data()
    pop_b_data = pop_b.get_data()

    # End simulation
    sim.end()

    return pop_a_data, pop_b_data

def simulate_spinnaker():
    import pynn_spinnaker as sim

    logger = logging.getLogger("pynn_spinnaker")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    setup_kwargs = {"spinnaker_hostname": "192.168.1.1"}

    return simulate(sim, setup_kwargs)

def simulate_nest():
    import pyNN.nest as sim
    return simulate(sim, {"spike_precision": "on_grid"})


def plot_spiketrains(axis, segment, **kwargs):
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations["source_index"]
        axis.scatter(spiketrain, y, linewidths=0.0, s=4, **kwargs)

def plot_signal(axis, signal_array, index, **kwargs):
    axis.plot(signal_array.times, signal_array[:, index], **kwargs)

# Run desired simulators
data = {}
if nest:
    data["NEST"] = (simulate_nest(), { "color": "blue" })

if spinnaker:
    data["SpiNNaker"] = (simulate_spinnaker(), { "color": "red" })

legacy_kwargs = { "color": "green", "alpha": 0.5 }

num_axes = 2
if record_voltages:
    num_axes += 2
if record_gsyn:
    num_axes += 2

figure, axes = plt.subplots(num_axes, sharex=True)

axes[0].set_title("Population A spikes")
axes[1].set_title("Population B spikes")
axes[0].set_ylabel("Neuron ID")
axes[1].set_ylabel("Neuron ID")

# Loop through all simulator's output data
for l, d in iteritems(data):
    # Loop through recorded populations and plot spike trains
    for i, p in enumerate(d[0]):
        plot_spiketrains(axes[i], p.segments[0], **d[1])

# Loop through all simulator's output data
for d in itervalues(data):
    # Loop through recorded populations and plot spike trains
    axis_idx = 2
    for pop_idx, p in enumerate(d[0]):
        for sig_idx, a in enumerate(p.segments[0].analogsignalarrays):
            axes[axis_idx].set_title("Population %s %s" % ("A" if pop_idx == 0 else "B", a.name))
            axes[axis_idx].set_ylabel("%s / %s" % (a.name, a.units._dimensionality.string))

            plot_signal(axes[axis_idx], a, 0, **d[1])

            axis_idx += 1

axes[-1].set_xlim((0, duration))

# Build legend
legend_handles = [plt.Line2D([], [], **d[1]) for d in itervalues(data)]
legend_labels = list(iterkeys(data))

figure.legend(legend_handles, legend_labels)
plt.show()