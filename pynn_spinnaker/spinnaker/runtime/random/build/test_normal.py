# Import modules
import matplotlib.pyplot as plt
import numpy as np
import struct
from spalloc import Job

# Import classes
from rig.machine_control import MachineController
from rig.type_casts import NumpyFixToFloatConverter
from StringIO import StringIO

NUM_SAMPLES = 100000

with Job(1) as j:

    # Create machine controller, booting machine if necessary
    mc = MachineController(j.hostname)
    mc.boot()

    # Select first processor of ethernet-connected chip
    with mc(x=0, y=0, p=1):
        # Allocation enough SDRAM for header word and samples - tag as 1
        sdram_data_pointer = mc.sdram_alloc(4 + (NUM_SAMPLES * 4), tag=1)

        # Write number of samples to first word of SDRAM
        mc.write(sdram_data_pointer, struct.pack("i", NUM_SAMPLES))

        # Load application
        mc.load_application("random.aplx", {(0, 0): set([1])})

        # Wait for it to exit
        mc.wait_for_cores_to_reach_state("exit", 1)

        # Read fixed point normal samples from SDRAM
        normal_fixed = np.fromstring(mc.read(sdram_data_pointer + 4, 4 * NUM_SAMPLES),
                               dtype=np.int32)
        # Stop the application
        mc.send_signal("stop")

# Convert samples to floating point
normal_float = NumpyFixToFloatConverter(15)(normal_fixed)
#np.savetxt('normal_samples', normal_float)

# Plot histogram
fig, axis = plt.subplots()
axis.hist(normal_float, 100)
plt.show()
