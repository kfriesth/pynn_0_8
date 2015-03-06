import numpy
from pyNN import common
from pyNN.standardmodels import StandardCellType
from pyNN.parameters import ParameterSpace, simplify
from . import simulator
from .recording import Recorder
from rig.neural_population import NeuralPopulation


class Assembly(common.Assembly):
    _simulator = simulator


class PopulationView(common.PopulationView):
    _assembly_class = Assembly
    _simulator = simulator

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            value = self.parent._parameters[name]
            if isinstance(value, numpy.ndarray):
                value = value[self.mask]
            parameter_dict[name] = simplify(value)
        return ParameterSpace(parameter_dict, shape=(self.size,)) # or local size?

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        #ps = self.parent._get_parameters(*self.celltype.get_native_names())
        for name, value in parameter_space.items():
            self.parent._parameters[name][self.mask] = value.evaluate(simplify=True)
            #ps[name][self.mask] = value.evaluate(simplify=True)
        #ps.evaluate(simplify=True)
        #self.parent._parameters = ps.as_dict()

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)



class Population(common.Population):
    __doc__ = common.Population.__doc__
    _simulator = simulator
    _recorder_class = Recorder
    _assembly_class = Assembly
    
    def __init__(self, size, cellclass, cellparams=None, structure=None,
                 initial_values={}, label=None):
        __doc__ = common.Population.__doc__
        super(Population, self).__init__(size, cellclass, cellparams, structure, initial_values, label)
        
        # Create empty list to hold incoming projections
        self.incoming_projections = []
        
        # Add population to simulator
        self._simulator.state.populations.append(self)
    
    def build(self):
        print("BUILDING POPULATION")
        if isinstance(self.celltype, StandardCellType):
            parameter_space = self.celltype.native_parameters
        else:
            parameter_space = self.celltype.parameter_space
        parameter_space.shape = (self.size,)
        
        # Evaluate parameter space
        parameter_space.evaluate(simplify=False)
        
        # Build numpy record datatype for neuron region
        # **TODO** this probably doesn't need to be a string
        record_datatype = ",".join(zip(*self.celltype.neuron_region_map)[1])
        
        # Build a numpy record array large enough for all neurons
        parameter_records = numpy.empty(self.size, dtype=(record_datatype))
        for f, n in zip(parameter_records.dtype.names, self.celltype.neuron_region_map):
            # If this map entry has a constant value, 
            # Write it into field for all neurons
            if len(n) == 2:
                parameter_records[f][:] = n[0]
            # Otherwise
            else:
                assert len(n) == 3
                
                # Extract correctly named parameter
                parameter = parameter_space[n[0]]
                
                # Apply translation function to parameter and write into field
                parameter_records[f] = n[2](parameter)
        
        # **TODO** pick correct population class
        self._spinnaker_population = NeuralPopulation(self.celltype, parameter_records)
        
        # Build incoming projections
        # **NOTE** this will result to multiple calls to convergent_connect
        for i in self.incoming_projections:
            i.build()
            
        with open("blob.dat", "wb") as f:
            key = 0
            vertex_slice = slice(0, self.size)
            self._spinnaker_population.write_to_file(key, vertex_slice, f)
    
    def convergent_connect(self, projection, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        self._spinnaker_population.convergent_connect(projection, presynaptic_indices, 
                                                      postsynaptic_index,
                                                      **connection_parameters)
        
    def _create_cells(self):
        id_range = numpy.arange(simulator.state.id_counter,
                                simulator.state.id_counter + self.size)
        self.all_cells = numpy.array([simulator.ID(id) for id in id_range],
                                     dtype=simulator.ID)
        
        # In terms of MPI, all SpiNNaker neurons are local
        self._mask_local = numpy.ones((self.size,), bool)
        
        for id in self.all_cells:
            id.parent = self
        simulator.state.id_counter += self.size

    def _set_initial_value_array(self, variable, initial_values):
        pass

    def _get_view(self, selector, label=None):
        return PopulationView(self, selector, label)

    def _get_parameters(self, *names):
        """
        return a ParameterSpace containing native parameters
        """
        parameter_dict = {}
        for name in names:
            parameter_dict[name] = simplify(self._parameters[name])
        return ParameterSpace(parameter_dict, shape=(self.local_size,))

    def _set_parameters(self, parameter_space):
        """parameter_space should contain native parameters"""
        #ps = self._get_parameters(*self.celltype.get_native_names())
        #ps.update(**parameter_space)
        #ps.evaluate(simplify=True)
        #self._parameters = ps.as_dict()
        parameter_space.evaluate(simplify=False, mask=self._mask_local)
        for name, value in parameter_space.items():
            self._parameters[name] = value
