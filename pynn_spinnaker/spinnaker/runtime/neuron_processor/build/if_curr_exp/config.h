#pragma once

// Model includes
#include "../../input_buffer.h"
#include "../../intrinsic_plasticity_models/stub.h"
#include "../../modular_neuron.h"
#include "../../neuron_dynamics_models/if.h"
#include "../../neuron_input_models/curr.h"
#include "../../neuron_threshold_models/constant.h"
#include "../../synapse_models/exp.h"

namespace NeuronProcessor
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef ModularNeuron<NeuronDynamicsModels::IF, NeuronInputModels::Curr,
                      NeuronThresholdModels::Constant> Neuron;
typedef SynapseModels::Exp Synapse;
typedef IntrinsicPlasticityModels::Stub IntrinsicPlasticity;

typedef InputBufferBase<uint32_t> InputBuffer;
};