#pragma once

// Rig CPP common includes
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/spinnaker.h"

// Namespaces
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// NeuronProcessor::SynapseModels::Exp
//-----------------------------------------------------------------------------
namespace NeuronProcessor
{
namespace SynapseModels
{
class Exp
{
public:
  //-----------------------------------------------------------------------------
  // MutableState
  //-----------------------------------------------------------------------------
  struct MutableState
  {
    // Excitatory input current
    S1615 m_ISynExc;

    // Inhibitory input current
    S1615 m_ISynInh;
  };

  //-----------------------------------------------------------------------------
  // ImmutableState
  //-----------------------------------------------------------------------------
  struct ImmutableState
  {
    // Excitatory decay constants
    U032 m_ExpTauSynExc;

    // Excitatory scale
    S1615 m_InitExc;

    // Inhibitory decay constant
    U032 m_ExpTauSynInh;

    // Inhibitory scale
    S1615 m_InitInh;
  };

  //-----------------------------------------------------------------------------
  // Static methods
  //-----------------------------------------------------------------------------
  static inline void ApplyInput(MutableState &mutableState, const ImmutableState &, S1615 input, unsigned int receptorType)
  {
    // Apply input to correct receptor
    if(receptorType == 0)
    {
      mutableState.m_ISynExc += input;
    }
    else
    {
      mutableState.m_ISynInh += input;
    }
  }

  static inline S1615 GetExcInput(const MutableState &mutableState, const ImmutableState &immutableState)
  {
    return MulS1615(mutableState.m_ISynExc, immutableState.m_InitExc);
  }

  static inline S1615 GetInhInput(const MutableState &mutableState, const ImmutableState &immutableState)
  {
    return MulS1615(mutableState.m_ISynInh, immutableState.m_InitInh);
  }

  static inline void Shape(MutableState &mutableState, const ImmutableState &immutableState)
  {
    // Decay both currents
    mutableState.m_ISynExc = MulS1615U032(mutableState.m_ISynExc, immutableState.m_ExpTauSynExc);
    mutableState.m_ISynInh = MulS1615U032(mutableState.m_ISynInh, immutableState.m_ExpTauSynInh);
  }

  static void Print(char *stream, const MutableState &mutableState, const ImmutableState &immutableState);
};
} // NeuronModels
} // NeuronProcessor