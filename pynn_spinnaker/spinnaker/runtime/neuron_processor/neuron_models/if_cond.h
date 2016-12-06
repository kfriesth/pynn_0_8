#pragma once

// Rig CPP common includes
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/log.h"

// Namespaces
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// NeuronModels
//-----------------------------------------------------------------------------
namespace NeuronProcessor
{
namespace NeuronModels
{
//-----------------------------------------------------------------------------
// IFCond
//-----------------------------------------------------------------------------
class IFCond
{
public:
  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  enum RecordingChannel
  {
    RecordingChannelV,
    RecordingChannelGSynExc,
    RecordingChannelGSynInh,
    RecordingChannelMax,
  };

  //-----------------------------------------------------------------------------
  // MutableState
  //-----------------------------------------------------------------------------
  struct MutableState
  {
    // Membrane voltage [mV]
    S1615 m_V_Membrane;

    // Countdown to end of next refractory period [machine timesteps]
    int32_t m_RefractoryTimer;
  };

  //-----------------------------------------------------------------------------
  // ImmutableState
  //-----------------------------------------------------------------------------
  struct ImmutableState
  {
    // Membrane voltage threshold at which neuron spikes [mV]
    S1615 m_V_Threshold;

    // Post-spike reset membrane voltage [mV]
    S1615 m_V_Reset;

    // Membrane resting voltage [mV]
    S1615 m_V_Rest;

    // Excitatory reversal voltage [mV]
    S1615 m_V_RevExc;

    // Inhibitory reversal voltage [mV]
    S1615 m_V_RevInh;

    // Offset current [nA] but take care because actually 'per timestep charge'
    S1615 m_I_Offset;

    // Membrane resistance [MegaOhm]
    S1615 m_R_Membrane;

    // 'Fixed' computation parameter - time constant multiplier for
    // Closed-form solution
    // exp( -(machine time step in ms)/(R * C) ) [.]
    S1615 m_ExpTC;

    // Refractory time of neuron [machine timesteps]
    int32_t m_T_Refractory;
  };

  //-----------------------------------------------------------------------------
  // Static methods
  //-----------------------------------------------------------------------------
  static inline bool Update(MutableState &mutableState, const ImmutableState &immutableState,
                            S1615 excInput, S1615 inhInput, S1615 extCurrent)
  {
    // If outside of the refractory period
    if (mutableState.m_RefractoryTimer <= 0)
    {
      // Get the input in nA
      S1615 inputThisTimestep = MulS1615(excInput, immutableState.m_V_RevExc - mutableState.m_V_Membrane) +
        MulS1615(inhInput, immutableState.m_V_RevInh - mutableState.m_V_Membrane) +
        extCurrent + immutableState.m_I_Offset;

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tInput this timestep:%.4knA", inputThisTimestep);

      // Convert input from current to voltage
      S1615 alpha = MulS1615(inputThisTimestep, immutableState.m_R_Membrane) + immutableState.m_V_Rest;

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tAlpha:%.4kmV", alpha);

      // Perform closed form update
      mutableState.m_V_Membrane = alpha - MulS1615(immutableState.m_ExpTC,
                                                   alpha - mutableState.m_V_Membrane);

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tMembrane voltage:%.4knA", mutableState.m_V_Membrane);

      // Neuron spikes if membrane voltage has crossed threshold
      if (mutableState.m_V_Membrane >= immutableState.m_V_Threshold)
      {
        // Reset membrane voltage
        mutableState.m_V_Membrane = immutableState.m_V_Reset;

        // Reset refractory timer
        mutableState.m_RefractoryTimer = immutableState.m_T_Refractory;

        return true;
      }
    }
    // Otherwise, count down refractory timer
    else
    {
      mutableState.m_RefractoryTimer--;
    }

    return false;
  }

  static S1615 GetRecordable(RecordingChannel c,
                             const MutableState &mutableState, const ImmutableState &,
                             S1615 excInput, S1615 inhInput, S1615)
  {
    switch(c)
    {
      case RecordingChannelV:
        return mutableState.m_V_Membrane;

      case RecordingChannelGSynExc:
        return excInput;

      case RecordingChannelGSynInh:
        return inhInput;

      default:
        LOG_PRINT(LOG_LEVEL_WARN, "Attempting to get data from non-existant recording channel %u", c);
        return 0;
    }
  }

  static void Print(char *stream, const MutableState &mutableState, const ImmutableState &immutableState);
};
};  // namespace NeuronModels
};  // namespace NeuronProcessor