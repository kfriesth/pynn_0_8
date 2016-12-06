#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/exp_decay_lut.h"
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/log.h"

// Namespaces
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// SynapseProcessor::Plasticity::TimingDependences::Pair
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
namespace Plasticity
{
namespace TimingDependences
{
template<unsigned int TauPlusLUTNumEntries, unsigned int TauPlusLUTShift,
         unsigned int TauMinusLUTNumEntries, unsigned int TauMinusLUTShift>
class Pair
{
public:
  //-----------------------------------------------------------------------------
  // Typedefines
  //-----------------------------------------------------------------------------
  typedef int16_t PostTrace;
  typedef int16_t PreTrace;

  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  PostTrace UpdatePostTrace(uint32_t tick, PostTrace lastTrace, uint32_t lastTick) const
  {
    // Get time since last spike
    uint32_t elapsedTicks = tick - lastTick;

    // Decay previous trace
    int32_t newTrace = Mul16S2011(lastTrace, m_TauMinusLUT.Get(elapsedTicks));

    // Add energy caused by new spike to trace
    newTrace += S2011One;

    LOG_PRINT(LOG_LEVEL_TRACE, "\tElapsed ticks:%d, New trace:%d",
              elapsedTicks, newTrace);

    // Return new trace_value
    return (PostTrace)newTrace;
  }

  PreTrace UpdatePreTrace(uint32_t tick, PreTrace lastTrace, uint32_t lastTick) const
  {
    // Get time since last spike
    uint32_t elapsedTicks = tick - lastTick;

    // Decay previous trace
    int32_t newTrace = Mul16S2011(lastTrace, m_TauPlusLUT.Get(elapsedTicks));

    // Add energy caused by new spike to trace
    newTrace += S2011One;

    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tElapsed ticks:%d, New trace:%d",
              elapsedTicks, newTrace);

    // Return new trace_value
    return (PreTrace)newTrace;
  }

  template<typename D, typename P>
  void ApplyPreSpike(D applyDepression, P,
                     uint32_t time, PreTrace,
                     uint32_t, PreTrace,
                     uint32_t lastPostTime, PostTrace lastPostTrace)
  {
    // Get time of event relative to last post-synaptic event
    uint32_t elapsedTicksSinceLastPost = time - lastPostTime;
    if (elapsedTicksSinceLastPost > 0)
    {
        S2011 decayedPostTrace = Mul16S2011(
          lastPostTrace, m_TauMinusLUT.Get(elapsedTicksSinceLastPost));

        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tElapsed ticks since last post:%u, last post trace:%d, decayed post trace=%d",
                  elapsedTicksSinceLastPost, lastPostTrace, decayedPostTrace);

        // Apply depression
        applyDepression(decayedPostTrace);
    }
  }

  template<typename D, typename P>
  void ApplyPostSpike(D, P applyPotentiation,
                      uint32_t time, PostTrace,
                      uint32_t lastPreTime, PreTrace lastPreTrace,
                      uint32_t, PostTrace)
  {
    // Get time of event relative to last pre-synaptic event
    uint32_t elapsedTicksSinceLastPre = time - lastPreTime;
    if (elapsedTicksSinceLastPre > 0)
    {
        S2011 decayedPreTrace = Mul16S2011(
          lastPreTrace, m_TauPlusLUT.Get(elapsedTicksSinceLastPre));

        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tElapsed ticks since last pre:%u, last pre trace:%d, decayed pre trace=%d",
                  elapsedTicksSinceLastPre, lastPreTrace, decayedPreTrace);

        // Apply potentiation
        applyPotentiation(decayedPreTrace);
    }
  }

  bool ReadSDRAMData(uint32_t *&region, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tPlasticity::TimingDependences::Pair::ReadSDRAMData");

    m_TauPlusLUT.ReadSDRAMData(region);
    m_TauMinusLUT.ReadSDRAMData(region);
    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  Common::ExpDecayLUT<TauPlusLUTNumEntries, TauPlusLUTShift> m_TauPlusLUT;
  Common::ExpDecayLUT<TauMinusLUTNumEntries, TauMinusLUTShift> m_TauMinusLUT;
};
} // TimingDependences
} // Plasticity
} // SynapseProcessor