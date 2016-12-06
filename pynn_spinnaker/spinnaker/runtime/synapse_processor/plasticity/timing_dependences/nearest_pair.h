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
// SynapseProcessor::Plasticity::TimingDependences::NearestPair
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
namespace Plasticity
{
namespace TimingDependences
{
template<unsigned int TauPlusLUTNumEntries, unsigned int TauPlusLUTShift,
         unsigned int TauMinusLUTNumEntries, unsigned int TauMinusLUTShift>
class NearestPair
{
public:
  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  template<typename D, typename P>
  void ApplyPreSpike(D applyDepression, P,
                     uint32_t time, uint32_t, uint32_t lastPostTime)
  {
    // Get time of event relative to last post-synaptic event
    uint32_t elapsedTicksSinceLastPost = time - lastPostTime;
    if (elapsedTicksSinceLastPost > 0)
    {
        S2011 decayedPostTrace = m_TauMinusLUT.Get(elapsedTicksSinceLastPost);

        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tElapsed ticks since last post:%u, decayed post trace=%d",
                  elapsedTicksSinceLastPost, decayedPostTrace);

        // Apply depression
        applyDepression(decayedPostTrace);
    }
  }

  template<typename D, typename P>
  void ApplyPostSpike(D, P applyPotentiation,
                      uint32_t time, uint32_t lastPreTime, uint32_t)
  {
    // Get time of event relative to last pre-synaptic event
    uint32_t elapsedTicksSinceLastPre = time - lastPreTime;
    if (elapsedTicksSinceLastPre > 0)
    {
        S2011 decayedPreTrace = m_TauPlusLUT.Get(elapsedTicksSinceLastPre);

        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tElapsed ticks since last pre:%u, decayed pre trace=%d",
                  elapsedTicksSinceLastPre, decayedPreTrace);

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