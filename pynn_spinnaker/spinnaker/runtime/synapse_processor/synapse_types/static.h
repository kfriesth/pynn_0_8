#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/log.h"

//-----------------------------------------------------------------------------
// SynapseProcessor::SynapseTypes::Static
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
namespace SynapseTypes
{
template<typename T, typename W, unsigned int D, unsigned int I>
class Static
{
public:
  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  // Three word for a synapse-count and delay extension data; and 1024 synapses
  static const unsigned int MaxRowWords = 1027;
    
  //-----------------------------------------------------------------------------
  // Public methods
  //-----------------------------------------------------------------------------
  template<typename F, typename E, typename R>
  bool ProcessRow(uint tick, uint32_t (&dmaBuffer)[MaxRowWords], uint32_t *, bool flush,
                  F applyInputFunction, E addDelayRowFunction, R)
  {
    // Flushes are irrelevant to static synapses
    // **NOTE** ideally they shouldn't be routed to these cores
    if(flush)
    {
      return true;
    }

    LOG_PRINT(LOG_LEVEL_TRACE, "\tProcessing static row with %u synapses",
              dmaBuffer[0]);

    // If this row has a delay extension, call function to add it
    // **NOTE** static rows have no need for flush so
    // set it to false so codepath can be optimised away
    if(dmaBuffer[1] != 0)
    {
      addDelayRowFunction(dmaBuffer[1] + tick, dmaBuffer[2], false);
    }

    register T *synapticWords = (T*)&dmaBuffer[3];
    register uint32_t count = dmaBuffer[0];
    for(; count > 0; count--)
    {
      // Get the next 32 bit word from the synaptic_row
      // (should autoincrement pointer in single instruction)
      T synapticWord = *synapticWords++;
       
      // Add weight to ring-buffer
      applyInputFunction(GetDelay(synapticWord) + tick, 
        GetIndex(synapticWord), GetWeight(synapticWord));
    }
    
    return true;
  }
  
  void AddPostSynapticSpike(uint, unsigned int)
  {
  }

  unsigned int GetRowWords(unsigned int rowSynapses) const
  {
    // Three header word and a synapse
    return 3 + ((rowSynapses * sizeof(T)) / 4);
  }

  bool ReadSDRAMData(uint32_t*, uint32_t, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "SynapseTypes::Static::ReadSDRAMData");
    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Constants
  //-----------------------------------------------------------------------------
  static const T DelayMask = ((1 << D) - 1);
  static const T IndexMask = ((1 << I) - 1);
  
  //-----------------------------------------------------------------------------
  // Private static methods
  //-----------------------------------------------------------------------------
  static T GetIndex(T word){ return (word & IndexMask); }
  static T GetDelay(T word){ return ((word >> I) & DelayMask); }
  static W GetWeight(T word){ return (W)(word >> (D + I)); }
};
} // SynapseTypes
} // SynapseProcessor