#pragma once

// Rig CPP common includes
#include "rig_cpp_common/circular_buffer.h"
namespace SynapseProcessor
{
  typedef Common::CircularBuffer<uint32_t, 256> SpikeInputBuffer;
}

// Synapse processor includes
#include "../../../common/key_lookup_binary_search.h"
namespace SynapseProcessor
{
  typedef Common::KeyLookupBinarySearch<10> KeyLookup;
}


#include "../../synapse_types/static.h"
namespace SynapseProcessor
{
  typedef SynapseTypes::Static<uint32_t, uint16_t, 3, 10> SynapseType;
}


#include "../../ring_buffer.h"
namespace SynapseProcessor
{
  typedef RingBufferBase<uint32_t, 3, 10> RingBuffer;
}

#include "../../delay_buffer.h"
namespace SynapseProcessor
{
  typedef DelayBufferBase<10> DelayBuffer;
}
