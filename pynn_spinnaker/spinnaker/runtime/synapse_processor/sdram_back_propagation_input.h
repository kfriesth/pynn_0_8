#pragma once

// Standard includes
#include <algorithm>
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/bit_field.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/utils.h"

// Namespaces
using namespace Common;
using namespace Common::Utils;

//-----------------------------------------------------------------------------
// SynapseProcessor::SDRAMBackPropagationInput
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
class SDRAMBackPropagationInput
{
public:
  SDRAMBackPropagationInput() : m_InputBuffers(NULL), m_NumInputBuffers(0), m_DMABuffer(NULL)
  {
  }

  //--------------------------------------------------------------------------
  // Public API
  //--------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *region, uint32_t, unsigned int numNeurons)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "SDRAMBackPropagationInput::ReadSDRAMData");

    // Read base address and num lookup entries from 1st 2 words
    m_NumInputBuffers = *region++;
    LOG_PRINT(LOG_LEVEL_INFO, "\tNum input buffers:%u", m_NumInputBuffers);

    // Copy key lookup entries
    if(!AllocateCopyStructArray(m_NumInputBuffers, region, m_InputBuffers))
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate key lookup array");
      return false;
    }

    // If we have any back propagation buffers
    if(m_NumInputBuffers > 0)
    {
      // Loop through buffers
      uint32_t maxBufferWords = 0;
      uint32_t totalNeurons = 0;
      for(unsigned int i = 0; i < m_NumInputBuffers; i++)
      {
        const auto &inputBuffer = m_InputBuffers[i];

        // Update maximum required buffer size
        maxBufferWords = std::max(maxBufferWords, inputBuffer.m_BufferWords);

        // Add number of neurons to read from buffer to total
        totalNeurons += (inputBuffer.m_EndNeuronBit - inputBuffer.m_StartNeuronBit);

        LOG_PRINT(LOG_LEVEL_INFO, "\t\tEntry:%u, Buffers:{%08x, %08x}, Buffer words:%u, Start neuron bit:%u, End neuron bit:%u",
          i, inputBuffer.m_Buffers[0], inputBuffer.m_Buffers[1], inputBuffer.m_BufferWords,
          inputBuffer.m_StartNeuronBit, inputBuffer.m_EndNeuronBit);
      }

      // If back propagation input isn't provided for all neurons
      if(totalNeurons != numNeurons)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "SDRAM back propogation buffers only provide back propogation for %u/%u neurons",
                  totalNeurons, numNeurons);
        return false;
      }

      // Allocate DMA buffer
      m_DMABuffer = (uint32_t*)spin1_malloc(sizeof(uint32_t) * maxBufferWords);
      if(m_DMABuffer == NULL)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate DMA buffer");
        return false;
      }
    }

    return true;
  }

  bool Fetch(unsigned int inputBufferIndex, unsigned int tick, uint tag) const
  {
    if(inputBufferIndex < m_NumInputBuffers)
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\tStarting DMA of back propagation buffer index:%u (%u)",
                inputBufferIndex, (tick + 1) % 2);

      // Get DMA buffer
      const auto &inputBuffer = m_InputBuffers[inputBufferIndex];
      const uint32_t *dmaBuffer = inputBuffer.m_Buffers[(tick + 1) % 2];

      // Start DMA of bitfield into local memory
      spin1_dma_transfer(tag, const_cast<uint32_t*>(dmaBuffer), m_DMABuffer,
                        DMA_READ, inputBuffer.m_BufferWords * sizeof(uint32_t));
      return false;
    }
    // Otherwise, all inputs are gathered - update neurons
    else
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\tAll back propagation buffers processed");
      return true;
    }
  }

  template<typename F>
  unsigned int Process(unsigned int inputBufferIndex, F processSpikeFunction) const
  {
    const auto &inputBuffer = m_InputBuffers[inputBufferIndex];

    LOG_PRINT(LOG_LEVEL_TRACE, "\tApplying back propagation buffer:%u", inputBufferIndex);

#if LOG_LEVEL <= LOG_LEVEL_TRACE
    BitField::PrintBits(IO_BUF, m_DMABuffer, inputBuffer.m_BufferWords);
    io_printf(IO_BUF, "\n");
#endif

    BitField::ForEach(m_DMABuffer,
                      inputBuffer.m_StartNeuronBit, inputBuffer.m_EndNeuronBit,
                      processSpikeFunction);

    // Return number of neurons processed
    return (inputBuffer.m_EndNeuronBit - inputBuffer.m_StartNeuronBit);
  }

private:
  //-----------------------------------------------------------------------------
  // Buffer
  //-----------------------------------------------------------------------------
  struct Buffer
  {
    const uint32_t *m_Buffers[2];
    uint32_t m_BufferWords;

    uint32_t m_StartNeuronBit;
    uint32_t m_EndNeuronBit;
  };

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Buffers into which neuron cores write spike vectors
  Buffer *m_InputBuffers;
  unsigned int m_NumInputBuffers;

  // DMA buffer into which we read data
  uint32_t *m_DMABuffer;
};
}