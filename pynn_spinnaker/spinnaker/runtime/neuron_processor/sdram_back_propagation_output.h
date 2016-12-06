#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/bit_field.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"

// Namespaces
using namespace Common;

//-----------------------------------------------------------------------------
// NeuronProcessor::SDRAMBackPropagationOutput
//-----------------------------------------------------------------------------
namespace NeuronProcessor
{
class SDRAMBackPropagationOutput
{
public:
  SDRAMBackPropagationOutput() : m_NumWords(0), m_SpikeBuffer(NULL), m_SDRAMBuffers{NULL, NULL}
  {
  }

  //--------------------------------------------------------------------------
  // Public API
  //--------------------------------------------------------------------------
  void TransferBuffer(unsigned int tick, uint tag)
  {
    // DMA spike buffer into correct SDRAM buffer for this timer tick
    if(IsEnabled())
    {
      spin1_dma_transfer(tag, m_SDRAMBuffers[tick % 2],
                         m_SpikeBuffer, DMA_WRITE,
                         m_NumWords * sizeof(uint32_t));
    }
  }

  void ClearBuffer()
  {
    // Zero spike buffer
    if(IsEnabled())
    {
      BitField::Clear(m_SpikeBuffer, m_NumWords);
    }
  }

  void RecordSpike(unsigned int neuron)
  {
    if(IsEnabled())
    {
      BitField::SetBit(m_SpikeBuffer, neuron);
    }
  }

  bool ReadSDRAMData(uint32_t *region, uint32_t, unsigned int numNeurons)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "SDRAMBackPropagationOutput::ReadSDRAMData");

    // If back propagation is enabled
    bool enabled = (*region++ != 0);
    if(enabled)
    {
      // Copy two SDRAM buffer pointers from region
      spin1_memcpy(m_SDRAMBuffers, region, 2 * sizeof(uint32_t*));

  #if LOG_LEVEL <= LOG_LEVEL_INFO
      for (uint32_t i = 0; i < 2; i++)
      {
        LOG_PRINT(LOG_LEVEL_INFO, "\tBuffer:%08x", m_SDRAMBuffers[i]);
      }
  #endif
      // Calculate number of words that are required to build a bitfield for ALL neurons
      m_NumWords = BitField::GetWordSize(numNeurons);
      LOG_PRINT(LOG_LEVEL_INFO, "\tBuffer words:%u", m_NumWords);

      // Allocate local spike buffer
      m_SpikeBuffer = (uint32_t*)spin1_malloc(m_NumWords * sizeof(uint32_t));
      if(m_SpikeBuffer == NULL)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate local spike buffer");
        return false;
      }

      // Clear newly allocated buffer
      ClearBuffer();
    }
    // Otherwise, zero num words
    else
    {
      m_NumWords = 0;
    }

    return true;
  }

  bool IsEnabled() const
  {
    return (m_NumWords > 0);
  }

private:
  //--------------------------------------------------------------------------
  // Members
  //--------------------------------------------------------------------------
  // How many words to write to SDRAM every time step
  unsigned int m_NumWords;

  // Bit-field of spikes from last timestep
  uint32_t *m_SpikeBuffer;

  // Addresses of SDRAM buffers
  uint32_t *m_SDRAMBuffers[2];
};
}