#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"

// Common includes
#include "../common/row_offset_length.h"

//-----------------------------------------------------------------------------
// SynapseProcessor::DelayBufferBase
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
template<unsigned int S>
class DelayBufferBase
{
public:
  //-----------------------------------------------------------------------------
  // Typedefines
  //-----------------------------------------------------------------------------
  typedef RowOffsetLength<S> R;

  DelayBufferBase() : m_DelayMask(0), m_BufferSize(0), m_FetchTick(UINT32_MAX),
    m_SDRAMRowBuffers(NULL), m_RowCount(NULL), m_DMABuffer(NULL)
  {
  }

  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *region, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "DelayBuffer::ReadSDRAMData");

    // Read number of delay slots and use to calculate mask
    const unsigned int numDelaySlots = (unsigned int)region[0];
    m_BufferSize = (unsigned int)region[1];
    m_DelayMask = numDelaySlots - 1;
    LOG_PRINT(LOG_LEVEL_INFO, "\tNum delay slots:%u, Delay mask:%x, Buffer size:%u",
              numDelaySlots, m_DelayMask, m_BufferSize);

    // Allocate SDRAM delay buffer pointers
    m_SDRAMRowBuffers = (R**)spin1_malloc(sizeof(R*) * numDelaySlots);
    if(m_SDRAMRowBuffers == NULL)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate pointers to SDRAM delay buffer");
      return false;
    }

    // Allocate delay buffer counts
    m_RowCount = (uint8_t*)spin1_malloc(sizeof(uint8_t) * numDelaySlots);
    if(m_RowCount == NULL)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate row counts");
      return false;
    }

    // Allocate DMA buffer
    m_DMABuffer = (R*)spin1_malloc(sizeof(R) * m_BufferSize);
    if(m_DMABuffer == NULL)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate DMA buffer");
      return false;
    }

    // Loop through delay slots
    R *delayBuffer = (R*)&region[2];
    for(unsigned int d = 0; d < numDelaySlots; d++, delayBuffer += m_BufferSize)
    {
      // Point this delay buffer
      m_SDRAMRowBuffers[d] = delayBuffer;
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tDelay buffer %u at %08x",
                d, m_SDRAMRowBuffers[d] );

      // Zero entry counter
      m_RowCount[d] = 0;
    }

    return true;
  }

  bool AddRow(unsigned int tick, R rowOffsetLength, bool flush)
  {
    // Calculate index of buffer to use
    const unsigned int d = tick & m_DelayMask;

    // **TODO** stash flushness SOMEHOW in rowOffsetLength

    // If there is space in this delay buffer, add row
    // offset length to SDRAM delay buffer and increment counter
    if(m_RowCount[d] < m_BufferSize)
    {
      m_SDRAMRowBuffers[d][m_RowCount[d]++] = rowOffsetLength;
      return true;
    }
    // Otherwise
    else
    {
      return false;
    }
  }

  bool Fetch(unsigned int tick, uint tag)
  {
    // If a previous row fetch is ongoing
    unsigned int rowCount = GetRowCount(tick);
    if(m_FetchTick != UINT32_MAX)
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "Cannot fetch delay row - previous fetch still in progress");
      return false;
    }
    // Otherwise, if row count is invalid
    else if(rowCount > m_BufferSize)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Cannot read %u rows) into DMA buffer of size %u",
                rowCount, m_BufferSize);
      return false;
    }
    // Otherwise, if there are any rows in this tick's buffer
    else if(rowCount > 0)
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "DMA reading %u entry delay row buffer for tick %u",
                rowCount, tick);

      // Start DMA of current tick's delay buffer into DMA buffer
      m_FetchTick = tick;
      spin1_dma_transfer(tag,
                        m_SDRAMRowBuffers[tick & m_DelayMask], m_DMABuffer,
                        DMA_READ, rowCount * sizeof(R));
    }
    return true;
  }

  template<typename P>
  void ProcessDMABuffer(P processDelayRowFunction)
  {
    if(m_FetchTick == UINT32_MAX)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Cannot process delay row - it hasn't been fetched");
      rt_error(RTE_ABORT);
    }
    else
    {
      // Loop through delay rows and call processing function
      for(unsigned int i = 0; i < GetRowCount(m_FetchTick); i++)
      {
        processDelayRowFunction(m_DMABuffer[i]);
      }

      // Reset count for this delay slot
      m_RowCount[m_FetchTick & m_DelayMask] = 0;
      m_FetchTick = UINT32_MAX;
    }
  }

  unsigned int GetRowCount(unsigned int tick) const
  {
    return (unsigned int)m_RowCount[tick & m_DelayMask];
  }

private:
  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Mask to apply to ticks to get delay buffer index
  unsigned int m_DelayMask;

  // How big is each delay buffer
  unsigned int m_BufferSize;

  // Which tick was delay buffer fetched for
  uint32_t m_FetchTick;

  // Pointers to heads of row buffers for each slot
  R **m_SDRAMRowBuffers;

  // Number of rows present in each delay buffer
  uint8_t *m_RowCount;

  // Buffer used to hold current timesteps buffer
  R *m_DMABuffer;
};
} // SynapseProcessor
