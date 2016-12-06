#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/bit_field.h"
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/utils.h"

// Namespaces
using namespace Common;
using namespace Common::FixedPointNumber;
using namespace Common::Utils;

//-----------------------------------------------------------------------------
// NeuronProcessor::AnalogueRecording
//-----------------------------------------------------------------------------
namespace NeuronProcessor
{
class AnalogueRecording
{
public:
  AnalogueRecording() : m_IndicesToRecord(NULL), m_SamplingIntervalTick(0),
    m_TicksUntilRecord(0), m_RecordSDRAM(NULL)  {}

  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *region, uint32_t, unsigned int numNeurons)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tAnalogueRecording::ReadSDRAMData");

    // Read sampling interval from region
    m_SamplingIntervalTick = *region++;
    LOG_PRINT(LOG_LEVEL_INFO, "\t\tSampling interval:%u (ticks)",
              m_SamplingIntervalTick);

    // Calculate number of words that are required to build a bitfield for ALL neurons
    unsigned int numWords = BitField::GetWordSize(numNeurons);
    LOG_PRINT(LOG_LEVEL_INFO, "\t\tNum words per population:%u", numWords);

    // Copy indices to record
    if(!AllocateCopyStructArray(numWords, region, m_IndicesToRecord))
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate indices to record array");
      return false;
    }

#if LOG_LEVEL <= LOG_LEVEL_TRACE
    BitField::PrintBits(IO_BUF, m_IndicesToRecord, numWords);
    io_printf(IO_BUF, "\n");
#endif

    // Cache pointer of subsequent data
    m_RecordSDRAM = (S1615*)region;
    LOG_PRINT(LOG_LEVEL_INFO, "\t\tRecording starting at %08x", m_RecordSDRAM);

    return true;
  }

  void RecordValue(unsigned int neuron, S1615 value)
  {
    // If we should record this neuron this tick
    if(m_TicksUntilRecord == 0 && BitField::TestBit(m_IndicesToRecord, neuron))
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tRecording neuron:%u, value:%k",
                neuron,  value);

      // Write value to SDRAM
      *m_RecordSDRAM++ = value;
    }
  }


  void EndTick()
  {
    // If we've been recording this tick, reset
    // ticks until record to sampling interval
    if(m_TicksUntilRecord == 0)
    {
      m_TicksUntilRecord = m_SamplingIntervalTick;
    }

    // Decrement counter
    m_TicksUntilRecord--;
  }

private:
  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Bit field specifying which neurons to record
  uint32_t *m_IndicesToRecord;

  // How often should we record
  uint32_t m_SamplingIntervalTick;

  // How many ticks until we should record next sample
  uint32_t m_TicksUntilRecord;

  // Pointer to SDRAM to write next value to
  S1615 *m_RecordSDRAM;
};
}