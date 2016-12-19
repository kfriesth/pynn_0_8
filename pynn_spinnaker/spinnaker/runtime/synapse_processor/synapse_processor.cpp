#include "synapse_processor.h"

// Standard includes
#include <climits>

// Rig CPP common includes
#include "rig_cpp_common/config.h"
#include "rig_cpp_common/disable_interrupts.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/profiler.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/statistics.h"

// Synapse processor includes
#include "sdram_back_propagation_input.h"

// Configuration include
#include "config.h"

// Namespaces
using namespace Common;
using namespace SynapseProcessor;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
enum DMATag
{
  DMATagRowRead,
  DMATagRowWrite,
  DMATagOutputWrite,
  DMATagDelayBufferRead,
  DMATagBackPropagationRead,
};

//----------------------------------------------------------------------------
// DMABuffer
//----------------------------------------------------------------------------
struct DMABuffer
{
  // Address of row in SDRAM
  uint32_t *m_SDRAMAddress;

  // Is this spike a flush message
  // (used to force an plasticity update)
  bool m_Flush;

  // Data read from SDRAM
  uint32_t m_Data[SynapseType::MaxRowWords];
};

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Config g_Config;
RingBuffer g_RingBuffer;
DelayBuffer g_DelayBuffer;
KeyLookup g_KeyLookup;
SpikeInputBuffer g_SpikeInputBuffer;
DelayRowBuffer g_DelayRowBuffer;
Statistics<StatWordMax> g_Statistics;
SynapseType g_Synapse;
SDRAMBackPropagationInput g_SDRAMBackPropagationInput;

uint32_t g_AppWords[AppWordMax];

uint32_t *g_OutputBuffers[2] = {NULL, NULL};

uint32_t *g_SynapticMatrixBaseAddress = NULL;

uint g_Tick = 0;

bool g_DMABusy = false;
DMABuffer g_DMABuffers[2];
unsigned int g_DMARowBufferIndex = 0;

unsigned int g_BackPropagationBufferBeingProcessed = UINT_MAX;
unsigned int g_BackPropagationBufferNeuronOffset = 0;

//-----------------------------------------------------------------------------
// Module inline functions
//-----------------------------------------------------------------------------
inline void DMASwapRowBuffers()
{
  g_DMARowBufferIndex ^= 1;
}
//-----------------------------------------------------------------------------
inline DMABuffer &DMACurrentRowBuffer()
{
  return g_DMABuffers[g_DMARowBufferIndex];
}
//-----------------------------------------------------------------------------
inline DMABuffer &DMANextRowBuffer()
{
  return g_DMABuffers[g_DMARowBufferIndex ^ 1];
}
//-----------------------------------------------------------------------------
inline void DMAStartRowFetchPipeline()
{
  // If we're not already processing synaptic dmas,
  // flag pipeline as busy and trigger a user event
  if(!g_DMABusy)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Triggering user event for new spike");

    if(spin1_trigger_user_event(0, 0))
    {
      g_DMABusy = true;
    }
    else
    {
      LOG_PRINT(LOG_LEVEL_WARN, "Could not trigger user event");
    }
  }
}

//-----------------------------------------------------------------------------
// Module functions
//-----------------------------------------------------------------------------
bool ReadSynapticMatrixRegion(uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadSynapticMatrixRegion");

  // Cache pointer to region as base address for synaptic matrices
  g_SynapticMatrixBaseAddress = region;

  LOG_PRINT(LOG_LEVEL_INFO, "\tSynaptic matrix base address:%08x",
            g_SynapticMatrixBaseAddress);

  return true;
}
//-----------------------------------------------------------------------------
bool ReadOutputBufferRegion(uint32_t *region, uint32_t)
{
  // Copy two output buffer pointers from region
  spin1_memcpy(g_OutputBuffers, region, 2 * sizeof(uint32_t*));

#if LOG_LEVEL <= LOG_LEVEL_INFO
  LOG_PRINT(LOG_LEVEL_INFO, "ReadOutputBufferRegion");
  for (uint32_t i = 0; i < 2; i++)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tIndex:%u, Address:%08x", i, g_OutputBuffers[i]);
  }
#endif

  return true;
}
//-----------------------------------------------------------------------------
bool ReadSDRAMData(uint32_t *baseAddress, uint32_t flags)
{
  // Verify data header
  if(!g_Config.VerifyHeader(baseAddress, flags))
  {
    return false;
  }

  // Read system region
  if(!g_Config.ReadSystemRegion(
    Config::GetRegionStart(baseAddress, RegionSystem),
    flags, AppWordMax, g_AppWords))
  {
    return false;
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tWeight fixed point:%u, Num post-neurons:%u",
      g_AppWords[AppWordWeightFixedPoint], g_AppWords[AppWordNumPostNeurons]);
  }

  // Read key lookup region
  if(!g_KeyLookup.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionKeyLookup),
    flags))
  {
    return false;
  }

  // Read synaptic matrix region
  if(!ReadSynapticMatrixRegion(
    Config::GetRegionStart(baseAddress, RegionSynapticMatrix),
    flags))
  {
    return false;
  }

   // Read plasticity region
  if(!g_Synapse.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionPlasticity),
    flags, g_AppWords[AppWordWeightFixedPoint]))
  {
    return false;
  }

  // Read output buffer region
  if(!ReadOutputBufferRegion(
    Config::GetRegionStart(baseAddress, RegionOutputBuffer),
    flags))
  {
    return false;
  }

  if(!g_DelayBuffer.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionDelayBuffer),
    flags))
  {
    return false;
  }

  if(!g_SDRAMBackPropagationInput.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionBackPropagationInput),
    flags, g_AppWords[AppWordNumPostNeurons]))
  {
    return false;
  }

  // Read profiler region
  if(!Profiler::ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionProfiler),
    flags))
  {
    return false;
  }

  if(!g_Statistics.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionStatistics),
    flags))
  {
    return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
void SetupNextDMARowRead()
{
  Profiler::TagDisableFIQ<ProfilerTagSetupNextDMARowRead> p;

  // If there are more delay rows than spikes left to be processed
  uint32_t key;
  DelayBuffer::R delayRow;
  if(g_DelayRowBuffer.GetAllocated() > g_SpikeInputBuffer.GetAllocated() &&
    g_DelayRowBuffer.Pop(delayRow))
  {
    // Convert number of synapses to words and get address from synaptic matrix base
    unsigned int delayRowWords = g_Synapse.GetRowWords(delayRow.GetNumSynapses());
    uint32_t *delayRowAddress = g_SynapticMatrixBaseAddress + delayRow.GetWordOffset();

    // Check row will fit in buffer
    if(delayRowWords > SynapseType::MaxRowWords)
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "%u word delay row requested at %08x",
                delayRowWords, delayRowAddress);
      rt_error(RTE_ABORT);
      return;
    }

    LOG_PRINT(LOG_LEVEL_TRACE, "Setting up DMA read for delay row synapse:%u, words:%u, address:%08x",
              delayRow.GetNumSynapses(), delayRowWords, delayRowAddress);

    // Store SDRAM address of row in buffer
    // so it can be written back if required
    DMANextRowBuffer().m_SDRAMAddress = delayRowAddress;
    DMANextRowBuffer().m_Flush = false;

    // Start a DMA transfer to fetch this synaptic row into next buffer
    g_Statistics[StatDelayRowRequested]++;
    spin1_dma_transfer(DMATagRowRead, delayRowAddress, DMANextRowBuffer().m_Data,
                      DMA_READ, delayRowWords * sizeof(uint32_t));

    // Flip DMA buffers and stop
    DMASwapRowBuffers();
    return;
  }
  // Otherwise, if there's another spike in the input buffer
  else if(g_SpikeInputBuffer.Pop(key))
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Setting up DMA read for spike %x", key);

    // Create lambda function to convert number of synapses to a row length in words
    auto getRowWordsLambda =
      [](unsigned int rowSynapses)
      {
        return g_Synapse.GetRowWords(rowSynapses);
      };

    // Use flush mask to determine if this key is a flush event
    bool flush = ((key & g_AppWords[AppWordFlushMask]) != 0);

    // Then clear these bit(s)
    key &= ~g_AppWords[AppWordFlushMask];

    // Decode key to get address and length of destination synaptic row
    unsigned int rowWords;
    uint32_t *rowAddress;
    if(g_KeyLookup.LookupRow(key, g_SynapticMatrixBaseAddress, getRowWordsLambda,
      rowWords, rowAddress))
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\tRow words:%u, Row address:%08x",
                rowWords, rowAddress);

      // Store SDRAM address of row in buffer
      // so it can be written back if required
      DMANextRowBuffer().m_SDRAMAddress = rowAddress;
      DMANextRowBuffer().m_Flush = flush;

      // Check row will fit in buffer
      if(rowWords > SynapseType::MaxRowWords)
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "Spike key %08x translated to %u word row at %08x",
                  key, rowWords, rowAddress);
        rt_error(RTE_ABORT);
        return;
      }

      // Start a DMA transfer to fetch this synaptic row into next buffer
      g_Statistics[StatRowRequested]++;
      spin1_dma_transfer(DMATagRowRead, rowAddress, DMANextRowBuffer().m_Data,
                        DMA_READ, rowWords * sizeof(uint32_t));

      // Flip DMA buffers and stop
      DMASwapRowBuffers();
      return;
    }
    else
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "Population associated with spike key %08x not found in key lookup", key);
      g_Statistics[StatWordKeyLookupFail]++;
    }
  }

  // Stop pipeline
  g_DMABusy = false;
}

//-----------------------------------------------------------------------------
// Event handler functions
//-----------------------------------------------------------------------------
void MCPacketReceived(uint key, uint)
{
  Profiler::Tag<ProfilerTagMCPacketReceived> p;

  LOG_PRINT(LOG_LEVEL_TRACE, "Received spike %x at tick %u, DMA Busy = %u",
            key, g_Tick, g_DMABusy);

  // If there was space to add spike to incoming
  // spike queue, start DMA row fetch pipeline
  if(g_SpikeInputBuffer.Push(key))
  {
    DMAStartRowFetchPipeline();
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Cannot add spike to input buffer");
    g_Statistics[StatWordInputBufferOverflows]++;
  }

}
//-----------------------------------------------------------------------------
void DMATransferDone(uint, uint tag)
{
  if(tag == DMATagRowRead)
  {
    // Create lambda function to add a weight to the ring-buffer
    auto addWeightLambda = 
      [](unsigned int tick, unsigned int index, uint32_t weight) 
      {
        LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tAdding weight %u to neuron %u for tick %u",
                  weight, index, tick);
        g_RingBuffer.AddWeight(tick, index, weight);
      };

    // Create lambda function to add a delay extension to the delay buffer
    auto addDelayRowLambda =
      [](unsigned int tick, uint32_t word, bool flush)
      {
        // **TODO** add flushness to word
        // **THINK** there IS at least one extra bit as plastic rows are limited to 512 synapses rather than 1024
        auto rowOffsetLength = DelayBuffer::R(word);
        LOG_PRINT(LOG_LEVEL_TRACE, "\t\tAdding delay extension row for tick %u, num synapses:%u, offset word:%u",
          tick, rowOffsetLength.GetNumSynapses(), rowOffsetLength.GetWordOffset());
        if(!g_DelayBuffer.AddRow(tick, rowOffsetLength, flush))
        {
          LOG_PRINT(LOG_LEVEL_TRACE, "Cannot schedule delay row read");
          g_Statistics[StatWordDelayBufferOverflows]++;
        }
      };

    // Create lambda function to write back row
    auto writeBackRowLambda =
      [](uint32_t *sdramAddress, uint32_t *localAddress, unsigned int numWords)
      {
        LOG_PRINT(LOG_LEVEL_TRACE, "\t\tWriting back %u words to SDRAM address:%08x",
                  numWords, sdramAddress);
        spin1_dma_transfer(DMATagRowWrite, sdramAddress, localAddress,
                           DMA_WRITE, numWords * sizeof(uint32_t));
      };

    // Cache the current row buffer as starting
    // a new row read will potentially flip buffers
    auto &dmaCurrentRowBuffer = DMACurrentRowBuffer();

    // Setup next row read so, ideally, data will be
    // available as soon as processing of current row completes
    SetupNextDMARowRead();

    // Process the current row, using this function to apply
    Profiler::WriteEntryDisableFIQ(Profiler::Enter | ProfilerTagProcessRow);
    if(!g_Synapse.ProcessRow(g_Tick, dmaCurrentRowBuffer.m_Data, dmaCurrentRowBuffer.m_SDRAMAddress, dmaCurrentRowBuffer.m_Flush,
      addWeightLambda, addDelayRowLambda, writeBackRowLambda))
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "Cannot process row");
      rt_error(RTE_ABORT);
    }
    Profiler::WriteEntryDisableFIQ(Profiler::Exit | ProfilerTagProcessRow);
  }
  else if(tag == DMATagOutputWrite)
  {
    // This timesteps output has been written from
    // the ring-buffer so we can now zero it
    g_RingBuffer.ClearOutputBuffer(g_Tick);

    // If there are no back propagation buffers to fetch, fetch delay buffer immediately
    g_BackPropagationBufferBeingProcessed = 0;
    g_BackPropagationBufferNeuronOffset = 0;
    if(g_SDRAMBackPropagationInput.Fetch(g_BackPropagationBufferBeingProcessed,
      g_Tick, DMATagBackPropagationRead))
    {
      // **NOTE** this will only cause a DMA if the buffer has any entries
      if(!g_DelayBuffer.Fetch(g_Tick, DMATagDelayBufferRead))
      {
        g_Statistics[StatWordDelayBufferFetchFail]++;
      }
    }
  }
  else if(tag == DMATagBackPropagationRead)
  {
    // Create lambda function to pass back propagating spike to synapse
    auto processSpikeLambda =
      [](unsigned int j)
      {
        g_Synapse.AddPostSynapticSpike(g_Tick - 1, g_BackPropagationBufferNeuronOffset + j);
      };

    // Process back propagated spikes using lambda function
    Profiler::WriteEntryDisableFIQ(Profiler::Enter | ProfilerTagProcessBackPropagation);
    g_BackPropagationBufferNeuronOffset += g_SDRAMBackPropagationInput.Process(g_BackPropagationBufferBeingProcessed,
                                                                               processSpikeLambda);
    Profiler::WriteEntryDisableFIQ(Profiler::Exit | ProfilerTagProcessBackPropagation);

    // Attempt to fetch next back propagation buffer
    g_BackPropagationBufferBeingProcessed++;
    if(g_SDRAMBackPropagationInput.Fetch(g_BackPropagationBufferBeingProcessed,
      g_Tick, DMATagBackPropagationRead))
    {
      // **NOTE** this will only cause a DMA if the buffer has any entries
      if(!g_DelayBuffer.Fetch(g_Tick, DMATagDelayBufferRead))
      {
        g_Statistics[StatWordDelayBufferFetchFail]++;
      }
    }
  }
  else if(tag == DMATagDelayBufferRead)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "DMA read of delay buffer for tick %u complete", g_Tick);

    // Create lambda function to add delay row to circular queue
    auto addDelayRowLambda =
      [](DelayBuffer::R rowOffset)
      {
        if(!g_DelayRowBuffer.Push(rowOffset))
        {
          g_Statistics[StatWordDelayBuffersNotProcessed]++;
        }
      };

    // Process delay buffer using lambda function
    g_DelayBuffer.ProcessDMABuffer(addDelayRowLambda);

    // Start DMA row fetch pipeline
    {
      DisableFIQ d;
      DMAStartRowFetchPipeline();
    }
  }
  else if(tag != DMATagRowWrite)
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "DMA transfer done with unknown tag %u", tag);
  }
}
//-----------------------------------------------------------------------------
void UserEvent(uint, uint)
{
  // Setup next row read
  SetupNextDMARowRead();
}
//-----------------------------------------------------------------------------
void TimerTick(uint tick, uint)
{
  Profiler::TagDisableIRQFIQ<ProfilerTagTimerTick> p;

  // Cache tick
  // **NOTE** ticks start at 1
  g_Tick = (tick - 1);

  // If a fixed number of simulation ticks are specified and these have passed
  if(g_Config.GetSimulationTicks() != UINT32_MAX
    && g_Tick >= g_Config.GetSimulationTicks())
  {
    LOG_PRINT(LOG_LEVEL_INFO, "Simulation complete");

    // Finalise profiling
    Profiler::Finalise();

    // Copy diagnostic stats out of spin1 API
    g_Statistics[StatWordTaskQueueFull] = diagnostics.task_queue_full;
    g_Statistics[StatWordNumTimerEventOverflows] = diagnostics.total_times_tick_tic_callback_overran;
    
    // Finalise statistics
    g_Statistics.Finalise();

    // Exit simulation
    spin1_exit(0);
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u, writing 'back' of ring-buffer to output buffer %u (%08x)",
              g_Tick, (g_Tick % 2), g_OutputBuffers[g_Tick % 2]);

    // Get output buffer from 'back' of ring-buffer
    const RingBuffer::Type *outputBuffer = g_RingBuffer.GetOutputBuffer(g_Tick);

#if LOG_LEVEL <= LOG_LEVEL_TRACE
    for(unsigned int i = 0; i < g_AppWords[AppWordNumPostNeurons]; i++)
    {
      io_printf(IO_BUF, "%u,", outputBuffer[i]);
    }
    io_printf(IO_BUF, "\n");
#endif

    // DMA output buffer into correct output buffer for this timer tick
    spin1_dma_transfer(DMATagOutputWrite, g_OutputBuffers[g_Tick % 2],
                      const_cast<RingBuffer::Type*>(outputBuffer), DMA_WRITE,
                      g_AppWords[AppWordNumPostNeurons] * sizeof(uint32_t));

  }
}
} // anonymous namespace

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
extern "C" void c_main()
{
  // Get this core's base address using alloc tag
  uint32_t *baseAddress = Config::GetBaseAddressAllocTag();

  // If reading SDRAM data fails
  if(!ReadSDRAMData(baseAddress, 0))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Error reading SDRAM data");
    rt_error(RTE_ABORT);
    return;
  }

  // Initialise
  g_DMABusy = false;
  g_DMARowBufferIndex = 0;

  // Set timer tick (in microseconds) in both timer and
  spin1_set_timer_tick(g_Config.GetTimerPeriod());

  // Register callbacks
  spin1_callback_on(MC_PACKET_RECEIVED, MCPacketReceived, -1);
  spin1_callback_on(DMA_TRANSFER_DONE,  DMATransferDone,   0);
  spin1_callback_on(USER_EVENT,         UserEvent,         0);
  spin1_callback_on(TIMER_TICK,         TimerTick,         2);

  // Start simulation
  spin1_start(SYNC_WAIT);
}