#include "current_input.h"

// Rig CPP common includes
#include "rig_cpp_common/config.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/profiler.h"
#include "rig_cpp_common/spinnaker.h"

// Common includes
#include "../common/spike_recording.h"

// Configuration include
#include "config.h"

// Namespaces
using namespace Common;
using namespace CurrentInput;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// Constants
//----------------------------------------------------------------------------
enum DMATag
{
  DMATagOutputWrite = Source::DMATagMax,
  DMATagSpikeRecordingWrite,
};

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Config g_Config;
uint32_t *g_OutputBuffers[2] = {NULL, NULL};

uint32_t *g_OutputWeights = NULL;

uint32_t *g_OutputBuffer = NULL;

uint32_t g_AppWords[AppWordMax];

SpikeRecording g_SpikeRecording;

Source g_SpikeSource;

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
bool ReadOutputWeightRegion(uint32_t *region, uint32_t)
{
  // Allocate and copy array of output weights from region
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumCurrentSources], region, g_OutputWeights))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate output weight array");
    return false;
  }

  // Allocate output buffer
  g_OutputBuffer = (uint32_t*)spin1_malloc(
    sizeof(uint32_t) * g_AppWords[AppWordNumCurrentSources]);
  if(g_OutputBuffer == NULL)
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate output buffer array");
    return false;
  }

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
    LOG_PRINT(LOG_LEVEL_INFO, "\tnum current sources=%u",
      g_AppWords[AppWordNumCurrentSources]);
  }

  // Read spike source region
  if(!g_SpikeSource.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionSpikeSource), flags,
    g_AppWords[AppWordNumCurrentSources]))
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

  // Read output weight region
  if(!ReadOutputWeightRegion(
    Config::GetRegionStart(baseAddress, RegionOutputWeight),
    flags))
  {
    return false;
  }

  // Read spike recording region
  if(!g_SpikeRecording.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionSpikeRecording), flags,
    g_AppWords[AppWordNumCurrentSources]))
  {
    return false;
  }

  // Read profiler region
  if(!Common::Profiler::ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionProfiler),
    flags))
  {
    return false;
  }

  return true;
}

//-----------------------------------------------------------------------------
// Event handler functions
//-----------------------------------------------------------------------------
void DMATransferDone(uint, uint tag)
{
  if(tag == DMATagSpikeRecordingWrite)
  {
    g_SpikeRecording.Reset();
  }
  else if(tag != DMATagOutputWrite && !g_SpikeSource.DMATransferDone(tag))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Spike source unable to handle DMA tag %u", tag);
  }
}
//-----------------------------------------------------------------------------
void TimerTick(uint tick, uint)
{
  Profiler::TagDisableIRQFIQ<ProfilerTagTimerTick> p;

  // Subtract 1 from tick as they start at 1
  tick--;

  // If a fixed number of simulation ticks are specified and these have passed
  if(g_Config.GetSimulationTicks() != UINT32_MAX
    && tick >= g_Config.GetSimulationTicks())
  {
    LOG_PRINT(LOG_LEVEL_INFO, "Simulation complete");

    // Finalise profiling
    Profiler::Finalise();
    
    // Exit simulation
    spin1_exit(0);
  }
  // Otherwise
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u", tick);

    // Zero output buffer
    for(unsigned int o = 0; o < g_AppWords[AppWordNumCurrentSources]; o++)
    {
      g_OutputBuffer[o] = 0;
    }

    // Create lambda function to add weight to output buffer
    auto emitSpikeLambda =
      [](unsigned int n)
      {
        g_OutputBuffer[n] += g_OutputWeights[n];
      };

    // Update poisson source
    g_SpikeSource.Update(tick, emitSpikeLambda, g_SpikeRecording,
      g_AppWords[AppWordNumCurrentSources]);

    // Transfer spike recording buffer to SDRAM
    g_SpikeRecording.TransferBuffer(DMATagSpikeRecordingWrite);


#if LOG_LEVEL <= LOG_LEVEL_TRACE
    for(unsigned int i = 0; i < g_AppWords[AppWordNumCurrentSources]; i++)
    {
      io_printf(IO_BUF, "%u,", g_OutputBuffer[i]);
    }
    io_printf(IO_BUF, "\n");
#endif

    // DMA output buffer into correct output buffer for this timer tick
    spin1_dma_transfer(DMATagOutputWrite, g_OutputBuffers[tick % 2],
                       g_OutputBuffer, DMA_WRITE,
                       g_AppWords[AppWordNumCurrentSources] * sizeof(uint32_t));
  }
}
} // anonymous namespace

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
extern "C" void c_main()
{
  //static_init();
  // Get this core's base address using alloc tag
  uint32_t *baseAddress = Config::GetBaseAddressAllocTag();

  // If reading SDRAM data fails
  if(!ReadSDRAMData(baseAddress, 0))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Error reading SDRAM data");
    rt_error(RTE_ABORT);
    return;
  }

  // Set timer tick (in microseconds) in both timer and
  spin1_set_timer_tick(g_Config.GetTimerPeriod());

  // Register callbacks
  spin1_callback_on(TIMER_TICK,         TimerTick,         2);
  spin1_callback_on(DMA_TRANSFER_DONE,  DMATransferDone,   0);

  // Start simulation
  spin1_start(SYNC_WAIT);
}