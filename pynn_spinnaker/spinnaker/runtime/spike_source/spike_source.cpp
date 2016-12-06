#include "spike_source.h"

// Standard includes
#include <climits>

// Rig CPP common includes
#include "rig_cpp_common/config.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/profiler.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/statistics.h"
#include "rig_cpp_common/utils.h"

// Common includes
#include "../common/flush.h"
#include "../common/spike_recording.h"

// Configuration include
#include "config.h"

// Namespaces
using namespace Common;
using namespace SpikeSource;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// Constants
//----------------------------------------------------------------------------
const uint DMATagOutputWrite = Source::DMATagMax;

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Config g_Config;
uint32_t g_AppWords[AppWordMax];
Statistics<StatWordMax> g_Statistics;

SpikeRecording g_SpikeRecording;

Flush g_Flush;

Source g_SpikeSource;

//----------------------------------------------------------------------------
// Functions
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
    LOG_PRINT(LOG_LEVEL_INFO, "\tspike key=%08x, flush key=%08x, num spike sources=%u",
      g_AppWords[AppWordSpikeKey], g_AppWords[AppWordFlushKey], g_AppWords[AppWordNumSpikeSources]);
  }

  // Read source region
  if(!g_SpikeSource.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionSpikeSource), flags,
    g_AppWords[AppWordNumSpikeSources]))
  {
    return false;
  }

  // Read flush region
  if(!g_Flush.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionFlush), flags,
    g_AppWords[AppWordNumSpikeSources]))
  {
    return false;
  }

  // Read spike recording region
  if(!g_SpikeRecording.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionSpikeRecording), flags,
    g_AppWords[AppWordNumSpikeSources]))
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
// Event handler functions
//-----------------------------------------------------------------------------
void DMATransferDone(uint, uint tag)
{
  if(tag == DMATagOutputWrite)
  {
    g_SpikeRecording.Reset();
  }
  else if(!g_SpikeSource.DMATransferDone(tag))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Spike source unable to handle DMA tag %u", tag);
  }
}
//-----------------------------------------------------------------------------
void TimerTick(uint tick, uint)
{
  // Subtract 1 from tick as they start at 1
  tick--;

  // If a fixed number of simulation ticks are specified and these have passed
  if(g_Config.GetSimulationTicks() != UINT32_MAX
    && tick >= g_Config.GetSimulationTicks())
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
  // Otherwise
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u", tick);

    // Create lambda function to emit spike
    auto emitSpikeLambda =
      [](unsigned int n)
      {
        // Send spike
        uint32_t key = g_AppWords[AppWordSpikeKey] | n;
        while(!spin1_send_mc_packet(key, 0, NO_PAYLOAD))
        {
          spin1_delay_us(1);
        }
      };

    // Update spike source
    Profiler::WriteEntry(Profiler::Enter | ProfilerTagUpdateNeurons);
    g_SpikeSource.Update(tick, emitSpikeLambda, g_SpikeRecording,
      g_AppWords[AppWordNumSpikeSources]
    );
    Profiler::WriteEntry(Profiler::Exit | ProfilerTagUpdateNeurons);

    // Transfer spike recording buffer to SDRAM
    g_SpikeRecording.TransferBuffer(DMATagOutputWrite);
  }
}
} // Anonymous namespace

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
extern "C" void c_main()
{
  // Get this core's base address using alloc tag
  uint32_t *baseAddress = Common::Config::GetBaseAddressAllocTag();
  
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
  spin1_callback_on(TIMER_TICK,         TimerTick,        2);
  spin1_callback_on(DMA_TRANSFER_DONE,  DMATransferDone,   0);
  
  // Start simulation
  spin1_start(SYNC_WAIT);
}