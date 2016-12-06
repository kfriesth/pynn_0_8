#include "neuron_processor.h"

// Standard includes
#include <climits>

// Rig CPP common includes
#include "rig_cpp_common/config.h"
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/profiler.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/statistics.h"
#include "rig_cpp_common/utils.h"

// Common includes
#include "../common/flush.h"
#include "../common/spike_recording.h"

// Neuron processor includes
#include "analogue_recording.h"
#include "input_buffer.h"
#include "sdram_back_propagation_output.h"

// Configuration include
#include "config.h"

// Namespaces
using namespace Common;
using namespace Common::FixedPointNumber;
using namespace Common::Utils;
using namespace NeuronProcessor;

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
  DMATagInputRead,
  DMATagSpikeRecordingWrite,
  DMATagBackPropagationWrite
};

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Common::Config g_Config;
uint32_t g_AppWords[AppWordMax];

uint16_t *g_NeuronImmutableStateIndices = NULL;
Neuron::MutableState *g_NeuronMutableState = NULL;
Neuron::ImmutableState *g_NeuronImmutableState = NULL;

uint16_t *g_SynapseImmutableStateIndices = NULL;
Synapse::MutableState *g_SynapseMutableState = NULL;
Synapse::ImmutableState *g_SynapseImmutableState = NULL;

InputBuffer g_InputBuffer;

SDRAMBackPropagationOutput g_BackPropagationOutput;

Flush g_Flush;

IntrinsicPlasticity g_IntrinsicPlasticity;

SpikeRecording g_SpikeRecording;
AnalogueRecording g_AnalogueRecording[Neuron::RecordingChannelMax + IntrinsicPlasticity::RecordingChannelMax];
Statistics<StatWordMax> g_Statistics;

unsigned int g_InputBufferBeingProcessed = UINT_MAX;

uint g_Tick = 0;

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
bool ReadNeuronRegion(uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadNeuronRegion");

  LOG_PRINT(LOG_LEVEL_TRACE, "\tNeuron mutable state");
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumNeurons], region, g_NeuronMutableState))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate neuron mutable state array");
    return false;
  }

  LOG_PRINT(LOG_LEVEL_TRACE, "\tNeuron immutable state");
  if(!AllocateCopyIndexedStructArray(g_AppWords[AppWordNumNeurons], region,
    g_NeuronImmutableStateIndices, g_NeuronImmutableState))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate neuron immutable state array");
    return false;
  }
  
#if LOG_LEVEL <= LOG_LEVEL_TRACE
  LOG_PRINT(LOG_LEVEL_TRACE, "Neurons");
  LOG_PRINT(LOG_LEVEL_TRACE, "------------------------------------------");
  for(unsigned int n = 0; n < g_AppWords[AppWordNumNeurons]; n++)
  {
    io_printf(IO_BUF, "Neuron %u:\n", n);
    Neuron::Print(IO_BUF, g_NeuronMutableState[n],
      g_NeuronImmutableState[g_NeuronImmutableStateIndices[n]]);
  }
  LOG_PRINT(LOG_LEVEL_TRACE, "------------------------------------------");
#endif
  
  return true;
}
//-----------------------------------------------------------------------------
bool ReadSynapseRegion(uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadSynapseRegion");

  LOG_PRINT(LOG_LEVEL_TRACE, "\tSynapse mutable state");
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumNeurons], region, g_SynapseMutableState))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate synapse mutable state array");
    return false;
  }

  LOG_PRINT(LOG_LEVEL_TRACE, "\tSynapse immutable state");
  if(!AllocateCopyIndexedStructArray(g_AppWords[AppWordNumNeurons], region,
    g_SynapseImmutableStateIndices, g_SynapseImmutableState))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate synapse immutable state array");
    return false;
  }

#if LOG_LEVEL <= LOG_LEVEL_TRACE
  LOG_PRINT(LOG_LEVEL_TRACE, "Synapses");
  LOG_PRINT(LOG_LEVEL_TRACE, "------------------------------------------");
  for(unsigned int n = 0; n < g_AppWords[AppWordNumNeurons]; n++)
  {
    io_printf(IO_BUF, "Neuron %u:\n", n);
    Synapse::Print(IO_BUF, g_SynapseMutableState[n],
      g_SynapseImmutableState[g_SynapseImmutableStateIndices[n]]
    );
  }
  LOG_PRINT(LOG_LEVEL_TRACE, "------------------------------------------");
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
    LOG_PRINT(LOG_LEVEL_INFO, "\tspike key=%08x, flush key=%08x, num neurons=%u",
      g_AppWords[AppWordSpikeKey], g_AppWords[AppWordFlushKey], g_AppWords[AppWordNumNeurons]);
  }
  
  // Read neuron region
  if(!ReadNeuronRegion(
    Config::GetRegionStart(baseAddress, RegionNeuron), flags))
  {
    return false;
  }

  // Read neuron region
  if(!ReadSynapseRegion(
    Config::GetRegionStart(baseAddress, RegionSynapse), flags))
  {
    return false;
  }

  // Read input buffer region
  if(!g_InputBuffer.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionInputBuffer), flags,
    g_AppWords[AppWordNumNeurons]))
  {
    return false;
  }

  // Read back propagation region
  if(!g_BackPropagationOutput.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionBackPropagationOutput), flags,
  g_AppWords[AppWordNumNeurons]))
  {
    return false;
  }

  // Read flush region
  if(!g_Flush.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionFlush), flags,
    g_AppWords[AppWordNumNeurons]))
  {
    return false;
  }

  // Read intrinsic plasticity region
  if(!g_IntrinsicPlasticity.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionIntrinsicPlasticity), flags,
    g_AppWords[AppWordNumNeurons]))
  {
    return false;
  }

  // Read spike recording region
  if(!g_SpikeRecording.ReadSDRAMData(
    Config::GetRegionStart(baseAddress, RegionSpikeRecording), flags,
    g_AppWords[AppWordNumNeurons]))
  {
    return false;
  }

  // Check that there are enough analogue recording regions for this neuron model
  static_assert(RegionAnalogueRecordingEnd - RegionAnalogueRecordingStart >= (Neuron::RecordingChannelMax + IntrinsicPlasticity::RecordingChannelMax),
                "Not enough analogue recording regions for neuron and intrinsic plasticity model channels");

  // Loop through neuron model's recording channels
  for(unsigned int r = 0; r < Neuron::RecordingChannelMax; r++)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "Neuron analogue recording channel %u", r);

    // Read analogue recording region
    if(!g_AnalogueRecording[r].ReadSDRAMData(
      Config::GetRegionStart(baseAddress, RegionAnalogueRecordingStart + r), flags,
      g_AppWords[AppWordNumNeurons]))
    {
      return false;
    }
  }

  // Loop through intrinsic plasticity model's recording channels
  for(unsigned int r = Neuron::RecordingChannelMax;
      r < (Neuron::RecordingChannelMax + IntrinsicPlasticity::RecordingChannelMax); r++)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "Intrinsic plasticity analogue recording channel %u", r);

    // Read analogue recording region
    if(!g_AnalogueRecording[r].ReadSDRAMData(
      Config::GetRegionStart(baseAddress, RegionAnalogueRecordingStart + r), flags,
      g_AppWords[AppWordNumNeurons]))
    {
      return false;
    }
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
void UpdateNeurons()
{
  Profiler::Tag<ProfilerTagUpdateNeurons> p;

  // If spike recording isn't reset, DMA to write previous time steps spikes is outstanding,
  if(!g_SpikeRecording.IsReset())
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Updating neurons with previous spike recorder transfer in progress");
    rt_error(RTE_ABORT);
    return;
  }

  // Loop through neurons
  auto *neuronMutableState = g_NeuronMutableState;
  const uint16_t *neuronImmutableStateIndex = g_NeuronImmutableStateIndices;
  auto *synapseMutableState = g_SynapseMutableState;
  const uint16_t *synapseImmutableStateIndex = g_SynapseImmutableStateIndices;
  for(unsigned int n = 0; n < g_AppWords[AppWordNumNeurons]; n++)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "\tSimulating neuron %u", n);

    // Get synaptic input
    auto &synMutable = *synapseMutableState++;
    const auto &synImmutable = g_SynapseImmutableState[*synapseImmutableStateIndex++];
    S1615 excInput = Synapse::GetExcInput(synMutable, synImmutable);
    S1615 inhInput = Synapse::GetInhInput(synMutable, synImmutable);

    // Get intrinsic plasticity input
    S1615 extCurrent = g_IntrinsicPlasticity.GetIntrinsicCurrent(n);

    // Update neuron
    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tExcitatory input:%k, Inhibitory input:%k, External current:%knA",
              excInput, inhInput, extCurrent);
    auto &neuronMutable = *neuronMutableState++;
    const auto &neuronImmutable = g_NeuronImmutableState[*neuronImmutableStateIndex++];
    bool spiked = Neuron::Update(neuronMutable, neuronImmutable,
      excInput, inhInput, extCurrent);

    // Record spike
    g_SpikeRecording.RecordSpike(n, spiked);

    // Update intrinsic plasticity based on new spike
    g_IntrinsicPlasticity.ApplySpike(n, spiked);

    // If this is an actual spike, record in back propagation system
    if(spiked)
    {
      g_BackPropagationOutput.RecordSpike(n);
    }

    // If it spikes or should flush
    bool flush = g_Flush.ShouldFlush(n, spiked);
    if(spiked || flush)
    {
      // If spike key and flush key are the same and both equal UINT32_MAX,
      // spikes from this neuron processor don't go anywhere thus haven't
      // been allocated a key so don't send a packet
      if(g_AppWords[AppWordSpikeKey] == UINT32_MAX &&
        g_AppWords[AppWordSpikeKey] == g_AppWords[AppWordFlushKey])
      {
        LOG_PRINT(LOG_LEVEL_TRACE, "\t\tNo routing for neuron - not sending packet");
      }
      else
      {
        if(spiked)
        {
          LOG_PRINT(LOG_LEVEL_TRACE, "\t\tEmitting spike");
        }
        else
        {
          LOG_PRINT(LOG_LEVEL_TRACE, "\t\tEmitting flush");
        }
        // Send spike/flush
        uint32_t key = g_AppWords[spiked ? AppWordSpikeKey : AppWordFlushKey] | n;

        while(!spin1_send_mc_packet(key, 0, NO_PAYLOAD))
        {
          spin1_delay_us(1);
        }
      }
    }

    // Loop through neuron model's analogue recording channels
    for(unsigned int r = 0; r < Neuron::RecordingChannelMax; r++)
    {
      // Record the value from each one
      g_AnalogueRecording[r].RecordValue(n,
        Neuron::GetRecordable((Neuron::RecordingChannel)r,
                              neuronMutable, neuronImmutable,
                              excInput, inhInput, extCurrent)
      );
    }

    // Loop through intrinsic plasticity model's analogue recording channels
    for(unsigned int r = Neuron::RecordingChannelMax;
        r < (Neuron::RecordingChannelMax + IntrinsicPlasticity::RecordingChannelMax); r++)
    {
      // Record the value from each one
      g_AnalogueRecording[r].RecordValue(n,
        g_IntrinsicPlasticity.GetRecordable(
          (IntrinsicPlasticity::RecordingChannel)(r - Neuron::RecordingChannelMax), n)
      );
    }

    // **HACK** sleep for 1us after every other neuron to better space spiking
    /*if((n % 2) != 0)
    {
      spin1_delay_us(1);
    }*/
  }

  // Transfer spike recording and back propagation buffers to SDRAM
  g_SpikeRecording.TransferBuffer(DMATagSpikeRecordingWrite);
  g_BackPropagationOutput.TransferBuffer(g_Tick, DMATagBackPropagationWrite);

  // Loop through all analogue recording regions and
  // end tick (updates sampling interval mechanism)
  for(unsigned int r = 0;
      r < (Neuron::RecordingChannelMax + IntrinsicPlasticity::RecordingChannelMax); r++)
  {
    g_AnalogueRecording[r].EndTick();
  }
}

//-----------------------------------------------------------------------------
// Event handler functions
//-----------------------------------------------------------------------------
static void DMATransferDone(uint, uint tag)
{
  LOG_PRINT(LOG_LEVEL_TRACE, "DMA transfer done tag:%u", tag);

  if(tag == DMATagInputRead)
  {
    // Create lambda function to apply scaled input to neuron
    auto applyInputLambda =
      [](unsigned int neuron, S1615 input, unsigned int receptorType)
      {
        Synapse::ApplyInput(g_SynapseMutableState[neuron],
                            g_SynapseImmutableState[g_SynapseImmutableStateIndices[neuron]],
                            input, receptorType);
      };

    // Apply input in DMA buffer
    Profiler::WriteEntry(Profiler::Enter | ProfilerTagApplyBuffer);
    g_InputBuffer.Process(g_InputBufferBeingProcessed,
                          applyInputLambda);
    Profiler::WriteEntry(Profiler::Exit | ProfilerTagApplyBuffer);

    // Advance to next input buffer
    g_InputBufferBeingProcessed++;

    // If there aren't any more input buffers to DMA, start neuron update
    if(g_InputBuffer.Fetch(g_InputBufferBeingProcessed, g_Tick, DMATagInputRead))
    {
      UpdateNeurons();
    }
  }
  else if(tag == DMATagSpikeRecordingWrite)
  {
    g_SpikeRecording.Reset();
  }
  else if(tag == DMATagBackPropagationWrite)
  {
    g_BackPropagationOutput.ClearBuffer();
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Dma transfer done with unknown tag %u", tag);
  }
}
//-----------------------------------------------------------------------------
static void TimerTick(uint tick, uint)
{
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
  // Otherwise
  else
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u", g_Tick);

    // Loop through neurons and shape synaptic inputs
    Profiler::WriteEntry(Profiler::Enter | ProfilerTagSynapseShape);
    auto *synapseMutableState = g_SynapseMutableState;
    const uint16_t *synapseImmutableStateIndex = g_SynapseImmutableStateIndices;
    for(uint n = 0; n < g_AppWords[AppWordNumNeurons]; n++)
    {
      Synapse::Shape(*synapseMutableState++,
                     g_SynapseImmutableState[*synapseImmutableStateIndex++]);
    }
    Profiler::WriteEntry(Profiler::Exit | ProfilerTagSynapseShape);

    // Start at first input buffer
    g_InputBufferBeingProcessed = 0;

    // If there aren't any input buffers to DMA, start neuron update
    if(g_InputBuffer.Fetch(g_InputBufferBeingProcessed, g_Tick, DMATagInputRead))
    {
      UpdateNeurons();
    }
  }

}
} // Anonymous namespace

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
  
  // Set timer tick (in microseconds) in both timer and 
  spin1_set_timer_tick(g_Config.GetTimerPeriod());
  
  // Register callbacks
  spin1_callback_on(DMA_TRANSFER_DONE,  DMATransferDone,  0);
  spin1_callback_on(TIMER_TICK,         TimerTick,        2);
  
  // Start simulation
  spin1_start(SYNC_WAIT);
}