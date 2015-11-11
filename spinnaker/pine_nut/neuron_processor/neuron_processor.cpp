#include "neuron_processor.h"

// Standard includes
#include <climits>

// Common includes
#include "../common/config.h"
#include "../common/fixed_point_number.h"
#include "../common/log.h"
#include "../common/spinnaker.h"
#include "../common/utils.h"

// Neuron processor includes
#include "input_buffer.h"

// Configuration include
#include "config.h"

// Namespaces
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
};

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Common::Config g_Config;
uint32_t g_AppWords[AppWordMax];

Neuron::MutableState *g_NeuronMutableState = NULL;
Neuron::ImmutableState *g_NeuronImmutableState = NULL;

Synapse::MutableState *g_SynapseMutableState = NULL;
Synapse::ImmutableState *g_SynapseImmutableState = NULL;

InputBuffer g_InputBuffer;

unsigned int g_InputBufferBeingProcessed = UINT_MAX;

uint g_Tick = 0;

//----------------------------------------------------------------------------
// Functions
//----------------------------------------------------------------------------
bool ReadNeuronRegion(const uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadNeuronRegion");

  LOG_PRINT(LOG_LEVEL_TRACE, "\tNeuron mutable state");
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumNeurons], region, g_NeuronMutableState))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate neuron mutable state array");
    return false;
  }

  LOG_PRINT(LOG_LEVEL_TRACE, "\tNeuron immutable state");
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumNeurons], region, g_NeuronImmutableState))
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
    Neuron::Print(IO_BUF, g_NeuronMutableState[n], g_NeuronImmutableState[n]);
  }
  LOG_PRINT(LOG_LEVEL_TRACE, "------------------------------------------");
#endif
  
  return true;
}
//-----------------------------------------------------------------------------
bool ReadSynapseRegion(const uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadSynapseRegion");

  LOG_PRINT(LOG_LEVEL_TRACE, "\tSynapse mutable state");
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumNeurons], region, g_SynapseMutableState))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Unable to allocate synapse mutable state array");
    return false;
  }

  LOG_PRINT(LOG_LEVEL_TRACE, "\tSynapse immutable state");
  if(!AllocateCopyStructArray(g_AppWords[AppWordNumNeurons], region, g_SynapseImmutableState))
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
    Synapse::Print(IO_BUF, g_SynapseMutableState[n], g_SynapseImmutableState[n]);
  }
  LOG_PRINT(LOG_LEVEL_TRACE, "------------------------------------------");
#endif
  return true;
}
//-----------------------------------------------------------------------------
bool ReadSDRAMData(const uint32_t *baseAddress, uint32_t flags)
{
  // Verify data header
  if(!g_Config.VerifyHeader(baseAddress, flags))
  {
    return false;
  }

  // Read system region
  if(!g_Config.ReadSystemRegion(
    Common::Config::GetRegionStart(baseAddress, RegionSystem),
    flags, AppWordMax, g_AppWords))
  {
    return false;
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tkey=%08x, num neurons=%u",
      g_AppWords[AppWordKey], g_AppWords[AppWordNumNeurons]);
  }
  
  // Read neuron region
  if(!ReadNeuronRegion(
    Common::Config::GetRegionStart(baseAddress, RegionNeuron), flags))
  {
    return false;
  }

  // Read neuron region
  if(!ReadSynapseRegion(
    Common::Config::GetRegionStart(baseAddress, RegionSynapse), flags))
  {
    return false;
  }

  // Read input buffer region
  if(!g_InputBuffer.ReadSDRAMData(
    Common::Config::GetRegionStart(baseAddress, RegionInputBuffer), flags,
    g_AppWords[AppWordNumNeurons]))
  {
    return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
void UpdateNeurons()
{
  // Loop through neurons
  auto *neuronMutableState = g_NeuronMutableState;
  const auto *neuronImmutableState = g_NeuronImmutableState;
  auto *synapseMutableState = g_SynapseMutableState;
  const auto *synapseImmutableState = g_SynapseImmutableState;
  for(uint n = 0; n < g_AppWords[AppWordNumNeurons]; n++)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "\tSimulating neuron %u", n);

    // Get synaptic input
    auto &synMutable = *synapseMutableState++;
    const auto &synImmutable = *synapseImmutableState++;
    S1615 excInput = Synapse::GetExcInput(synMutable, synImmutable);
    S1615 inhInput = Synapse::GetInhInput(synMutable, synImmutable);

    // Update neuron, if it spikes
    S1615 extCurrent = 0;
    LOG_PRINT(LOG_LEVEL_TRACE, "\t\tExcitatory input:%k, Inhibitory input:%k, External current:%knA",
              excInput, inhInput, extCurrent);
    if(Neuron::Update(*neuronMutableState++, *neuronImmutableState++,
      excInput, inhInput, extCurrent))
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\tEmitting spike");

      // Send spike
      uint32_t key = g_AppWords[AppWordKey] | n;
      while(!spin1_send_mc_packet(key, 0, NO_PAYLOAD))
      {
        spin1_delay_us(1);
      }
    }
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
                            g_SynapseImmutableState[neuron],
                            input, receptorType);
      };

    // Apply input in DMA buffer
    g_InputBuffer.ApplyDMABuffer(g_InputBufferBeingProcessed,
                                 g_AppWords[AppWordNumNeurons],
                                 applyInputLambda);

    // Advance to next input buffer
    g_InputBufferBeingProcessed++;

    // If there aren't any more input buffers to DMA, start neuron update
    if(g_InputBuffer.SetupBufferDMA(g_InputBufferBeingProcessed, g_Tick,
      g_AppWords[AppWordNumNeurons], DMATagInputRead))
    {
      UpdateNeurons();
    }
  }
  else
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Dma transfer done with unknown tag %u", tag);
  }
}
//-----------------------------------------------------------------------------
static void TimerTick(uint tick, uint)
{
  LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u", tick);

  // Cache tick
  g_Tick = tick;

  // If a fixed number of simulation ticks are specified and these have passed
  if(g_Config.GetSimulationTicks() != UINT32_MAX
    && tick >= g_Config.GetSimulationTicks())
  {
    LOG_PRINT(LOG_LEVEL_INFO, "Simulation complete");

    // Finalise any recordings that are in progress, writing
    // back the final amounts of samples recorded to SDRAM
    //recording_finalise();
    spin1_exit(0);
  }
  // Otherwise
  else
  {
    // Loop through neurons and shape synaptic inputs
    auto *synapseMutableState = g_SynapseMutableState;
    const auto *synapseImmutableState = g_SynapseImmutableState;
    for(uint n = 0; n < g_AppWords[AppWordNumNeurons]; n++)
    {
      Synapse::Shape(*synapseMutableState++, *synapseImmutableState++);
    }

    // Start at first input buffer
    g_InputBufferBeingProcessed = 0;

    // If there aren't any input buffers to DMA, start neuron update
    if(g_InputBuffer.SetupBufferDMA(g_InputBufferBeingProcessed, tick,
      g_AppWords[AppWordNumNeurons], DMATagInputRead))
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
  const uint32_t *baseAddress = Common::Config::GetBaseAddressAllocTag();
  
  // If reading SDRAM data fails
  if(!ReadSDRAMData(baseAddress, 0))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Error reading SDRAM data");
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