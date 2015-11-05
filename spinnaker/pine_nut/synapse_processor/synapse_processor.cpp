#include "synapse_processor.h"

// Common includes
#include "../common/config.h"
#include "../common/log.h"
#include "../common/spinnaker.h"

// Configuration include
#include "config.h"

// Namespaces
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
};

typedef uint32_t DMABuffer[SynapseType::MaxRowWords];

//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Common::Config g_Config;
RingBuffer g_RingBuffer;
KeyLookup g_KeyLookup;
SpikeInputBuffer g_SpikeInputBuffer;

uint32_t g_AppWords[AppWordMax];

uint32_t *g_OutputBuffers[2] = {NULL, NULL};

const uint32_t *g_SynapticMatrixBaseAddress = NULL;

uint g_Tick = 0;


bool g_DMABusy = false;
DMABuffer g_DMABuffers[2];
unsigned int g_DMARowBufferIndex = 0;

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
// Module functions
//-----------------------------------------------------------------------------
bool ReadSynapticMatrixRegion(const uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadSynapticMatrixRegion");

  // Cache pointer to region as base address for synaptic matrices
  g_SynapticMatrixBaseAddress = region;

  LOG_PRINT(LOG_LEVEL_INFO, "\tSynaptic matrix base address:%08x",
            g_SynapticMatrixBaseAddress);

  return true;
}
//-----------------------------------------------------------------------------
bool ReadOutputBufferRegion(const uint32_t *region, uint32_t)
{
  // Copy two output buffer pointers from region
  spin1_memcpy(g_OutputBuffers, region, 2 * sizeof(uint32_t*));

#if LOG_LEVEL <= LOG_LEVEL_INFO
  LOG_PRINT(LOG_LEVEL_INFO, "output_buffer\n");
  LOG_PRINT(LOG_LEVEL_INFO, "------------------------------------------");
  for (uint32_t i = 0; i < 2; i++)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "index %u, buffer:%08x", i, g_OutputBuffers[i]);
  }
  LOG_PRINT(LOG_LEVEL_INFO, "------------------------------------------");
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
    LOG_PRINT(LOG_LEVEL_INFO, "\tWeight fixed point=%u",
      g_AppWords[AppWordWeightFixedPoint]);
  }

  // Read key lookup region
  if(!g_KeyLookup.ReadSDRAMData(
    Common::Config::GetRegionStart(baseAddress, RegionKeyLookup),
    flags))
  {
    return false;
  }

  // Read synaptic matrix region
  if(!ReadSynapticMatrixRegion(
    Common::Config::GetRegionStart(baseAddress, RegionSynapticMatrix),
    flags))
  {
    return false;
  }

  // Read output buffer region
  if(!ReadOutputBufferRegion(
    Common::Config::GetRegionStart(baseAddress, RegionOutputBuffer),
    flags))
  {
    return false;
  }

  return true;
}
//-----------------------------------------------------------------------------
void SetupNextDMARowRead()
{
  // If there's more incoming spikes
  uint32_t key;
  if(g_SpikeInputBuffer.GetNextSpike(&key))
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "Setting up DMA read for spike %x", key);
    
    // Create lambda function to convert number of synapses to a row length in words
    auto getRowWordsLambda = 
      [](unsigned int rowSynapses) 
      { 
        return SynapseType::GetRowWords(rowSynapses);
      };
    
    // Decode key to get address and length of destination synaptic row
    unsigned int rowWords;
    const uint32_t *rowAddress;
    if(g_KeyLookup.LookupRow(key, g_SynapticMatrixBaseAddress, getRowWordsLambda,
      rowWords, rowAddress))
    {
      LOG_PRINT(LOG_LEVEL_TRACE, "\tRow words:%u, Row address:%08x",
                rowWords, rowAddress);
      
      // Start a DMA transfer to fetch this synaptic row into current buffer
      spin1_dma_transfer(DMATagRowRead, const_cast<uint32_t*>(rowAddress), DMACurrentRowBuffer(), DMA_READ, rowWords * sizeof(uint32_t));

      // Flip DMA buffers
      DMASwapRowBuffers();

      return;
    }
  }

  g_DMABusy = false;
}

//-----------------------------------------------------------------------------
// Event handler functions
//-----------------------------------------------------------------------------
void MCPacketReceived(uint key, uint)
{
  LOG_PRINT(LOG_LEVEL_TRACE, "Received spike %x at tick %u, DMA Busy = %u",
            key, g_Tick, g_DMABusy);

  // If there was space to add spike to incoming spike queue
  if(g_SpikeInputBuffer.AddSpike(key))
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
  else
  {
    LOG_PRINT(LOG_LEVEL_WARN, "Cannot add spike to input buffer");
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
        LOG_PRINT(LOG_LEVEL_TRACE, "\t\tAdding weight %u to neuron %u for tick %u",
                  weight, index, tick);
        g_RingBuffer.AddWeight(tick, index, weight);
      };
    
    // Process the next row in the DMA buffer, using this function to apply
    SynapseType::ProcessRow(g_Tick, DMANextRowBuffer(), addWeightLambda);
    
    // Setup next row read
    SetupNextDMARowRead();
  }
  else if(tag == DMATagOutputWrite)
  {
    // This timesteps output has been written from
    // the ring-buffer so we can now zero it
    g_RingBuffer.ClearOutputBuffer(g_Tick);
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

  LOG_PRINT(LOG_LEVEL_TRACE, "Timer tick %u, writing 'back' of ring-buffer to output buffer %u (%08x)",
            tick, (tick % 2), g_OutputBuffers[tick % 2]);

  // Get output buffer from 'back' of ring-buffer
  const RingBuffer::Type *pOutputBuffer = g_RingBuffer.GetOutputBuffer(tick);

  // DMA output buffer into correct output buffer for this timer tick
  //spin1_dma_transfer(DMATagOutputWrite,
  //  g_OutputBuffers[tick % 2],
  //  output_buffer,
  //  DMA_WRITE,
  //  output_buffer_bytes);
}
} // anonymous namespace

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
extern "C" void c_main()
{
  //static_init();
  // Get this core's base address using alloc tag
  const uint32_t *baseAddress = Common::Config::GetBaseAddressAllocTag();

  // If reading SDRAM data fails
  if(!ReadSDRAMData(baseAddress, 0))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Error reading SDRAM data");
    return;
  }

  // Initialise
  g_DMABusy = false;
  g_DMARowBufferIndex = 0;

  // Initialize modules
  //ring_buffer_init();

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