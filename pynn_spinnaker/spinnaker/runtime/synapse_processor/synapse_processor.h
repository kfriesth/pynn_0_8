#pragma once

namespace SynapseProcessor
{
//-----------------------------------------------------------------------------
// Enumerations
//-----------------------------------------------------------------------------
// Indexes of synapse executable regions
enum Region
{
  RegionSystem,
  RegionKeyLookup,
  RegionSynapticMatrix,
  RegionPlasticity,
  RegionOutputBuffer,
  RegionDelayBuffer,
  RegionBackPropagationInput,
  RegionConnectionBuilder,
  RegionProfiler,
  RegionStatistics,
};

// Indices of application words
enum AppWord
{
  AppWordWeightFixedPoint,
  AppWordNumPostNeurons,
  AppWordFlushMask,
  AppWordMax,
};

enum ProfilerTag
{
  ProfilerTagTimerTick,
  ProfilerTagMCPacketReceived,
  ProfilerTagSetupNextDMARowRead,
  ProfilerTagProcessRow,
  ProfilerTagProcessBackPropagation,

};

// Indices of statistic words
enum StatWord
{
  StatRowRequested,
  StatDelayRowRequested,
  StatWordDelayBuffersNotProcessed,
  StatWordInputBufferOverflows,
  StatWordKeyLookupFail,
  StatWordDelayBufferOverflows,
  StatWordDelayBufferFetchFail,
  StatWordTaskQueueFull,
  StatWordNumTimerEventOverflows,
  StatWordMax,
};

} // SynapseProcessor