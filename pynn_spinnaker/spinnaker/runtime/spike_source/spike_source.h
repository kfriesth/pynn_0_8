#pragma once

namespace SpikeSource
{
//-----------------------------------------------------------------------------
// Enumerations
//-----------------------------------------------------------------------------
// Indices or regions
enum Region
{
  RegionSystem,
  RegionSpikeSource,
  RegionFlush = 5,
  RegionSpikeRecording = 7,
  RegionProfiler = 12,
  RegionStatistics,
};

// Indexes of application words
enum AppWord
{
  AppWordSpikeKey,
  AppWordFlushKey,
  AppWordNumSpikeSources,
  AppWordMax,
};

enum ProfilerTag
{
  ProfilerTagSynapseShape,
  ProfilerTagUpdateNeurons,
  ProfilerTagApplyBuffer,
};

// Indices of statistic words
enum StatWord
{
  StatWordTaskQueueFull,
  StatWordNumTimerEventOverflows,
  StatWordMax,
};

}  // namespace SpikeSource