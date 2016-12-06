#pragma once

namespace NeuronProcessor
{
//-----------------------------------------------------------------------------
// Enumerations
//-----------------------------------------------------------------------------
// Indices or regions
enum Region
{
  RegionSystem,
  RegionNeuron,
  RegionSynapse,
  RegionInputBuffer,
  RegionBackPropagationOutput,
  RegionFlush,
  RegionIntrinsicPlasticity,
  RegionSpikeRecording,
  RegionAnalogueRecordingStart,
  RegionAnalogueRecordingEnd = RegionAnalogueRecordingStart + 4,
  RegionProfiler = RegionAnalogueRecordingEnd,
  RegionStatistics,
};

// Indexes of application words
enum AppWord
{
  AppWordSpikeKey,
  AppWordFlushKey,
  AppWordNumNeurons,
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

};  // namespace NeuronProcessor