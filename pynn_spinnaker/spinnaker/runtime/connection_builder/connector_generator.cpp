#include "connector_generator.h"

// Rig CPP common includes
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/random/mars_kiss64.h"

//-----------------------------------------------------------------------------
// ConnectionBuilder::ConnectorGenerator::AllToAll
//-----------------------------------------------------------------------------
ConnectionBuilder::ConnectorGenerator::AllToAll::AllToAll(uint32_t *&)
{
  LOG_PRINT(LOG_LEVEL_INFO, "\t\tAll-to-all connector");
}
//-----------------------------------------------------------------------------
unsigned int ConnectionBuilder::ConnectorGenerator::AllToAll::Generate(
  unsigned int, unsigned int numPostNeurons, MarsKiss64 &, uint32_t (&indices)[1024]) const
{
  // Write indices
  for(unsigned int i = 0; i < numPostNeurons; i++)
  {
    indices[i] = i;
  }

  return numPostNeurons;
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::ConnectorGenerator::FixedProbability
//-----------------------------------------------------------------------------
ConnectionBuilder::ConnectorGenerator::FixedProbability::FixedProbability(uint32_t *&region)
{
  m_Probability = *region++;

  LOG_PRINT(LOG_LEVEL_INFO, "\t\tFixed-probability connector: probability:%u",
    m_Probability
  );
}
//-----------------------------------------------------------------------------
unsigned int ConnectionBuilder::ConnectorGenerator::FixedProbability::Generate(
  unsigned int, unsigned int numPostNeurons, MarsKiss64 &rng, uint32_t (&indices)[1024]) const
{
  // Write indices
  unsigned int k = 0;
  for(unsigned int i = 0; i < numPostNeurons; i++)
  {
    // If draw if less than probability, add index to row
    if(rng.GetNext() < m_Probability)
    {
      indices[k++] = i;
    }
  }

  return k;
}