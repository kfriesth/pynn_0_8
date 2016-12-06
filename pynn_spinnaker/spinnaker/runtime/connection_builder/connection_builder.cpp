#include "connection_builder.h"

// Rig CPP common includes
#include "rig_cpp_common/config.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/spinnaker.h"
#include "rig_cpp_common/random/mars_kiss64.h"

// Common includes
#include "../common/key_lookup_binary_search.h"

// Connection builder includes
#include "connector_generator.h"
#include "generator_factory.h"
#include "matrix_generator.h"
#include "param_generator.h"

// Namespaces
using namespace Common;
using namespace Common::Random;
using namespace ConnectionBuilder;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{
//----------------------------------------------------------------------------
// Module level variables
//----------------------------------------------------------------------------
Config g_Config;
KeyLookupBinarySearch<10> g_KeyLookup;

uint32_t g_AppWords[AppWordMax];

uint32_t *g_SynapticMatrixBaseAddress = NULL;

// Factories to create matrix, connector and parameter generators by ID
GeneratorFactory<MatrixGenerator::Base, 3> g_MatrixGeneratorFactory;
GeneratorFactory<ConnectorGenerator::Base, 5> g_ConnectorGeneratorFactory;
GeneratorFactory<ParamGenerator::Base, 10> g_ParamGeneratorFactory;

// Memory buffers to placement new generators into
void *g_MatrixGeneratorBuffer = NULL;
void *g_ConnectorGeneratorBuffer = NULL;
void *g_DelayParamGeneratorBuffer = NULL;
void *g_WeightParamGeneratorBuffer = NULL;

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
bool ReadConnectionBuilderRegion(uint32_t *region, uint32_t)
{
  LOG_PRINT(LOG_LEVEL_INFO, "ReadConnectionBuilderRegion");

  // Read RNG seed
  uint32_t seed[MarsKiss64::StateSize];
  LOG_PRINT(LOG_LEVEL_TRACE, "\tSeed:");
  for(unsigned int s = 0; s < MarsKiss64::StateSize; s++)
  {
    seed[s] = *region++;
    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t%u", seed[s]);
  }

  // Create RNG with this seed
  // **TODO** multiple RNGs multiple seeds
  MarsKiss64 rng(seed);

  // Loop through matrices to generate
  const uint32_t numMatricesToGenerate = *region++;
  for(unsigned int i = 0; i < numMatricesToGenerate; i++)
  {
    // Read basic matrix properties
    const uint32_t key = *region++;
    const uint32_t sizeWords = *region++;
    const uint32_t numRows = *region++;
    const uint32_t matrixTypeHash = *region++;
    const uint32_t connectorTypeHash = *region++;
    const uint32_t delayTypeHash = *region++;
    const uint32_t weightTypeHash = *region++;
    LOG_PRINT(LOG_LEVEL_INFO, "\tMatrix %u: key %08x, size words:%u, num rows:%u, matrix type hash:%u, connector type hash:%u, delay type hash:%u, weight type hash:%u",
              i, key, sizeWords, numRows, matrixTypeHash, connectorTypeHash, delayTypeHash, weightTypeHash);

    // Generate matrix, connector, delays and weights
    const auto matrixGenerator = g_MatrixGeneratorFactory.Create(matrixTypeHash, region,
                                                                 g_MatrixGeneratorBuffer);
    const auto connectorGenerator = g_ConnectorGeneratorFactory.Create(connectorTypeHash, region,
                                                                       g_ConnectorGeneratorBuffer);

    LOG_PRINT(LOG_LEVEL_INFO, "\t\tDelay");
    const auto delayGenerator = g_ParamGeneratorFactory.Create(delayTypeHash, region,
                                                               g_DelayParamGeneratorBuffer);

    LOG_PRINT(LOG_LEVEL_INFO, "\t\tWeight");
    const auto weightGenerator = g_ParamGeneratorFactory.Create(weightTypeHash, region,
                                                               g_WeightParamGeneratorBuffer);

    // If any components couldn't be created return false
    if(matrixGenerator == NULL || connectorGenerator == NULL
      || delayGenerator == NULL || weightGenerator == NULL)
    {
      return false;
    }

    // Find matrix in key lookup
    unsigned int matrixRowSynapses;
    unsigned int matrixWordOffset;
    uint32_t matrixKeyMask;
    if(g_KeyLookup.LookupMatrix(key, matrixRowSynapses, matrixWordOffset, matrixKeyMask))
    {
      // Calculate start address of matrix
      uint32_t *matrixAddress = g_SynapticMatrixBaseAddress + matrixWordOffset;

      // Generate matrix
      LOG_PRINT(LOG_LEVEL_INFO, "\t\tAddress:%08x, row synapses:%u",
                matrixAddress, matrixRowSynapses);
      if(!matrixGenerator->Generate(g_SynapticMatrixBaseAddress, matrixAddress,
                                    matrixRowSynapses,
                                    g_AppWords[AppWordWeightFixedPoint],
                                    g_AppWords[AppWordNumPostNeurons],
                                    sizeWords, numRows,
                                    connectorGenerator, delayGenerator, weightGenerator,
                                    rng))
      {
        LOG_PRINT(LOG_LEVEL_ERROR, "\tMatrix generation failed");
        return false;
      }

    }
    else
    {
      LOG_PRINT(LOG_LEVEL_ERROR, "\tMatrix not found in key lookup");
      return false;
    }
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

  // Read connection builder region
  if(!ReadConnectionBuilderRegion(
    Config::GetRegionStart(baseAddress, RegionConnectionBuilder),
    flags))
  {
    return false;
  }

  return true;
}
} // anonymous namespace

//-----------------------------------------------------------------------------
// Entry point
//-----------------------------------------------------------------------------
extern "C" void __cxa_pure_virtual()
{
  LOG_PRINT(LOG_LEVEL_ERROR, "Pure virtual function call");
}
//-----------------------------------------------------------------------------
extern "C" void c_main()
{
  // Register matrix generators with factories
  // **NOTE** plastic matrix generator is capable of generating
  // both standard and extended plastic matrices
  LOG_PRINT(LOG_LEVEL_INFO, "Matrix generators");
  REGISTER_FACTORY_CLASS("StaticSynapticMatrix", MatrixGenerator, Static);
  REGISTER_FACTORY_CLASS("PlasticSynapticMatrix", MatrixGenerator, Plastic);
  REGISTER_FACTORY_CLASS("ExtendedPlasticSynapticMatrix", MatrixGenerator, Plastic);

  // Register connector generators with factories
  LOG_PRINT(LOG_LEVEL_INFO, "Connector generators");
  REGISTER_FACTORY_CLASS("AllToAllConnector", ConnectorGenerator, AllToAll);
  REGISTER_FACTORY_CLASS("FixedProbabilityConnector", ConnectorGenerator, FixedProbability);

  // Register parameter generators with factories
  LOG_PRINT(LOG_LEVEL_INFO, "Parameter generators");
  REGISTER_FACTORY_CLASS("constant", ParamGenerator, Constant);
  REGISTER_FACTORY_CLASS("uniform", ParamGenerator, Uniform);
  REGISTER_FACTORY_CLASS("normal", ParamGenerator, Normal);
  REGISTER_FACTORY_CLASS("normal_clipped", ParamGenerator, NormalClipped);
  REGISTER_FACTORY_CLASS("normal_clipped_to_boundary", ParamGenerator, NormalClippedToBoundary);
  REGISTER_FACTORY_CLASS("exponential", ParamGenerator, Exponential);

  // Allocate buffers for placement new from factories
  // **NOTE** we need to be able to simultaneously allocate a delay and
  // a weight generator so we need two buffers for parameter allocation
  g_MatrixGeneratorBuffer = g_MatrixGeneratorFactory.Allocate();
  g_ConnectorGeneratorBuffer = g_ConnectorGeneratorFactory.Allocate();
  g_DelayParamGeneratorBuffer = g_ParamGeneratorFactory.Allocate();
  g_WeightParamGeneratorBuffer = g_ParamGeneratorFactory.Allocate();

  // Get this core's base address using alloc tag
  uint32_t *baseAddress = Config::GetBaseAddressAllocTag();

  // If reading SDRAM data fails
  if(!ReadSDRAMData(baseAddress, 0))
  {
    LOG_PRINT(LOG_LEVEL_ERROR, "Error reading SDRAM data");
    rt_error(RTE_ABORT);
    return;
  }
}
