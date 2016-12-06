#include "param_generator.h"

// Rig CPP common includes
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/arm_intrinsics.h"
#include "rig_cpp_common/log.h"
#include "rig_cpp_common/maths/normal.h"
#include "rig_cpp_common/random/non_uniform.h"
#include "rig_cpp_common/random/mars_kiss64.h"

// Namespaces
using namespace Common::ARMIntrinsics;
using namespace Common::Maths;
using namespace Common::Random::NonUniform;

//-----------------------------------------------------------------------------
// ConnectionBuilder::ParamGenerator::Constant
//-----------------------------------------------------------------------------
ConnectionBuilder::ParamGenerator::Constant::Constant(uint32_t *&region)
{
  m_Value = *reinterpret_cast<int32_t*>(region++);

  LOG_PRINT(LOG_LEVEL_INFO, "\t\t\tConstant parameter: value:%d", m_Value);
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::ParamGenerator::Constant::Generate(unsigned int number,
  unsigned int, MarsKiss64 &, int32_t (&output)[1024]) const
{
  // Copy constant into output
  for(uint32_t i = 0; i < number; i++)
  {
    output[i] = m_Value;
  }
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::ParamGenerators::Uniform
//-----------------------------------------------------------------------------
ConnectionBuilder::ParamGenerator::Uniform::Uniform(uint32_t *&region)
{
  m_Low = *reinterpret_cast<int32_t*>(region++);
  const int32_t high = *reinterpret_cast<int32_t*>(region++);
  m_Range = high - m_Low;

  LOG_PRINT(LOG_LEVEL_INFO, "\t\t\tUniform parameter: low:%d, high:%d, range:%d",
            m_Low, high, m_Range);
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::ParamGenerator::Uniform::Generate(unsigned int number,
  unsigned int, MarsKiss64 &rng, int32_t (&output)[1024]) const
{
  for(uint32_t i = 0; i < number; i++)
  {
    // Draw random number (0, UINT32_MAX) clear top bit and cast to signed
    int32_t fraction = (int32_t)(rng.GetNext() & 0x7FFFFFFF);

    // Multiply the resultant fraction by the range and shift down
    output[i] = m_Low + (int32_t)(__smull(fraction, m_Range) >> 31);
  }
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::ParamGenerators::Normal
//-----------------------------------------------------------------------------
ConnectionBuilder::ParamGenerator::Normal::Normal(uint32_t *&region)
{
  m_Mu = *reinterpret_cast<int32_t*>(region++);
  m_Sigma = *reinterpret_cast<int32_t*>(region++);
  LOG_PRINT(LOG_LEVEL_INFO, "\t\t\tNormal parameter: mu:%d, sigma:%d",
            m_Mu, m_Sigma);
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::ParamGenerator::Normal::Generate(unsigned int number,
  unsigned int, MarsKiss64 &rng, int32_t (&output)[1024]) const
{
  for(uint32_t i = 0; i < number; i++)
  {

    // **TODO** why is sign later being thrown away?

    // Draw random number (0, UINT32_MAX)
    uint32_t uniform = rng.GetNext();
    // Transform into standard uniform random variable
    // add location parameter and scale
    int32_t normal = m_Mu + MulS1615(NormalU032(uniform), m_Sigma);

    output[i] = normal;
  }
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::ParamGenerators::NormalClipped
//-----------------------------------------------------------------------------
ConnectionBuilder::ParamGenerator::NormalClipped::NormalClipped(uint32_t *&region)
{
  m_Mu = *reinterpret_cast<int32_t*>(region++);
  m_Sigma = *reinterpret_cast<int32_t*>(region++);
  m_Low = *reinterpret_cast<int32_t*>(region++);
  m_High = *reinterpret_cast<int32_t*>(region++);
  LOG_PRINT(LOG_LEVEL_INFO, "\t\t\tNormal clipped parameter: mu:%d, sigma:%d, low:%d, high:%d",
            m_Mu, m_Sigma, m_Low, m_High);
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::ParamGenerator::NormalClipped::Generate(unsigned int number,
  unsigned int, MarsKiss64 &rng, int32_t (&output)[1024]) const
{
  for(uint32_t i = 0; i < number; i++)
  {

    int32_t normal;

    // **TODO** some kind of check to make sure we don't end up in inf. loop
    do
    {
      // Draw random number (0, UINT32_MAX)
      uint32_t uniform = rng.GetNext();
      // Transform into standard uniform random variable, add location parameter and scale
      normal = m_Mu + MulS1615(NormalU032(uniform), m_Sigma);
    } while (normal > m_High || normal < m_Low);

    output[i] = normal;
  }
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::ParamGenerators::NormalClippedToBoundary
//-----------------------------------------------------------------------------
ConnectionBuilder::ParamGenerator::NormalClippedToBoundary::NormalClippedToBoundary(uint32_t *&region)
{
  m_Mu = *reinterpret_cast<int32_t*>(region++);
  m_Sigma = *reinterpret_cast<int32_t*>(region++);
  m_Low = *reinterpret_cast<int32_t*>(region++);
  m_High = *reinterpret_cast<int32_t*>(region++);
  LOG_PRINT(LOG_LEVEL_INFO, "\t\t\tNormal clipped to boundary parameter:\
                             mu:%d, sigma:%d, low:%d, high:%d",
            m_Mu, m_Sigma, m_Low, m_High);
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::ParamGenerator::NormalClippedToBoundary::Generate(unsigned int number,
  unsigned int, MarsKiss64 &rng, int32_t (&output)[1024]) const
{
  for(uint32_t i = 0; i < number; i++)
  {

    // Draw random number (0, UINT32_MAX)
    uint32_t uniform = rng.GetNext();
    // Transform into standard uniform random variable, add location parameter and scale
    int32_t normal = m_Mu + MulS1615(NormalU032(uniform), m_Sigma);

    // Clip output to boundary
    if (normal < m_Low)
    {
      normal = m_Low;
    }
    if (normal > m_High)
    {
      normal = m_High;
    }
    output[i] = normal;
  }
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::ParamGenerators::Exponential
//-----------------------------------------------------------------------------
ConnectionBuilder::ParamGenerator::Exponential::Exponential(uint32_t *&region)
{
  m_Beta = *reinterpret_cast<int32_t*>(region++);

  LOG_PRINT(LOG_LEVEL_INFO, "\t\t\tExponential parameter: beta:%d",
            m_Beta);
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::ParamGenerator::Exponential::Generate(unsigned int number,
  unsigned int, MarsKiss64 &rng, int32_t (&output)[1024]) const
{
  for(uint32_t i = 0; i < number; i++)
  {
    int32_t exp = MulS1615(m_Beta, ExponentialDistVariate(rng));
    output[i] = exp;
  }
}
