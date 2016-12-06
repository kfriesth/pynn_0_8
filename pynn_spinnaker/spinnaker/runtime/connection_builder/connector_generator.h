#pragma once

// Standard includes
#include <cstdint>

// Connection builder includes
#include "generator_factory.h"

// Forward declarations
namespace Common
{
  namespace Random
  {
    class MarsKiss64;
  }
}

// Namespaces
using namespace Common::Random;

//-----------------------------------------------------------------------------
// ConnectionBuilder::ConnectorGenerators
//-----------------------------------------------------------------------------
namespace ConnectionBuilder
{
namespace ConnectorGenerator
{
//-----------------------------------------------------------------------------
// Base
//-----------------------------------------------------------------------------
class Base
{
public:
  //-----------------------------------------------------------------------------
  // Declared virtuals
  //-----------------------------------------------------------------------------
  virtual unsigned int Generate(unsigned int row, unsigned int numPostNeurons,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]) const = 0;

};

//-----------------------------------------------------------------------------
// AllToAll
//-----------------------------------------------------------------------------
class AllToAll : public Base
{
public:
  ADD_FACTORY_CREATOR(AllToAll);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual unsigned int Generate(unsigned int row, unsigned int numPostNeurons,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]) const;

private:
  AllToAll(uint32_t *&);
};

//-----------------------------------------------------------------------------
// FixedProbability
//-----------------------------------------------------------------------------
class FixedProbability : public Base
{
public:
  ADD_FACTORY_CREATOR(FixedProbability);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual unsigned int Generate(unsigned int row, unsigned int numPostNeurons,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]) const;

private:
  FixedProbability(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  uint32_t m_Probability;
};

} // ConnectorGenerators
} // ConnectionBuilder