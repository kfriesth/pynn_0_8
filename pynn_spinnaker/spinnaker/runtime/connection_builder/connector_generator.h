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
                                unsigned int vertexPostSlice, unsigned int vertexPreSlice,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]) = 0;

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
                                unsigned int vertexPostSlice, unsigned int vertexPreSlice,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]);

private:
  AllToAll(uint32_t *&);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  uint32_t m_AllowSelfConnections;
};

//-----------------------------------------------------------------------------
// OneToOne
//-----------------------------------------------------------------------------
class OneToOne : public Base
{
public:
  ADD_FACTORY_CREATOR(OneToOne);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual unsigned int Generate(unsigned int row, unsigned int numPostNeurons,
                                unsigned int vertexPostSlice, unsigned int vertexPreSlice,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]);

private:
  OneToOne(uint32_t *&);
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
                                unsigned int vertexPostSlice, unsigned int vertexPreSlice,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]);

private:
  FixedProbability(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  uint32_t m_Probability;
  uint32_t m_AllowSelfConnections;
};

//-----------------------------------------------------------------------------
// FixedTotalNumber
//-----------------------------------------------------------------------------
class FixedTotalNumber : public Base
{
public:
  ADD_FACTORY_CREATOR(FixedTotalNumber);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual unsigned int Generate(unsigned int row, unsigned int numPostNeurons,
                                unsigned int vertexPostSlice, unsigned int vertexPreSlice,
                                MarsKiss64 &rng, uint32_t (&indices)[1024]);

private:
  FixedTotalNumber(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  uint32_t m_AllowSelfConnections;
  uint32_t m_WithReplacement;
  uint32_t m_ConnectionsInSubmatrix;
  uint32_t m_SubmatrixSize;
};

} // ConnectorGenerators
} // ConnectionBuilder
