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
// ConnectionBuilder::ParamGenerator
//-----------------------------------------------------------------------------
namespace ConnectionBuilder
{
namespace ParamGenerator
{
/*binomial':       ('n', 'p'),
'gamma':          ('k', 'theta'),
'exponential':    ('beta',),
'lognormal':      ('mu', 'sigma'),
'normal':         ('mu', 'sigma'),
'normal_clipped': ('mu', 'sigma', 'low', 'high'),
'normal_clipped_to_boundary':
                  ('mu', 'sigma', 'low', 'high'),
'poisson':        ('lambda_',),
'uniform':        ('low', 'high'),
'uniform_int':    ('low', 'high'),*/
//-----------------------------------------------------------------------------
// Base
//-----------------------------------------------------------------------------
class Base
{
public:
  //-----------------------------------------------------------------------------
  // Declared virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&integers)[1024]) const = 0;
};

//-----------------------------------------------------------------------------
// Constant
//-----------------------------------------------------------------------------
class Constant : public Base
{
public:
  ADD_FACTORY_CREATOR(Constant);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&output)[1024]) const;

private:
  Constant(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Value;
};

//-----------------------------------------------------------------------------
// Uniform
//-----------------------------------------------------------------------------
class Uniform : public Base
{
public:
  ADD_FACTORY_CREATOR(Uniform);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&output)[1024]) const;

private:
  Uniform(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Low;
  int32_t m_Range;
};

//-----------------------------------------------------------------------------
// Normal
//-----------------------------------------------------------------------------
class Normal : public Base
{
public:
  ADD_FACTORY_CREATOR(Normal);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&output)[1024]) const;

private:
  Normal(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Mu;
  int32_t m_Sigma;
};

//-----------------------------------------------------------------------------
// Normal clipped
//-----------------------------------------------------------------------------
class NormalClipped : public Base
{
public:
  ADD_FACTORY_CREATOR(NormalClipped);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&output)[1024]) const;

private:
  NormalClipped(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Mu;
  int32_t m_Sigma;
  int32_t m_Low;
  int32_t m_High;
};

//-----------------------------------------------------------------------------
// Normal clipped to boundary
//-----------------------------------------------------------------------------
class NormalClippedToBoundary : public Base
{
public:
  ADD_FACTORY_CREATOR(NormalClippedToBoundary);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&output)[1024]) const;

private:
  NormalClippedToBoundary(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Mu;
  int32_t m_Sigma;
  int32_t m_Low;
  int32_t m_High;
};

//-----------------------------------------------------------------------------
// Exponential
//-----------------------------------------------------------------------------
class Exponential : public Base
{
public:
  ADD_FACTORY_CREATOR(Exponential);

  //-----------------------------------------------------------------------------
  // Base virtuals
  //-----------------------------------------------------------------------------
  virtual void Generate(unsigned int number, unsigned int fixedPoint,
                        MarsKiss64 &rng, int32_t (&output)[1024]) const;

private:
  Exponential(uint32_t *&region);

  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  int32_t m_Beta;
};
 
} // ParamGenerator
} // ConnectionBuilder
