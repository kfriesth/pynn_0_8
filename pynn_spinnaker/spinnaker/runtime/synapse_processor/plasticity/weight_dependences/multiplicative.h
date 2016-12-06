#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/log.h"

// Namespaces
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// SynapseProcessor::Plasticity::WeightDependences::Multiplicative
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
namespace Plasticity
{
namespace WeightDependences
{
template<typename W>
class Multiplicative
{
public:
  //-----------------------------------------------------------------------------
  // Typedefines
  //-----------------------------------------------------------------------------
  typedef W Weight;

  //-----------------------------------------------------------------------------
  // WeightState
  //-----------------------------------------------------------------------------
  class WeightState
  {
  public:
    WeightState(Weight weight) : m_Weight(weight)
    {
    }

    //-----------------------------------------------------------------------------
    // Public API
    //-----------------------------------------------------------------------------
    void ApplyDepression(S2011 depression, const Multiplicative<Weight> &weightDependence)
    {
      // Calculate scale
      // **NOTE** this calculation must be done at runtime-defined weight
      // fixed-point format
      int32_t scale = __smulbb(m_Weight - weightDependence.m_MinWeight,
                               weightDependence.m_A2Minus);
      scale >>= weightDependence.m_WeightFixedPoint;

      // Multiply scale by depression and subtract
      // **NOTE** using standard STDP fixed-point format handles format conversion
      m_Weight -= Mul16S2011(scale, depression);
    }

    void ApplyPotentiation(S2011 potentiation, const Multiplicative<Weight> &weightDependence)
    {
      // Calculate scale
      // **NOTE** this calculation must be done at runtime-defined weight
      // fixed-point format
      int32_t scale = __smulbb(weightDependence.m_MaxWeight - m_Weight,
                               weightDependence.m_A2Plus);
      scale >>= weightDependence.m_WeightFixedPoint;

      // Multiply scale by potentiation and add
      // **NOTE** using standard STDP fixed-point format handles format conversion
      m_Weight += Mul16S2011(scale, potentiation);
    }

    Weight CalculateFinalWeight(const Multiplicative<Weight> &) const
    {
      return (Weight)m_Weight;
    }

  private:
    //-----------------------------------------------------------------------------
    // Members
    //-----------------------------------------------------------------------------
    int32_t m_Weight;
  };

  Multiplicative() : m_A2Plus(0), m_A2Minus(0), m_MinWeight(0), m_MaxWeight(0)
  {
  }

  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *&region, uint32_t, uint32_t weightFixedPoint)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tPlasticity::WeightDependences::Multiplicative::ReadSDRAMData");

    // Read region parameters
    m_A2Plus = *reinterpret_cast<int32_t*>(region++);
    m_A2Minus = *reinterpret_cast<int32_t*>(region++);
    m_MinWeight = *reinterpret_cast<int32_t*>(region++);
    m_MaxWeight = *reinterpret_cast<int32_t*>(region++);
    m_WeightFixedPoint = weightFixedPoint;

    LOG_PRINT(LOG_LEVEL_INFO, "\t\tA2+:%d, A2-:%d, Min weight:%d, Max weight:%d, Weight fixed point:%u",
              m_A2Plus, m_A2Minus, m_MinWeight, m_MaxWeight, m_WeightFixedPoint);

    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Potentiation and depression scaling factors in runtime weight format
  int32_t m_A2Plus;
  int32_t m_A2Minus;

  // Minimum and maximum synaptic weight in runtime weight format
  int32_t m_MinWeight;
  int32_t m_MaxWeight;

  // Where is the fixed point in the fixed-point
  // numeric format used to represent weights
  uint32_t m_WeightFixedPoint;
};
} // WeightDependences
} // Plasticity
} // SynapseProcessor