#pragma once

// Standard includes
#include <cstdint>

// Rig CPP common includes
#include "rig_cpp_common/fixed_point_number.h"
#include "rig_cpp_common/log.h"

// Namespaces
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// SynapseProcessor::Plasticity::WeightDependences::Additive
//-----------------------------------------------------------------------------
namespace SynapseProcessor
{
namespace Plasticity
{
namespace WeightDependences
{
template<typename W>
class Additive
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
    WeightState(Weight weight) : m_InitialWeight(weight), m_Potentiation(0), m_Depression(0)
    {
    }

    //-----------------------------------------------------------------------------
    // Public API
    //-----------------------------------------------------------------------------
    void ApplyDepression(S2011 depression, const Additive<Weight> &)
    {
      m_Depression += depression;
    }

    void ApplyPotentiation(S2011 potentiation, const Additive<Weight> &)
    {
      m_Potentiation += potentiation;
    }

    Weight CalculateFinalWeight(const Additive<Weight> &weightDependence) const
    {
      // Scale potentiation and depression and combine together
      int32_t weightChange = __smulbb(m_Potentiation, weightDependence.m_A2Plus);
      weightChange = __smlabb(m_Depression, weightDependence.m_MinusA2Minus, weightChange);
      weightChange >>= 11;

      // Apply weight change and clamp
      // **NOTE** we know that m_MinWeight is always less than m_MaxWeight so doing 
      // this rather than calling std::min and std::max saves a couple of cycles
      int32_t newWeight = m_InitialWeight + weightChange;
      if(newWeight < weightDependence.m_MinWeight)
      {
        newWeight = weightDependence.m_MinWeight;
      }
      else if(newWeight > weightDependence.m_MaxWeight)
      {
        newWeight = weightDependence.m_MaxWeight;
      }

      LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tInitial weight:%d, Potentiation:%d, Depression:%d, Weight change:%d, New weight:%d",
                m_InitialWeight, m_Potentiation, m_Depression, weightChange, newWeight);

      return (Weight)newWeight;
    }

  private:
    //-----------------------------------------------------------------------------
    // Members
    //-----------------------------------------------------------------------------
    int32_t m_InitialWeight;

    S2011 m_Potentiation;
    S2011 m_Depression;
  };

  Additive() : m_A2Plus(0), m_MinusA2Minus(0), m_MinWeight(0), m_MaxWeight(0)
  {
  }

  //-----------------------------------------------------------------------------
  // Public API
  //-----------------------------------------------------------------------------
  bool ReadSDRAMData(uint32_t *&region, uint32_t, uint32_t)
  {
    LOG_PRINT(LOG_LEVEL_INFO, "\tPlasticity::WeightDependences::Additive::ReadSDRAMData");

    // Read region parameters
    U032 a2Plus = *region++;
    U032 a2Minus = *region++;
    m_MinWeight = *reinterpret_cast<int32_t*>(region++);
    m_MaxWeight = *reinterpret_cast<int32_t*>(region++);

    // Multiply A2+ and A2- by weight range
    // **NOTE** shift down by 32 bits as A2+ and A2- are in U0.32,
    // m_MinWeight and m_MaxWeight are in runtime-defined weight format
    // and we want the result to be left in the runtime-defined weight format
    // **NOTE** 64-bit casting as result of multiplication is going to be 48-bit
    const int32_t weightRange = m_MaxWeight - m_MinWeight;
    m_A2Plus = ((int64_t)a2Plus * (int64_t)weightRange) >> 32;
    m_MinusA2Minus = -(((int64_t)a2Minus * (int64_t)weightRange) >> 32);

    LOG_PRINT(LOG_LEVEL_INFO, "\t\tA2+:%d, -A2-:%d, Min weight:%d, Max weight:%d",
              m_A2Plus, m_MinusA2Minus, m_MinWeight, m_MaxWeight);

    return true;
  }

private:
  //-----------------------------------------------------------------------------
  // Members
  //-----------------------------------------------------------------------------
  // Potentiation and depression scaling factors in runtime weight format
  int32_t m_A2Plus;
  int32_t m_MinusA2Minus;

  // Minimum and maximum synaptic weight in runtime weight format
  int32_t m_MinWeight;
  int32_t m_MaxWeight;
};
} // WeightDependences
} // Plasticity
} // SynapseProcessor