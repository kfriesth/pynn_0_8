// Common includes
#include "../fixed_point_number.h"

// Maths includes
#include "recip.h"

// Namespaces
using namespace Common::Maths;
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{

const S1615 recips[64] = {
  327680, 283398, 249660, 223101, 201649, 183960, 169125, 156503,
  145635, 136178, 127875, 120525, 113975, 108100, 102801,  97997,
  93622,  89621,  85948,  82565,  79437,  76538,  73843,  71331,
  68985,  66788,  64726,  62788,  60963,  59241,  57614,  56073,
  54613,  53227,  51909,  50655,  49461,  48321,  47233,  46192,
  45197,  44243,  43329,  42452,  41610,  40800,  40021,  39272,
  38550,  37854,  37183,  36535,  35910,  35305,  34721,  34155,
  33608,  33078,  32564,  32066,  31583,  31115,  30660,  30218
};

S1615 reciprocal_core(S1615 x)
{
  int32_t i0 = (x - 3276) >> 9;
  S1615 x0 = (i0 << 9) + 3276;
    
  S1615 y = recips[i0];
  S1615 h = (x - x0);
    
  S1615 k1 = -MulS1615(y, y);
  
  S1615 k2 = y + MulS1615(h, k1>>1);
  k2 = -MulS1615(k2, k2);

  S1615 k3 = y + MulS1615(h, k2>>1);
  k3 = -MulS1615(k3, k3);

  S1615 k4 = y + MulS1615(h, k3);
  k4 = -MulS1615(k4, k4);
  
  y = y + MulS1615(MulS1615(h, (k1 + (k2 << 1) + (k3 << 1) + k4)), 5461);
    
  return y;

}
  
}

//-----------------------------------------------------------------------------
// Common::Maths
//-----------------------------------------------------------------------------
namespace Common
{
namespace Maths
{

// TODO we only handle the positive case here
S1615 Reciprocal(S1615 x)
{
  int32_t s = 0;
  while (x >= 36044)
  {
    x >> 1;
    s -= 1;
  }
  while (x < 3276)
  {
    x << 1;
    s += 1;
  }

  return (reciprocal_core(x) << s);
}
    
} // Maths
} // Common
