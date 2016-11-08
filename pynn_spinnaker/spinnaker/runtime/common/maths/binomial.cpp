#include "binomial.h"

// Common includes
#include "../fixed_point_number.h"
#include "../random/mars_kiss64.h"
#include "ln.h"
#include "recip.h"

// Namespaces
using namespace Common::Maths;
using namespace Common::FixedPointNumber;

//-----------------------------------------------------------------------------
// Anonymous namespace
//-----------------------------------------------------------------------------
namespace
{

uint32_t randbin_bg_core(uint32_t n, S1615 ln_1_min_p,
			 MarsKiss64 &rng)
{
  uint32_t y = 0, x = 0;
  if (ln_1_min_p >= 0)
    return x;

  // TODO implement reciprocal for negative values
  // replace neg_recip with recip
  // replace -Ln(u) with Ln(u) below
  S1615 neg_recip_ln_1_min_p = Reciprocal(-ln_1_min_p);

  while (1)
  {
    S1615 u = (S1615)(rng.GetNext() & 0x00007fff);
    y += (MulS1615(-Ln(u), neg_recip_ln_1_min_p) >> 15) + 1;
    if (y > n)
      break;
    x += 1;
  }

  return x;
}

} // anonymous namespace

//-----------------------------------------------------------------------------
// Common::Maths
//-----------------------------------------------------------------------------
namespace Common
{
namespace Maths
{

uint32_t Binomial(uint32_t n, S1615 p, MarsKiss64 &rng)
{
  if (p > 16384)
    return n - randbin_bg_core(n, Ln(p), rng);
  else
    return randbin_bg_core(n, Ln(32768-p), rng);
}

uint32_t Binomial(uint32_t n, uint32_t num, uint32_t denom, MarsKiss64 &rng)
{
  if ((num<<1) > denom)
    return n - randbin_bg_core(n, Ln((int32_t)num) - Ln((int32_t)denom), rng);
  else
    return randbin_bg_core(n, Ln((int32_t)denom-(int32_t)num) - Ln((int32_t)denom), rng);
}
  
} // Maths
} // Common
