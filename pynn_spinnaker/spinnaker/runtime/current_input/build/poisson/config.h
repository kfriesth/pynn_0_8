#pragma once

// Rig CPP common includes
#include "rig_cpp_common/random/mars_kiss64.h"

// Common includes
#include "../../../common/poisson_source.h"

namespace CurrentInput
{
//-----------------------------------------------------------------------------
// Typedefines
//-----------------------------------------------------------------------------
typedef Common::PoissonSource<MarsKiss64> Source;
};