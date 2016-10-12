#include "if.h"

#include "../../common/spinnaker.h"

//-----------------------------------------------------------------------------
// NeuronProcessor::NeuronDynamicsModels::IF
//-----------------------------------------------------------------------------
namespace NeuronProcessor
{
namespace NeuronDynamicsModels
{
void IF::Print(char *stream, const MutableState &mutableState, const ImmutableState &immutableState)
{
  io_printf(stream, "\tMutable state:\n");
  io_printf(stream, "\t\tV_Membrane       = %11.4k [mV]\n", mutableState.m_V_Membrane);
  io_printf(stream, "\t\tRefractoryTimer  = %10d [timesteps]\n", mutableState.m_RefractoryTimer);

  io_printf(stream, "\tImmutable state:\n");
  io_printf(stream, "\t\tV_Reset          = %11.4k [mV]\n", immutableState.m_V_Reset);
  io_printf(stream, "\t\tV_Rest           = %11.4k [mV]\n", immutableState.m_V_Rest);
  io_printf(stream, "\t\tI_Offset         = %11.4k [nA]\n", immutableState.m_I_Offset);
  io_printf(stream, "\t\tR_Membrane       = %11.4k [MegaOhm]\n", immutableState.m_R_Membrane);
  io_printf(stream, "\t\tExpTC            = %11.4k\n", immutableState.m_ExpTC);
  io_printf(stream, "\t\tT_Refractory     = %10d [timesteps]\n", immutableState.m_T_Refractory);
}
};  // namespace NeuronModels
};  // namespace NeuronProcessor