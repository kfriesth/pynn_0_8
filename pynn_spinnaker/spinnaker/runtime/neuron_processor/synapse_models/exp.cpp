#include "exp.h"

// Rig CPP common includes
#include "rig_cpp_common/spinnaker.h"

//-----------------------------------------------------------------------------
// NeuronProcessor::SynapseModels::Exp
//-----------------------------------------------------------------------------
namespace NeuronProcessor
{
namespace SynapseModels
{
void Exp::Print(char *stream, const MutableState &mutableState, const ImmutableState &immutableState)
{
  io_printf(stream, "\tMutable state:\n");
  io_printf(stream, "\t\tm_ISynExc        = %11.4k [nA]\n", mutableState.m_ISynExc);
  io_printf(stream, "\t\tm_ISynInh        = %11.4k [nA]\n", mutableState.m_ISynInh);

  io_printf(stream, "\tImmutable state:\n");
  io_printf(stream, "\t\tExpTauSynExc      = %11.4k\n", (S1615)(immutableState.m_ExpTauSynExc >> 17));
  io_printf(stream, "\t\tInitExc           = %11.4k [nA]\n", immutableState.m_InitExc);
  io_printf(stream, "\t\tExpTauSynInh      = %11.4k\n", (S1615)(immutableState.m_ExpTauSynInh >> 17));
  io_printf(stream, "\t\tInitInh           = %11.4k [nA]\n", immutableState.m_InitInh);
}
};  // namespace NeuronModels
};  // namespace NeuronProcessor