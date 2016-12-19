#include "matrix_generator.h"

// Standard includes
#include <algorithm>

// Rig CPP common includes
#include "rig_cpp_common/log.h"

// Connection builder includes
#include "connector_generator.h"
#include "param_generator.h"

// **YUCK** standard algorithms tend to rely on memcpy under the hood.
// This wraps the SpiNNaker memcpy routine in a suitable form
// it probably belongs somewhere else though
void* memcpy(void* dest, const void* src, std::size_t count)
{
  spin1_memcpy(dest, src, count);
  return dest;
}


//-----------------------------------------------------------------------------
// ConnectionBuilder::MatrixGenerator::Base
//-----------------------------------------------------------------------------
ConnectionBuilder::MatrixGenerator::Base::Base(uint32_t *&region)
{
  m_SignedWeight = *region++;
}
//-----------------------------------------------------------------------------
bool ConnectionBuilder::MatrixGenerator::Base::Generate(uint32_t *synapticMatrixBaseAddress, uint32_t *matrixAddress,
  unsigned int maxRowSynapses, unsigned int weightFixedPoint, unsigned int numPostNeurons,
  unsigned int sizeWords, unsigned int numRows,
  unsigned int vertexPostSlice, unsigned int vertexPreSlice,
  ConnectorGenerator::Base *connectorGenerator,
  const ParamGenerator::Base *delayGenerator,
  const ParamGenerator::Base *weightGenerator,
  MarsKiss64 &rng) const
{
  // Calculate the maximum number of words in a row
  const unsigned int maxRowWords = GetMaxRowWords(maxRowSynapses);

  // End address of matrix
  uint32_t *endAddress = matrixAddress + sizeWords;

  // Ragged section of matrix begins at matrix address
  uint32_t *raggedMatrixAddress = matrixAddress;

  // Delay extension section of matrix begins after padded ragged matrix
  uint32_t *delayMatrixAddress = matrixAddress + ((NumHeaderWords + maxRowWords) * numRows);

  // Loop through rows
  unsigned int numSynapses = 0;
  for(unsigned int i = 0; i < numRows; i++)
  {
    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\tRow %u", i);

    // Generate postsynaptic indices for row
    // **TODO** these can be 16 bit and thus we can 
    // have twice as many to allow for more multapses
    uint32_t indices[1024];
    LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tGenerating indices");
    const unsigned int numIndices = connectorGenerator->Generate(
      i, numPostNeurons, vertexPostSlice, vertexPreSlice, rng, indices);
    TraceUInt(indices, numIndices);

    // If this row should be empty
    if(numIndices == 0)
    {
      // Write zeros for the number of synapses in
      // row and for the delay extension fields
      *raggedMatrixAddress++ = 0;
      *raggedMatrixAddress++ = 0;
      *raggedMatrixAddress++ = 0;

      // Skip to correct address to start next ragged row
      raggedMatrixAddress += maxRowWords;
    }
    // Otherwise
    else
    {
      // Generate delays for each index
      int32_t delays[1024];
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tGenerating delays");
      delayGenerator->Generate(numIndices, 0, rng, delays);
      TraceInt(delays, numIndices);

      // Generate weights for each index
      int32_t weights[1024];
      LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tGenerating weights");
      weightGenerator->Generate(numIndices, weightFixedPoint, rng, weights);
      TraceInt(weights, numIndices);

      // Update total number of synapses
      numSynapses += numIndices;

      // Generate indices so as to begin partitioning
      // with row in the order it was generated
      uint16_t sortedRowIndices[1024];
      for(unsigned int i = 0; i < numIndices; i++)
      {
        sortedRowIndices[i] = (uint16_t)i;
      }

      // First sub-row starts at next ragged address
      uint32_t *rowAddress = raggedMatrixAddress;

      // There is no previous sub-row at this point
      uint32_t *previousSubRowDelayAddress = NULL;
      uint32_t previousSubRowStartDelay = 0;

      // Loop through possible sub-row delay ranges
      uint16_t *subRowStartIndex = &sortedRowIndices[0];
      uint16_t *subRowEndIndex = &sortedRowIndices[numIndices];
      for(int32_t startDelay = 1; subRowStartIndex != subRowEndIndex; startDelay += MaxDTCMDelaySlots)
      {
        // Is this the first sub-row?
        const bool firstSubRow = (startDelay == 1);
        const int32_t endDelay = startDelay + MaxDTCMDelaySlots;

        // Indirectly partition the delays to determine which are in current sub-row
        uint16_t *newSubRowStartIndex = std::partition(subRowStartIndex, subRowEndIndex,
          [delays, endDelay](uint16_t i)
          {
            return delays[i] < endDelay;
          }
        );

        // Calculate the number of synapses in this section of the sub-row
        unsigned int numSubRowSynapses = newSubRowStartIndex - subRowStartIndex;

        // If there are any synapses in the sub-row or
        // this is the first sub-row - which is always written
        if(numSubRowSynapses > 0 || firstSubRow)
        {
          LOG_PRINT(LOG_LEVEL_TRACE, "\t\t\t\tSub-row (%08x) with delay [%u, %u) - %u synapses",
            rowAddress, startDelay, endDelay, numSubRowSynapses);

          // If this is the first sub-row and we have exceeded
          // the maximum number of synapses per ragged-row
          if(firstSubRow && numSubRowSynapses > maxRowSynapses)
          {
            LOG_PRINT(LOG_LEVEL_WARN, "Generated matrix with %u synapses in first sub-row when maximum is %u",
                      numSubRowSynapses, maxRowSynapses);

            // Reduce number of synapses to maximum
            numSubRowSynapses = maxRowSynapses;
          }

          // If the resulting sub-row is too long for
          // the synapse processor's delay buffer
          if(numSubRowSynapses > 1024)
          {
            LOG_PRINT(LOG_LEVEL_WARN, "Generated matrix with %u synapses in row when only 1024 are supported",
                      numSubRowSynapses);

            // Reduce number of synapses to maximum
            numSubRowSynapses = 1024;
          }

          // If this isn't the first sub-row
          if(!firstSubRow)
          {
            // Build a row offset-length object based on this row's number of
            // synapses and its offset from the start of the matrix
            RowOffsetLength rowOffsetLength(numSubRowSynapses,
                                            rowAddress - synapticMatrixBaseAddress);

            // Write row's delay offset and the word representation of it's
            // offset and length to the correct part of the previous sub-row
            previousSubRowDelayAddress[0] = startDelay - previousSubRowStartDelay;
            previousSubRowDelayAddress[1] = rowOffsetLength.GetWord();
          }

          // Write number of indices in sub-row to correct address
          *rowAddress++ = numSubRowSynapses;

          // The next sub-row will want to write its delay details
          // here so cache its pointer and uint32_t *synapticMatrixBaseAddress, its starting delay
          previousSubRowDelayAddress = rowAddress;
          previousSubRowStartDelay = startDelay;

          // BUT, incase they don't zero delay extension entries
          *rowAddress++ = 0;
          *rowAddress++ = 0;

          // As sub-row has now been potentially truncated to fit
          // within ragged matrix, calculate new pointer to end index
          const uint16_t *subRowEndIndex = subRowStartIndex + numSubRowSynapses;

          // Write row
          unsigned int rowWords = WriteRow(rowAddress, startDelay - 1,
                                           subRowStartIndex, subRowEndIndex,
                                           indices, delays, weights);

          // If this row went past end of memory allocated for matrix
          if((rowAddress + rowWords) > endAddress)
          {
            LOG_PRINT(LOG_LEVEL_ERROR, "Matrix overflowed memory allocated for it");
            return false;
          }

          // If this is the first delay sub-row, advance the
          // ragged matrix past the padded row and update row
          // address so it writes to the start of the delay matrix
          if(firstSubRow)
          {
            raggedMatrixAddress += (NumHeaderWords + maxRowWords);
            rowAddress = delayMatrixAddress;
          }
          // Otherwise, advance the delay matrix address past
          // the sub-row and update row-address so it writes here
          else
          {
            delayMatrixAddress += (NumHeaderWords + rowWords);
            rowAddress = delayMatrixAddress;
          }
        }

        // Advance to next sub-row
        subRowStartIndex = newSubRowStartIndex;
      }
    }
  }

  LOG_PRINT(LOG_LEVEL_INFO, "\t\tGenerated %u synapses", numSynapses);
  return true;
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::MatrixGenerator::Base::TraceUInt(uint32_t (&values)[1024],
                                                         unsigned int number) const
{
#if LOG_LEVEL <= LOG_LEVEL_TRACE
  for(unsigned int i = 0; i < number; i++)
  {
    io_printf(IO_BUF, "%u,", values[i]);
  }
  io_printf(IO_BUF, "\n");
#endif
}
//-----------------------------------------------------------------------------
void ConnectionBuilder::MatrixGenerator::Base::TraceInt(int32_t (&values)[1024],
                                                        unsigned int number) const
{
#if LOG_LEVEL <= LOG_LEVEL_TRACE
  for(unsigned int i = 0; i < number; i++)
  {
    io_printf(IO_BUF, "%u,", values[i]);
  }
  io_printf(IO_BUF, "\n");
#endif
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::MatrixGenerator::Static
//-----------------------------------------------------------------------------
ConnectionBuilder::MatrixGenerator::Static::Static(uint32_t *&region) : Base(region)
{
  LOG_PRINT(LOG_LEVEL_INFO, "\t\tStatic synaptic matrix: %u signed weights",
    IsSignedWeight());
}
//-----------------------------------------------------------------------------
unsigned int ConnectionBuilder::MatrixGenerator::Static::WriteRow(uint32_t *rowAddress,
  int32_t startDelay, const uint16_t *subRowStartIndex, const uint16_t *subRowEndIndex,
  const uint32_t (&indices)[1024], const int32_t (&delays)[1024], const int32_t (&weights)[1024]) const
{
  // Loop through newly-sorted sub-row indices
  for(const uint16_t *j = subRowStartIndex; j != subRowEndIndex; j++)
  {
    // Extract index pointed to by sorted index
    const uint32_t postIndex = indices[*j];

    // Clamp delays and weights pointed to be sorted index
    const int32_t delay = ClampDelay(delays[*j] - startDelay);
    const int32_t weight = ClampWeight(weights[*j]);

    // Build synaptic word
    const uint32_t word = (postIndex & IndexMask) |
      (((uint32_t)delay & DelayMask) << IndexBits) |
      (weight << (DelayBits + IndexBits));

#if LOG_LEVEL <= LOG_LEVEL_TRACE
    io_printf(IO_BUF, "%u,", word);
#endif
    // Write word to matrix
    *rowAddress++ = word;
  }

#if LOG_LEVEL <= LOG_LEVEL_TRACE
  io_printf(IO_BUF, "\n");
#endif

  // Return number of words written to row
  return (subRowEndIndex - subRowStartIndex);
}
//-----------------------------------------------------------------------------
unsigned int ConnectionBuilder::MatrixGenerator::Static::GetMaxRowWords(unsigned int maxRowSynapses) const
{
  return maxRowSynapses;
}

//-----------------------------------------------------------------------------
// ConnectionBuilder::MatrixGenerator::Plastic
//-----------------------------------------------------------------------------
ConnectionBuilder::MatrixGenerator::Plastic::Plastic(uint32_t *&region) : Base(region)
{
  // Read number of presynaptic state words from region
  const uint32_t preStateBytes = *region++;
  m_SynapseTraceBytes = *region++;

  // Round up to words
  m_PreStateWords = (preStateBytes / 4)
      + (((preStateBytes & 3) != 0) ? 1 : 0);

  LOG_PRINT(LOG_LEVEL_INFO, "\t\tPlastic synaptic matrix: %u signed weights, %u bytes presynaptic state (%u words), %u bytes synapse trace",
            IsSignedWeight(), preStateBytes, m_PreStateWords, m_SynapseTraceBytes);
}
//-----------------------------------------------------------------------------
unsigned int ConnectionBuilder::MatrixGenerator::Plastic::WriteRow(uint32_t *rowAddress,
  int32_t startDelay, const uint16_t *subRowStartIndex, const uint16_t *subRowEndIndex,
  const uint32_t (&indices)[1024], const int32_t (&delays)[1024], const int32_t (&weights)[1024]) const
{
  // Zero presynaptic state words
  for(unsigned int w = 0; w < m_PreStateWords; w++)
  {
    *rowAddress++ = 0;
  }

  // Calculate row length in synapses
  const unsigned int numIndices = subRowEndIndex - subRowStartIndex;

  // Calculate the size of the plastic and control parts of row
  const unsigned int numPlasticArrayWords = GetNumPlasticWords(numIndices);
  const unsigned int numControlArrayWords = GetNumControlWords(numIndices);

  // From this get 8-bit pointer to synapses and 16-bit pointer to control half words
  uint8_t *synapseAddress = reinterpret_cast<uint8_t*>(rowAddress);
  uint16_t *controlAddress = reinterpret_cast<uint16_t*>(rowAddress + numPlasticArrayWords);

  // Loop through newly-sorted sub-row indices
  for(const uint16_t *j = subRowStartIndex; j != subRowEndIndex; j++)
  {
    // Extract index pointed to by sorted index
    const uint32_t postIndex = indices[*j];

    // Clamp delays and weights pointed to be sorted index
    const int32_t delay = ClampDelay(delays[*j] - startDelay);
    const int32_t weight = ClampWeight(weights[*j]);

    // Write weight to first two synapse bytes
    uint16_t *weightAddress = reinterpret_cast<uint16_t*>(synapseAddress);
    *weightAddress = (uint16_t)weight;

    // Go onto next synapse
    synapseAddress += 2;

    // Write synapse trace bytes
    for(unsigned int s = 0; s < m_SynapseTraceBytes; s++)
    {
      *synapseAddress++ = 0;
    }

    // Build control word
    const uint16_t controlWord = (uint16_t)(postIndex & IndexMask) |
      (((uint32_t)delay & DelayMask) << IndexBits);

#if LOG_LEVEL <= LOG_LEVEL_TRACE
    io_printf(IO_BUF, "%u/%u,", weight, controlWord);
#endif
      // Write control word
    *controlAddress++ = controlWord;
  }

#if LOG_LEVEL <= LOG_LEVEL_TRACE
  io_printf(IO_BUF, "\n");
#endif

  // Return total size
  return m_PreStateWords + numPlasticArrayWords + numControlArrayWords;
}
//-----------------------------------------------------------------------------
unsigned int ConnectionBuilder::MatrixGenerator::Plastic::GetMaxRowWords(unsigned int maxRowSynapses) const
{
  return m_PreStateWords + GetNumPlasticWords(maxRowSynapses) + GetNumControlWords(maxRowSynapses);
}
