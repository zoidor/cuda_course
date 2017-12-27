// for uchar4 struct
#include <cuda_runtime.h>

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols)
{
  for (size_t r = 0; r < numRows; ++r) {
    for (size_t c = 0; c < numCols; ++c) {
      const int pos = r * numCols + c;
      uchar4 rgba = rgbaImage[pos];
      float channelSum = (.299f * rgba.x + .587f * rgba.y + .114f * rgba.z);
      greyImage[pos] = channelSum;
    }
  }
}

