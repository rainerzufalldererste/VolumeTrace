#ifndef __CUDA__CUDAOCTREE_H__
#define __CUDA__CUDAOCTREE_H__

#include "Octree.h"
#include "default.h"

__host__ __device__
inline uchar4 cast_float4_to_uchar4(float4 input)
{
  return make_uchar4(
    (uint8_t)roundf(input.x * (float)0xFF),
    (uint8_t)roundf(input.y * (float)0xFF),
    (uint8_t)roundf(input.z * (float)0xFF),
    (uint8_t)roundf(input.w * (float)0xFF)
  );
}

#endif // !__CUDA__CUDAOCTREE_H__
