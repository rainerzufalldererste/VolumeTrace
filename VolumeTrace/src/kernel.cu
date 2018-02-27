#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif // !__CUDACC__

// CUDA
#include "vector_types.h"
#include "vector_functions.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include "default.h"
#include "cudaHelper.h"
#include "Octree.h"

enum
{
	BlockSize = 64,
};

__device__ OctreeNode *g__cuda__pOctData;

__host__ __device__ inline float cheapLength(float3 f) { return f.x + f.y + f.z; }

__device__
uint8_t renderRecursive(uOctPtr_t octNode, ray r, float tmin, float3 blockA, float3 blockB, uint64_t size, mat4x4 &cameraMatrix, uchar4 &__cuda__pRenderBuffer, uOctPtr_t &__cuda__pStreamerData, float3 &cameraPosition, const float maxDistance, int stackSize, int &index)
{
  float tmax;
  float3 pBlock[2];
  float tmins[8];

  OctreeNode node = g__cuda__pOctData[octNode];

  if (!node.m_isSolid)
  {
    return 0;
  }
  else if (size <= 1 || /*cheapLength((blockA + blockB / 2.0f) - cameraPosition) > maxDistance || */stackSize == 20)
  {
    __cuda__pRenderBuffer = cast_float4_to_uchar4(node.m_color);
    return 1;
  }
  else if (node.m_childFlags)
  {
    if (node.m_unloadedChildren)
    {
      __cuda__pStreamerData = octNode;
      __cuda__pRenderBuffer = cast_float4_to_uchar4(node.m_color);
      return 1;
    }
    else
    {
      size >>= 1;

      float3 intersection = r.origin + r.direction * tmin;

      uint8_t found = 0;
      uint8_t childIndex = (intersection.x >= blockA.x + size) + ((intersection.y >= blockA.y + size) << 1) + ((intersection.z >= blockA.z + size) << 2);

      if (node.m_childFlags & (1 << childIndex))
      {
        pBlock[0] = make_float3((childIndex & 1), (childIndex & 2) >> 1, (childIndex & 4) >> 2) * size;
        pBlock[1] = blockB - make_float3(size - pBlock[0].x, size - pBlock[0].y, size - pBlock[0].z);

        pBlock[0] += blockA;
        intersection_distances_no_if(r, pBlock, tmins[childIndex], tmax);

        if (tmins[childIndex] < tmax) // redundant?
        {
          found = renderRecursive(node.m_childIndex + childIndex, r, tmins[childIndex], pBlock[0], pBlock[1], size, cameraMatrix, __cuda__pRenderBuffer, __cuda__pStreamerData, cameraPosition, maxDistance, stackSize + 1, index);

          if (found)
            return found;
        }
      }

      int indexes[7];
      int indexT = 0;

#pragma unroll
      for (size_t i = 0; i < 8; i++)
      {
        if (i == childIndex)
          continue;

        if (node.m_childFlags & (1 << i))
        {
          pBlock[0] = make_float3((i & 1), (i & 2) >> 1, (i & 4) >> 2) * size;
          pBlock[1] = blockB - make_float3(size - pBlock[0].x, size - pBlock[0].y, size - pBlock[0].z);

          pBlock[0] += blockA;

          intersection_distances_no_if(r, pBlock, tmins[i], tmax);

          if (tmins[i] < tmax)
          {
            indexes[indexT] = i;
            indexT++;
          }
        }
      }

      int lswap = indexT;
      int llswap;

      for (size_t i = 0; i < lswap; i++)
      {
        llswap = lswap;

        for (size_t j = 1; j < llswap; j++)
        {
          if (tmins[indexes[j - 1]] > tmins[indexes[j]])
          {
            // swap
            indexes[j - 1] ^= indexes[j];
            indexes[j] ^= indexes[j - 1];
            indexes[j - 1] ^= indexes[j];
            lswap = j;
          }
        }
      }

      llswap = 0;

      while (llswap < indexT)
      {
        pBlock[0] = make_float3((indexes[llswap] & 1), (indexes[llswap] & 2) >> 1, (indexes[llswap] & 4) >> 2) * size;
        pBlock[1] = make_float3(size - pBlock[0].x, size - pBlock[0].y, size - pBlock[0].z);

        found = renderRecursive(node.m_childIndex + indexes[llswap], r, tmins[indexes[llswap]], blockA + pBlock[0], blockB - pBlock[1], size, cameraMatrix, __cuda__pRenderBuffer, __cuda__pStreamerData, cameraPosition, maxDistance, stackSize + 1, index);

        if (found)
          return found;

        llswap++;
      }

      return 0;
    }
  }
  else
  {
    __cuda__pRenderBuffer = cast_float4_to_uchar4(node.m_color);
    return 1;
  }
}

__global__
void renderKernel(size_t count, size_t width, size_t height, size_t layers, vec3 cameraPosition, uchar4 *__cuda__pRenderBuffer, OctreeNode *__cuda__pOctData, uOctPtr_t *__cuda__pStreamerData, mat4x4 cameraMatrix)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int x = i % width;
  int y = i / width;

  if (i >= count) 
  {
    return;
  }

  int renderSize = 1 << layers;
  float4 dir = cameraMatrix * make_float4((x / (float)width - 0.5f), (y / (float)height - 0.5f), 0.725f, 1);
  float3 dir3 = /*normalize*/(*(float3 *)&dir);
  ray r = make_ray(cameraPosition, dir3);
  float3 block[2] = { make_float3(0, 0, 0), make_float3(renderSize, renderSize, renderSize) };
  float tmin, tmax;
  intersection_distances_no_if(r, block, tmin, tmax);

  if(tmin >= tmax || !renderRecursive(1, r, tmin, block[0], block[1], renderSize, cameraMatrix, __cuda__pRenderBuffer[i], __cuda__pStreamerData[i], r.origin, renderSize, 1, i))
    __cuda__pRenderBuffer[i] = make_uchar4(15, 15, 15, 0);
}

extern "C" 
{
  void Init(size_t width, size_t height, uchar4 **p__cuda__pRenderBuffer, size_t gpuOctreeBufferSize, void **p__cuda__pOctreeData, uOctPtr_t **p__cuda__pStreamerData)
  {
    cudaError_t error = cudaMalloc(p__cuda__pRenderBuffer, sizeof(uchar4) * width * height);
    ASSERT(error == cudaSuccess);

    error = cudaMalloc(p__cuda__pOctreeData, gpuOctreeBufferSize);
    ASSERT(error == cudaSuccess);
    error = cudaMemset(*p__cuda__pOctreeData, 0, gpuOctreeBufferSize);
    ASSERT(error == cudaSuccess);

    error = cudaMalloc(p__cuda__pStreamerData, sizeof(uOctPtr_t) * width * height);
    ASSERT(error == cudaSuccess);

    error = cudaDeviceSetLimit(cudaLimitStackSize, 1024 * 40);
    ASSERT(error == cudaSuccess);
  }

	void Render(size_t width, size_t height, size_t layers, vec3 cameraPosition, uchar4 *__cuda__pRenderBuffer, void *__cuda__pOctreeData, uOctPtr_t *__cuda__pStreamerData, mat4x4 cameraMatrix)
	{
    cudaError_t error = cudaMemset(__cuda__pStreamerData, 0, sizeof(uOctPtr_t) * width * height);
    ASSERT(error == cudaSuccess);

    error = cudaMemcpyToSymbol(g__cuda__pOctData, &__cuda__pOctreeData, sizeof(OctreeNode *), 0, cudaMemcpyHostToDevice);
    ASSERT(error == cudaSuccess);

    renderKernel<<<width * height, BlockSize>>>(width * height, width, height, layers, cameraPosition, __cuda__pRenderBuffer, (OctreeNode *)__cuda__pOctreeData, __cuda__pStreamerData, cameraMatrix);

    error = cudaDeviceSynchronize();
    ASSERT(error == cudaSuccess);
	}

  void Cleanup(uchar4 **p__cuda__pRenderBuffer, void **p__cuda__pOctreeData, uOctPtr_t **p__cuda__pStreamerData)
  {
    cudaFree(*p__cuda__pRenderBuffer);
    cudaFree(*p__cuda__pOctreeData);
    cudaFree(*p__cuda__pStreamerData);
  }
}