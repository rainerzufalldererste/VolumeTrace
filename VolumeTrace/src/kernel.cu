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
	BlockSize = 512,
};

__device__ OctreeNode *g__cuda__pOctData;

__host__ __device__ inline float cheapLength(float3 f) { return f.x + f.y + f.z; }

__host__ __device__
uint8_t renderRecursive(uOctPtr_t octNode, ray r, float3 blockA, float3 blockB, uint64_t size, mat4x4 &cameraMatrix, uchar4 &__cuda__pRenderBuffer, uOctPtr_t &__cuda__pStreamerData, float3 &cameraPosition, const float maxDistance, int stackSize = 1)
{
  if (stackSize == 10)
  {
    __cuda__pRenderBuffer = make_uchar4(0, 255, 255, 0);
    return stackSize;
  }

  float tmin, tmax;
  float3 pBlock[2] = { blockA, blockB };
  intersection_distances_no_if(r, pBlock, tmin, tmax);

  if (tmin < tmax)
  {
    OctreeNode node = g__cuda__pOctData[octNode];

    if (!node.m_isSolid)
      return 0;
    else if (size <= 1 || cheapLength((blockA + blockB / 2.0f) - cameraPosition) < maxDistance)
    {
      __cuda__pRenderBuffer = cast_float4_to_uchar4(node.m_color);
      return stackSize;
    }
    else if (node.m_childFlags)
    {
      if (node.m_unloadedChildren)
      {
        __cuda__pStreamerData = octNode;
        __cuda__pRenderBuffer = cast_float4_to_uchar4(node.m_color);
        return stackSize;
      }
      else
      {
        size >>= 1;

        float3 intersection = r.origin + r.direction * tmin;

        uint8_t childIndex = (intersection.x >= blockA.x + size) + ((intersection.y >= blockA.y + size) << 1) + ((intersection.z >= blockA.z + size) << 2);

        if (node.m_childFlags & (1 << childIndex))
        {
          float3 offsetA = make_float3((childIndex & 1), (childIndex & 2) >> 1, (childIndex & 4) >> 2) * size;
          float3 offsetB = make_float3(size - offsetA.x, size - offsetA.y, size - offsetA.z);

          return renderRecursive(node.m_childIndex + childIndex, r, blockA + offsetA, blockB - offsetB, size, cameraMatrix, __cuda__pRenderBuffer, __cuda__pStreamerData, cameraPosition, maxDistance, stackSize + 1);
        }
        else
        {
          __cuda__pRenderBuffer = make_uchar4(stackSize * 20, 27, 15, 0);
          return stackSize; //!!!!!!!!!!!!!!
        }
      }
    }
    else
    {
      __cuda__pRenderBuffer = cast_float4_to_uchar4(node.m_color);
      return stackSize;
    }
  }
  else
  {
    return 0;
  }
}

__global__
void renderKernel(size_t count, size_t width, size_t height, size_t samples, uchar4 *__cuda__pRenderBuffer, OctreeNode *__cuda__pOctData, uOctPtr_t *__cuda__pStreamerData, mat4x4 cameraMatrix)
{
  // Load first 48kb of octree into a shared buffer sounds good, doesn't it?
  // or every 100 frames recalculate which nodes should be streamed into every block (messes up indexes...)

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int x = i % width;
  int y = i / width;

  int halfX = x >> 1;
  int halfY = y >> 1;

  if (i >= count) 
  {
    return;
  }

  uint32_t node = 1;
  OctreeNode currentNode = __cuda__pOctData[node];
  int renderSize = 1 << 8;
  uint8_t layer = 0;

  float tmin, tmax;
  float4 dir = cameraMatrix * make_float4((x / (float)width - 0.5f), (y / (float)height - 0.5f), 0.725f, 1);
  float3 dir3 = normalize(*(float3 *)&dir);
  ray r = make_ray(make_float3(-5.f, -5.f, -5.f), dir3);
  float3 block[2] = { make_float3(0, 0, 0), make_float3(renderSize, renderSize, renderSize) };

  if (renderRecursive(1, r, block[0], block[1], renderSize, cameraMatrix, __cuda__pRenderBuffer[i], __cuda__pStreamerData[i], r.origin, 1.0f / width))
  {
    return;
  }
  else
  {
    __cuda__pRenderBuffer[i] = make_uchar4(15, 15, 15, 0);
    return;
  }
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

	void Render(size_t width, size_t height, size_t samples, uchar4 *__cuda__pRenderBuffer, void *__cuda__pOctreeData, uOctPtr_t *__cuda__pStreamerData, mat4x4 cameraMatrix)
	{
    cudaError_t error = cudaMemset(__cuda__pStreamerData, 0, sizeof(uOctPtr_t) * width * height);
    ASSERT(error == cudaSuccess);

    error = cudaMemcpyToSymbol(g__cuda__pOctData, &__cuda__pOctreeData, sizeof(OctreeNode *), 0, cudaMemcpyHostToDevice);
    ASSERT(error == cudaSuccess);

    renderKernel<<<width * height, BlockSize>>>(width * height, width, height, samples, __cuda__pRenderBuffer, (OctreeNode *)__cuda__pOctreeData, __cuda__pStreamerData, cameraMatrix);

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