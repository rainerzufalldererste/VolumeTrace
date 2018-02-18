#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif // !__CUDACC__

// CUDA
#include "vector_types.h"
#include "vector_functions.h"
#include "cuda_runtime.h"

#include "default.h"
#include "cudaHelper.h"
#include "Octree.h"

enum
{
	BlockSize = 512,
};

__global__
void renderKernel(size_t count, size_t width, size_t height, size_t samples, uchar4 *__cuda__pRenderBuffer, OctreeNode *__cuda__pOctData)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int x = i % width;
  int y = i / width;

  if (i >= count) 
  {
    return;
  }

  uint32_t node = 1;
  OctreeNode currentNode = __cuda__pOctData[node];
  int renderSize = 1 << 8;
  uint8_t layer = 0;

  x -= 128;
  y -= 128;

  if (x < 0 || x >= renderSize || y < 0 || y >= renderSize)
  {
    __cuda__pRenderBuffer[i] = make_uchar4(15, 15, 15, 0);
    return;
  }

  while (node != 0 && renderSize > 0)
  {
    renderSize >>= 1;
    layer++;

    if (currentNode.m_isSolid)
    {
      if (currentNode.m_unloadedChildren == 1)
      {
        // TODO: Tell streamer.
        break;
      }
      else if (currentNode.m_childFlags != 0)
      {
        uint8_t xx = x >= renderSize;
        uint8_t yy = (y >= renderSize) * 2;
        uint8_t zz = 0;

        uint8_t nextNodeIndex = (xx | yy | zz);
        uint8_t nextNodeFlag = 1 << nextNodeIndex;

        if ((currentNode.m_childFlags & nextNodeFlag) != 0)
        {
          node = currentNode.m_childIndex + nextNodeIndex;
          currentNode = __cuda__pOctData[node];
        }
        else
        {
          nextNodeIndex += 4; // the node behind this node. (z = 4)
          nextNodeFlag <<= 4;

          if ((currentNode.m_childFlags & nextNodeFlag) != 0)
          {
            node = currentNode.m_childIndex + nextNodeIndex;
            currentNode = __cuda__pOctData[node];
          }
          else
          {
            node = 0;
            currentNode = __cuda__pOctData[node];

            goto END;
          }
        }
      }
      else
      {
        goto END;
      }
    }
    else
    {
      goto END;
    }
  }

  END:

  __cuda__pRenderBuffer[i] = cast_float4_to_uchar4(currentNode.m_color);
}

extern "C" 
{
  void Init(size_t width, size_t height, uchar4 **p__cuda__pRenderBuffer, size_t gpuOctreeBufferSize, void **p__cuda__pOctreeData)
  {
    cudaMalloc(p__cuda__pRenderBuffer, sizeof(uchar4) * width * height);

    cudaMalloc(p__cuda__pOctreeData, gpuOctreeBufferSize);
    cudaMemset(p__cuda__pOctreeData, 0, gpuOctreeBufferSize);
  }

	void Render(size_t width, size_t height, size_t samples, uchar4 *__cuda__pRenderBuffer, void *__cuda__pOctreeData)
	{
    renderKernel<<<width * height, BlockSize>>>(width * height, width, height, samples, __cuda__pRenderBuffer, (OctreeNode *)__cuda__pOctreeData);

    ASSERT(cudaDeviceSynchronize() == cudaSuccess);
	}

  void Cleanup(uchar4 **p__cuda__pRenderBuffer, void **p__cuda__pOctreeData)
  {
    cudaFree(*p__cuda__pRenderBuffer);
    cudaFree(*p__cuda__pOctreeData);
  }
}