#ifndef __CUDACC__
#include "device_launch_parameters.h"
#endif // !__CUDACC__

#include "vector_types.h"
#include "vector_functions.h"
#include "cuda_runtime.h"
#include "default.h"

enum
{
	BlockSize = 512,
};

__global__
void renderKernel(size_t count, size_t width, size_t height, size_t samples, float3 *__cuda__pRenderBuffer)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int x = i % width;
  int y = i / width;

  if (i >= count)
    return;

  __cuda__pRenderBuffer[i] = make_float3(x / (float)width, y / (float)height, (x + y) / (float)(width + height));
}

extern "C" 
{
  void Init(size_t width, size_t height, float3 **p__cuda__pRenderBuffer)
  {
    cudaMalloc(p__cuda__pRenderBuffer, sizeof(float3) * width * height);
  }

	void Render(size_t width, size_t height, size_t samples, float3 *__cuda__pRenderBuffer)
	{
    renderKernel<<<width * height, BlockSize>>>(width * height, width, height, samples, __cuda__pRenderBuffer);

    cudaDeviceSynchronize();
	}

  void Cleanup(float3 **p__cuda__pRenderBuffer)
  {
    cudaFree(*p__cuda__pRenderBuffer);
  }
}