#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "default.h"

extern "C"
{
  void Init(size_t width, size_t height, float3 **p__cuda__pRenderBuffer);
  void Render(size_t width, size_t height, size_t samples, float3 *__cuda__pRenderBuffer);
  void Cleanup(float3 **p__cuda__pRenderBuffer);
}

#endif // !__KERNEL_H__