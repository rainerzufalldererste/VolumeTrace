#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "default.h"

extern "C"
{
  void Init(size_t width, size_t height, uchar4 **p__cuda__pRenderBuffer);
  void Render(size_t width, size_t height, size_t samples, uchar4 *__cuda__pRenderBuffer);
  void Cleanup(uchar4 **p__cuda__pRenderBuffer);
}

#endif // !__KERNEL_H__