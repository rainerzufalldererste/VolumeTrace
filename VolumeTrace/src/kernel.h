#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "default.h"
#include "Octree.h"

extern "C"
{
  void Init(size_t width, size_t height, uchar4 **p__cuda__pRenderBuffer, size_t gpuOctreeBufferSize, void **p__cuda__pOctreeData, uOctPtr_t **p__cuda__pStreamerData);
  void Render(size_t width, size_t height, size_t layers, vec3 cameraPosition, uchar4 *__cuda__pRenderBuffer, void *p__cuda__pOctreeData, uOctPtr_t *__cuda__pStreamerData, mat4x4 cameraMatrix);
  void Cleanup(uchar4 **p__cuda__pRenderBuffer, void **p__cuda__pOctreeData, uOctPtr_t **p__cuda__pStreamerData);
}

#endif // !__KERNEL_H__