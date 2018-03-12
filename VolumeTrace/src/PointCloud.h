#ifndef __POINTCLOUD_H__
#define __POINTCLOUD_H__

#include "default.h"
#include "Octree.h"

struct Point
{
  int32_t x;
  int32_t y;
  int32_t z;
  uint32_t color;
};

class PointCloud
{
public:
  PointCloud(const char * filename);
  ~PointCloud();

  void GetExtents(vec3i *pExtents, vec3i *pOffset);
  size_t Size();
  Octree *GetOctree(uint8_t layerDepth);

  Point operator [] (size_t index);

private:
  vec3i m_extents;
  vec3i m_offset;
  size_t m_size;
  Point *m_pPoints;
};

#endif // !__POINTCLOUD_H__
