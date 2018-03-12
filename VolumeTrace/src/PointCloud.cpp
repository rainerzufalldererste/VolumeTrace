#include "PointCloud.h"

PointCloud::PointCloud(const char * filename)
{
  FILE *pFile = fopen(filename, "r");

  ASSERT(pFile);

  fseek(pFile, 0, SEEK_END);

  size_t size = ftell(pFile);
  m_size = size / sizeof(Point);
  fclose(pFile);

  printf("Reading %llu Points from '%s'...\n", m_size, filename);

  m_pPoints = MALLOC(Point, m_size);

  pFile = fopen(filename, "r");
  //fseek(pFile, 0, SEEK_SET);
  fread(m_pPoints, sizeof(Point), m_size, pFile);
  fclose(pFile);

  m_size = 50;

  if (m_size)
  {
    m_extents = vec3i(INT64_MIN);
    m_offset = vec3i(INT64_MAX);

    for (size_t i = 0; i < m_size; i++)
    {
      Point p = m_pPoints[i];

      if ((int64_t)p.x > m_extents.x)
        m_extents.x = (int64_t)p.x;
      if ((int64_t)p.y > m_extents.y)
        m_extents.y = (int64_t)p.y;
      if ((int64_t)p.z > m_extents.z)
        m_extents.z = (int64_t)p.z;

      if ((int64_t)p.x < m_offset.x)
        m_offset.x = (int64_t)p.x;
      if ((int64_t)p.y < m_offset.y)
        m_offset.y = (int64_t)p.y;
      if ((int64_t)p.z < m_offset.z)
        m_offset.z = (int64_t)p.z;
    }

    m_extents;// -= m_offset;
  }
}

PointCloud::~PointCloud()
{
  FREE(m_pPoints);
}

void PointCloud::GetExtents(vec3i * pExtents, vec3i * pOffset)
{
  *pExtents = m_extents;
  *pOffset = m_offset;
}

size_t PointCloud::Size()
{
  return m_size;
}

Octree * PointCloud::GetOctree(uint8_t layerDepth)
{
  Octree *pOctree = new Octree(layerDepth);
  vec3i minPos = m_offset + vec3i(1LL << layerDepth);

  for (size_t i = 0; i < m_size; i++)
  {
    Point p = m_pPoints[i];

    if (p.x >= minPos.x && p.y >= minPos.y && p.z >= minPos.z)
      continue;

    pOctree->AddNode(
      vec3u
      (
        (uint64_t)((int64_t)p.x - m_offset.x), 
        (uint64_t)((int64_t)p.y - m_offset.y), 
        (uint64_t)((int64_t)p.z - m_offset.z)
      )
    )->m_color = float4_from_uint32_t(p.color);

    if(i % 1000000 == 0)
      printf("\rAdded %llu Million / %llu Million Voxels to Octree.", i / 1000000, m_size / 1000000);
  }

  printf("\n");

  return pOctree;
}

Point PointCloud::operator[](size_t index)
{
  ASSERT(index < m_size);

  return m_pPoints[index];
}
