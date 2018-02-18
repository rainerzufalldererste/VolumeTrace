#ifndef __OCTREE_H__
#define __OCTREE_H__

#include "default.h"
#include "vector"
#include "ChunkedArray.h"

using namespace std;

typedef uint32_t uOctPtr_t;

class Octree;

class OctreeNode
{
public:
  OctreeNode(bool isSolid = false);
  
  uOctPtr_t GetChild(uint8_t childIndex);
  static OctreeNode *AddChild(uOctPtr_t parentIndex, Octree *pOctree, uint8_t childIndex, uOctPtr_t *pChildIndex);
  bool IsLeaf();

  uOctPtr_t m_childIndex = 0;
  uint8_t m_childFlags = 0;
  uint8_t m_isSolid : 1;
  uint8_t m_unusedFlags : 7;
  float4 m_color;
  uint8_t m_layersToChange = 0xFF;
};

class Octree
{
public:
  Octree(uint8_t defaultOctreeDepth);
  Octree(char *filename);
  uOctPtr_t GetNewChildIndex();
  OctreeNode *GetNode(uOctPtr_t index);
  OctreeNode *AddNode(ulonglong3 position);
  void Save(char *filename);

private:
  ChunkedArray<OctreeNode, 256> m_nodes;
  uint8_t m_defaultOctreeDepth;
  uint64_t m_center;

  OctreeNode *AddNode();
};

#endif // !__OCTREE_H__
