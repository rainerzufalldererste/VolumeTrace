#ifndef __OCTREE_H__
#define __OCTREE_H__

#include "default.h"
#include "ChunkedArray.h"
#include "FlushQueue.h"

using namespace std;

typedef uint32_t uOctPtr_t;
typedef void (UploadFunc)(void * pData, size_t size, size_t position);
typedef void (FinishUploadFunc)();

class Octree;

class OctreeNode
{
public:
  OctreeNode(bool isSolid = false);
  
  uOctPtr_t GetChild(uint8_t childIndex, Octree *pOctree, uOctPtr_t parentIndex);
  OctreeNode *AddChild(Octree *pOctree, uint8_t childIndex, uOctPtr_t *pOutChildIndex);
  bool IsLeaf();

  uOctPtr_t m_childIndex = 0;
  uint8_t m_childFlags = 0;
  uint8_t m_isSolid : 1;
  uint8_t m_unloadedChildren : 1;
  uint8_t m_unusedFlags : 6;
  float4 m_color;
  uint8_t m_layersToChange = 0xFF;
  uint64_t m_lastRenderedFrame; // we don't need to save this.
};

class Octree
{
public:
  Octree(uint8_t defaultOctreeDepth);
  Octree(char *filename, bool streaming);
  uOctPtr_t GetNewChildIndex();
  OctreeNode *GetNode(uOctPtr_t index);
  OctreeNode *AddNode(ulonglong3 position);
  void Save(char *filename);
  void Enqueue(uOctPtr_t parentIndex);
  void Update();
  void IncreaseFrames();
  void SetMaxSize(size_t maxSize);
  void SetUpload(UploadFunc *pCallback);
  void SetFinishUpload(FinishUploadFunc *pCallback);

private:
  ChunkedArray<OctreeNode, 256> m_nodes; // blockSize no less than 8
  FlushQueue<uOctPtr_t, 1024> m_streamerQueue;
  uint8_t m_defaultOctreeDepth;
  uint64_t m_center;
  FILE *m_pFile;
  uint64_t m_currentFrame = 0;
  size_t m_maxSize = 0;
  UploadFunc *m_pUploadFuncCallback = nullptr;
  FinishUploadFunc *m_pFinishUploadCallback = nullptr;

  OctreeNode *AddNode();
};

#endif // !__OCTREE_H__
