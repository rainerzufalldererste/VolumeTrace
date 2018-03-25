#include "Octree.h"

enum 
{
  FirstOctreeIndex = 8,
};

OctreeNode::OctreeNode(bool isSolid)
{
  m_isSolid = isSolid;
}

uOctPtr_t OctreeNode::GetChild(uint8_t childIndex, Octree *pOctree, uOctPtr_t parentIndex)
{
  if (m_childIndex == (uOctPtr_t)0)
    return (uOctPtr_t)0;

  if (childIndex > 8)
    return (uOctPtr_t)0;

  if (m_unloadedChildren == 1)
  {
    pOctree->Enqueue(parentIndex);
    return (uOctPtr_t)0;
  }

  return m_childIndex + childIndex;
}

OctreeNode *OctreeNode::AddChild(Octree *pOctree, uint8_t childIndex, uOctPtr_t *pOutChildIndex)
{
  if (childIndex > 8)
    return nullptr;

  m_isSolid = 1;
  m_childFlags |= (1 << childIndex);

  if (m_childIndex == (uOctPtr_t)0)
  {
    m_childIndex = pOctree->GetNewChildIndex();
  }

  *pOutChildIndex = m_childIndex + childIndex;
  OctreeNode *pChildNode = pOctree->GetNode(*pOutChildIndex);
  
  if(pChildNode)
    *pChildNode = OctreeNode(true);

  return pChildNode;
}

bool OctreeNode::IsLeaf()
{
  return m_childIndex == 0;
}

Octree::Octree(uint8_t defaultOctreeDepth)
{
  m_pFile = nullptr;

  for (size_t i = 0; i < FirstOctreeIndex; i++)
    m_nodes.AddEntry();

  m_defaultOctreeDepth = defaultOctreeDepth;
  m_center = uint64_t(1) << (defaultOctreeDepth - 1);
}

Octree::Octree(char *filename, bool streaming)
{
  FILE *pFile = fopen(filename, "r");
  void *pData = MALLOC(uint64_t, 1);

  fread(pData, sizeof(uint64_t), 1, pFile);
  uint64_t nodeCount = ((uint64_t *)pData)[0];

  fread(pData, sizeof(uint64_t), 1, pFile);
  m_center = ((uint64_t *)pData)[0];

  fread(pData, sizeof(uint8_t), 1, pFile);
  m_defaultOctreeDepth = ((uint8_t *)pData)[0];

  FREE(pData);

  if (streaming)
  {
    OctreeNode *pFirstEntry = m_nodes.AddEntry();

    for (int64_t i = 1; i < 8; i++)
      m_nodes.AddEntry();

    fread(pFirstEntry, sizeof(OctreeNode), 8, pFile);

    OctreeNode *pTopNode = GetNode(1);
    
    if (pTopNode->m_isSolid == 1 && pTopNode->m_childFlags != 0)
      pTopNode->m_unloadedChildren = 1;

    m_pFile = pFile;
  }
  else
  {
    size_t blockSize = m_nodes.GetBlockSize();

    for (int64_t i = 0; i < int64_t(nodeCount - blockSize) && i > 0; i += blockSize)
      fread(m_nodes.GetNewBlock(), sizeof(OctreeNode), blockSize, pFile);

    for (size_t i = (nodeCount / blockSize) * blockSize; i < nodeCount; i++)
      fread(m_nodes.AddEntry(), sizeof(OctreeNode), 1, pFile);

    fclose(pFile);
    m_pFile = nullptr;
  }
}

uOctPtr_t Octree::GetNewChildIndex()
{
  uOctPtr_t ret = (uOctPtr_t)m_nodes.Size();

  for (size_t i = 0; i < 8; i++)
    AddNode();

  return ret;
}

OctreeNode * Octree::GetNode(uOctPtr_t index)
{
  return m_nodes.GetPtrAt(index);
}

OctreeNode * Octree::AddNode(vec3u position)
{
  if (m_pFile != nullptr)
    return nullptr;

  uint64_t midx = m_center;
  uint64_t midy = m_center;
  uint64_t midz = m_center;
  uint64_t halfsize = (uint64_t)1 << (uint64_t)(m_defaultOctreeDepth - 1);

  uint8_t layersLeft = m_defaultOctreeDepth;
  OctreeNode *pNode = GetNode(1);
  uOctPtr_t nodeIndex = 1;

  while (layersLeft > 0)
  {
    layersLeft--;
    halfsize >>= 1;

    uint8_t x = (position.x >= midx);
    uint8_t y = (position.y >= midy) << 1;
    uint8_t z = (position.z >= midz) << 2;

    if (x) midx += halfsize;
    else midx -= halfsize;

    if (y) midy += halfsize;
    else midy -= halfsize;

    if (z) midz += halfsize;
    else midz -= halfsize;

    uint8_t index = x | y | z;

    pNode->m_isSolid = 1;

    if (pNode->m_childIndex == 0)
    {
      pNode = pNode->AddChild(this, index, &nodeIndex);
    }
    else
    {
      pNode->m_childFlags |= 1 << index;
      nodeIndex = pNode->m_childIndex + index;
      pNode = GetNode(nodeIndex);
    }
  }

  pNode->m_isSolid = 1;

  return pNode;
}

void Octree::Save(char * filename)
{
  FILE *pFile = fopen(filename, "w");

  uint64_t nodeCount = m_nodes.Size();

  fwrite(&nodeCount, sizeof(uint64_t), 1, pFile);
  fwrite(&m_center, sizeof(uint64_t), 1, pFile);
  fwrite(&m_defaultOctreeDepth, sizeof(uint8_t), 1, pFile);

  size_t blockSize = m_nodes.GetBlockSize();

  for(int64_t i = 0; i < int64_t(nodeCount - blockSize) && i > 0; i += blockSize)
    fwrite(m_nodes.GetPtrAt(i), sizeof(OctreeNode), blockSize, pFile);

  fwrite(m_nodes.GetPtrAt((nodeCount / blockSize) * blockSize), sizeof(OctreeNode), blockSize - (nodeCount % blockSize), pFile);

  fclose(pFile);
}

void Octree::Enqueue(uOctPtr_t parentIndex)
{
  if (m_pFile == nullptr)
    return;

  m_streamerQueue.Enqueue(parentIndex);
}

void Octree::Update()
{
  if (m_pFile == nullptr)
    return;

  for (size_t i = 0; i < m_streamerQueue.Size(); i++)
  {
    OctreeNode *pParent = GetNode(m_streamerQueue[i]);

    if (pParent->m_unloadedChildren == 0)
      continue;

    OctreeNode nodes[8];

    fseek(m_pFile, 2 * sizeof(uint64_t) + sizeof(uint8_t) + pParent->m_childIndex * sizeof(OctreeNode), SEEK_SET);
    fread(nodes, sizeof(OctreeNode), 8, m_pFile);

    for (size_t i = 0; i < 8; i++)
    {
      if (nodes[i].m_isSolid == 1 && nodes[i].m_childFlags != 0)
        nodes[i].m_unloadedChildren = 1;

      nodes[i].m_lastRenderedFrame = m_currentFrame;
    }

    uOctPtr_t destination = GetNewChildIndex();
    OctreeNode *pDestination = GetNode(destination);

    MEMCPY(pDestination, nodes, OctreeNode, 8);

    pParent->m_childIndex = destination;
    pParent->m_unloadedChildren = 0;

    if (m_pUploadFuncCallback)
    {
      (*m_pUploadFuncCallback)(&nodes, 8 * sizeof(OctreeNode), destination * sizeof(OctreeNode));
      (*m_pUploadFuncCallback)(pParent, 1 * sizeof(OctreeNode), m_streamerQueue[i] * sizeof(OctreeNode));
    }
  }

  m_streamerQueue.Flush();

  if (m_pFinishUploadCallback)
    (*m_pFinishUploadCallback)();
}

void Octree::IncreaseFrames()
{
  m_currentFrame++;
}

void Octree::SetMaxSize(size_t maxSize)
{
  m_maxSize = maxSize / sizeof(OctreeNode);
}

void Octree::SetUpload(UploadFunc * pCallback)
{
  m_pUploadFuncCallback = pCallback;

  size_t blockSize = m_nodes.GetBlockSize();

  // TODO: check for maximum size.

  for (int64_t i = 0; i < int64_t(m_nodes.Size() - blockSize); i += blockSize)
    m_pUploadFuncCallback(m_nodes.GetPtrAt(i), blockSize * sizeof(OctreeNode), i);

  size_t lastBlockIndex = (m_nodes.Size() / blockSize) * blockSize;

  if(m_nodes.Size())
    m_pUploadFuncCallback(m_nodes.GetPtrAt(lastBlockIndex), (m_nodes.Size() % blockSize) * sizeof(OctreeNode), lastBlockIndex * sizeof(OctreeNode));
}

void Octree::SetFinishUpload(FinishUploadFunc * pCallback)
{
  m_pFinishUploadCallback = pCallback;
}

float4 AverageColors(OctreeNode *pNode, Octree *pOctree)
{
  if (pNode->m_isSolid && pNode->m_childIndex)
  {
    uint8_t count = 0;
    float4 avgCol = make_float4(0, 0, 0, 0);
    uint8_t flag = 1;

    for (uOctPtr_t i = 0; i < 8; i++)
    {
      if (pNode->m_childFlags & flag)
      {
        avgCol = alphaAdd(avgCol, AverageColors(pOctree->GetNode(pNode->m_childIndex + i), pOctree));
        count++;
      }

      flag <<= 1;
    }

    if (count > 0)
      return pNode->m_color = alphaDivide(avgCol, (float)count);
  }

  return pNode->m_color;
}

void Octree::CalculateParentNodes()
{
  if (m_pFile != nullptr)
    return;

  AverageColors(GetNode(1), this);
}

uint8_t Octree::GetLayerDepth()
{
  return m_defaultOctreeDepth;
}

OctreeNode * Octree::AddNode()
{
  return m_nodes.AddEntry();
}
