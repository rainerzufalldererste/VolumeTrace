#include "Octree.h"

enum 
{
  FirstOctreeIndex = 8,
};

OctreeNode::OctreeNode(bool isSolid)
{
  m_isSolid = isSolid;
}

uOctPtr_t OctreeNode::GetChild(uint8_t childIndex)
{
  if (childIndex > 8)
    return (uOctPtr_t)0;

  if (m_childIndex == (uOctPtr_t)0)
    return (uOctPtr_t)0;

  return m_childIndex + childIndex;
}

OctreeNode *OctreeNode::AddChild(uOctPtr_t parentIndex, Octree *pOctree, uint8_t childIndex, uOctPtr_t *pChildIndex)
{
  if (childIndex > 8)
    return nullptr;

  OctreeNode *pNode = pOctree->GetNode(parentIndex);

  pNode->m_isSolid = 1;
  pNode->m_childFlags |= (1 << childIndex);

  if (pNode->m_childIndex == (uOctPtr_t)0)
  {
    uOctPtr_t index = pOctree->GetNewChildIndex();
    
    pNode = pOctree->GetNode(parentIndex);
    pNode->m_childIndex = index;
  }

  *pChildIndex = pNode->m_childIndex + childIndex;
  OctreeNode *pChildNode = pOctree->GetNode(*pChildIndex);
  
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
  for (size_t i = 0; i < FirstOctreeIndex; i++)
    m_nodes.AddEntry();

  m_defaultOctreeDepth = defaultOctreeDepth;
  m_center = uint64_t(1) << (defaultOctreeDepth - 2);
}

Octree::Octree(char *filename)
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

  size_t blockSize = m_nodes.GetBlockSize();

  for (int64_t i = 0; i < int64_t(nodeCount - blockSize) && i > 0; i += blockSize)
    fread(m_nodes.GetNewBlock(), sizeof(OctreeNode), blockSize, pFile);

  for (size_t i = (nodeCount / blockSize) * blockSize; i < nodeCount; i++)
    fread(m_nodes.AddEntry(), sizeof(OctreeNode), 1, pFile);

  fclose(pFile);
}

uOctPtr_t Octree::GetNewChildIndex()
{
  uOctPtr_t ret = (uOctPtr_t)m_nodes.Size();

  for (size_t i = 0; i < 8; i++)
    AddNode()->m_isSolid = false;

  return ret;
}

OctreeNode * Octree::GetNode(uOctPtr_t index)
{
  return m_nodes.GetPtrAt(index);
}

OctreeNode * Octree::AddNode(ulonglong3 position)
{
  uint64_t midx = m_center;
  uint64_t midy = m_center;
  uint64_t midz = m_center;

  uint8_t layersLeft = m_defaultOctreeDepth - 1;
  OctreeNode *pNode = GetNode(1);
  uOctPtr_t nodeIndex = 1;

  while (layersLeft > 0)
  {
    layersLeft--;

    uint8_t x = (position.x >= midx);
    uint8_t y = (position.y >= midy) << 1;
    uint8_t z = (position.z >= midz) << 2;

    if (x) midx += (midx >> 1);
    else midx -= (midx >> 1);

    if (y) midy += (midy >> 1);
    else midy -= (midy >> 1);

    if (z) midz += (midz >> 1);
    else midz -= (midz >> 1);

    uint8_t index = x | y | z;

    pNode = OctreeNode::AddChild(nodeIndex, this, index, &nodeIndex);
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

OctreeNode * Octree::AddNode()
{
  return m_nodes.AddEntry();
}
