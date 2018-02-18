#ifndef __CHUNKEDARRAY_H__
#define __CHUNKEDARRAY_H__

#include "default.h"

template <typename T, size_t TBlockSize>
class ChunkedArray
{
public:
  ChunkedArray();
  ~ChunkedArray();

  size_t GetBlockSize();
  size_t Size();
  T *AddEntry();
  T *GetPtrAt(size_t index);
  T& operator [](size_t index);
  T* GetNewBlock(); // Only possible if Size % BlockSize == 0

private:
  size_t m_size;
  size_t m_blocks;
  size_t m_count;

  T **m_pBlocks;
};

#include "ChunkedArray.h"

template<typename T, size_t TBlockSize>
inline ChunkedArray<T, TBlockSize>::ChunkedArray()
{
  m_size = 0;
  m_blocks = 0;
  m_count = 0;
  m_pBlocks = nullptr;
}

template<typename T, size_t TBlockSize>
ChunkedArray<T, TBlockSize>::~ChunkedArray()
{
  for (size_t i = 0; i < m_blocks; i++)
    FREE(m_pBlocks[i]);

  FREE(m_pBlocks);
}

template<typename T, size_t TBlockSize>
inline size_t ChunkedArray<T, TBlockSize>::GetBlockSize()
{
  return TBlockSize;
}

template<typename T, size_t TBlockSize>
size_t ChunkedArray<T, TBlockSize>::Size()
{
  return m_count;
}

template<typename T, size_t TBlockSize>
T * ChunkedArray<T, TBlockSize>::AddEntry()
{
  if (m_blocks == 0)
  {
    m_blocks++;
    m_pBlocks = MALLOC(T*, m_blocks);
    m_pBlocks[m_blocks - 1] = MALLOC(T, TBlockSize);
    MEMSET(m_pBlocks[m_blocks - 1], 0, T, TBlockSize);
    m_size += TBlockSize;
  }

  size_t oldCount = m_count;
  m_count++;

  if (m_count > m_size)
  {
    m_blocks++;
    REALLOC(m_pBlocks, T*, m_blocks);
    m_pBlocks[m_blocks - 1] = MALLOC(T, TBlockSize);
    MEMSET(m_pBlocks[m_blocks - 1], 0, T, TBlockSize);
    m_size += TBlockSize;
  }

  return GetPtrAt(oldCount);
}

template<typename T, size_t TBlockSize>
T * ChunkedArray<T, TBlockSize>::GetPtrAt(size_t index)
{
  size_t block = index / TBlockSize;
  size_t i = index % TBlockSize;

  ASSERT(block < m_blocks || i < m_count % TBlockSize);

  return (m_pBlocks[block]) + i;
}

template<typename T, size_t TBlockSize>
T & ChunkedArray<T, TBlockSize>::operator[](size_t index)
{
  return *GetPtrAt(index);
}

template<typename T, size_t TBlockSize>
T * ChunkedArray<T, TBlockSize>::GetNewBlock()
{
  ASSERT(m_count % TBlockSize == 0);
  if (m_count % TBlockSize != 0)
    return nullptr;

  m_blocks++;

  if (m_blocks == 0)
    m_pBlocks = MALLOC(T*, m_blocks);
  else
    REALLOC(m_pBlocks, T*, m_blocks);

  m_pBlocks[m_blocks - 1] = MALLOC(T, TBlockSize);
  MEMSET(m_pBlocks[m_blocks - 1], 0, T, TBlockSize);
  m_size += TBlockSize;
  m_count += TBlockSize;

  return m_pBlocks[m_blocks - 1];
}


#endif // !__CHUNKEDARRAY_H__
