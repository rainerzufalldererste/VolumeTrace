#ifndef __FLUSHQUEUE_H__
#define __FLUSHQUEUE_H__

#include "default.h"

template <typename T, size_t TCapacity>
class FlushQueue
{
public:
  FlushQueue();
  ~FlushQueue();

  void Flush();
  size_t Size();
  Error Enqueue(T element);

  T& operator[](size_t index);

private:
  T *m_pData;
  size_t m_size;
};

template<typename T, size_t TCapacity>
inline FlushQueue<T, TCapacity>::FlushQueue() : m_size(0)
{
  m_pData = MALLOC(T, TCapacity);
}

template<typename T, size_t TCapacity>
inline FlushQueue<T, TCapacity>::~FlushQueue()
{
  FREE(m_pData);
}

template<typename T, size_t TCapacity>
inline void FlushQueue<T, TCapacity>::Flush()
{
  m_size = 0;
}

template<typename T, size_t TCapacity>
inline size_t FlushQueue<T, TCapacity>::Size()
{
  return m_size;
}

template<typename T, size_t TCapacity>
inline Error FlushQueue<T, TCapacity>::Enqueue(T element)
{
  if (m_size >= TCapacity)
    return IndexOutOfBoundsError;

  if (m_size > 0 && m_pData[m_size - 1] == element)
    return Success;

  m_pData[m_size++] = element;

  return Success;
}

template<typename T, size_t TCapacity>
inline T &FlushQueue<T, TCapacity>::operator[](size_t index)
{
  ASSERT(index < m_size);

  return m_pData[index];
}

#endif // !__FLUSHQUEUE_H__
