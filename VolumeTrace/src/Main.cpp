#include "default.h"
#include "kernel.h"
#include "Window.h"
#include "Octree.h"

enum
{
  SizeX = 720,
  SizeY = 480,
  Samples = 5,
  GpuBufferSize = 256 * 1024 * 1024,
};

int main(int argc, char * pArgv[]);

void UploadCallback(void *pData, size_t size, size_t position);
void FinishUploadCallback();

cudaStream_t cudaStream;
void *__cuda__pOctreeData;

int main(int argc, char* pArgv[])
{
  UNUSED(argc);
  UNUSED(pArgv);

  Window *pWindow = new Window("VolumeTrace", SizeX, SizeY);
  uchar4 *__cuda__pRenderBuffer;
  Octree *pOctree = new Octree(3);

  pOctree->AddNode(make_ulonglong3(0, 0, 0))->m_color = make_float4(1, 1, 1, 1);
  pOctree->AddNode(make_ulonglong3(1, 0, 0))->m_color = make_float4(1, 0, 0, 1);
  pOctree->AddNode(make_ulonglong3(0, 1, 0))->m_color = make_float4(0, 1, 0, 1);
  pOctree->AddNode(make_ulonglong3(0, 0, 1))->m_color = make_float4(0, 0, 1, 1);

  pOctree->CalculateParentNodes();

  pOctree->Save("octree.oct");

  delete(pOctree);

  cudaStreamCreate(&cudaStream);
  Init(SizeX, SizeY, &__cuda__pRenderBuffer, GpuBufferSize, &__cuda__pOctreeData);

  pOctree = new Octree("octree.oct", true);
  pOctree->SetMaxSize(GpuBufferSize);
  pOctree->SetUpload(UploadCallback);
  pOctree->SetFinishUpload(FinishUploadCallback);

  uint32_t *pRenderBuffer = pWindow->GetPixels();

  while (true)
  {
    uint32_t start = SDL_GetTicks();

    Render(SizeX, SizeY, Samples, __cuda__pRenderBuffer, __cuda__pOctreeData);

    uint32_t cuda = SDL_GetTicks() - start;
    start = SDL_GetTicks();

    cudaMemcpy(pRenderBuffer, __cuda__pRenderBuffer, sizeof(uchar4) * SizeX * SizeY, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    uint32_t gpuDownload = SDL_GetTicks() - start;
    start = SDL_GetTicks();

    pOctree->GetNode(1)->GetChild(1, pOctree, 1);
    pOctree->Update();
    pOctree->IncreaseFrames();

    uint32_t octUpdate = SDL_GetTicks() - start;
    start = SDL_GetTicks();

    pWindow->Swap();

    uint32_t swap = SDL_GetTicks() - start;
    start = SDL_GetTicks();

    static uint32_t second = SDL_GetTicks() + 1000;
    static uint32_t _cuda = 0;
    static uint32_t _gpuDownload = 0;
    static uint32_t _octUpdate = 0;
    static uint32_t _swap = 0;
    static uint32_t _frames = 0;

    _cuda += cuda;
    _gpuDownload += gpuDownload;
    _octUpdate += octUpdate;
    _swap += swap;
    _frames++;

    if (SDL_GetTicks() > second)
    {
      second = SDL_GetTicks() + 1000;
      uint32_t total = _cuda + _gpuDownload + _octUpdate + _swap;

      printf("\rcuda: %.2f%% | gpuDownload: %.2f%% | octUpdate: %.2f%% | swap: %.2f%% (%u frames/s)", (float)_cuda / total * 100.f, (float)_gpuDownload / total * 100.f, (float)_octUpdate / total * 100.f, (float)_swap / total * 100.f, _frames);
    
      _cuda = 0;
      _gpuDownload = 0;
      _octUpdate = 0;
      _swap = 0;
      _frames = 0;
    }
  }

  Cleanup(&__cuda__pRenderBuffer, &__cuda__pOctreeData);

  pWindow->Close();
  delete(pWindow);
}

void UploadCallback(void * pData, size_t size, size_t position)
{
#if _DEBUG
  printf("\rUPLOADING %llu bytes to %llu.\n", size, position);
#endif
  cudaError_t error = cudaMemcpyAsync((void *)((size_t)__cuda__pOctreeData + position), pData, size, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStream);

  ASSERT(error == cudaSuccess);
  UNUSED(error);
}

void FinishUploadCallback()
{
  cudaEvent_t cudaEvent;

  cudaError_t error = cudaEventCreate(&cudaEvent);
  ASSERT(error == cudaSuccess);

  error = cudaEventRecord(cudaEvent, cudaStream);
  ASSERT(error == cudaSuccess);

  error = cudaStreamWaitEvent(cudaStream, cudaEvent, 0);
  ASSERT(error == cudaSuccess);

  UNUSED(error);
}
