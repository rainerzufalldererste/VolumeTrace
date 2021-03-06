#include "default.h"
#include "kernel.h"
#include "Window.h"
#include "Octree.h"
#include "PointCloud.h"

enum
{
  SizeX = 720,
  SizeY = 480,
  Samples = 5,
  GpuBufferSize = 512 * 1024 * 1024,
};

int main(int argc, char * pArgv[]);

void UploadCallback(void *pData, size_t size, size_t position);
void FinishUploadCallback();

cudaStream_t cudaStream;
void *__cuda__pOctreeData;

#define GENERATE_OCTREE

int main(int argc, char* pArgv[])
{
  UNUSED(argc);
  UNUSED(pArgv);

  Window *pWindow = new Window("VolumeTrace", SizeX, SizeY);
  uchar4 *__cuda__pRenderBuffer;
  uOctPtr_t *__cuda__pStreamerData;
  uOctPtr_t *pStreamerData = MALLOC2D(uOctPtr_t, SizeX, SizeY);
  vec3 cameraPosition = vec3(-5.0f, -5.0f, -5.0f);
  mat4x4 viewMat;
  D3DXMatrixLookAtLH(&viewMat, &cameraPosition, &vec3(0, 0, 0), &vec3(0, 1, 0));

  Octree *pOctree;

#ifdef GENERATE_OCTREE
  pOctree = new Octree(8);

  //pOctree->AddNode(vec3u(0, 0, 0))->m_color = make_float4(1, 1, 1, 1);
  //pOctree->AddNode(vec3u(1, 0, 0))->m_color = make_float4(1, 0, 0, 1);
  //pOctree->AddNode(vec3u(0, 1, 0))->m_color = make_float4(0, 1, 0, 1);
  //pOctree->AddNode(vec3u(0, 0, 1))->m_color = make_float4(0, 0, 1, 1);
  //pOctree->AddNode(vec3u(0, 0, 0))->m_color = make_float4(1, 1, 1, 1);
  //pOctree->AddNode(vec3u(2, 1, 0))->m_color = make_float4(1, 1, 0, 1);
  //pOctree->AddNode(vec3u(1, 1, 2))->m_color = make_float4(0, 1, 1, 1);
  //pOctree->AddNode(vec3u(2, 2, 4))->m_color = make_float4(0, 0.8f, 1, 1);
  //pOctree->AddNode(vec3u(4, 4, 8))->m_color = make_float4(0, 0.6f, 1, 1);
  //pOctree->AddNode(vec3u(8, 8, 16))->m_color = make_float4(0, 0.4f, 1, 1);
  //pOctree->AddNode(vec3u(16, 16, 32))->m_color = make_float4(0, 0.2f, 1, 1);
  //pOctree->AddNode(vec3u(2, 0, 1))->m_color = make_float4(1, 0, 1, 1);
  //pOctree->AddNode(vec3u(1, 3, 2))->m_color = make_float4(1, 0.5, 1, 1);
  //pOctree->AddNode(vec3u(1, 2, 3))->m_color = make_float4(0.5, 0.5, 1, 1);
  //pOctree->AddNode(vec3u(3, 2, 1))->m_color = make_float4(1, 0.5, 0.5, 1);
  //pOctree->AddNode(vec3u(7, 3, 5))->m_color = make_float4(0.25, 0.5, 0.75, 1);
  //pOctree->AddNode(vec3u(13, 15, 18))->m_color = make_float4(0.5, 0.5, 0.75, 1);
  //pOctree->AddNode(vec3u(5, 3, 7))->m_color = make_float4(0.75, 0.5, 0.5, 1);
  //pOctree->AddNode(vec3u(70, 70, 52))->m_color = make_float4(0.75, 0.5, 0.25, 1);
  //pOctree->AddNode(vec3u(98, 83, 35))->m_color = make_float4(0.5, 0.5, 0.25, 1);
  //pOctree->AddNode(vec3u(51, 45, 73))->m_color = make_float4(0.25, 0.5, 0.5, 1);
  //
  vec3u offsetVector = vec3u(1, 5, 1);
  
  pOctree->AddNode(vec3u(2, 2, 2) + vec3u(1) + offsetVector)->m_color = make_float4(1, 1, 1, 1);
  pOctree->AddNode(vec3u(2, 2, 1) + vec3u(1) + offsetVector)->m_color = make_float4(1, 0, 0, 1);
  pOctree->AddNode(vec3u(1, 2, 2) + vec3u(1) + offsetVector)->m_color = make_float4(0, 1, 0, 1);
  pOctree->AddNode(vec3u(2, 1, 2) + vec3u(1) + offsetVector)->m_color = make_float4(0, 0, 1, 1);
  pOctree->AddNode(vec3u(2, 2, 0) + vec3u(1) + offsetVector)->m_color = make_float4(1, 0.5f, 0, 1);
  pOctree->AddNode(vec3u(0, 2, 2) + vec3u(1) + offsetVector)->m_color = make_float4(0, 1, 0.5f, 1);
  pOctree->AddNode(vec3u(2, 0, 2) + vec3u(1) + offsetVector)->m_color = make_float4(0.5f, 0, 1, 1);
  //pOctree->AddNode(vec3u(3, 3, 0) + offsetVector)->m_color = make_float4(1, 0.25f, 0, 1);
  //pOctree->AddNode(vec3u(0, 3, 3) + offsetVector)->m_color = make_float4(0, 1, 0.25f, 1);
  //pOctree->AddNode(vec3u(3, 0, 3) + offsetVector)->m_color = make_float4(0.25f, 0, 1, 1);

  pOctree->CalculateParentNodes();
  pOctree->Save("carrickHill.oct");

  delete(pOctree);
#endif // !GENERATE_OCTREE

  cudaStreamCreate(&cudaStream);
  Init(SizeX, SizeY, &__cuda__pRenderBuffer, GpuBufferSize, &__cuda__pOctreeData, &__cuda__pStreamerData);

  pOctree = new Octree("carrickHill.oct", true);
  pOctree->SetMaxSize(GpuBufferSize);
  pOctree->SetFinishUpload(FinishUploadCallback);
  pOctree->SetUpload(UploadCallback);

  uint32_t *pRenderBuffer = pWindow->GetPixels();
  float halfOctreeSize = (float)((1 << pOctree->GetLayerDepth()) >> 1);

  while (true)
  {
    mat4x4 rotMat;
    vec4 camPos4;
    cameraPosition = vec3(-5.0f, -5.0f, -5.0f);
    D3DXMatrixRotationYawPitchRoll(&rotMat, sinf(SDL_GetTicks() * 0.001f), cosf(SDL_GetTicks() * 0.0007f), 0.0f);
    D3DXVec3Transform(&camPos4, &cameraPosition, &rotMat);
    cameraPosition = vec3(camPos4.x, camPos4.y, camPos4.z);
    D3DXMatrixLookAtLH(&viewMat, &cameraPosition, &vec3(halfOctreeSize, halfOctreeSize, halfOctreeSize), &vec3(0, 1, 0));

    uint32_t start = SDL_GetTicks();

    Render(SizeX, SizeY, pOctree->GetLayerDepth(), cameraPosition, __cuda__pRenderBuffer, __cuda__pOctreeData, __cuda__pStreamerData, viewMat);

    uint32_t cuda = SDL_GetTicks() - start;
    start = SDL_GetTicks();

    cudaMemcpy(pRenderBuffer, __cuda__pRenderBuffer, sizeof(uchar4) * SizeX * SizeY, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(pStreamerData, __cuda__pStreamerData, sizeof(uOctPtr_t) * SizeX * SizeY, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    uint32_t gpuDownload = SDL_GetTicks() - start;
    start = SDL_GetTicks();

    for (size_t i = 0; i < SizeX * SizeY; i++)
      if (pStreamerData[i] != 0)
        pOctree->Enqueue(pStreamerData[i]);

    uint32_t processStreamer = SDL_GetTicks() - start;
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
    static uint32_t _processStreamer = 0;
    static uint32_t _octUpdate = 0;
    static uint32_t _swap = 0;
    static uint32_t _frames = 0;

    _cuda += cuda;
    _gpuDownload += gpuDownload;
    _octUpdate += octUpdate;
    _swap += swap;
    _processStreamer += processStreamer;
    _frames++;

    if (SDL_GetTicks() > second)
    {
      second = SDL_GetTicks() + 1000;
      uint32_t total = _cuda + _gpuDownload + _processStreamer + _octUpdate + _swap;

      printf("\rcuda: %.2f%% | gpuDownload: %.2f%% | processStreamer: %.2f%% | octUpdate: %.2f%% | swap: %.2f%% (%u frames/s)", (float)_cuda / total * 100.f, (float)_gpuDownload / total * 100.f, (float)_processStreamer / total * 100.f, (float)_octUpdate / total * 100.f, (float)_swap / total * 100.f, _frames);
    
      _cuda = 0;
      _gpuDownload = 0;
      _processStreamer = 0;
      _octUpdate = 0;
      _swap = 0;
      _frames = 0;
    }
  }

  Cleanup(&__cuda__pRenderBuffer, &__cuda__pOctreeData, &__cuda__pStreamerData);

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
