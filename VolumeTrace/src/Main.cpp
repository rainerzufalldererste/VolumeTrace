#include "default.h"
#include "kernel.h"
#include "Window.h"
#include "Octree.h"
//#include "d3dx9math.h"
//#include "d3dx9math.inl"

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
  uOctPtr_t *__cuda__pStreamerData;
  uOctPtr_t *pStreamerData = MALLOC2D(uOctPtr_t, SizeX, SizeY);
  Octree *pOctree = new Octree(6);

  mat4x4 viewMat, projMat;
  D3DXMatrixLookAtLH(&viewMat, &vec3(-5.0f, -5.0f, -5.f), &vec3(0, 0, 0), &vec3(0, 1, 0));
  D3DXMatrixPerspectiveFovLH(&projMat, 65, SizeX / (float)SizeY, 0.01f, 1000.0f);
  mat4x4 cameraMatrix = projMat * viewMat;
  //D3DXMatrixTranspose(&cameraMatrix, &cameraMatrix);
  mat4x4 inverseCameraMatrix;
  float determinant = D3DXMatrixDeterminant(&cameraMatrix);
  D3DXMatrixInverse(&inverseCameraMatrix, &determinant, &cameraMatrix);

  pOctree->AddNode(make_ulonglong3(0, 0, 0))->m_color = make_float4(1, 1, 1, 1);
  pOctree->AddNode(make_ulonglong3(1, 0, 0))->m_color = make_float4(1, 0, 0, 1);
  pOctree->AddNode(make_ulonglong3(0, 1, 0))->m_color = make_float4(0, 1, 0, 1);
  pOctree->AddNode(make_ulonglong3(0, 0, 1))->m_color = make_float4(0, 0, 1, 1);

  pOctree->CalculateParentNodes();

  pOctree->Save("octree.oct");

  delete(pOctree);

  cudaStreamCreate(&cudaStream);
  Init(SizeX, SizeY, &__cuda__pRenderBuffer, GpuBufferSize, &__cuda__pOctreeData, &__cuda__pStreamerData);

  pOctree = new Octree("octree.oct", true);
  pOctree->SetMaxSize(GpuBufferSize);
  pOctree->SetUpload(UploadCallback);
  pOctree->SetFinishUpload(FinishUploadCallback);

  uint32_t *pRenderBuffer = pWindow->GetPixels();

  while (true)
  {
    //for (size_t y = 0; y < SizeX; y++)
    //{
    //  for (size_t x = 0; x < SizeY; x++)
    //  {
    //    float tmin, tmax;
    //    vec4 dir;
    //    D3DXVec3Transform(&dir, &vec3((x / (float)SizeX - 0.5f), (y / (float)SizeY - 0.5f), 0.725f), &viewMat);//-0.8565f, 1);
    //    float3 dir3 = normalize(*(float3 *)&dir);
    //    ray r = make_ray(make_float3(-5.f, -5.f, -5.f), dir3);
    //    float3 block[2] = { make_float3(0, 0, 0), make_float3(2 << 5, 2 << 5, 2 << 5) };
    //
    //    intersection_distances_no_if(r, block, tmin, tmax);
    //
    //    if (tmin < tmax)
    //      pRenderBuffer[x + y * SizeX] = 0xFFFF00 + (uint8_t)(tmin * 255.0f);
    //    else
    //      pRenderBuffer[x + y * SizeX] = 0x111111;
    //  }
    //}
    //
    //pWindow->Swap();
    //continue;

    uint32_t start = SDL_GetTicks();

    Render(SizeX, SizeY, Samples, __cuda__pRenderBuffer, __cuda__pOctreeData, __cuda__pStreamerData, viewMat);

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
