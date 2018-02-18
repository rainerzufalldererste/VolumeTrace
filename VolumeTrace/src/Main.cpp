#include "default.h"
#include "kernel.h"
#include "Window.h"
#include "Octree.h"

enum
{
  SizeX = 720,
  SizeY = 480,
  Samples = 5
};

int main(int argc, char* pArgv[])
{
  UNUSED(argc);
  UNUSED(pArgv);

  Window *pWindow = new Window("VolumeTrace", SizeX, SizeY);
  float3 *pFloatRenderBuffer = MALLOC2D(float3, SizeX, SizeY);
  float3 *__cuda__pRenderBuffer;
  Octree *pOctree = new Octree(2);

  pOctree->AddNode(make_ulonglong3(0, 0, 0));
  pOctree->AddNode(make_ulonglong3(1, 0, 0));
  pOctree->AddNode(make_ulonglong3(0, 1, 0));
  pOctree->AddNode(make_ulonglong3(0, 0, 1));

  pOctree->Save("octree.oct");

  Init(SizeX, SizeY, &__cuda__pRenderBuffer);

  while (true)
  {
    uint32_t *pRenderBuffer = pWindow->GetPixels();

    Render(SizeX, SizeY, Samples, __cuda__pRenderBuffer);
    cudaMemcpy(pFloatRenderBuffer, __cuda__pRenderBuffer, sizeof(float3) * SizeX * SizeY, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < SizeX * SizeY; i++)
    {
      float3 color = pFloatRenderBuffer[i];

      pRenderBuffer[i] = uint32_t(uint8_t(color.x * (float)0xFF)) | (uint32_t(uint8_t(color.y * (float)0xFF)) << 0x08) | (uint32_t(uint8_t(color.z * (float)0xFF)) << 0x10);
    }

    pWindow->Swap();
  }

  Cleanup(&__cuda__pRenderBuffer);

  pWindow->Close();
  delete(pWindow);
  FREE(pFloatRenderBuffer);
}