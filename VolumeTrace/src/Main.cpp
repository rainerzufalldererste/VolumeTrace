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
  uchar4 *__cuda__pRenderBuffer;
  Octree *pOctree = new Octree(2);

  pOctree->AddNode(make_ulonglong3(0, 0, 0));
  pOctree->AddNode(make_ulonglong3(1, 0, 0));
  pOctree->AddNode(make_ulonglong3(0, 1, 0));
  pOctree->AddNode(make_ulonglong3(0, 0, 1));

  pOctree->Save("octree.oct");

  delete(pOctree);

  pOctree = new Octree("octree.oct", true);

  Init(SizeX, SizeY, &__cuda__pRenderBuffer);

  while (true)
  {
    uint32_t *pRenderBuffer = pWindow->GetPixels();

    Render(SizeX, SizeY, Samples, __cuda__pRenderBuffer);
    cudaMemcpy(pRenderBuffer, __cuda__pRenderBuffer, sizeof(uchar4) * SizeX * SizeY, cudaMemcpyKind::cudaMemcpyDeviceToHost);

    pOctree->GetNode(1)->GetChild(1, pOctree, 1);
    pOctree->Update();
    pOctree->IncreaseFrames();

    pWindow->Swap();
  }

  Cleanup(&__cuda__pRenderBuffer);

  pWindow->Close();
  delete(pWindow);
}