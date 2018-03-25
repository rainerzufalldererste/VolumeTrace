#include "default.h"
#include "Window.h"

enum
{
  SizeX = 1280,
  SizeY = 720
};

int main(int argc, char **pArgv)
{
  Window *pWindow = new Window("ImgQuadScan", SizeX, SizeY);

  while (true)
  {
    uint32_t *pPixels = pWindow->GetPixels();

    memset(pPixels, 0, sizeof(uint32_t) * SizeX * SizeY);

    pWindow->Swap();
  }

  pWindow->Close();
}