#ifndef __WINDOW_H__
#define __WINDOW_H__

#include "default.h"

class Window
{
public:
  Window(char *title, size_t sizeX, size_t sizeY);
  Window(const char *title, size_t sizeX, size_t sizeY);
  ~Window();
  
  uint32_t *GetPixels();
  void Swap();
  SDL_Window *GetSDLWindow();
  SDL_Surface *GetSDLSurface();
  void Close();

private:
  SDL_Window *m_pWindow;
  SDL_Surface *m_pSurface;
};

#endif // !__WINDOW_H__
