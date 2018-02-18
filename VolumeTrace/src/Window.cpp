#include "Window.h"

Window::Window(char * title, size_t sizeX, size_t sizeY)
{
  m_pWindow = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, (int)sizeX, (int)sizeY, SDL_WINDOW_SHOWN);
  m_pSurface = SDL_GetWindowSurface(m_pWindow);
}

Window::~Window()
{
  Close();
}

uint32_t * Window::GetPixels()
{
  return (uint32_t *)m_pSurface->pixels;
}

void Window::Swap()
{
  SDL_UpdateWindowSurface(m_pWindow);

  SDL_Event sdl_event;

  while (SDL_PollEvent(&sdl_event))
    ;
}

SDL_Window * Window::GetSDLWindow()
{
  return m_pWindow;
}

SDL_Surface * Window::GetSDLSurface()
{
  return m_pSurface;
}

void Window::Close()
{
  if (m_pWindow == nullptr)
    return;

  SDL_DestroyWindow(m_pWindow);
  m_pWindow = nullptr;
}
