#ifndef __DEFAULT_H__
#define __DEFAULT_H__

#define _USE_MATH_DEFINES 1

#include <stdint.h>
#include <climits>
#include <math.h>
#include <malloc.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include "SDL.h"

// CUDA
#include "cuda.h"
#include "cuda_runtime.h"
#include "vector_types.h"
#include "vector_functions.h"

#define MALLOC(type, count) ((type *)malloc(sizeof(type) * count))
#define MALLOC2D(type, countX, countY) ((type *)malloc(sizeof(type) * countX * countY))
#define MALLOC3D(type, countX, countY, countZ) ((type *)malloc(sizeof(type) * countX * countY * countZ))
#define REALLOC(ptr, type, count) (ptr = (type *)realloc(ptr, sizeof(type) * count))
#define REALLOC2D(ptr, type, countX, countY) (ptr = (type *)realloc(ptr, sizeof(type) * countX * countY))
#define REALLOC3D(ptr, type, countX, countY, countZ) (ptr = (type *)realloc(ptr, sizeof(type) * countX * countY * countZ))
#define FREE(ptr) {free(ptr); ptr = nullptr;}
#define MEMSET(ptr, value, type, count) (memset(ptr, value, sizeof(type) * count))
#define MEMSET2D(ptr, value, type, countX, countY) (memset(ptr, value, sizeof(type) * countX * countY))
#define MEMSET3D(ptr, value, type, countX, countY, countZ) (memset(ptr, value, sizeof(type) * countX * countY * countZ))
#define MEMCPY(dst, src, type, count) (memcpy(dst, src, sizeof(type) * count))
#define UNUSED(v) ((void)v)
#define RELASSERT(booleanExpression, ...) {assert(booleanExpression);}

#ifdef _DEBUG
#define ASSERT(booleanExpression, ...) {assert(booleanExpression);}
#else
#define ASSERT(booleanExpression, ...) /**/
#endif

#define PI M_PI
#define SQRT2 M_SQRT2
#define INVSQRT2 M_SQRT1_2

enum Error
{
  Success = 0,
  InternalError = 1,
  IndexOutOfBoundsError = 2,
};

#endif // !__DEFAULT_H__

