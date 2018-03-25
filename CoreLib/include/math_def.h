#ifndef __MATH_DEF_H__
#define __MATH_DEF_H__

#ifndef D3DXMATHTYPES

#include "d3dx9math.h"
#include "d3dx9math.inl"

typedef D3DXVECTOR2 vec2;
typedef D3DXVECTOR3 vec3;
typedef D3DXVECTOR4 vec4;
typedef D3DXMATRIX mat4x4;

#endif // !D3DXMATHTYPES

#include "vector_types.h"

template <typename T>
struct vec2t
{
  T x, y;

  __host__ __device__ inline vec2t() : x(0), y(0) {}
  __host__ __device__ inline vec2t(T _v) : x(_v), y(_v) {}
  __host__ __device__ inline vec2t(T _x, T _y) : x(_x), y(_y) {}

  __host__ __device__ inline vec2t(vec2t<T> &&move) : x(move.x), y(move.y) {}
  __host__ __device__ inline vec2t(const vec2t<T> &copy) : x(copy.x), y(copy.y) {}

  __host__ __device__ inline vec2t<T>& operator =  (vec2t<T> &&move) { x = move.x; y = move.y; return *this; }
  __host__ __device__ inline vec2t<T>& operator =  (const vec2t<T> &copy) { x = copy.x; y = copy.y; return *this; }

  __host__ __device__ inline vec2t<T>  operator +  (vec2t<T> a) { return vec2t<T>(x + a.x, y + a.y); };
  __host__ __device__ inline vec2t<T>  operator -  (vec2t<T> a) { return vec2t<T>(x - a.x, y - a.y); };
  __host__ __device__ inline vec2t<T>  operator *  (vec2t<T> a) { return vec2t<T>(x * a.x, y * a.y); };
  __host__ __device__ inline vec2t<T>  operator /  (vec2t<T> a) { return vec2t<T>(x / a.x, y / a.y); };
  __host__ __device__ inline vec2t<T>& operator += (vec2t<T> a) { return *this = vec2t<T>(x + a.x, y + a.y); };
  __host__ __device__ inline vec2t<T>& operator -= (vec2t<T> a) { return *this = vec2t<T>(x - a.x, y - a.y); };
  __host__ __device__ inline vec2t<T>& operator *= (vec2t<T> a) { return *this = vec2t<T>(x * a.x, y * a.y); };
  __host__ __device__ inline vec2t<T>& operator /= (vec2t<T> a) { return *this = vec2t<T>(x / a.x, y / a.y); };
  __host__ __device__ inline vec2t<T>  operator *  (T a) { return vec2t<T>(x * a, y * a); };
  __host__ __device__ inline vec2t<T>  operator /  (T a) { return vec2t<T>(x / a, y / a); };
  __host__ __device__ inline vec2t<T>& operator *= (T a) { return *this = vec2t<T>(x * a, y * a); };
  __host__ __device__ inline vec2t<T>& operator /= (T a) { return *this = vec2t<T>(x / a, y / a); };

  __host__ __device__ inline vec2t<T>  operator -  () { x = -x; y = -y; return *this; };
};

typedef vec2t<int64_t> vec2i;
typedef vec2t<uint64_t> vec2u;

template <typename T>
struct vec3t
{
  T x, y, z;

  __host__ __device__ inline vec3t() : x(0), y(0), z(0) {}
  __host__ __device__ inline vec3t(T _v) : x(_v), y(_v), z(_v) {}
  __host__ __device__ inline vec3t(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

  __host__ __device__ inline vec3t(vec3t<T> &&move) : x(move.x), y(move.y), z(move.z) {}
  __host__ __device__ inline vec3t(const vec3t<T> &copy) : x(copy.x), y(copy.y), z(copy.z) {}

  __host__ __device__ inline vec3t<T>& operator =  (vec3t<T> &&move) { x = move.x; y = move.y; z = move.z; return *this; }
  __host__ __device__ inline vec3t<T>& operator =  (const vec3t<T> &copy) { x = copy.x; y = copy.y; z = copy.z; return *this; }

  __host__ __device__ inline vec3t<T>  operator +  (vec3t<T> a) { return vec3t<T>(x + a.x, y + a.y, z + a.z); };
  __host__ __device__ inline vec3t<T>  operator -  (vec3t<T> a) { return vec3t<T>(x - a.x, y - a.y, z - a.z); };
  __host__ __device__ inline vec3t<T>  operator *  (vec3t<T> a) { return vec3t<T>(x * a.x, y * a.y, z * a.z); };
  __host__ __device__ inline vec3t<T>  operator /  (vec3t<T> a) { return vec3t<T>(x / a.x, y / a.y, z / a.z); };
  __host__ __device__ inline vec3t<T>& operator += (vec3t<T> a) { return *this = vec3t<T>(x + a.x, y + a.y, z + a.z); };
  __host__ __device__ inline vec3t<T>& operator -= (vec3t<T> a) { return *this = vec3t<T>(x - a.x, y - a.y, z - a.z); };
  __host__ __device__ inline vec3t<T>& operator *= (vec3t<T> a) { return *this = vec3t<T>(x * a.x, y * a.y, z * a.z); };
  __host__ __device__ inline vec3t<T>& operator /= (vec3t<T> a) { return *this = vec3t<T>(x / a.x, y / a.y, z / a.z); };
  __host__ __device__ inline vec3t<T>  operator *  (T a) { return vec3t<T>(x * a, y * a, z * a); };
  __host__ __device__ inline vec3t<T>  operator /  (T a) { return vec3t<T>(x / a, y / a, z / a); };
  __host__ __device__ inline vec3t<T>& operator *= (T a) { return *this = vec3t<T>(x * a, y * a, z * a); };
  __host__ __device__ inline vec3t<T>& operator /= (T a) { return *this = vec3t<T>(x / a, y / a, z / a); };

  __host__ __device__ inline vec3t<T>  operator -  () { x = -x; y = -y; z = -z; return *this; };
};

typedef vec3t<int64_t> vec3i;
typedef vec3t<uint64_t> vec3u;

template <typename T>
struct vec4t
{
  T x, y, z, w;

  __host__ __device__ inline vec4t() : x(0), y(0), z(0), w(0) {}
  __host__ __device__ inline vec4t(T _v) : x(_v), y(_v), z(_v), w(_v) {}
  __host__ __device__ inline vec4t(T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {}

  __host__ __device__ inline vec4t(vec4t<T> &&move) : x(move.x), y(move.y), z(move.z), w(move.w) {}
  __host__ __device__ inline vec4t(const vec4t<T> &copy) : x(copy.x), y(copy.y), z(copy.z), w(copy.w) {}

  __host__ __device__ inline vec4t<T>& operator =  (vec4t<T> &&move) { x = move.x; y = move.y; z = move.z; w = move.w; return *this; }
  __host__ __device__ inline vec4t<T>& operator =  (const vec4t<T> &copy) { x = copy.x; y = copy.y; z = copy.z; w = copy.w; return *this; }

  __host__ __device__ inline vec4t<T>  operator +  (vec4t<T> a) { return vec4t<T>(x + a.x, y + a.y, z + a.z, w + a.w); };
  __host__ __device__ inline vec4t<T>  operator -  (vec4t<T> a) { return vec4t<T>(x - a.x, y - a.y, z - a.z, w - a.w); };
  __host__ __device__ inline vec4t<T>  operator *  (vec4t<T> a) { return vec4t<T>(x * a.x, y * a.y, z * a.z, w * a.w); };
  __host__ __device__ inline vec4t<T>  operator /  (vec4t<T> a) { return vec4t<T>(x / a.x, y / a.y, z / a.z, w / a.w); };
  __host__ __device__ inline vec4t<T>& operator += (vec4t<T> a) { return *this = vec4t<T>(x + a.x, y + a.y, z + a.z, a.w + a.w); };
  __host__ __device__ inline vec4t<T>& operator -= (vec4t<T> a) { return *this = vec4t<T>(x - a.x, y - a.y, z - a.z, a.w - a.w); };
  __host__ __device__ inline vec4t<T>& operator *= (vec4t<T> a) { return *this = vec4t<T>(x * a.x, y * a.y, z * a.z, a.w * a.w); };
  __host__ __device__ inline vec4t<T>& operator /= (vec4t<T> a) { return *this = vec4t<T>(x / a.x, y / a.y, z / a.z, a.w / a.w); };
  __host__ __device__ inline vec4t<T>  operator *  (T a) { return vec4t<T>(x * a, y * a, z * a, w * a); };
  __host__ __device__ inline vec4t<T>  operator /  (T a) { return vec4t<T>(x / a, y / a, z / a, w / a); };
  __host__ __device__ inline vec4t<T>& operator *= (T a) { return *this = vec4t<T>(x * a, y * a, z * a, w * a); };
  __host__ __device__ inline vec4t<T>& operator /= (T a) { return *this = vec4t<T>(x / a, y / a, z / a, w / a); };

  __host__ __device__ inline vec4t<T>  operator -  () { x = -x; y = -y; z = -z; w = -w; return *this; };
};

typedef vec4t<int64_t> vec4i;
typedef vec4t<uint64_t> vec4u;

__host__ __device__ inline float4 float4_from_uint32_t(uint32_t c)
{ 
  return make_float4(
    ((c & 0xFF0000) >> 0x10) / (float)0xFF, 
    ((c & 0xFF00) >> 0x08) / (float)0xFF, 
    ((c & 0xFF) >> 0x00) / (float)0xFF, 
    ((c & 0xFF000000) >> 0x18) / (float)0xFF); 
}


#endif // !__MATH_DEF_H__
