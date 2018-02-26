#ifndef __CUDAHELPER_H__
#define __CUDAHELPER_H__

#ifdef __CUDACC__

#define __MATH_DEF_H__

struct mat4x4
{
  float _1_1, _1_2, _1_3, _1_4;
  float _2_1, _2_2, _2_3, _2_4;
  float _3_1, _3_2, _3_3, _3_4;
  float _4_1, _4_2, _4_3, _4_4;
};

typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;
#endif

#include "default.h"

__host__ __device__ inline float4 operator +(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__host__ __device__ inline float4 operator -(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__host__ __device__ inline float4 operator *(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__host__ __device__ inline float4 operator /(float4 a, float4 b) { return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
__host__ __device__ inline float4 operator *(float4 a, float  b) { return make_float4(a.x * b  , a.y * b  , a.z * b  , a.w * b  ); }
__host__ __device__ inline float4 operator /(float4 a, float  b) { return make_float4(a.x / b  , a.y / b  , a.z / b  , a.w / b  ); }
__host__ __device__ inline float4 &operator +=(float4 &a, float4 b) { return a = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__host__ __device__ inline float4 &operator -=(float4 &a, float4 b) { return a = make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__host__ __device__ inline float4 &operator *=(float4 &a, float4 b) { return a = make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__host__ __device__ inline float4 &operator /=(float4 &a, float4 b) { return a = make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); }
__host__ __device__ inline float4 &operator *=(float4 &a, float  b) { return a = make_float4(a.x * b  , a.y * b  , a.z * b  , a.w * b  ); }
__host__ __device__ inline float4 &operator /=(float4 &a, float  b) { return a = make_float4(a.x / b  , a.y / b  , a.z / b  , a.w / b  ); }

__host__ __device__ inline float4 alphaAdd(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 1.0f - ((1.0f - a.w) * (1.0f - b.w))); };
__host__ __device__ inline float4 alphaDivide(float4 a, float b) { return make_float4(a.x / b, a.y / b, a.z / b, a.w); };


__host__ __device__ inline float3 operator +(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 operator -(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 operator *(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ inline float3 operator /(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__host__ __device__ inline float3 operator *(float3 a, float  b) { return make_float3(a.x * b, a.y * b, a.z * b); }
__host__ __device__ inline float3 operator /(float3 a, float  b) { return make_float3(a.x / b, a.y / b, a.z / b); }
__host__ __device__ inline float3 &operator +=(float3 &a, float3 b) { return a = make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__host__ __device__ inline float3 &operator -=(float3 &a, float3 b) { return a = make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__host__ __device__ inline float3 &operator *=(float3 &a, float3 b) { return a = make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
__host__ __device__ inline float3 &operator /=(float3 &a, float3 b) { return a = make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
__host__ __device__ inline float3 &operator *=(float3 &a, float  b) { return a = make_float3(a.x * b, a.y * b, a.z * b); }
__host__ __device__ inline float3 &operator /=(float3 &a, float  b) { return a = make_float3(a.x / b  , a.y / b  , a.z / b  ); }


__host__ __device__ inline float2 operator +(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__host__ __device__ inline float2 operator -(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
__host__ __device__ inline float2 operator *(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }
__host__ __device__ inline float2 operator /(float2 a, float2 b) { return make_float2(a.x / b.x, a.y / b.y); }
__host__ __device__ inline float2 operator *(float2 a, float  b) { return make_float2(a.x * b, a.y * b); }
__host__ __device__ inline float2 operator /(float2 a, float  b) { return make_float2(a.x / b, a.y / b); }
__host__ __device__ inline float2 &operator +=(float2 &a, float2 b) { return a = make_float2(a.x + b.x, a.y + b.y); }
__host__ __device__ inline float2 &operator -=(float2 &a, float2 b) { return a = make_float2(a.x - b.x, a.y - b.y); }
__host__ __device__ inline float2 &operator *=(float2 &a, float2 b) { return a = make_float2(a.x * b.x, a.y * b.y); }
__host__ __device__ inline float2 &operator /=(float2 &a, float2 b) { return a = make_float2(a.x / b.x, a.y / b.y); }
__host__ __device__ inline float2 &operator *=(float2 &a, float  b) { return a = make_float2(a.x * b, a.y * b); }
__host__ __device__ inline float2 &operator /=(float2 &a, float  b) { return a = make_float2(a.x / b  , a.y / b  ); }

__host__ __device__ inline float4 normalize(float4 a) { float f = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w; if (f == 0) return a; f = sqrtf(f); return a / f; };
__host__ __device__ inline float3 normalize(float3 a) { float f = a.x * a.x + a.y * a.y + a.z * a.z; if (f == 0) return a; f = sqrtf(f); return a / f; };
__host__ __device__ inline float2 normalize(float2 a) { float f = a.x * a.x + a.y * a.y; if (f == 0) return a; f = sqrtf(f); return a / f; };

__host__ __device__
inline uchar4 cast_float4_to_uchar4(float4 input)
{
  return make_uchar4(
    (uint8_t)roundf(input.x * (float)0xFF),
    (uint8_t)roundf(input.y * (float)0xFF),
    (uint8_t)roundf(input.z * (float)0xFF),
    (uint8_t)roundf(input.w * (float)0xFF)
  );
}

struct ray {
  float3 origin;
  float3 direction;
  float4 inv_direction;
  int sign[3];
};

#ifdef __CUDACC__
__host__ __device__ inline float4 operator *(mat4x4 b, float4 a)
{ 
  float *pF = (float *)&b;

  return make_float4(
    pF[0] * a.x + pF[1] * a.y + pF[2] * a.z + pF[3] * a.w,
    pF[4] * a.x + pF[5] * a.y + pF[6] * a.z + pF[7] * a.w,
    pF[8] * a.x + pF[9] * a.y + pF[10] * a.z + pF[11] * a.w,
    pF[12] * a.x + pF[13] * a.y + pF[14] * a.z + pF[15] * a.w);
    //pF[0] * a.x + pF[4] * a.y + pF[8] * a.z + pF[12] * a.w,
    //pF[1] * a.x + pF[5] * a.y + pF[9] * a.z + pF[13] * a.w,
    //pF[2] * a.x + pF[6] * a.y + pF[10] * a.z + pF[14] * a.w,
    //pF[3] * a.x + pF[7] * a.y + pF[11] * a.z + pF[15] * a.w);
}

__host__ __device__
inline ray make_ray(float3 origin, float3 direction)
{
  float4 inv_direction = make_float4(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z, 1.0f);
  ray r;
  r.origin = origin;
  r.direction = direction;
  r.inv_direction = inv_direction;
  r.sign[0] = (inv_direction.x < 0 ? 1 : 0);
  r.sign[1] = (inv_direction.y < 0 ? 1 : 0);
  r.sign[2] = (inv_direction.z < 0 ? 1 : 0);

  return r;
}

__host__ __device__
inline void intersection_distances_no_if(ray ray, float3 aabb[2], float &tmin, float &tmax)
{
  float tymin, tymax, tzmin, tzmax;

  tmin = (aabb[ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;
  tmax = (aabb[1 - ray.sign[0]].x - ray.origin.x) * ray.inv_direction.x;

  tymin = (aabb[ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
  tymax = (aabb[1 - ray.sign[1]].y - ray.origin.y) * ray.inv_direction.y;
  tzmin = (aabb[ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;
  tzmax = (aabb[1 - ray.sign[2]].z - ray.origin.z) * ray.inv_direction.z;

  tmin = fmaxf(fmaxf(tmin, tymin), tzmin);
  tmax = fminf(fminf(tmax, tymax), tzmax);

  // post condition:
  // if tmin > tmax (in the code above this is represented by a return value of INFINITY)
  //     no intersection
  // else
  //     front intersection point = ray.origin + ray.direction * tmin (normally only this point matters)
  //     back intersection point  = ray.origin + ray.direction * tmax
}
#endif

#endif // !__CUDAHELPER_H__
