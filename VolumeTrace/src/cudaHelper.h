#ifndef __CUDAHELPER_H__
#define __CUDAHELPER_H__

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

#endif // !__CUDAHELPER_H__
