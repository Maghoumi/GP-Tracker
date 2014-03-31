#pragma once

/**
 * Increments a float4 struct by the given uchar4 struct
 */
__device__
inline void incFloat4ByUchar4(float4* base, uchar4* value) {
	(*base).x += (*value).x;
	(*base).y += (*value).y;
	(*base).z += (*value).z;
	(*base).w += (*value).w;
}

/**
 * Divides the given float4 struct by the given value
 */
__device__
inline void divFloat4(float4* value, int divValue) {
	(*value).x /= divValue;
	(*value).y /= divValue;
	(*value).z /= divValue;
	(*value).w /= divValue;
}

/**
 * Calculates the square root of the given float4 struct and stores
 * the result in the passed input. (In other words, this is a destructive operation)
 */
__device__
inline void sqrt(float4* input) {
	(*input). x = sqrt((*input).x);
	(*input). y = sqrt((*input).y);
	(*input). z = sqrt((*input).z);
	(*input). w = sqrt((*input).w);
}
