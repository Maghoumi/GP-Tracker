#pragma once

/**
 * Overlays the given color by the provided overlay color. You should provide the mix
 * percentage for the overlay color. This is obviously a destructive operation.
 */
__device__
inline void colorOverlay(uchar4 *color, uchar4 overlay, float overlayMix) {
	float colorMix = 1 - overlayMix;

	(*color).x = 255;
	(*color).y = colorMix * (*color).y + overlayMix * overlay.y;
	(*color).z = colorMix * (*color).z + overlayMix * overlay.z;
	(*color).w = colorMix * (*color).w + overlayMix * overlay.w;
}

/**
 * Converts a float4 to uchar4 color struct;
 */
__device__
inline uchar4* float4ToUchar4(float4 in) {
	uchar4 result;
	result.x = (unsigned char) in.x;
	result.y = (unsigned char) in.y;
	result.z = (unsigned char) in.z;
	result.w = (unsigned char) in.w;
	return &result;
}
