#pragma once

#include "../struct-math.h"
texture<uchar4, 2, cudaReadModeElementType> inputTexture;

/**
 * Calculates the average and standard deviation filters for the provided input.
 * Note that it does not tamper with the channels BUT creates an output with 3 channels.
 * Assumes that the input of the textures has the ALPHA component at the beginning.
 *
 */
extern "C"
__global__ void avgSdFilter(float4 *smallAvg, float4 *smallSdm,
		float4 *mediumAvg, float4 *mediumSdm,
		float4 *largeAvg, float4 *largeSdm,
		const int imageWidth, const int imageHeight,
		const int smallMaskWidth, const int mediumMaskWidth, const int largeMaskWidth) {

	// get indices
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	int width = imageWidth;
	int height = imageHeight;

	// thread should not fall out of bounds
	if (xIndex >= width || yIndex >= height)
		return;

	int index = yIndex * width + xIndex;	// For linear addressing later

	int smallOffset = smallMaskWidth / 2;
	int mediumOffset = mediumMaskWidth / 2;
	int largeOffset = largeMaskWidth / 2;

	uchar4 fetched = {0, 0, 0, 0};
	float4 smallSum = {0, 0, 0, 0};
	float4 mediumSum = {0, 0, 0, 0};
	float4 largeSum = {0, 0, 0, 0};

	// first calculate the average
	for (int i = -largeOffset ; i <= largeOffset ; i++)
			for (int j = -largeOffset ; j <= largeOffset ; j++) {
				// Do the largest filter first
				fetched = tex2D(inputTexture, xIndex + i, yIndex + j);
				incFloat4ByUchar4(&largeSum, &fetched);

				// Is it in the range of the medium sum?
				if (i >= -mediumOffset && i <= mediumOffset
						&& j >= -mediumOffset && j <= mediumOffset) {
					incFloat4ByUchar4(&mediumSum, &fetched);
				}

				// Is it in the range of the small sum?
				if (i >= -smallOffset && i <= smallOffset
						&& j >= -smallOffset && j <= smallOffset) {
					incFloat4ByUchar4(&smallSum, &fetched);
				}
			}

	int largeCount = largeMaskWidth * largeMaskWidth;
	int mediumCount = mediumMaskWidth * mediumMaskWidth;
	int smallCount = smallMaskWidth * smallMaskWidth;

	divFloat4(&largeSum, largeCount);
	largeAvg[index] = largeSum;

	divFloat4(&mediumSum, mediumCount);
	mediumAvg[index] = mediumSum;

	divFloat4(&smallSum, smallCount);
	smallAvg[index] = smallSum;

	float4 smallSd = {0, 0, 0, 0};
	float4 mediumSd = {0, 0, 0, 0};
	float4 largeSd = {0, 0, 0, 0};

	// first calculate the average
	for (int i = -largeOffset ; i <= largeOffset ; i++)
			for (int j = -largeOffset ; j <= largeOffset ; j++) {
				// Do the largest filter first
				fetched = tex2D(inputTexture, xIndex + i, yIndex + j);
				largeSd.x += pow(fetched.x - largeSum.x, 2);
				largeSd.y += pow(fetched.y - largeSum.y, 2);
				largeSd.z += pow(fetched.z - largeSum.z, 2);
				largeSd.w += pow(fetched.w - largeSum.w, 2);

				// Is it in the range of the medium sum?
				if (i >= -mediumOffset && i <= mediumOffset
						&& j >= -mediumOffset && j <= mediumOffset) {
					mediumSd.x += pow(fetched.x - mediumSum.x, 2);
					mediumSd.y += pow(fetched.y - mediumSum.y, 2);
					mediumSd.z += pow(fetched.z - mediumSum.z, 2);
					mediumSd.w += pow(fetched.w - mediumSum.w, 2);
				}

				// Is it in the range of the small sum?
				if (i >= -smallOffset && i <= smallOffset
						&& j >= -smallOffset && j <= smallOffset) {
					smallSd.x += pow(fetched.x - smallSum.x, 2);
					smallSd.y += pow(fetched.y - smallSum.y, 2);
					smallSd.z += pow(fetched.z - smallSum.z, 2);
					smallSd.w += pow(fetched.w - smallSum.w, 2);
				}
			}

	divFloat4(&largeSd, largeCount);
	divFloat4(&mediumSd, mediumCount);
	divFloat4(&smallSd, smallCount);

	// Calculate final standard deviation values and write the results
	sqrt(&largeSd);
	largeSdm[index] = largeSd;

	sqrt(&mediumSd);
	mediumSdm[index] = mediumSd;

	sqrt(&smallSd);
	smallSdm[index] = smallSd;
}
