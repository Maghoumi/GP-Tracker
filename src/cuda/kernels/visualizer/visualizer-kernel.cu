#include "../types.h"
#include "../macros.h"
#include "../color-utils.h"
#include "../postfix-parser.h"
#include "visualizer-filters.h"

/**
 * A function that runs the GP individual and directly outputs the results
 * to the OpenGL buffer.
 */
__device__
void directWriteDescribe(char* expressions, int pitchInElements, int expCount,
		char* enabilityMap, uchar4 *overlayColors, const char showConflicts, const float opacity,
		DescribeData data, uchar4 *output,
		const int segmentX, const int segmentY,
		const int segmentWidth, const int segmentHeight,
		const int outputWidth, const int outputHeight
		)
{
	int bid = blockIdx.x;
	int threadIndex = threadIdx.x;

	float stack[STACK_SIZE];
	int sp = -1;

	int tid = bid * blockDim.x + threadIndex;	// Thread to data index mapping

	if(tid >= segmentWidth * segmentHeight)
		return;

	int outputIndex = (segmentY + bid) * outputWidth + segmentX + threadIdx.x;	// the OpenGL buffer index for this segment and this thread

	int tidConflicts = 0;
	// First copy the input to the output and then filter output
	uchar4* overlayed = float4ToUchar4(data.input[tid]);

	for (int i = 0 ; i < expCount ; i++) {
		// Obtain the next expression
		char* expression = expressions + i * pitchInElements;
		// Obtain the overlay color of the next expression
		uchar4 overlay = overlayColors[i];
		bool enabled = enabilityMap[i] == 1;

		// If there are no expressions passed to this kernel, the kernel would just copy
		// the input data into the output buffer! (Acting as a simple image/video displayer)
		if (expression[0] == 0) {
			continue;
		}

		bool obtained = parseExpression(expression, data);

		if (obtained && enabled) {
			colorOverlay(overlayed, overlay, opacity);
			tidConflicts++;	// Update the conflict table
		}

		output[outputIndex] = *overlayed;

	} // end-for (the loop that iterates through the expressions)

	if (showConflicts && tidConflicts > 1)
		output[outputIndex] = make_uchar4(255, 0, 0, 255);
}


/**
 * A function that runs the GP tree and produces the data that is necessary for thresholding
 * back in Java
 */
__device__
void thresholdDescribe(char* expressions, int pitchInElements, int expCount,
		float* scratchPad,
		DescribeData data,
		const int segmentX, const int segmentY,
		const int segmentWidth, const int segmentHeight,
		const int segmentPitchInElements
		)
{
	int bid = blockIdx.x;
	int threadIndex = threadIdx.x;

	int tid = bid * blockDim.x + threadIndex;	// Thread to data index mapping

	if(tid >= segmentWidth * segmentHeight)
		return;


	for (int i = 0 ; i < expCount ; i++) {
		// Obtain the next expression
		char* expression = expressions + i * pitchInElements;

		// If there are no expressions passed to this kernel, the kernel would just copy
		// the input data into the output buffer! (Acting as a simple image/video displayer)
		if (expression[0] == 0) {
			continue;
		}

		bool obtained = parseExpression(expression, data);

		if (obtained) {
			atomicAdd(&scratchPad[i], 1.0);
		}

	} // end-for (the loop that iterates through the expressions)
}


/**
 * Runs the given individuals on whole image. Depending on the mode, it will either filter
 * the final image directly or will provide a the number of positive classifications per
 * individual so that thresholding will become possible.
 */
extern "C"
__global__ void describe(const char shouldThreshold, char* expressions, int pitchInElements, int expCount,
		char* enabilityMap, uchar4 *overlayColors, const char showConflicts, const float opacity,
		float* scratchPad,
		float4 *input, uchar4 *output,
		float4 *smallAvg, float4 *mediumAvg, float4 *largeAvg,
		float4 *smallSd, float4 *mediumSd, float4 *largeSd,
		const int segmentX, const int segmentY,
		const int segmentWidth, const int segmentHeight, const int segmentPitchInElements,
		const int outputWidth, const int outputHeight
		)
{
	DescribeData data = make_DescribeData(input, smallAvg, mediumAvg, largeAvg, smallSd, mediumSd, largeSd);

	if (shouldThreshold) {
		thresholdDescribe(
				expressions, pitchInElements, expCount,
				scratchPad,
				data,
				segmentX, segmentY,
				segmentWidth, segmentHeight,
				segmentPitchInElements
				);
	}
	else {
		directWriteDescribe(
				expressions, pitchInElements, expCount,
				enabilityMap, overlayColors, showConflicts, opacity,
				data, output,
				segmentX, segmentY,
				segmentWidth, segmentHeight,
				outputWidth, outputHeight
				);
	}
}
