#include <curand_kernel.h>
#include "filters.cu"

#define STACK_SIZE 128
#define FIT_CASES_COUNT /*@@fitness-cases@@*/
#define EVAL_BLOCK_SIZE /*@@eval-block-size@@*/
#define DESC_BLOCK_SIZE /*@@desc-block-size@@*/
#define POSITIVE_EXAMPLES /*@@positive-examples@@*/
#define NEGATIVE_EXAMPLES /*@@negative-examples@@*/
#define MAX_GRID_SIZE /*@@max-grid-size@@*/

// stack macros
#define push(A) do { sp++;stack[sp]=A; if(sp >= STACK_SIZE) printf("FUCK!");} while(false)
#define pop(A) do{ A=stack[sp];sp--; }while(false)

__device__ inline void reduce(int *positives, int *negatives) {
	for (int stride = blockDim.x >> 1 ; stride > 0 ; stride >>= 1) {
		if (threadIdx.x < stride) {
			positives[threadIdx.x]+= positives[threadIdx.x + stride];
			negatives[threadIdx.x]+= negatives[threadIdx.x + stride];
		}
		__syncthreads();
	}
}

extern "C"
__global__ void evaluate(const char* __restrict__ individuals, const int indCounts, const int maxLength,
		float4 *input,
		float4 *smallAvg, float4 *mediumAvg, float4 *largeAvg,
		float4 *smallSd, float4 *mediumSd, float4 *largeSd,
		int *labels,
		float *fitnesses)
{
	int bid = blockIdx.x;
	int threadIndex = threadIdx.x;

	if (bid >= indCounts)
		return;

	const char* __restrict__ expression = &(individuals[bid * maxLength]);

	// the first thread should reset these values
	if (threadIndex == 0) {
		fitnesses[bid] = 0;
	}

	__shared__ int totTp[EVAL_BLOCK_SIZE];
	__shared__ int totTn[EVAL_BLOCK_SIZE];

	int myTp = 0;
	int myTn = 0;

	float stack[STACK_SIZE];
	int sp;
	totTp[threadIndex] = totTn[threadIndex] = 0;

	// Determine how many fitness cases this thread should process
	int divNumbers = (FIT_CASES_COUNT - 1) / EVAL_BLOCK_SIZE + 1;

	for (int i = 0 ; i < divNumbers; i++) {
		int tid = divNumbers * threadIndex + i;

		if (tid >= FIT_CASES_COUNT)
			break;

		sp = - 1;

		int k = 0;
		while(expression[k] != 0)
		{
			switch(expression[k])
			{
				/*@@actions@@*/
			}

			k++;
		}

		float result;
		pop(result);
		bool obtained = result > 0;
		bool expected = labels[tid] == 1;

		if(sp!=-1)
			printf("Stack pointer not -1 but is %d", sp);

		if (obtained && expected)
			myTp++;
		else if (!obtained && !expected)
			myTn++;
	}

	// At this point, all fitnesses are calculated by all threads in this block
	totTp[threadIndex] = myTp;
	totTn[threadIndex] = myTn;
	__syncthreads();	// all threads should be finished with writing their results here

	// reduction phase
	reduce(totTp, totTn);

	// calculate the total fitness and assign it
	if (threadIndex == 0) {
		fitnesses[bid] = (totTp[0] + totTn[0]) / (float)FIT_CASES_COUNT;
	}
}



/**
 * Runs one individual on whole image and returns a **ABGR*** image as
 * output.
 */
extern "C"
__global__ void describe(const char* __restrict__ expression,
		float4 *input, char4 *output,
		float4 *smallAvg, float4 *mediumAvg, float4 *largeAvg,
		float4 *smallSd, float4 *mediumSd, float4 *largeSd,
		const int imageWidth, const int imageHeight
		)
{
	int bid = blockIdx.x;
	int threadIndex = threadIdx.x;


	float stack[STACK_SIZE];
	int sp;


	int tid = bid * DESC_BLOCK_SIZE + threadIndex;

	if(tid >= imageWidth * imageHeight)
		return;

	sp = - 1;

	int k = 0;
	while(expression[k] != 0)
	{
		switch(expression[k])
		{
			/*@@actions@@*/
		}

		k++;
	}

	float result;
	pop(result);
	bool obtained = result > 0;

	if (obtained) {
		char4 green = {255, 0, 255, 0};
		output[tid] = green;
	}
	else {
		float4 orig = input[tid];
		char4 origColor = {255, (char)orig.y, (char)orig.z, (char)orig.w};
		output[tid] = origColor;
	}

}
