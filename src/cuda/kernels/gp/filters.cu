texture<uchar4, 2, cudaReadModeElementType> inputTexture;


/**
 * Calculates the average and standard deviation filters for the provided input.
 * Note that it does not tamper with the channels BUT creates an output with 3 channels.
 * Assumes that the input of the textures has the ALPHA component at the beginning.
 *
 */
extern "C"
__global__ void avgSdFilter(float4 *average, float4 *stdDev,
		const int imageWidth, const int imageHeight, const int pitchInElements,
		const int maskWidth) {

	// get indices
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	int width = imageWidth;
	int height = imageHeight;

	// thread should not fall out of bounds
	if (xIndex >= width || yIndex >= height)
		return;

	int count = maskWidth * maskWidth;

	int offsetX = maskWidth / 2;

	// first calculate the average

	float4 sum = {0, 0, 0, 0};
	uchar4 fetched = {0, 0, 0, 0};

	// iterate the area around the center data
	for (int i = -offsetX ; i <= offsetX ; i++)
		for (int j = -offsetX ; j <= offsetX ; j++) {
			fetched = tex2D(inputTexture, xIndex + i, yIndex + j);

			sum.x += fetched.x;
			sum.y += fetched.y;
			sum.z += fetched.z;
			sum.w += fetched.w;
		}

	sum.x /= count;
	sum.y /= count;
	sum.z /= count;
	sum.w /= count;



	// now calculate the standard deviation
	float4 sumSd = {0, 0, 0, 0};

	// iterate the area around the center data
	for (int i = -offsetX ; i <= offsetX ; i++)
		for (int j = -offsetX ; j <= offsetX ; j++) {
			fetched = tex2D(inputTexture, xIndex + i, yIndex + j);
			sumSd.x += pow(fetched.x - sum.x, 2);
			sumSd.y += pow(fetched.y - sum.y, 2);
			sumSd.z += pow(fetched.z - sum.z, 2);
			sumSd.w += pow(fetched.w - sum.w, 2);
		}

	sumSd.x /= count;
	sumSd.y /= count;
	sumSd.z /= count;
	sumSd.w /= count;

	sumSd.x = sqrt(sumSd.x);
	sumSd.y = sqrt(sumSd.y);
	sumSd.z = sqrt(sumSd.z);
	sumSd.w = sqrt(sumSd.w);

	int index = yIndex * pitchInElements + xIndex;
	average[index] = sum;
	stdDev[index] = sumSd;
}
