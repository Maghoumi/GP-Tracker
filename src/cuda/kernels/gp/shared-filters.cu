/**
 * Calculates the average and standard deviation filters for the provided input.
 * Note that it does not tamper with the channels BUT creates an output with 3 channels.
 * Assumes that the input of the textures has the ALPHA component at the beginning.
 *
 */

#define O_TILE_WIDTH 16

extern "C"
__global__ void newFilter(uchar4 *input, const int iPitchInElements,
		float4 *average, float4 *stdDev, const int  oPitchInElements,
		const int width, const int height,
		const int maskWidth) {

	// get indices
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int col_o = blockIdx.x * O_TILE_WIDTH + tx;
	int row_o = blockIdx.y * O_TILE_WIDTH + ty;

	int row_i = row_o - maskWidth / 2;
	int col_i = col_o - maskWidth / 2;

	__shared__ uchar4 tile[30][30];

	if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < iPitchInElements)
		tile[ty][tx] = input[row_i * iPitchInElements + col_i];
//	else
//		tile[ty][tx] = make_uchar4(255, 0, 0, 0);

	__syncthreads();

	int count = maskWidth * maskWidth;

	// first calculate the average

	float4 sum = {0, 0, 0, 0};
	float4 sumSd = {0, 0, 0, 0};
	uchar4 fetched;

	if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
		for (int i = 0 ; i < maskWidth ; i++)
			for (int j = 0 ; j < maskWidth ; j++) {
				fetched = tile[i + ty][j + tx];

				sum.x += fetched.x;
				sum.y += fetched.y;
				sum.z += fetched.z;
				sum.w += fetched.w;
			}

		sum.x /= count;
		sum.y /= count;
		sum.z /= count;
		sum.w /= count;

		for (int i = 0 ; i < maskWidth ; i++)
			for (int j = 0 ; j < maskWidth ; j++) {
				fetched = tile[i + ty][j + tx];

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

		if (row_o < height && col_o < oPitchInElements) {
			average[row_o * oPitchInElements + col_o] = sum;
			stdDev[row_o * oPitchInElements + col_o] = sumSd;
		}
	}
}
