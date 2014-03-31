extern "C"
__global__ void copyme (uchar4 *input, uchar4 *output,
		const int width, const int height) {

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= height || col >= width)
		return;

	uchar4 in = input[row * width + col];
	uchar4 out;
	out.x = in.w;
	out.y = in.z;
	out.z = in.y;
	out.w = in.x;

//	printf("I'm called! %d ", in.y);


	output[row * width + col] = out;
}
