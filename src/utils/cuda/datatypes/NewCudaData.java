package utils.cuda.datatypes;

import static jcuda.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
import static jcuda.driver.CUfilter_mode.CU_TR_FILTER_MODE_POINT;
import static jcuda.driver.JCudaDriver.CU_TRSA_OVERRIDE_FORMAT;
import static jcuda.driver.JCudaDriver.CU_TRSF_READ_AS_INTEGER;
import static jcuda.driver.JCudaDriver.*;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUarray;
import jcuda.driver.CUarray_format;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;

/**
 * Trying to replace the original CudaData with this
 * @author mm12tm
 *TODO document me
 *
 */
public class NewCudaData extends CudaData {
	
	public static final int FILTER_SIZE_SMALL = 15;
	public static final int FILTER_SIZE_MEDIUM = 17;
	public static final int FILTER_SIZE_LARGE = 19;
	
	public static final int FILTER_BLOCK_SIZE = 16;
	
	
	private CUmodule module;
	private CUfunction fncFilter;
	
	private int numChannels;
	
	private byte[] byteInput;
	
	/**
	 * Instantiates a new object using the provided ByteImage. 
	 * @param image
	 */
	public NewCudaData(CUmodule module, CUfunction fncFilter, ByteImage image) {
		this.module = module;
		this.fncFilter = fncFilter;
		
		this.imageHeight = image.getHeight();
		this.imageWidth = image.getWidth();
		this.numChannels = image.getNumChannels();
		
		this.input = image.getFloatData();
		this.byteInput = image.getByteData();
		
		if (module != null && fncFilter != null)
			prepareDevPointers();
	}
	
	public NewCudaData(ByteImage image) {
		this(null, null, image);
	}
	
	/**
	 * Initializes the devPointer fields and allocates CUDA memory using the calling context
	 */
	private void prepareDevPointers() {
		dev_input = new CUdeviceptr();
		dev_smallAvg = new CUdeviceptr();
		dev_mediumAvg = new CUdeviceptr();
		dev_largeAvg = new CUdeviceptr();
		dev_smallSd = new CUdeviceptr();
		dev_mediumSd = new CUdeviceptr();
		dev_largeSd = new CUdeviceptr();
		
		int size = this.imageWidth * this.imageHeight * this.numChannels * Sizeof.FLOAT;		
		cuMemAlloc(dev_input, size);
		cuMemAlloc(dev_smallAvg, size);
		cuMemAlloc(dev_mediumAvg, size);
		cuMemAlloc(dev_largeAvg, size);
		cuMemAlloc(dev_smallSd, size);
		cuMemAlloc(dev_mediumSd, size);
		cuMemAlloc(dev_largeSd, size);
	}
	
	/**
	 * Performs the three level of convolution filters for the data stored in this
	 * instance of the class
	 * 
	 * @param inputImage
	 *            The input image
	 * @param input
	 *            The result is stored in this object
	 */
	public void performFilters() {
		int copySize = this.imageWidth * this.imageHeight * this.numChannels * Sizeof.FLOAT;

		cuMemcpyHtoD(this.dev_input, Pointer.to(this.input), copySize);

		// Allocate device array
		CUarray devTexture = new CUarray();
		CUDA_ARRAY_DESCRIPTOR desc = new CUDA_ARRAY_DESCRIPTOR();
		desc.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
		desc.NumChannels = this.numChannels;
		desc.Width = imageWidth;
		desc.Height = imageHeight;
		JCudaDriver.cuArrayCreate(devTexture, desc);

		// Copy the host input to the array
		CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
		copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
		copyHD.srcHost = Pointer.to(byteInput);
		copyHD.srcPitch = imageWidth * Sizeof.BYTE * this.numChannels;
		copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
		copyHD.dstArray = devTexture;
		copyHD.WidthInBytes = imageWidth * Sizeof.BYTE * this.numChannels;
		copyHD.Height = imageHeight;

		cuMemcpy2D(copyHD);

		// Set texture reference properties
		CUtexref inputTexRef = new CUtexref();
		cuModuleGetTexRef(inputTexRef, this.module, "inputTexture");
		cuTexRefSetFilterMode(inputTexRef, CU_TR_FILTER_MODE_POINT);
		cuTexRefSetAddressMode(inputTexRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
		cuTexRefSetAddressMode(inputTexRef, 1, CU_TR_ADDRESS_MODE_CLAMP);
		cuTexRefSetFlags(inputTexRef, CU_TRSF_READ_AS_INTEGER);
		cuTexRefSetFormat(inputTexRef, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, this.numChannels);
		cuTexRefSetArray(inputTexRef, devTexture, CU_TRSA_OVERRIDE_FORMAT);

		// Allocate results array
		Pointer kernelParams = Pointer.to(Pointer.to(this.dev_smallAvg), Pointer.to(this.dev_smallSd),
				Pointer.to(this.dev_mediumAvg), Pointer.to(this.dev_mediumSd),
				Pointer.to(this.dev_largeAvg), Pointer.to(this.dev_largeSd),
				Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }),
				Pointer.to(new int[] { NewCudaData.FILTER_SIZE_SMALL }), Pointer.to(new int[] { NewCudaData.FILTER_SIZE_MEDIUM }), Pointer.to(new int[] { NewCudaData.FILTER_SIZE_LARGE }));

		// Call kernel
		cuLaunchKernel(fncFilter,
				(imageWidth + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, (imageHeight + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, 1,
				FILTER_BLOCK_SIZE, FILTER_BLOCK_SIZE, 1,
				0, null,
				kernelParams, null);
		cuCtxSynchronize();

		// A little housekeeping
		cuArrayDestroy(devTexture);
		
		//TODO copy back?? necessary??
		int length = this.input.length;
		
		this.smallAvg = new float[length];
		cuMemcpyDtoH(Pointer.to(this.smallAvg), this.dev_smallAvg, length * Sizeof.FLOAT);
		this.mediumAvg = new float[length];
		cuMemcpyDtoH(Pointer.to(this.mediumAvg), this.dev_mediumAvg, length * Sizeof.FLOAT);
		this.largeAvg = new float[length];
		cuMemcpyDtoH(Pointer.to(this.largeAvg), this.dev_largeAvg, length * Sizeof.FLOAT);
		
		this.smallSd = new float[length];
		cuMemcpyDtoH(Pointer.to(this.smallSd), this.dev_smallSd, length * Sizeof.FLOAT);
		this.mediumSd = new float[length];
		cuMemcpyDtoH(Pointer.to(this.mediumSd), this.dev_mediumSd, length * Sizeof.FLOAT);
		this.largeSd = new float[length];
		cuMemcpyDtoH(Pointer.to(this.largeSd), this.dev_largeSd, length * Sizeof.FLOAT);
	}
	
	/**
	 * Set the necessary CUDA objects for this instance
	 * 
	 * @param module	The module to use
	 * @param fncFilter	The handle to the filter function to use for performing filters
	 */
	public void setCudaObjects(CUmodule module, CUfunction fncFilter) {
		this.module = module;
		this.fncFilter = fncFilter;
		
		this.freeAll();
		prepareDevPointers();
		performFilters();
	}
	
	@Override
	public void freeAll() {
		if (this.dev_input != null)
			cuMemFree(this.dev_input);
		
		if (dev_smallAvg != null)
			cuMemFree(dev_smallAvg);
		if (dev_mediumAvg != null)
			cuMemFree(dev_mediumAvg);
		if (dev_largeAvg != null)
			cuMemFree(dev_largeAvg);
		
		if (dev_smallSd != null)
			cuMemFree(dev_smallSd);
		if (dev_mediumSd != null)
			cuMemFree(dev_mediumSd);
		if (dev_largeSd != null)
			cuMemFree(dev_largeSd);
		
		if (dev_labels != null)
			cuMemFree(dev_labels);
		
		if(dev_output != null)
			cuMemFree(dev_output);
		
		this.dev_input = null;
		this.dev_smallAvg = null;
		this.dev_mediumAvg = null;
		this.dev_largeAvg = null;
		this.dev_smallSd = null;
		this.dev_mediumSd = null;
		this.dev_largeSd = null;
		this.dev_labels = null;
		this.dev_output = null;
		
	}
}
