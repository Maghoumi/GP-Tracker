package utils.cuda;

import static jcuda.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
import static jcuda.driver.CUfilter_mode.CU_TR_FILTER_MODE_POINT;
import static jcuda.driver.JCudaDriver.CU_TRSA_OVERRIDE_FORMAT;
import static jcuda.driver.JCudaDriver.CU_TRSF_READ_AS_INTEGER;
import static jcuda.driver.JCudaDriver.cuMemcpy2D;
import static jcuda.driver.JCudaDriver.cuModuleGetTexRef;
import static jcuda.driver.JCudaDriver.cuTexRefSetAddressMode;
import static jcuda.driver.JCudaDriver.cuTexRefSetArray;
import static jcuda.driver.JCudaDriver.cuTexRefSetFilterMode;
import static jcuda.driver.JCudaDriver.cuTexRefSetFlags;
import static jcuda.driver.JCudaDriver.cuTexRefSetFormat;
import utils.cuda.datatypes.FilterResult;
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
import jcuda.utils.KernelLauncher;

/**
 * Utility class to perform various statistical filters on a given image
 * The filters are performed using CUDA
 * 
 * @author Mehran Maghoumi
 */
public class ImageFilters {
	
	private KernelLauncher launcher = null;
	
	private CUmodule module = null;		// CUDA module
	private CUfunction fAvgSd = null;		// CUDA function for average filter
	
	public int blockSizeX = 16;
	public int blockSizeY = 16; 
	
	public ImageFilters(boolean recompileKernel) {
		JCudaDriver.setExceptionsEnabled(true);
		
		if (recompileKernel)
			System.out.println("Recompiling kernel...");
		
		this.launcher = KernelLauncher.create("bin/utils/cuda/kernels/image-filter.cu", "avgSdFilter", recompileKernel, "-arch sm_20");
		this.module = launcher.getModule();
		
		this.fAvgSd = new CUfunction();
		
		JCudaDriver.cuModuleGetFunction(fAvgSd, module, "avgSdFilter");
	}
	
	
	/**
	 * Performs the average and standard deviation box filter on the graphics card
	 * and returns the result.
	 * 
	 * @param input
	 * 		The input image to be filtered
	 * @param maskWidth
	 * 		The width of the mask
	 * @param maskHeight
	 * 		The height of the mask
	 * @param filterType
	 * 		The type of filter to perform
	 * @return
	 * 		The filtered image in form of a 2D Float4 array
	 */
	public FilterResult filterImage(byte[] input, int imageWidth, int imageHeight, int maskWidth, int maskHeight) {
		String filterLabel;
		CUfunction filterFunc = fAvgSd;
		
		System.out.println();
		System.out.print("Calculating " + maskWidth + "x" + maskHeight + " AVG and SD filters...");
		
		// Allocate device array
		CUarray dev_average = new CUarray();
		CUDA_ARRAY_DESCRIPTOR desc = new CUDA_ARRAY_DESCRIPTOR();
		desc.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
		desc.NumChannels = 4;
		desc.Width = imageWidth;
		desc.Height = imageHeight;
		JCudaDriver.cuArrayCreate(dev_average, desc);
		
		// Copy the host input to the array
        CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
        copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        copyHD.srcHost = Pointer.to(input);
        copyHD.srcPitch = imageWidth * Sizeof.BYTE * 4;
        copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
        copyHD.dstArray = dev_average;
        copyHD.WidthInBytes = imageWidth * Sizeof.BYTE * 4;
        copyHD.Height = imageHeight;
        
        cuMemcpy2D(copyHD);
        
        // Set texture reference properties
        CUtexref inputTexRef = new CUtexref();
        cuModuleGetTexRef(inputTexRef, launcher.getModule(), "inputTexture");
        cuTexRefSetFilterMode(inputTexRef, CU_TR_FILTER_MODE_POINT);
        cuTexRefSetAddressMode(inputTexRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetAddressMode(inputTexRef, 1, CU_TR_ADDRESS_MODE_CLAMP);
        cuTexRefSetFlags(inputTexRef, CU_TRSF_READ_AS_INTEGER);
        cuTexRefSetFormat(inputTexRef, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, 4);
        cuTexRefSetArray(inputTexRef, dev_average, CU_TRSA_OVERRIDE_FORMAT);
        
        // Allocate results array
        CUdeviceptr dev_average1 = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(dev_average1, imageWidth * imageHeight * Sizeof.FLOAT * 4);
        CUdeviceptr dev_stdDev = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(dev_stdDev, imageWidth * imageHeight * Sizeof.FLOAT * 4);
        
        Pointer kernelParams = Pointer.to(
        		Pointer.to(dev_average1),
        		Pointer.to(dev_stdDev),
                Pointer.to(new int[]{imageWidth}),
                Pointer.to(new int[]{imageHeight}),
                Pointer.to(new int[]{maskWidth}),
                Pointer.to(new int[]{maskHeight})
            );
        
        // Call kernel
        JCudaDriver.cuLaunchKernel(filterFunc,
        		(imageWidth + blockSizeX - 1) / blockSizeX, (imageHeight + blockSizeY - 1)/blockSizeY, 1,
        		blockSizeX, blockSizeY, 1,
        		0, null, kernelParams, null);
        JCudaDriver.cuCtxSynchronize();
        
        // Retrieve results
        float[] average = new float[imageWidth * imageHeight * 4];
        float[] stdDev = new float[imageWidth * imageHeight * 4];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(average), dev_average1, imageWidth * imageHeight * Sizeof.FLOAT * 4);
        JCudaDriver.cuMemcpyDtoH(Pointer.to(stdDev), dev_stdDev, imageWidth * imageHeight * Sizeof.FLOAT * 4);
        
        System.out.print(" DONE");
        
        // A little housekeeping
        JCudaDriver.cuArrayDestroy(dev_average);
        JCudaDriver.cuMemFree(dev_average1);
        JCudaDriver.cuMemFree(dev_stdDev);
        
        return new FilterResult(average, stdDev);
	}
}