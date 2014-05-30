package visualizer;

import static jcuda.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
import static jcuda.driver.CUfilter_mode.CU_TR_FILTER_MODE_POINT;
import static jcuda.driver.JCudaDriver.*;
import invoker.Invoker;

import java.awt.Color;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import javax.media.opengl.GLAutoDrawable;

import utils.Classifier;
import utils.FilteredImage;
import utils.ImageFilterProvider;
import utils.Segment;
import utils.ClassifierSet.ClassifierAllocationResult;
import utils.cuda.pointers.CudaByte2D;
import utils.cuda.pointers.CudaFloat2D;
import utils.opengl.OpenGLUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

/**
 * A wrapper class for the CUDA visualization kernel. A kernel call request
 * is passed to an object of this class. (Software engineering principles at
 * their best ;-] )
 * The objects of this class are also able to directly draw to the OpenGL buffer
 * the results of the kernel invocation. 
 * 
 * @author Mehran Maghoumi
 *
 */
public class GLVisualizerKernel implements ImageFilterProvider {
	
	/** Recompile flag ** FOR DEBUGGING PURPOSES** */
	public static final boolean RECOMPILE = false;
	/** Generate debug info ** FOR DEBUGGING PURPOSES** */
	public static final boolean GEN_DEBUG = false;
	
	/** The size of the small image filter */
	public static final int FILTER_SIZE_SMALL = 15;
	
	/** The size of the medium image filter */
	public static final int FILTER_SIZE_MEDIUM = 17;
	
	/** The size of the large image filter */
	public static final int FILTER_SIZE_LARGE = 19;
	
	/** The block size to use for the filter function */
	protected static final int FILTER_BLOCK_SIZE = 16;
	
	/** The CUDA module of the compiled kernel */
	private CUmodule codeModule = new CUmodule();

	/** Handle to the convolution filter kernel */
	private CUfunction fncFilter = new CUfunction();

	/** Handle to the describe kernel */
	private CUfunction fncDescribe;

	/** Pointer to the registered OpenGL buffer */
	private CUdeviceptr devOutput;
	
	/** CUDA graphics resource that has been used to registered the buffer */
	private CUgraphicsResource bufferResource;
	
	/** The handle to the OpenGL pixel buffer object */
	private int glBuffer;
	
	/**
	 * Initializes an instance of this class. The instantiation will also cause the 
	 * kernel code to be compiled and also the handles to the CUDA functions be created 
	 */
	public GLVisualizerKernel() {
		JCuda.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);

		// Create a device and a context
		cuInit(0);
		CUdevice dev = new CUdevice();
		cuDeviceGet(dev, 0);
		CUcontext glCtx = new CUcontext();
		cuCtxCreate(glCtx, 0, dev);

		// Prepare the PTX file containing the kernel
		String ptxFileName = "";

		try {
			ptxFileName = preparePtxFile("bin/cuda/kernels/visualizer/visualizer-kernel.cu");
		} catch (IOException e) {
			System.err.println("Could not create PTX file");
			throw new RuntimeException("Could not create PTX file", e);
		}

		// Load the PTX file containing the kernel
		CUmodule module = new CUmodule();
		cuModuleLoad(module, ptxFileName);

		// Obtain fncDescribe handle
		fncDescribe = new CUfunction();
		cuModuleGetFunction(fncDescribe, module, "describe");
		cuModuleGetFunction(fncFilter, module, "avgSdFilter");
		codeModule = module;
		devOutput = new CUdeviceptr();
	}
	
	/**
	 * Register an OpenGL buffer with this kernel for CUDA/OpenGL interop
	 * @param glBuffer	The handle to the OpenGL buffer
	 */
	public void registerGLBuffer(int glBuffer) {
		this.bufferResource = new CUgraphicsResource();
		cuGraphicsGLRegisterBuffer(bufferResource, glBuffer, CUgraphicsMapResourceFlags.CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
		this.glBuffer = glBuffer;
	}
	
	/**
	 * Calls the visualization CUDA kernel and displays the results using OpenGL. Also
	 * returns the number of classifiers that have claimed the current segment
	 * 
	 * @param gp	The GPEngine object that is used for retraining classifiers
	 * @param drawable	The OpenGL drawable
	 * @param classifiers	A set of classifiers to visualize
	 * @param segment	The image segment to display the data for
	 * @param shouldThreshold	Boolean flag indicating whether the function should do thresholding or not
	 * @param threshold	The threshold percentage (in the range [0, 1])
	 * @param opacity	The opacity of the color overlay drawn on the segment
	 * @param showConflicts	A flag indicating whether the conflicting areas should be painted red
	 * @param imageWidth	The image (frame) width
	 * @param imageHeight	The image (frame) height
	 * 
	 * @return Returns the number of classifiers that have claimed this segment. Will return -1
	 * 			if thresholding was not enabled
	 */
	public int call(Invoker invoker, GLAutoDrawable drawable, ClassifierAllocationResult pointerToAll, Segment segment,
			boolean shouldThreshold, float threshold, float opacity,
			boolean showConflicts, int imageWidth, int imageHeight) {
		
		CudaByte2D devExpression = pointerToAll.expressions;
		CudaByte2D overlayColors = pointerToAll.overlayColors;
		CudaByte2D enabilityMap = pointerToAll.enabilityMap;
		
		// Determine the number of GP expressions
		int numClassifiers = devExpression.getHeight();
		
		// First, we must filter the segment using the context associated with this thread! Otherwise, CUDA will complain!
		segment.filterImage(this);
		FilteredImage filteredImage = segment.getFilteredImage();
		
		// Allocate and transfer the scratchpad
		float[] scratchPad = new float[numClassifiers];
		CUdeviceptr devScratchPad = new CUdeviceptr();
		cuMemAlloc(devScratchPad, numClassifiers * Sizeof.FLOAT);
		cuMemcpyHtoD(devScratchPad, Pointer.to(scratchPad), numClassifiers * Sizeof.FLOAT);
		
		// Map the OpenGL buffer to a CUDA pointer
		cuGraphicsMapResources(1, new CUgraphicsResource[] { bufferResource }, null);
		cuGraphicsResourceGetMappedPointer(devOutput, new long[1], bufferResource);
		
		// Setup kernel parameters
		Pointer kernelParams = Pointer.to(Pointer.to(new byte[] {(byte) (shouldThreshold ? 1 : 0)}),
				Pointer.to(devExpression),Pointer.to(devExpression.getDevPitchInElements()), Pointer.to(new int[] { numClassifiers }),
				Pointer.to(enabilityMap),Pointer.to(overlayColors), Pointer.to(new byte[] {(byte) (showConflicts ? 1 : 0)}), Pointer.to(new float[] {opacity}),
				Pointer.to(devScratchPad),
				filteredImage.getInput().toPointer(), Pointer.to(devOutput),
				filteredImage.getSmallAvg().toPointer(), filteredImage.getMediumAvg().toPointer(), filteredImage.getLargeAvg().toPointer(),
				filteredImage.getSmallSd().toPointer(), filteredImage.getMediumSd().toPointer(), filteredImage.getLargeSd().toPointer(),
				Pointer.to(new int[] {segment.getBounds().x}), Pointer.to(new int[] {segment.getBounds().y}), Pointer.to(new int[] {segment.getBounds().width}), Pointer.to(new int[] {segment.getBounds().height}), Pointer.to(new int[] { (int) filteredImage.getInput().getDevPitchInElements()[0] }),
				Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }));
		
		
		// Call kernel
		cuLaunchKernel(fncDescribe,
				segment.getBounds().height, 1, 1, 	// segment height is equal to the number of blocks
				segment.getBounds().width, 1, 1,	// segment width is equal to the number of threads in each block
				0, null,
				kernelParams, null);
		cuCtxSynchronize();
		
		// Unmap OpenGL buffer from CUDA
		cuGraphicsUnmapResources(1, new CUgraphicsResource[] { bufferResource }, null);
		filteredImage.freeAll();
		
		// Copy the scratchpad back 
		cuMemcpyDtoH(Pointer.to(scratchPad), devScratchPad, Sizeof.FLOAT * numClassifiers);
		
		// Housekeeping
		cuMemFree(devScratchPad);
		
		// Should we do the thresholding here??
		if (!shouldThreshold)
			return -1;
		
		// Do thresholding:
		// A list of classifiers that have claimed this segment
		List<Classifier> claimers = new ArrayList<Classifier>();
		
		for (int i = 0 ; i < scratchPad.length ; i++) {
			scratchPad[i] /= segment.getBounds().width * segment.getBounds().height;
			if (scratchPad[i] > threshold) {
				Classifier claimer = pointerToAll.classifiers.get(i);
				claimers.add(claimer);
				claimer.addClaim(segment);
			}
		}
		
		if (claimers.size() == 1) {
			if (claimers.get(0).isEnabled())	// Overlay time! If a classifier is disabled, we shouldn't paint overlays!
				OpenGLUtils.drawRegionOverlay(drawable, glBuffer,
					claimers.get(0).getColor(), opacity,
					imageWidth, imageHeight ,segment.getBounds());
			return claimers.size();
		} 
		else if (claimers.size() != 0) {	// if 0 => no classifiers have been active!!
			OpenGLUtils.drawRegionOverlay(drawable, glBuffer,
					Color.RED, opacity,
					imageWidth, imageHeight, segment.getBounds());
		}
		
		return claimers.size();
	}	
	
	/**
	 * @return	The CUmodule of the compiled kernel
	 */
	public CUmodule getModule() {
		return this.codeModule;
	}
	
	/**
	 * @return	The handle to the image filter function
	 */
	public CUfunction getFilterFunction() {
		return this.fncFilter;
	}
	
	/**
	 * The extension of the given file name is replaced with "ptx". If the file
	 * with the resulting name does not exist, it is compiled from the given
	 * file using NVCC. The name of the PTX file is returned.
	 * 
	 * @param cuFileName
	 *            The name of the .CU file
	 * @return The name of the PTX file
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	private static String preparePtxFile(String cuFileName) throws IOException {
		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1) {
			endIndex = cuFileName.length() - 1;
		}
		String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
		File ptxFile = new File(ptxFileName);

		if (ptxFile.exists() && !RECOMPILE) {
			return ptxFileName;
		}

		File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new IOException("Input file not found: " + cuFileName);
		}
		String modelString = "-m" + System.getProperty("sun.arch.data.model");
		String command = "nvcc " + modelString + " -arch=sm_21" + (GEN_DEBUG ? " -G" : "") +" -ptx " + cuFile.getPath() + " -o " + ptxFileName;

		System.out.println("Executing\n" + command);
		Process process = Runtime.getRuntime().exec(command);

		String errorMessage = new String(toByteArray(process.getErrorStream()));
		String outputMessage = new String(toByteArray(process.getInputStream()));
		int exitValue = 0;
		try {
			exitValue = process.waitFor();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted while waiting for nvcc output", e);
		}

		if (exitValue != 0) {
			System.out.println("nvcc process exitValue " + exitValue);
			System.out.println("errorMessage:\n" + errorMessage);
			System.out.println("outputMessage:\n" + outputMessage);
			throw new IOException("Could not create .ptx file: " + errorMessage);
		}

		System.out.println("Finished creating PTX file");
		return ptxFileName;
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array
	 * 
	 * @param inputStream
	 *            The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	private static byte[] toByteArray(InputStream inputStream) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte buffer[] = new byte[8192];
		while (true) {
			int read = inputStream.read(buffer);
			if (read == -1) {
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}

	@Override
	public void performFilters(CudaByte2D byteInput, CudaFloat2D smallAvg, CudaFloat2D mediumAvg, CudaFloat2D largeAvg, CudaFloat2D smallSd, CudaFloat2D mediumSd, CudaFloat2D largeSd) {
		byteInput.reallocate();
		smallAvg.reallocate();
		mediumAvg.reallocate();
		largeAvg.reallocate();
		
		smallSd.reallocate();
		mediumSd.reallocate();
		largeSd.reallocate();
		
		int imageWidth = byteInput.getWidth();
		int imageHeight = byteInput.getHeight();
		int numChannels = byteInput.getNumFields();
		
		// Allocate device array
		CUarray devTexture = new CUarray();
		CUDA_ARRAY_DESCRIPTOR desc = new CUDA_ARRAY_DESCRIPTOR();
		desc.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
		desc.NumChannels = numChannels;
		desc.Width = imageWidth;
		desc.Height = imageHeight;
		JCudaDriver.cuArrayCreate(devTexture, desc);

		// Copy the host input to the array
		CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
		copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
		copyHD.srcHost = byteInput.hostDataToPointer();
		copyHD.srcPitch = byteInput.getSourcePitch();
		copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
		copyHD.dstArray = devTexture;
		copyHD.WidthInBytes = imageWidth * byteInput.getElementSizeInBytes() * numChannels;
		copyHD.Height = imageHeight;

		cuMemcpy2D(copyHD);

		// Set texture reference properties
		CUtexref inputTexRef = new CUtexref();
		cuModuleGetTexRef(inputTexRef, this.codeModule, "inputTexture");
		cuTexRefSetFilterMode(inputTexRef, CU_TR_FILTER_MODE_POINT);
		cuTexRefSetAddressMode(inputTexRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
		cuTexRefSetAddressMode(inputTexRef, 1, CU_TR_ADDRESS_MODE_CLAMP);
		cuTexRefSetFlags(inputTexRef, CU_TRSF_READ_AS_INTEGER);
		cuTexRefSetFormat(inputTexRef, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, numChannels);
		cuTexRefSetArray(inputTexRef, devTexture, CU_TRSA_OVERRIDE_FORMAT);

		// Allocate results array
		Pointer kernelParams = Pointer.to(smallAvg.toPointer(), smallSd.toPointer(),
				mediumAvg.toPointer(), mediumSd.toPointer(),
				largeAvg.toPointer(), largeSd.toPointer(),
				Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }), Pointer.to(new int[] { (int) smallAvg.getDevPitchInElements()[0] }),
				Pointer.to(new int[] { getSmallFilterSize() }), Pointer.to(new int[] { getMediumFilterSize() }), Pointer.to(new int[] { getLargeFilterSize() }));

		// Call kernel
		cuLaunchKernel(this.fncFilter,
				(imageWidth + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, (imageHeight + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, 1,
				FILTER_BLOCK_SIZE, FILTER_BLOCK_SIZE, 1,
				0, null,
				kernelParams, null);
		cuCtxSynchronize();
		
		smallAvg.refresh();
		mediumAvg.refresh();
		largeAvg.refresh();
		
		smallSd.refresh();
		mediumSd.refresh();
		largeSd.refresh();
		
		// A little housekeeping
		cuArrayDestroy(devTexture);		
	}

	@Override
	public int getSmallFilterSize() {
		return FILTER_SIZE_SMALL;
	}

	@Override
	public int getMediumFilterSize() {
		return FILTER_SIZE_MEDIUM;
	}

	@Override
	public int getLargeFilterSize() {
		return FILTER_SIZE_LARGE;
	}
	
}
