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
import java.util.HashMap;
import java.util.List;

import javax.media.opengl.GLAutoDrawable;

import cuda.multigpu.KernelAddJob;
import cuda.multigpu.KernelArgSetter;
import cuda.multigpu.KernelInvoke;
import cuda.multigpu.TransScale;
import cuda.multigpu.Trigger;
import utils.Classifier;
import utils.ClassifierSet;
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
	
	public static final String KERNEL_DESCRIBE = "describe";
	public static final String KERNEL_FILTER = "avgSdFilter";
	public static final String KERNEL_FILTER_ID = "avgSdFilter2";
	
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
			ptxFileName = TransScale.preparePtxFile("bin/cuda/kernels/visualizer/visualizer-kernel.cu", RECOMPILE, GEN_DEBUG);
		} catch (IOException e) {
			System.err.println("Could not create PTX file");
			throw new RuntimeException("Could not create PTX file", e);
		}
		
		KernelAddJob job = new KernelAddJob();
		job.ptxFile = new File(ptxFileName);
		job.functionMapping = new HashMap<>();
		job.functionMapping.put(KERNEL_DESCRIBE, KERNEL_DESCRIBE);
		job.functionMapping.put(KERNEL_FILTER_ID, KERNEL_FILTER);
		TransScale.getInstance().addKernel(job);

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
	public int call(Invoker invoker, GLAutoDrawable drawable, final ClassifierSet classifiers, final Segment segment,
			final boolean shouldThreshold, float threshold, final float opacity,
			final boolean showConflicts, final int imageWidth, final int imageHeight) {
		
		// First, we must filter the segment using the context associated with this thread! Otherwise, CUDA will complain!
		segment.filterImage(this);
		final FilteredImage filteredImage = segment.getFilteredImage();
		
		/*
		 * Note: Lazy transfer is active, so we just obtain everything but
		 * no GPU API call will be made, so we're safe :-)
		 */
		final ClassifierAllocationResult pointerToAll = classifiers.getPointerToAll();
		
		final CudaByte2D devExpression = pointerToAll.expressions;
		final CudaByte2D overlayColors = pointerToAll.overlayColors;
		final CudaByte2D enabilityMap = pointerToAll.enabilityMap;
		// Determine the number of GP expressions
		final int numClassifiers = devExpression.getHeight();
		
		// Allocate the scratchpad
		float[] temp = new float[numClassifiers];	// Initial values of 0 for scratchpad
		final CudaFloat2D devScratchPad = new CudaFloat2D(numClassifiers, 1, 1, temp, true);
		
		Trigger pre = new Trigger() {
			@Override
			public void doTask(CUmodule module) {
				filteredImage.allocateAndTransfer();
				
				devExpression.reallocate();
				overlayColors.reallocate();
				enabilityMap.reallocate();
				devScratchPad.reallocate();
			}
		};
		
		
				
		
		
		
		
		
		
		
		
		
		
		
//		CUdeviceptr devScratchPad = new CUdeviceptr();
//		cuMemAlloc(devScratchPad, numClassifiers * Sizeof.FLOAT);
//		cuMemcpyHtoD(devScratchPad, Pointer.to(scratchPad), numClassifiers * Sizeof.FLOAT);
		
		// Map the OpenGL buffer to a CUDA pointer
//		cuGraphicsMapResources(1, new CUgraphicsResource[] { bufferResource }, null);
//		cuGraphicsResourceGetMappedPointer(devOutput, new long[1], bufferResource);
		
		// Setup kernel parameters
		
		KernelArgSetter setter = new KernelArgSetter() {
			
			@Override
			public Pointer getArgs() {
				return Pointer.to(Pointer.to(new byte[] {(byte) (shouldThreshold ? 1 : 0)}),
						Pointer.to(devExpression),Pointer.to(devExpression.getDevPitchInElements()), Pointer.to(new int[] { numClassifiers }),
						Pointer.to(enabilityMap),Pointer.to(overlayColors), Pointer.to(new byte[] {(byte) (showConflicts ? 1 : 0)}), Pointer.to(new float[] {opacity}),
						Pointer.to(devScratchPad),
						filteredImage.getInput().toPointer(), Pointer.to(new CUdeviceptr()),
						filteredImage.getSmallAvg().toPointer(), filteredImage.getMediumAvg().toPointer(), filteredImage.getLargeAvg().toPointer(),
						filteredImage.getSmallSd().toPointer(), filteredImage.getMediumSd().toPointer(), filteredImage.getLargeSd().toPointer(),
						Pointer.to(new int[] {segment.getBounds().x}), Pointer.to(new int[] {segment.getBounds().y}), Pointer.to(new int[] {segment.getBounds().width}), Pointer.to(new int[] {segment.getBounds().height}), Pointer.to(new int[] { (int) filteredImage.getInput().getDevPitchInElements()[0] }),
						Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }));
			}
		};
//		Pointer kernelParams = Pointer.to(Pointer.to(new byte[] {(byte) (shouldThreshold ? 1 : 0)}),
//				Pointer.to(devExpression),Pointer.to(devExpression.getDevPitchInElements()), Pointer.to(new int[] { numClassifiers }),
//				Pointer.to(enabilityMap),Pointer.to(overlayColors), Pointer.to(new byte[] {(byte) (showConflicts ? 1 : 0)}), Pointer.to(new float[] {opacity}),
//				Pointer.to(devScratchPad),
//				filteredImage.getInput().toPointer(), Pointer.to(new CUdeviceptr()),
//				filteredImage.getSmallAvg().toPointer(), filteredImage.getMediumAvg().toPointer(), filteredImage.getLargeAvg().toPointer(),
//				filteredImage.getSmallSd().toPointer(), filteredImage.getMediumSd().toPointer(), filteredImage.getLargeSd().toPointer(),
//				Pointer.to(new int[] {segment.getBounds().x}), Pointer.to(new int[] {segment.getBounds().y}), Pointer.to(new int[] {segment.getBounds().width}), Pointer.to(new int[] {segment.getBounds().height}), Pointer.to(new int[] { (int) filteredImage.getInput().getDevPitchInElements()[0] }),
//				Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }));
		
		final float[] scratchpad = new float[numClassifiers];
		
		// Unmap OpenGL buffer from CUDA
//		cuGraphicsUnmapResources(1, new CUgraphicsResource[] { bufferResource }, null);
		Trigger post = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				filteredImage.freeAll();
				devScratchPad.refresh();
				System.arraycopy(devScratchPad.getUnclonedArray(), 0, scratchpad, 0, scratchpad.length);
				devScratchPad.free();
				pointerToAll.freeAll();
				// fetch scratchPad and clear memory
			}
		};
		
		KernelInvoke kernelJob = new KernelInvoke();
		kernelJob.functionId = KERNEL_DESCRIBE;
		kernelJob.preTrigger = pre;
		kernelJob.postTrigger = post;
		
		kernelJob.gridDimX = segment.getBounds().height;
		kernelJob.gridDimY = 1;
		
		kernelJob.blockDimX = segment.getBounds().width;
		kernelJob.blockDimY = 1;
		kernelJob.blockDimZ = 1;
		
		kernelJob.argSetter = setter;
		
		// Queue kernel and wait for it		
		TransScale.getInstance().queueJob(kernelJob);
		kernelJob.waitFor();
		
		
		
//				cuLaunchKernel(fncDescribe,
//						segment.getBounds().height, 1, 1, 	// segment height is equal to the number of blocks
//						segment.getBounds().width, 1, 1,	// segment width is equal to the number of threads in each block
//						0, null,
//						kernelParams, null);
//				cuCtxSynchronize();
		
		
//		// Copy the scratchpad back 
//		cuMemcpyDtoH(Pointer.to(scratchPad), devScratchPad, Sizeof.FLOAT * numClassifiers);
//		
//		// Housekeeping
//		cuMemFree(devScratchPad);
		
		// Should we do the thresholding here??
		if (!shouldThreshold)
			return -1;
		
		// Do thresholding:
		// A list of classifiers that have claimed this segment
		List<Classifier> claimers = new ArrayList<Classifier>();
		
		for (int i = 0 ; i < scratchpad.length ; i++) {
			scratchpad[i] /= segment.getBounds().width * segment.getBounds().height;
			if (scratchpad[i] > threshold) {
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
	
	

	@Override
	public void performFilters(final CudaByte2D byteInput, final CudaFloat2D smallAvg, final CudaFloat2D mediumAvg, final CudaFloat2D largeAvg, final CudaFloat2D smallSd, final CudaFloat2D mediumSd, final CudaFloat2D largeSd) {
		final int imageWidth = byteInput.getWidth();
		final int imageHeight = byteInput.getHeight();
		final int numChannels = byteInput.getNumFields();
		
		byteInput.reallocate();
		smallAvg.reallocate();
		mediumAvg.reallocate();
		largeAvg.reallocate();
		
		smallSd.reallocate();
		mediumSd.reallocate();
		largeSd.reallocate();
		
		final CUarray devTexture = new CUarray();
		
		Trigger pre = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				// Allocate device array
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
				cuModuleGetTexRef(inputTexRef, module, "inputTexture");
				cuTexRefSetFilterMode(inputTexRef, CU_TR_FILTER_MODE_POINT);
				cuTexRefSetAddressMode(inputTexRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
				cuTexRefSetAddressMode(inputTexRef, 1, CU_TR_ADDRESS_MODE_CLAMP);
				cuTexRefSetFlags(inputTexRef, CU_TRSF_READ_AS_INTEGER);
				cuTexRefSetFormat(inputTexRef, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, numChannels);
				cuTexRefSetArray(inputTexRef, devTexture, CU_TRSA_OVERRIDE_FORMAT);
				
				smallAvg.reallocate();
				mediumAvg.reallocate();
				largeAvg.reallocate();
				
				smallSd.reallocate();
				mediumSd.reallocate();
				largeSd.reallocate();
			}
		};
		
		// Allocate results array
		KernelArgSetter setter = new KernelArgSetter() {
			
			@Override
			public Pointer getArgs() {
				return Pointer.to(smallAvg.toPointer(), smallSd.toPointer(),
						mediumAvg.toPointer(), mediumSd.toPointer(),
						largeAvg.toPointer(), largeSd.toPointer(),
						Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }), Pointer.to(new int[] { (int) smallAvg.getDevPitchInElements()[0] }),
						Pointer.to(new int[] { getSmallFilterSize() }), Pointer.to(new int[] { getMediumFilterSize() }), Pointer.to(new int[] { getLargeFilterSize() }));				
			}
		};

		// Call kernel
//		cuLaunchKernel(this.fncFilter,
//				, , 1,
//				FILTER_BLOCK_SIZE, FILTER_BLOCK_SIZE, 1,
//				0, null,
//				kernelParams, null);
//		cuCtxSynchronize();
		
		Trigger post = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				smallAvg.refresh();
				mediumAvg.refresh();
				largeAvg.refresh();
				
				smallSd.refresh();
				mediumSd.refresh();
				largeSd.refresh();
				
				smallAvg.free();
				mediumAvg.free();
				largeAvg.free();
				
				smallSd.free();
				mediumSd.free();
				largeSd.free();
				
				// A little housekeeping
				cuArrayDestroy(devTexture);				
			}
		};
		
		KernelInvoke kernelJob = new KernelInvoke();
		kernelJob.functionId = KERNEL_FILTER_ID;
		kernelJob.preTrigger = pre;
		kernelJob.postTrigger = post;
		
		kernelJob.gridDimX = (imageWidth + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
		kernelJob.gridDimY = (imageHeight + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
		
		kernelJob.blockDimX = FILTER_BLOCK_SIZE;
		kernelJob.blockDimY = FILTER_BLOCK_SIZE;
		kernelJob.blockDimZ = 1;
		
		kernelJob.argSetter = setter;
		
		TransScale.getInstance().queueJob(kernelJob);
		kernelJob.waitFor();
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
