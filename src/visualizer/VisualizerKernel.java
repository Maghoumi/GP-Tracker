package visualizer;

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

import utils.cuda.datatypes.CUdeviceptr2D;
import utils.cuda.datatypes.Classifier;
import utils.cuda.datatypes.ClassifierSet;
import utils.cuda.datatypes.ClassifierSet.ClassifierAllocationResult;
import utils.cuda.datatypes.NewCudaData;
import utils.cuda.datatypes.Segment;
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
public class VisualizerKernel {
	
	/** Recompile flag ** FOR DEBUGGING PURPOSES** */
	public static final boolean RECOMPILE = false;
	/** Generate debug info ** FOR DEBUGGING PURPOSES** */
	public static final boolean GEN_DEBUG = false;
	
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
	public VisualizerKernel() {
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
	 * @param gp	The GPSystem object that is used for retraining classifiers
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
			boolean showConflicts, boolean autoRetrain,
			int imageWidth, int imageHeight) {
		
		CUdeviceptr2D devExpression = pointerToAll.expressions;
		CUdeviceptr overlayColors = pointerToAll.overlayColors;
		CUdeviceptr enabilityMap = pointerToAll.enabilityMap;
		
		// Determine the number of GP expressions
		int numClassifiers = devExpression.getHeight();
		
		NewCudaData descData = segment.getImageData();
		// Perform filters for the describe data
		descData.setCudaObjects(codeModule, fncFilter);
		
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
				Pointer.to(devExpression),Pointer.to(devExpression.getPitchInElements()), Pointer.to(new int[] { numClassifiers }),
				Pointer.to(enabilityMap),Pointer.to(overlayColors), Pointer.to(new byte[] {(byte) (showConflicts ? 1 : 0)}), Pointer.to(new float[] {opacity}),
				Pointer.to(devScratchPad),
				Pointer.to(descData.dev_input), Pointer.to(devOutput),
				Pointer.to(descData.dev_smallAvg), Pointer.to(descData.dev_mediumAvg), Pointer.to(descData.dev_largeAvg),
				Pointer.to(descData.dev_smallSd), Pointer.to(descData.dev_mediumSd), Pointer.to(descData.dev_largeSd),
				Pointer.to(new int[] {segment.x}), Pointer.to(new int[] {segment.y}), Pointer.to(new int[] {segment.width}), Pointer.to(new int[] {segment.height}), 
				Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }));
		
		
		// Call kernel
		cuLaunchKernel(fncDescribe,
				segment.height, 1, 1, 	// segment height is equal to the number of blocks
				segment.width, 1, 1,	// segment width is equal to the number of threads in each block
				0, null,
				kernelParams, null);
		cuCtxSynchronize();
		
		// Unmap OpenGL buffer from CUDA
		cuGraphicsUnmapResources(1, new CUgraphicsResource[] { bufferResource }, null);
		descData.freeAll();
		
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
			scratchPad[i] /= segment.width * segment.height;
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
					imageWidth, imageHeight ,segment.getRectangle());
			return claimers.size();
		} 
		else if (claimers.size() != 0) {	// if 0 => no classifiers have been active!!
			OpenGLUtils.drawRegionOverlay(drawable, glBuffer,
					Color.RED, opacity,
					imageWidth, imageHeight, segment.getRectangle());
		}
		
		if (!autoRetrain)
			return claimers.size();
		
		// Retrain all asserters
		//FIXME should I retrain only specific ones??
		for (Classifier c : claimers) {
			invoker.retrain(c, false);
		}
		
		return claimers.size();
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
	
}
