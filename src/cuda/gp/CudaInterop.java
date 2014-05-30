package cuda.gp;

import static jcuda.driver.JCudaDriver.*;

import java.awt.Color;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.commons.io.FileUtils;

import cuda.multigpu.KernelAddJob;
import cuda.multigpu.KernelInvoke;
import cuda.multigpu.TransScale;
import cuda.multigpu.Trigger;
import ec.EvolutionState;
import ec.Singleton;
import ec.util.Parameter;
import gnu.trove.list.array.TByteArrayList;
import gp.datatypes.CudaTrainingInstance;
import gp.datatypes.Job;
import gp.datatypes.TrainingInstance;
import utils.ByteImage;
import utils.FilteredImage;
import utils.ImageFilterProvider;
import utils.cuda.pointers.CudaByte2D;
import utils.cuda.pointers.CudaFloat2D;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import jcuda.utils.KernelLauncher;
import static jcuda.driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP;
import static jcuda.driver.CUfilter_mode.CU_TR_FILTER_MODE_POINT;

/**
 * A helper class to manage CUDA/GP interoperability. This class can evaluate individuals
 * on the graphics card and can also transfer data to and from GPU.
 * Most of the operations of this class require a CudaData instance.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaInterop implements Singleton, ImageFilterProvider {
	
	private static final boolean RECOMPILE = false;
	
	private static final String KERNEL_EVALUATE = "evaluate";
	private static final String KERNEL_FILTER = "avgSdFilter";

	private KernelLauncher kernel = null;
	private String kernelCode;

	private int FILTER_BLOCK_SIZE = 16;

	private int EVAL_BLOCK_SIZE = 512; // no point increasing the number of threads per block (CUDA OCCUPANCY CALCULATOR says so!)
	private int EVAL_MAX_GRID_SIZE;

	private int EVAL_FITNESS_CASES;
	private int POSITIVE_COUNT;
	private int NEGATIVE_COUNT;

	private final int DESC_BLOCK_SIZE = 512;

	private int smallFilterSize; // The size of the small filter
	private int mediumFilterSize; // The size of the medium filter
	private int largeFilterSize; // The size of the large filter

//	private CUcontext context = null;
//	private CUmodule codeModule = null; // Stores the module of the compiled code.
//	private CUfunction fncEvaluate = null; // Function handle for the "Evaluate" function
//	private CUfunction fncFilter = null; // Function handle for the image filter function

	@Override
	public void setup(EvolutionState state, Parameter base) {
		// Set the number of training points as well as the maximum CUDA grid size
		this.EVAL_MAX_GRID_SIZE = state.parameters.getInt(new Parameter("pop.subpop.0.size"), null);
		this.POSITIVE_COUNT = state.parameters.getInt(new Parameter("problem.positiveExamples"), null);
		this.NEGATIVE_COUNT = state.parameters.getInt(new Parameter("problem.negativeExamples"), null);
		this.EVAL_FITNESS_CASES = POSITIVE_COUNT + NEGATIVE_COUNT;
		
		// Obtain the filter sizes
		this.smallFilterSize = state.parameters.getInt(new Parameter("problem.smallWindowSize"), null);
		this.mediumFilterSize = state.parameters.getInt(new Parameter("problem.mediumWindowSize"), null);
		this.largeFilterSize = state.parameters.getInt(new Parameter("problem.largeWindowSize"), null);

		try {
			prepareKernel(RECOMPILE);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(0);
		}
	}
	
	/**
	 * Prepares the data for an evolutionary run. This method must be called before 
	 * starting the evolutionary loop. Essentially after calling this method, the 
	 * training data is transferred to the GPU and the training samples are selected. 
	 * 
	 * @param job	The job object that is queued on the GPEngine
	 * 
	 */
	public void prepareDataForRun(CudaEvolutionState state, Job job) {
		prepareImages(job);
		randomSample(state, job);
	}

	/**
	 * Obtains the passed samples to the CudaEvolutionState, performs filters on
	 * the training images and stores them in CUDA memory.
	 * 
	 * @param job	The current active job
	 */
	private void prepareImages(Job job) {
		
		switch(job.getJobType()) {
		case Job.TYPE_GT:
			gtPrepare(job);
			break;
			
		case Job.TYPE_POS_NEG:
			posNegPrepare(job);
			break;
		}
	}
	
	/**
	 * A helper function that prepares the images (ie. calculates the filters)
	 * for the evolutionary run for the GT training mode.
	 * 
	 * @param job	The current active job
	 */
	private void gtPrepare(Job job) {
		FilteredImage image = new FilteredImage(job.getTrainingImage(), this);
		job.setFilteredTrainingImage(image);
	}
	
	/**
	 * A helper function that prepares the images (ie. calculates the filters)
	 * for the evolutionary run for the positive-negative training mode.
	 * 
	 * @param job The current active job
	 */
	private void posNegPrepare(Job job) {
		for (ByteImage image : job.getPositiveExamples()) {
			FilteredImage filtered = new FilteredImage(image, this);
			job.addFilteredPositiveImage(filtered);
		}

		for (ByteImage image : job.getNegativeExamples()) {
			FilteredImage filtered = new FilteredImage(image, this);
			job.addFilteredNegativeImage(filtered);
		}
	}

	private void randomSample(CudaEvolutionState state, Job job) {
		
		switch(job.getJobType()) {
		case Job.TYPE_GT:
			gtSample(state, job);
			break;
			
		case Job.TYPE_POS_NEG:
			posNegSample(state, job);
			break;
		}
	}
	
	/**
	 * Helper function to sample the training points using a ground truth image.
	 * 
	 * @param cuState	The EvolutionState object to use
	 */
	private void gtSample(CudaEvolutionState state, Job job) {
		HashSet<Integer> trainingPoints = new HashSet<Integer>(); // To prevent duplicated training points		
		List<TrainingInstance> instances = new ArrayList<TrainingInstance>(this.POSITIVE_COUNT + this.NEGATIVE_COUNT);
		ByteImage inputImage = job.getTrainingImage();
		ByteImage gtImage = job.getGtImage();
		
		int pos = 0, neg = 0;
		int width = inputImage.getWidth();
		int height = inputImage.getHeight();
		int numChannels = inputImage.getNumChannels();
		
		while (pos < this.POSITIVE_COUNT || neg < this.NEGATIVE_COUNT) {
			int index = state.random[0].nextInt(width * height);
			
			if (trainingPoints.contains(new Integer(index)))
				continue;	// This training point was selected before...
			
			Color c = gtImage.getColor(index);
			boolean haveInstance = false;
			int label = -1;
			
			if (c.equals(Color.green) &&  pos < this.POSITIVE_COUNT) {	// positive sample
				pos++;
				haveInstance = true;
				label = 1;
			}
			else if (neg < this.NEGATIVE_COUNT) {
				neg++;
				haveInstance = true;
				label = 2;
			}
			
			if (haveInstance) {
				FilteredImage filtered = job.getFilteredTrainingImage();
				TrainingInstance instance = new TrainingInstance(filtered, index, label);
				instances.add(instance);
				trainingPoints.add(new Integer(index));
			} // end-if
				
		} // end-while
		
		// Shuffle the samples
		Collections.shuffle(instances);
			
		// Create a CudaTrainingData instance to hold the same training points
		// This call will also transfer the training data to GPU
		job.setCudaTrainingInstances(new CudaTrainingInstance(instances, numChannels));
	}
	
	
	/**
	 * Helper function to sample training points from a set of positive examples 
	 * and negative examples.
	 * 
	 * @param cuState	The EvolutionState object to use
	 */
	private void posNegSample(CudaEvolutionState state, Job job) {
		HashSet<Integer> trainingPoints = new HashSet<Integer>(); // To prevent duplicated training points
		
		List<TrainingInstance> instances = new ArrayList<TrainingInstance>(this.POSITIVE_COUNT + this.NEGATIVE_COUNT);

		int sampleCount;
		List<ByteImage> samples;
		List<FilteredImage> filteredSamples;

		samples = job.getPositiveExamples();
		filteredSamples = job.getFilteredPositiveImages();
		int numChannels = 0;

		// Would like to evenly select positive examples from all existing positive images
		// Therefore, we find the portion for each sample image (using the ceiling value)
		int portion = (POSITIVE_COUNT - 1) / samples.size() + 1;

		for (int i = 0; i < samples.size(); i++) {
			ByteImage image = samples.get(i);
			FilteredImage filteredImage = filteredSamples.get(i);
			sampleCount = 0;
			numChannels = image.getNumChannels();

			while (sampleCount < portion) { // Should loop until we have sampled the desired number of pixels from this image
				int index = state.random[0].nextInt(image.getWidth() * image.getHeight());

				if (trainingPoints.contains(new Integer(index)))
					continue; // Oops! We have already sampled this pixel
				
				// This pixel should not have alpha == 0. If it has, it means it is background!
				if (image.getColor(index).getAlpha() == 0)
					continue;

				// Add this point as the already used point
				trainingPoints.add(new Integer(index));				
				
				TrainingInstance instance = new TrainingInstance(filteredImage, index, 1);
				instances.add(instance);
				sampleCount++;
			}
		}
		
		// Could adjust the number of POSITIVE_COUNT here, but who cares...? ;)
		sampleCount = 0;
		trainingPoints.clear();
		
		samples = job.getNegativeExamples();
		filteredSamples = job.getFilteredNegativeImages();
		
		portion = (NEGATIVE_COUNT - 1) / samples.size() + 1;

		for (int i = 0; i < samples.size(); i++) {
			ByteImage image = samples.get(i);
			FilteredImage filteredImage = filteredSamples.get(i);
			sampleCount = 0;

			while (sampleCount < portion) { // Should loop until we have sampled the desired number of pixels from this image
				int index = state.random[0].nextInt(image.getWidth() * image.getHeight());

				if (trainingPoints.contains(new Integer(index)))
					continue; // Oops! We have already sampled this pixel
				
				if (image.getColor(index).getAlpha() == 0)
					continue;

				trainingPoints.add(new Integer(index));

				TrainingInstance instance = new TrainingInstance(filteredImage, index, 2);
				instances.add(instance);
				sampleCount++;
			}
		}
		
		// Shuffle the samples
		Collections.shuffle(instances);
		
		// Create a CudaTrainingData instance to hold the same training points
		// This call will also transfer the training data to GPU
		job.setCudaTrainingInstances(new CudaTrainingInstance(instances, numChannels));	 
	}

	/**
	 * Sets the kernel's code template. This method must be called before the kernel is compiled.
	 * Also note that the method should be called when the function sets and everything else has
	 * been setup.
	 * 
	 * @param code	the kernel template code to use for kernel compilation
	 */
	public void setKernelCode(String code) {
		this.kernelCode = code;
	}

	private void prepareKernel(boolean recompile) throws Exception {
		JCuda.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);

		String kernelTemplate = "";
//		fncEvaluate = new CUfunction();
//		fncFilter = new CUfunction();

		File kernelCodeFile = new File("bin/cuda/kernels/gp/cuda-kernels.cu");

		if (recompile || !kernelCodeFile.exists()) {
			// Read the template code and insert the actions for the functions
			kernelTemplate = FileUtils.readFileToString(new File("bin/cuda/kernels/gp/evaluator-template.cu")).replace("/*@@actions@@*/", kernelCode);
			// Set the value of the constants in the kernel
			kernelTemplate = kernelTemplate.replace("/*@@fitness-cases@@*/", String.valueOf(EVAL_FITNESS_CASES));
			kernelTemplate = kernelTemplate.replace("/*@@eval-block-size@@*/", String.valueOf(EVAL_BLOCK_SIZE));
			kernelTemplate = kernelTemplate.replace("/*@@max-grid-size@@*/", String.valueOf(EVAL_MAX_GRID_SIZE));
			kernelTemplate = kernelTemplate.replace("/*@@positive-examples@@*/", String.valueOf(POSITIVE_COUNT));
			kernelTemplate = kernelTemplate.replace("/*@@negative-examples@@*/", String.valueOf(NEGATIVE_COUNT));

			kernelTemplate = kernelTemplate.replace("/*@@desc-block-size@@*/", String.valueOf(DESC_BLOCK_SIZE));
			// Save the template
			FileUtils.write(kernelCodeFile, kernelTemplate);
		}

		KernelAddJob kernelAdd = new KernelAddJob();
		kernelAdd.ptxFile = new File("bin/cuda/kernels/gp/cuda-kernels.ptx");
		kernelAdd.functionMapping = new HashMap<>();
		kernelAdd.functionMapping.put(KERNEL_EVALUATE, KERNEL_EVALUATE);
		kernelAdd.functionMapping.put(KERNEL_FILTER, KERNEL_FILTER);
		
		TransScale.getInstance().addKernel(kernelAdd);
		if (true)
		return;
		
		
		
		// Compile or load the kernel
		kernel = KernelLauncher.create("bin/cuda/kernels/gp/cuda-kernels.cu", "evaluate", recompile, "-arch=compute_20 -code=sm_30 -use_fast_math");

		System.out.println(recompile ? "Kernel COMPILED" : "Kernel loaded");

		// Save the current context. I want to use this context later for multithreaded operations
//		this.context = new CUcontext();
//		cuCtxGetCurrent(context);

		// Get the module so that I can get handles to kernel functions
//		this.codeModule = kernel.getModule();
//		cuModuleGetFunction(fncEvaluate, codeModule, "evaluate");
//		cuModuleGetFunction(fncFilter, codeModule, "avgSdFilter");
	}

	/**
	 * This method will switch the context to the calling thread so that another host
	 * thread can access the CUDA functionality offered by this class.
	 */
	public synchronized void switchContext() {
//		cuCtxSetCurrent(context); DO NOTHING!
	}

	/**
	 * Performs the average and standard deviation filters on the provided input
	 * and stores the results in the provided CudaFloat2D objects
	 * 
	 * @param byteInput	The input image data
	 * @param averageResult Placeholder to store the result of the average filter
	 * @param sdResult Placeholder to store the result of the standard deviation filter
	 * @param maskSize	the mask size to use for the filter
	 */
	private void performFilter(final CudaByte2D byteInput, final CudaFloat2D averageResult, final CudaFloat2D sdResult, int maskSize) {
		final int imageWidth = byteInput.getWidth();
		final int imageHeight = byteInput.getHeight();
		final int numChannels = byteInput.getNumFields();
		
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
				cuTexRefSetFormat(inputTexRef, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, 4);
				cuTexRefSetArray(inputTexRef, devTexture, CU_TRSA_OVERRIDE_FORMAT);
				
				averageResult.reallocate();
				sdResult.reallocate();
			}
		};
		
		

		Pointer kernelParams = Pointer.to(averageResult.toPointer(), sdResult.toPointer(),
				Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }), Pointer.to(new int[] { (int) averageResult.getDevPitchInElements()[0] }),
				Pointer.to(new int[] { maskSize }));

//		// Call kernel
//		cuLaunchKernel(fncFilter,
//				(imageWidth + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, (imageHeight + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, 1,
//				16, 16, 1,
//				0, null,
//				kernelParams, null);
//		cuCtxSynchronize();

		Trigger post = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				// Retrieve results
				averageResult.refresh();
				sdResult.refresh();

				// A little housekeeping
				cuArrayDestroy(devTexture);				
			}
		};
		
		KernelInvoke kernelJob = new KernelInvoke();
		kernelJob.functionId = KERNEL_FILTER;
		kernelJob.preTrigger = pre;
		kernelJob.postTrigger = post;
		
		kernelJob.gridDimX = (imageWidth + FILTER_BLOCK_SIZE - 1);
		kernelJob.gridDimY = (imageHeight + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
		
		kernelJob.blockDimX = 16;
		kernelJob.blockDimY = 16;
		kernelJob.blockDimZ = 1;
		
		kernelJob.pointerToArguments = kernelParams;
		
		TransScale.getInstance().queueJob(kernelJob);
		kernelJob.waitFor();
	}
	
	@Override
	public void performFilters(CudaByte2D byteInput, CudaFloat2D smallAvg, CudaFloat2D mediumAvg, CudaFloat2D largeAvg, CudaFloat2D smallSd, CudaFloat2D mediumSd, CudaFloat2D largeSd) {
		performFilter(byteInput, smallAvg, smallSd, getSmallFilterSize());
		performFilter(byteInput, mediumAvg, mediumSd, getMediumFilterSize());
		performFilter(byteInput, largeAvg, largeSd, getLargeFilterSize());
	}
	
	@Override
	public int getSmallFilterSize() {
		return smallFilterSize;
	}

	@Override
	public int getMediumFilterSize() {
		return mediumFilterSize;
	}

	@Override
	public int getLargeFilterSize() {
		return largeFilterSize;
	}
	
	/**
	 * Evaluates a list of invidivuals on the GPU and returns their fitness array
	 * 
	 * @param expressions	List of list of list of expressions that are unevaluated per each evaluator thread! 
	 * @param job	The active job
	 * @return	The fitness array
	 */
	public float[] evaluatePopulation(List<List<TByteArrayList>> expressions, Job job) {
		// First determine how many unevals we have in total
		int indCount = 0;
		int maxExpLength = 0;

		for (List<TByteArrayList> thExps : expressions) {
			indCount += thExps.size();

			// Determine the longest expression
			for (TByteArrayList exp : thExps)
				if (exp.size() > maxExpLength)
					maxExpLength = exp.size();
		}

		// Convert expressions to byte[]
		byte[] population = new byte[indCount * maxExpLength];
		int i = 0;

		for (List<TByteArrayList> thExps : expressions) {
			for (TByteArrayList currExp : thExps) {
				int length = currExp.size();
				currExp.toArray(population, 0, i * maxExpLength, length);
				i++;
			}
		}

		final CudaTrainingInstance ti = job.getTrainingInstances();		
		// Allocate expressions memory
		final CudaByte2D devExpressions = new CudaByte2D(population.length, 1, 1, population, true);
		/**/Trigger pre = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				ti.allocateAndTransfer();
				devExpressions.reallocate();
			}
		};
		
		/**/Trigger post = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				// Refresh fitnesses (aka outputs)
				ti.getOutputs().refresh();
				devExpressions.free();
			}
		};
		
		KernelInvoke kernelJob = new KernelInvoke();
		kernelJob.functionId = KERNEL_EVALUATE;
		kernelJob.preTrigger = pre;
		kernelJob.postTrigger = post;
		
		kernelJob.gridDimX = indCount;
		kernelJob.gridDimY = 1;
		
		kernelJob.blockDimX = EVAL_BLOCK_SIZE;
		kernelJob.blockDimY = 1;
		kernelJob.blockDimZ = 1;
		
		kernelJob.pointerToArguments =
				Pointer.to(
					devExpressions.toPointer(), Pointer.to(new int[] {indCount}), Pointer.to(new int[] {maxExpLength}),
					ti.getInputs().toPointer(),
					ti.getSmallAvgs().toPointer(), ti.getMediumAvgs().toPointer(), ti.getLargeAvgs().toPointer(),
					ti.getSmallSds().toPointer(), ti.getMediumSds().toPointer(), ti.getLargeSds().toPointer(),
					ti.getLabels().toPointer(), ti.getOutputs().toPointer()
				);
		
		TransScale.getInstance().queueJob(kernelJob);
		kernelJob.waitFor();
		
		
		
//		kernel.setGridSize(indCount, 1);
//		kernel.setBlockSize(EVAL_BLOCK_SIZE, 1, 1);
//		kernel.call(devExpressions, indCount, maxExpLength,
//				ti.getInputs(),
//				ti.getSmallAvgs(), ti.getMediumAvgs(), ti.getLargeAvgs(),
//				ti.getSmallSds(), ti.getMediumSds(), ti.getLargeSds(),
//				ti.getLabels(), ti.getOutputs());
//		
//
//
		
		// Refresh fitnesses (aka outputs)
		
//		ti.getOutputs().refresh();
//		devExpressions.free();

		return ti.getOutputs().getUnclonedArray();
	}
}
