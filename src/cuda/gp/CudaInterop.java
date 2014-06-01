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

import cuda.multigpu.*;
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
	
	private void prepareKernel(boolean recompile) throws Exception {
		JCuda.setExceptionsEnabled(true);
		JCudaDriver.setExceptionsEnabled(true);

		String kernelTemplate = "";

		File kernelCodeFile = new File(TransScale.preparePtxFile("bin/cuda/kernels/gp/cuda-kernels.cu", RECOMPILE, false));

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
		System.out.println(recompile ? "Kernel COMPILED" : "Kernel loaded");
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
	private void performFilter(final CudaByte2D byteInput, final CudaFloat2D averageResult, final CudaFloat2D sdResult, final int maskSize) {
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
		
		KernelArgSetter setter = new KernelArgSetter() {
			
			@Override
			public Pointer getArgs() {
				return Pointer.to(averageResult.toPointer(), sdResult.toPointer(),
						Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }), Pointer.to(new int[] { (int) averageResult.getDevPitchInElements()[0] }),
						Pointer.to(new int[] { maskSize }));
			}
		}; 

		Trigger post = new Trigger() {
			
			@Override
			public void doTask(CUmodule module) {
				// Retrieve results
				averageResult.refresh();
				sdResult.refresh();
				averageResult.free();
				sdResult.free();

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
		
		kernelJob.argSetter = setter;
		
		TransScale.getInstance().queueJob(kernelJob);
		kernelJob.waitFor();
	}
	
	@Override
	public void performFilters(final CudaByte2D byteInput, final CudaFloat2D smallAvg, final CudaFloat2D mediumAvg, final CudaFloat2D largeAvg, final CudaFloat2D smallSd, final CudaFloat2D mediumSd, final CudaFloat2D largeSd) {
		Thread smallFilterRunner = new Thread( new Runnable() {
			
			@Override
			public void run() {
				performFilter(byteInput, smallAvg, smallSd, getSmallFilterSize());
			}
		});
		smallFilterRunner.start();
		
		Thread mediumFilterRunner = new Thread(new Runnable() {
			
			@Override
			public void run() {
				performFilter(byteInput, mediumAvg, mediumSd, getMediumFilterSize());
			}
		});
		mediumFilterRunner.start();
		
		Thread largeFilterRunner = new Thread(new Runnable() {
			
			@Override
			public void run() {
				performFilter(byteInput, largeAvg, largeSd, getLargeFilterSize());
			}
		});
		largeFilterRunner.start();
		
		try {
			smallFilterRunner.join();
			mediumFilterRunner.join();
			largeFilterRunner.join();
		}
		catch (Throwable e) {
			
		}
		
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
		int maxExpLengthtmp = 0;

		for (List<TByteArrayList> thExps : expressions) {
			indCount += thExps.size();

			// Determine the longest expression
			for (TByteArrayList exp : thExps)
				if (exp.size() > maxExpLengthtmp)
					maxExpLengthtmp = exp.size();
		}
		
		final int maxExpLength = maxExpLengthtmp;

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
		
		// Break population into chunks, each chunk is evaluated by a single GPU
		TransScale scaler = TransScale.getInstance();
		int gpuCount = scaler.getNumberOfDevices();
		float[] fitnessResults = new float[indCount];	// The evaluated fitness values
		
		// Create gpuCount number of threads. Each thread will call "waitFor" for a single job 
		Thread[] gpuInteropThreads = new Thread[gpuCount];
		
		int arrayCpyOffset = 0;	// Offset variable
		int assignmentOffset = 0;	// Offset variable
		
		List<byte[]> chunks = new ArrayList<>();
		
		for (i = 0 ; i < gpuCount ; i++) {
			final GpuRunnerThread runner = new GpuRunnerThread();
			runner.scaler = scaler;
			runner.destination = fitnessResults;
			runner.start = assignmentOffset;
			
			final int thisOutputShare;
			int thisPopShare;
			
			// *Evenly* divide the number of individuals
			if (i == gpuCount - 1) {
				thisOutputShare = indCount - i * (indCount / gpuCount);
			}
			else {
				thisOutputShare = indCount / gpuCount;
			}
			
			thisPopShare = thisOutputShare * maxExpLength;
			
			final CudaTrainingInstance ti = (CudaTrainingInstance) job.getTrainingInstances().clone();
			final CudaFloat2D chunkOutput = new CudaFloat2D(thisOutputShare, 1, 1, true);	// Allocate the output pointer for this portion
			
			byte[] popChunk  = new byte[thisPopShare];
			System.arraycopy(population, arrayCpyOffset, popChunk, 0, thisPopShare);	// Copy this GPU's chunk of expressions
			chunks.add(popChunk);
			final CudaByte2D devExpression = new CudaByte2D(thisPopShare, 1, 1, popChunk, true);	// Allocate device expression pointer
			
			arrayCpyOffset += thisPopShare;
			assignmentOffset += thisOutputShare;
			
			Trigger pre = new Trigger() {
				
				@Override
				public void doTask(CUmodule module) {
					ti.allocateAndTransfer();
					chunkOutput.allocate();
					devExpression.reallocate();
				}
			};
			
			Trigger post = new Trigger() {
				
				@Override
				public void doTask(CUmodule module) {
					chunkOutput.refresh();
					devExpression.free();
					runner.result = chunkOutput.getUnclonedArray();
					ti.freeAll();
					chunkOutput.free();
				}
			}; 
			
			// Create a kernel job
			KernelInvoke kernelJob = new KernelInvoke();
			
			kernelJob.functionId = KERNEL_EVALUATE;
			kernelJob.preTrigger = pre;
			kernelJob.postTrigger = post;
			
			kernelJob.gridDimX = thisOutputShare;
			kernelJob.gridDimY = 1;
			
			kernelJob.blockDimX = EVAL_BLOCK_SIZE;
			kernelJob.blockDimY = 1;
			kernelJob.blockDimZ = 1;
			
			kernelJob.argSetter = new KernelArgSetter() {
				
				@Override
				public Pointer getArgs() {
					return Pointer.to(
							devExpression.toPointer(), Pointer.to(new int[] {thisOutputShare}), Pointer.to(new int[] {maxExpLength}),
							ti.getInputs().toPointer(),
							ti.getSmallAvgs().toPointer(), ti.getMediumAvgs().toPointer(), ti.getLargeAvgs().toPointer(),
							ti.getSmallSds().toPointer(), ti.getMediumSds().toPointer(), ti.getLargeSds().toPointer(),
							ti.getLabels().toPointer(), chunkOutput.toPointer()
						);
				}
			};
					
			
			kernelJob.id = "Popsize: " + indCount + " (array length of  " + population.length + ") my share is " + thisOutputShare + " processing " + thisPopShare + " maxExpLength:" + maxExpLength;
			
			runner.kernelJob = kernelJob;
			gpuInteropThreads[i] = new Thread(runner);
			// Run this job on this thread, wait for the job to finish, then copy back the fitness
			gpuInteropThreads[i].start();			
		}
		
//		i = 0;
//		for (byte[] chunk : chunks) {
//			for (int j = 0 ; j < chunk.length ; j++) {
//				if (population[i] != chunk[j]) {
//					throw new RuntimeException("elements not equal");
//				}
//				i++;
//			}
//			
//		}
//		
//		for (i = 0 ; i < gpuCount ; i++) {
//			gpuInteropThreads[i].start();
//		}
		
		
		
		// Wait for auxiliary threads to finish their job
		for (i = 0 ; i < gpuCount ; i++) {
			try {
				gpuInteropThreads[i].join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}		
		
		// No need to merge anything! The threads have done that :-)
		return fitnessResults;
	}
	
	/**
	 * Runs using a thread, queues a job on GPU, waits for the job to finish and
	 * then obtains the results and copies its portion of the output to the final
	 * output.
	 * 
	 * @author Mehran Maghoumi
	 *
	 */
	private class GpuRunnerThread implements Runnable {
		
		public TransScale scaler;
		public KernelInvoke kernelJob;
		
		public float[] destination;
		public float[] result;
		public int start;

		@Override
		public void run() {
			scaler.queueJob(kernelJob);
			kernelJob.waitFor();
			
			// Copy my fitness values :-)
			System.arraycopy(result, 0, destination, start, result.length);
		}
		
	}
}
