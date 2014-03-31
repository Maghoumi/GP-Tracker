package cuda;

import static jcuda.driver.JCudaDriver.*;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import javax.imageio.ImageIO;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLProfile;

import org.apache.commons.io.FileUtils;

import cuda.gp.CudaEvolutionState;
import ec.EvolutionState;
import ec.Individual;
import ec.Singleton;
import ec.gp.GPIndividual;
import ec.util.Parameter;
import gnu.trove.list.array.TByteArrayList;
import utils.PreciseTimer;
import utils.StringUtils;
import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.CudaData;
import utils.cuda.datatypes.Float4;
import visualizer.Visualizer;
import m2xfilter.datatypes.DataInstance;
import m2xfilter.datatypes.ProblemData;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUarray;
import jcuda.driver.CUcontext;
import jcuda.driver.CUarray_format;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUmodule;
import jcuda.driver.CUtexref;
import jcuda.driver.JCudaDriver;
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
public class CudaInterop implements Singleton {

	public void cleanUp() {
		// TODO: should deallocate memories, should deallocate CUDA examples and everything else
		throw new RuntimeException("Not implemented yet!");
	}

	private KernelLauncher kernel = null;
	private String kernelCode;

	private ProblemData gpData; // for CPU evaluation purposes

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

	private CUcontext context = null;
	private CUmodule codeModule = null; // Stores the module of the compiled code.
	private CUfunction fncEvaluate = null; // Function handle for the "Evaluate" function
	private CUfunction fncDescribe = null; // Function handle for the "Describe" function
	private CUfunction fncFilter = null; // Function handle for the image filter function

	/** The visualizer instance that can display data in realtime */
	public Visualizer visualizer;

	/** Flag indicating whether the data should be visualized using OpenGL */
	private boolean visualize;
	
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
			prepareKernel(false);
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
	 * @param state
	 * 		An evolution state to use for seting up the data
	 */
	public void prepareDataForRun(CudaEvolutionState state) {
		prepareImages(state);
		randomSample(state);
	}

	/**
	 * Obtains the passed samples to the CudaEvolutionState, performs filters on
	 * the training images and stores them in CUDA memory.
	 */
	private void prepareImages(EvolutionState state) {
		CudaEvolutionState cuState = (CudaEvolutionState) state;
		
		if (cuState.getTrainingMode() == CudaEvolutionState.TRAINING_MODE_GT) {
			gtPrepare(cuState);
		}
		else if (cuState.getTrainingMode() == CudaEvolutionState.TRAINING_MODE_POS_NEG) {
			posNegPrepare(cuState);
		}
	}
	
	/**
	 * A helper function that prepares the images (ie. calculates the filters)
	 * for the evolutionary run for the GT training mode.
	 * 
	 * @param state	The EvolutionState object to work with
	 */
	private void gtPrepare(CudaEvolutionState cuState) {
		CudaData data = new CudaData();
		fillAndPerformFilters(cuState.getTrainingImage(), data);
		cuState.setDevTrainingImage(data);
	}
	
	/**
	 * A helper function that prepares the images (ie. calculates the filters)
	 * for the evolutionary run for the positive-negative training mode.
	 * 
	 * @param state	The EvolutionState object to work with
	 */
	private void posNegPrepare(CudaEvolutionState cuState) {
		// FIXME is it even necessary to have everything on the GPU??
		// FIXME for all I know, there is so much traffic on the PCI bus
		// FIXME and maybe we just need the training data to reside on
		// FIXME the GPU memory... Will have too look at it further
		for (ByteImage image : cuState.getPositiveExamples()) {
			CudaData data = new CudaData();
			fillAndPerformFilters(image, data);
			cuState.addDevPositiveExample(data);
		}

		for (ByteImage image : cuState.getNegativeExamples()) {
			CudaData data = new CudaData();
			fillAndPerformFilters(image, data);
			cuState.addDevNegativeExample(data);
		}
	}

	private void randomSample(EvolutionState state) {
		state.output.message("Selecting training points...");
		CudaEvolutionState cuState = (CudaEvolutionState) state;
		
		if (cuState.getTrainingMode() == CudaEvolutionState.TRAINING_MODE_GT) {
			gtSample(cuState);			
		}
		else if (cuState.getTrainingMode() == CudaEvolutionState.TRAINING_MODE_POS_NEG) {
			posNegSample(cuState);
		}
	}
	
	/**
	 * Helper function to sample the training points using a ground truth image.
	 * 
	 * @param cuState	The EvolutionState object to use
	 */
	private void gtSample(CudaEvolutionState cuState) {
		HashSet<Integer> trainingPoints = new HashSet<Integer>(); // To prevent duplicated training points		
		ArrayList<DataInstance> instances = cuState.trainingInstances;
		ByteImage inputImage = cuState.getTrainingImage();
		CudaData cudaDataInput = cuState.getDevTrainingImage();
		ByteImage gtImage = cuState.getGtImage();
		
		int pos = 0, neg = 0;
		int width = inputImage.getWidth();
		int height = inputImage.getHeight();
		
		while (pos < this.POSITIVE_COUNT || neg < this.NEGATIVE_COUNT) {
			int index = cuState.random[0].nextInt(width * height);
			
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
				
				Float4 input = Float4.getFloat4(cudaDataInput.input, index);
				
				Float4 smallAvg = Float4.getFloat4(cudaDataInput.smallAvg, index);
				Float4 mediumAvg = Float4.getFloat4(cudaDataInput.mediumAvg, index);
				Float4 largeAvg = Float4.getFloat4(cudaDataInput.largeAvg, index);
				
				
				Float4 smallSd = Float4.getFloat4(cudaDataInput.smallSd, index);
				Float4 mediumSd = Float4.getFloat4(cudaDataInput.mediumSd, index);
				Float4 largeSd = Float4.getFloat4(cudaDataInput.largeSd, index);
				
				DataInstance instance = new DataInstance(input,
						smallAvg, mediumAvg, largeAvg,
						smallSd, mediumSd, largeSd,
						label);
				instances.add(instance);
				trainingPoints.add(new Integer(index));
			} // end-if
				
		} // end-while
		
		// Shuffle the samples
		Collections.shuffle(instances);
			
		// Create a CudaData instance to hold the same training points
		// This call will also transfer the training data to GPU
		cuState.devTrainingInstances = new CudaData(instances);
	}
	
	
	/**
	 * Helper function to sample training points from a set of positive examples 
	 * and negative examples.
	 * 
	 * @param cuState	The EvolutionState object to use
	 */
	private void posNegSample(CudaEvolutionState cuState) {
		HashSet<Integer> trainingPoints = new HashSet<Integer>(); // To prevent duplicated training points
		
		ArrayList<DataInstance> instances = cuState.trainingInstances;

		int sampleCount;
		ArrayList<ByteImage> samples;
		ArrayList<CudaData> cuSamples;

		samples = cuState.getPositiveExamples();
		cuSamples = cuState.getDevPositiveExamples();

		// Would like to evenly select positive examples from all existing positive images
		// Therefore, we find the portion for each sample image (using the ceiling value)
		int portion = (POSITIVE_COUNT - 1) / samples.size() + 1;

		for (int i = 0; i < samples.size(); i++) {
			ByteImage image = samples.get(i);
			CudaData cuImage = cuSamples.get(i);
			sampleCount = 0;

			while (sampleCount < portion) { // Should loop until we have sampled the desired number of pixels from this image
				int index = cuState.random[0].nextInt(image.getWidth() * image.getHeight());

				if (trainingPoints.contains(new Integer(index)))
					continue; // Oops! We have already sampled this pixel
				
				// This pixel should not have alpha == 0. If it has, it means it is background!
				if (image.getColor(index).getAlpha() == 0)
					continue;

				trainingPoints.add(new Integer(index));

				Float4 input = Float4.getFloat4(cuImage.input, index);

				Float4 smallAvg = Float4.getFloat4(cuImage.smallAvg, index);
				Float4 mediumAvg = Float4.getFloat4(cuImage.mediumAvg, index);
				Float4 largeAvg = Float4.getFloat4(cuImage.largeAvg, index);

				Float4 smallSd = Float4.getFloat4(cuImage.smallSd, index);
				Float4 mediumSd = Float4.getFloat4(cuImage.mediumSd, index);
				Float4 largeSd = Float4.getFloat4(cuImage.largeSd, index);

				DataInstance instance = new DataInstance(input, smallAvg, mediumAvg, largeAvg, smallSd, mediumSd, largeSd, 1);
				instances.add(instance);
				sampleCount++;
			}
		}
		
		// FIXME could adjust the number of POSITIVE_COUNT here, but who cares...? ;)
		sampleCount = 0;
		trainingPoints.clear();
		
		samples = cuState.getNegativeExamples();
		cuSamples = cuState.getDevNegativeExamples();
		
		portion = (NEGATIVE_COUNT - 1) / samples.size() + 1;

		for (int i = 0; i < samples.size(); i++) {
			ByteImage image = samples.get(i);
			CudaData cuImage = cuSamples.get(i);
			sampleCount = 0;

			while (sampleCount < portion) { // Should loop until we have sampled the desired number of pixels from this image
				int index = cuState.random[0].nextInt(image.getWidth() * image.getHeight());

				if (trainingPoints.contains(new Integer(index)))
					continue; // Oops! We have already sampled this pixel
				
				if (image.getColor(index).getAlpha() == 0)
					continue;

				trainingPoints.add(new Integer(index));

				Float4 input = Float4.getFloat4(cuImage.input, index);

				Float4 smallAvg = Float4.getFloat4(cuImage.smallAvg, index);
				Float4 mediumAvg = Float4.getFloat4(cuImage.mediumAvg, index);
				Float4 largeAvg = Float4.getFloat4(cuImage.largeAvg, index);

				Float4 smallSd = Float4.getFloat4(cuImage.smallSd, index);
				Float4 mediumSd = Float4.getFloat4(cuImage.mediumSd, index);
				Float4 largeSd = Float4.getFloat4(cuImage.largeSd, index);

				DataInstance instance = new DataInstance(input, smallAvg, mediumAvg, largeAvg, smallSd, mediumSd, largeSd, 2);
				instances.add(instance);
				sampleCount++;
			}
		}
		
		// Shuffle the samples
		Collections.shuffle(instances);
		
		// Create a CudaData instance to hold the same training points
		// This call will also transfer the training data to GPU
		cuState.devTrainingInstances = new CudaData(instances);		 
	}

	public CudaInterop(boolean visualize) {
		this.visualize = visualize;
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
		fncEvaluate = new CUfunction();
		fncDescribe = new CUfunction();
		fncFilter = new CUfunction();

		File kernelCodeFile = new File("bin/cuda/kernels/gp/cuda-kernels.cu");

		if (recompile || !kernelCodeFile.exists()) {
			// Read the template code and insert the actions for the functions
			kernelTemplate = StringUtils.TextToString("bin/cuda/kernels/gp/evaluator-template.cu").replace("/*@@actions@@*/", kernelCode);
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

		// Compile or load the kernel
		kernel = KernelLauncher.create("bin/cuda/kernels/gp/cuda-kernels.cu", "evaluate", recompile, "-arch=compute_20 -code=sm_30 -use_fast_math");

		System.out.println(recompile ? "Kernel COMPILED" : "Kernel loaded");

		// Save the current context. I want to use this context later for
		// multithreaded
		// operations
		this.context = new CUcontext();
		cuCtxGetCurrent(context);

		// Get the module so that I can get handles to kernel functions
		this.codeModule = kernel.getModule();
		cuModuleGetFunction(fncEvaluate, codeModule, "evaluate");
		cuModuleGetFunction(fncDescribe, codeModule, "describe");
		cuModuleGetFunction(fncFilter, codeModule, "avgSdFilter");
	}

	/**
	 * Sets the CPU data (for CPU evaluation purposes) and the visualizer data.
	 * Note that if the use of visualizer was indicated in the setup file, this
	 * method will initialize the visualizer and block the calling thread untill
	 * the visualizer instance is fully loaded and ready.
	 * 
	 * 
	 * @param gpData
	 *            The CPU-evaluation data
	 * @param describeData
	 *            The visualizer data
	 */
	public void setDescribeData(CudaEvolutionState state, ProblemData gpData, CudaData describeData) {
		this.gpData = gpData;

		if (visualize)
			initVisualizer(state, describeData);
	}
	
	/**
	 * This method will switch the context to the calling thread so that another host
	 * thread can access the CUDA functionality offered by this class.
	 */
	public synchronized void switchContext() {
		cuCtxSetCurrent(context);
	}

	/**
	 * Initializes the visualizer using the provided data. This will block the
	 * calling thread until the visualizer instance is ready and fully loaded.
	 * 
	 * @param testingData
	 *            The data that should be visualized
	 */
	private void initVisualizer(CudaEvolutionState state, CudaData testingData) {
		// Create a proper title for the window
		String title = "CUDA/OpenGL Interop -- Job #" + state.job[0].toString();
		throw new RuntimeException("OOps! Commented out!");
//		visualizer = new Visualizer(testingData.lightClone(), title);
//		visualizer.waitReady();
	}

	/**
	 * Performs all filters on the input image and stores the results in an
	 * instance of CudaData that was provided to this function
	 */
	public void fillAndPerformFilters(ByteImage inputImage, CudaData input) {

		PreciseTimer timer = new PreciseTimer();
		timer.start();

		byte[] byteData = inputImage.getByteData();
		int imageWidth = inputImage.getWidth();
		int imageHeight = inputImage.getHeight();
		input.imageHeight = imageHeight;
		input.imageWidth = imageWidth;

		// Fill the input values
		input.input = inputImage.getFloatData();
		input.dev_input = allocTransFloat(input.input);

		System.out.print("Performing small filters...");
		FilterResult result = performFilter(byteData, imageWidth, imageHeight, smallFilterSize);
		input.smallAvg = result.averageResult;

		input.smallSd = result.sdResult;
		input.dev_smallAvg = result.averagePointer;
		input.dev_smallSd = result.sdPointer;
		System.out.println("\tDone!");

		System.out.print("Performing medium filters...");
		result = performFilter(byteData, imageWidth, imageHeight, mediumFilterSize);
		input.mediumAvg = result.averageResult;
		input.mediumSd = result.sdResult;
		input.dev_mediumAvg = result.averagePointer;
		input.dev_mediumSd = result.sdPointer;
		System.out.println("\tDone!");

		System.out.print("Performing large filters...");
		result = performFilter(byteData, imageWidth, imageHeight, largeFilterSize);
		input.largeAvg = result.averageResult;
		input.largeSd = result.sdResult;
		input.dev_largeAvg = result.averagePointer;
		input.dev_largeSd = result.sdPointer;
		System.out.println("\tDone!");

		timer.stopAndLog("Filtering");
	}

	/**
	 * Performs the average and standard deviation filters on the provided input
	 * and returns the results
	 * 
	 * @param input
	 * @param maskSize
	 * @return
	 */
	private FilterResult performFilter(byte[] input, int imageWidth, int imageHeight, int maskSize) {
		// Allocate device array
		CUarray devTexture = new CUarray();
		CUDA_ARRAY_DESCRIPTOR desc = new CUDA_ARRAY_DESCRIPTOR();
		desc.Format = CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8;
		desc.NumChannels = 4;
		desc.Width = imageWidth;
		desc.Height = imageHeight;
		JCudaDriver.cuArrayCreate(devTexture, desc);

		// Copy the host input to the array
		CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
		copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
		copyHD.srcHost = Pointer.to(input);
		copyHD.srcPitch = imageWidth * Sizeof.BYTE * 4;
		copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_ARRAY;
		copyHD.dstArray = devTexture;
		copyHD.WidthInBytes = imageWidth * Sizeof.BYTE * 4;
		copyHD.Height = imageHeight;

		cuMemcpy2D(copyHD);

		// Set texture reference properties
		CUtexref inputTexRef = new CUtexref();
		cuModuleGetTexRef(inputTexRef, codeModule, "inputTexture");
		cuTexRefSetFilterMode(inputTexRef, CU_TR_FILTER_MODE_POINT);
		cuTexRefSetAddressMode(inputTexRef, 0, CU_TR_ADDRESS_MODE_CLAMP);
		cuTexRefSetAddressMode(inputTexRef, 1, CU_TR_ADDRESS_MODE_CLAMP);
		cuTexRefSetFlags(inputTexRef, CU_TRSF_READ_AS_INTEGER);
		cuTexRefSetFormat(inputTexRef, CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8, 4);
		cuTexRefSetArray(inputTexRef, devTexture, CU_TRSA_OVERRIDE_FORMAT);

		// Allocate results array
		CUdeviceptr devAverage = new CUdeviceptr();
		cuMemAlloc(devAverage, imageWidth * imageHeight * Sizeof.FLOAT * 4);
		CUdeviceptr devSd = new CUdeviceptr();
		cuMemAlloc(devSd, imageWidth * imageHeight * Sizeof.FLOAT * 4);

		Pointer kernelParams = Pointer.to(Pointer.to(devAverage), Pointer.to(devSd), Pointer.to(new int[] { imageWidth }), Pointer.to(new int[] { imageHeight }), Pointer.to(new int[] { maskSize }));

		// Call kernel
		cuLaunchKernel(fncFilter, (imageWidth + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, (imageHeight + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, 1, 16, 16, 1, 0, null, kernelParams, null);
		cuCtxSynchronize();

		// Retrieve results
		float[] average = new float[imageWidth * imageHeight * 4];
		float[] sd = new float[imageWidth * imageHeight * 4];
		cuMemcpyDtoH(Pointer.to(average), devAverage, imageWidth * imageHeight * Sizeof.FLOAT * 4);
		cuMemcpyDtoH(Pointer.to(sd), devSd, imageWidth * imageHeight * Sizeof.FLOAT * 4);

		// A little housekeeping
		cuArrayDestroy(devTexture);
		return new FilterResult(average, sd, devAverage, devSd);
	}

	/**
	 * Evaluates a population of individuals given as a list of list of byte[].
	 * 
	 * @param expressions
	 * @param data
	 * @param indCount
	 * @return
	 */
	public float[] evaluatePopulation(List<ArrayList<TByteArrayList>> expressions, CudaData data) {
		// Convert expressions to byte[]
		// First determine how many unevals we have in total
		int indCount = 0;
		int maxExpLength = 0;

		for (ArrayList<TByteArrayList> thExps : expressions) {
			indCount += thExps.size();

			// Determine the longest expression
			for (TByteArrayList exp : thExps)
				if (exp.size() > maxExpLength)
					maxExpLength = exp.size();
		}

		byte[] population = new byte[indCount * maxExpLength];

		int i = 0;

		for (ArrayList<TByteArrayList> thExps : expressions) {
			for (TByteArrayList currExp : thExps) {
				int length = currExp.size();
				currExp.toArray(population, 0, i * maxExpLength, length);
				i++;
			}
		}

		return evaluatePopulation(population, indCount, maxExpLength, data);
	}

	/**
	 * A helper function to evaluate individuals that are represented in a
	 * NON-RAGGED array of byte[]. All of the individuals are transferred to
	 * memory in one go.
	 * 
	 * @param expressions
	 * @param indCount
	 * @param maxLength
	 * @param data
	 * @return returns the fitness vector of the evaluated individuals.
	 */
	public float[] evaluatePopulation(byte[] expressions, int indCount, int maxLength, CudaData data) {
		// Allocate expressions memory
		CUdeviceptr devExpressions = allocTransByte(expressions);

		// Create the output fitnesses array
		CUdeviceptr devFitnesses = new CUdeviceptr();
		cuMemAlloc(devFitnesses, indCount * Sizeof.FLOAT);

		kernel.setGridSize(indCount, 1);
		kernel.setBlockSize(EVAL_BLOCK_SIZE, 1, 1);
		kernel.call(devExpressions, indCount, maxLength, data.dev_input, data.dev_smallAvg, data.dev_mediumAvg, data.dev_largeAvg, data.dev_smallSd, data.dev_mediumSd, data.dev_largeSd, data.dev_labels, devFitnesses);

		// grab the fitnesses array
		float[] fitnesses = new float[indCount];
		cuMemcpyDtoH(Pointer.to(fitnesses), devFitnesses, indCount * Sizeof.FLOAT);
		cuMemFree(devFitnesses);
		cuMemFree(devExpressions);

		return fitnesses;
	}

	public ByteImage describeIndividual(byte[] expression, CudaData data) {
		CUdeviceptr devExpression = allocTransByte(expression);
		data.initOutput();
		Pointer kernelParams = Pointer.to(Pointer.to(devExpression), Pointer.to(data.dev_input), Pointer.to(data.dev_output), Pointer.to(data.dev_smallAvg), Pointer.to(data.dev_mediumAvg), Pointer.to(data.dev_largeAvg), Pointer.to(data.dev_smallSd), Pointer.to(data.dev_mediumSd), Pointer.to(data.dev_largeSd), Pointer.to(new int[] { data.imageWidth }), Pointer.to(new int[] { data.imageHeight }));

		int gridSize = (int) Math.ceil(data.imageWidth * data.imageHeight / (double) DESC_BLOCK_SIZE);

		cuLaunchKernel(fncDescribe, gridSize, 1, 1, DESC_BLOCK_SIZE, 1, 1, 0, null, kernelParams, null);
		cuCtxSynchronize();

		// Copy the output image
		byte[] byteData = new byte[data.imageWidth * data.imageHeight * 4];
		cuMemcpyDtoH(Pointer.to(byteData), data.dev_output, data.imageWidth * data.imageHeight * 4 * Sizeof.BYTE);
		cuMemFree(devExpression);

		return new ByteImage(byteData, data.imageWidth, data.imageHeight);
	}

	/**
	 * Evaluates an individual on the CPU. Use it for debugging purposes only
	 * :-)
	 * 
	 * @param state
	 * @param ind
	 * @param trainingInstances
	 * @return
	 */
	public synchronized float cpuEvaluate(final EvolutionState state, final Individual ind, ArrayList<DataInstance> trainingInstances) {
		int tp = 0, tn = 0;
		int fp = 0, fn = 0;

		for (DataInstance instance : trainingInstances) {
			// set the instance as the input
			gpData.instance = instance;

			((GPIndividual) ind).trees[0].child.eval(state, 0, gpData, null, ((GPIndividual) ind), null);

			boolean obtained = gpData.value > 0;
			boolean expected = instance.label == 1;

			if (obtained && expected)
				tp++;
			else if (!obtained && !expected)
				tn++;
			else if (obtained && !expected)
				fp++;
			else
				fn++;
		}

		float fitness = ((float) (tp + tn)) / (ProblemData.positiveExamples + ProblemData.negativeExamples);
		return fitness;
	}

	// /**
	// * Destroys the CUDA context that was created when this instance of the
	// * class compiled (loaded) the kernel.
	// *
	// */
	// public void destroy() {
	// CUcontext currentCtx = new CUcontext();
	// cuCtxGetCurrent(currentCtx);
	// cuCtxDestroy(currentCtx);
	// }

	/**
	 * Allocates and transfers a float array to the CUDA memory.
	 * 
	 * @param array
	 *            The input float array
	 * @return A device pointer to the allocated array
	 */
	public static CUdeviceptr allocTransFloat(float[] array) {
		CUdeviceptr devPointer = new CUdeviceptr();
		cuMemAlloc(devPointer, array.length * Sizeof.FLOAT);
		cuMemcpyHtoD(devPointer, Pointer.to(array), array.length * Sizeof.FLOAT);

		return devPointer;
	}

	/**
	 * Allocates and transfers an integer array to the CUDA memory.
	 * 
	 * @param array
	 *            The input integer array
	 * @return A device pointer to the allocated array
	 */
	public static CUdeviceptr allocTransInt(int[] array) {
		CUdeviceptr devPointer = new CUdeviceptr();
		cuMemAlloc(devPointer, array.length * Sizeof.INT);
		cuMemcpyHtoD(devPointer, Pointer.to(array), array.length * Sizeof.INT);

		return devPointer;
	}

	/**
	 * Allocates and transfers a byte array to the CUDA memory.
	 * 
	 * @param array
	 *            The input byte array
	 * @return A device pointer to the allocated array
	 */
	public static CUdeviceptr allocTransByte(byte[] array) {
		CUdeviceptr devPointer = new CUdeviceptr();
		int x = cuMemAlloc(devPointer, array.length * Sizeof.BYTE);
		x = cuMemcpyHtoD(devPointer, Pointer.to(array), array.length * Sizeof.BYTE);

		return devPointer;
	}

	/**
	 * Allocates a single double pointer on the device memory with the given
	 * initial value
	 * 
	 * @param initialValue
	 *            The initial value for the allocated double
	 * @return The pointer to the allocated memory.
	 */
	public static CUdeviceptr allocateDouble(double initialValue) {
		CUdeviceptr devPointer = new CUdeviceptr();
		cuMemAlloc(devPointer, Sizeof.DOUBLE);
		cuMemcpyHtoD(devPointer, Pointer.to(new double[] { initialValue }), Sizeof.DOUBLE);

		return devPointer;
	}

	class FilterResult {
		float[] averageResult;
		float[] sdResult;
		CUdeviceptr averagePointer;
		CUdeviceptr sdPointer;

		public FilterResult(float[] averageResult, float[] sdResult, CUdeviceptr averagePoiner, CUdeviceptr sdPointer) {
			this.averageResult = averageResult;
			this.sdResult = sdResult;
			this.averagePointer = averagePoiner;
			this.sdPointer = sdPointer;
		}
	}
}
