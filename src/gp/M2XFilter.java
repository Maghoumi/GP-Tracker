package gp;

import static jcuda.driver.JCudaDriver.*;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import javax.imageio.ImageIO;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

import org.apache.commons.io.IOUtils;

import cuda.gp.CudaEvolutionState;
import cuda.gp.CudaInterop;
import cuda.gp.CudaNode;
import utils.ByteImage;
import utils.PreciseTimer;
import utils.cuda.ImageFilters;
import ec.EvolutionState;
import ec.Evolve;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.gp.GPProblem;
import ec.simple.SimpleFitness;
import ec.simple.SimpleProblemForm;
import ec.util.Parameter;
import gp.datatypes.ProblemData;

/**
 * @deprecated This class is no longer used. Only kept for a rainy day!
 * @author Mehran Maghoumi
 * 
 */
public class M2XFilter extends GPProblem implements SimpleProblemForm
{
	public ProblemData input;

	@Override
	public Object clone()
	{
		M2XFilter newobj = (M2XFilter) (super.clone());
		newobj.input = (ProblemData) (input.clone());
		return newobj;
	}

	@Override
	public void setup(final EvolutionState state, final Parameter base)
	{
		// very important, remember this
		super.setup(state, base);

		// set up our input -- don't want to use the default base, it's unsafe here
		input = (ProblemData) state.parameters.getInstanceForParameterEq(base.push(P_DATA), null, ProblemData.class);
		input.setup(state, base.push(P_DATA));
	}

	private void initCuda(int maxGridSize) {
		// setup the CUDA caller for this problem
		//		cuState.cudaInterop.setup(ProblemData.smallWindowSize, ProblemData.mediumWindowSize,
		//				ProblemData.largeWindowSize, ProblemData.positiveExamples, ProblemData.negativeExamples, maxGridSize);
		//		
		//		CudaData inputData = new CudaData();
		//		ProblemData.inputData = inputData;
		//		
		//		// Perform the filters on the original image
		//		cuState.cudaInterop.fillAndPerformFilters(ProblemData.byteInputImage, inputData);
		//		
		//		// Do the stuff for the testing image
		//		CudaData testingData = new CudaData();
		//		cuState.cudaInterop.fillAndPerformFilters(ProblemData.byteTestImage, testingData);
		//		ProblemData.testingData = testingData;

		//		cuState.cudaInterop.setDescribeData(cuState, input, testingData); // for CPU evaluation purposes
	}

//	private void randomSample(EvolutionState state, List<DataInstance> instances) throws IOException {
//		int pos = 0, neg = 0;
//		int width = ProblemData.byteInputImage.getWidth();
//		int height = ProblemData.byteInputImage.getHeight();
//
//		state.output.message("Selecting training points...");
//
//		while (pos < ProblemData.positiveExamples || neg < ProblemData.negativeExamples) {
//			int index = state.random[0].nextInt(width * height);
//
//			if (trainingPoints.contains(new Integer(index)))
//				continue; // This training point was selected before...
//
//			Color c = ProblemData.byteInputGt.getColor(index);
//			boolean haveInstance = false;
//			int label = -1;
//
//			if (c.equals(Color.green) && pos < ProblemData.positiveExamples) { // positive sample
//				pos++;
//				haveInstance = true;
//				label = 1;
//			}
//			else if (neg < ProblemData.negativeExamples) {
//				neg++;
//				haveInstance = true;
//				label = 2;
//			}
//
//			if (haveInstance) {
//
//				Float4 input = Float4.getFloat4(ProblemData.inputData.input, index);
//
//				Float4 smallAvg = Float4.getFloat4(ProblemData.inputData.smallAvg, index);
//				Float4 mediumAvg = Float4.getFloat4(ProblemData.inputData.mediumAvg, index);
//				Float4 largeAvg = Float4.getFloat4(ProblemData.inputData.largeAvg, index);
//
//				Float4 smallSd = Float4.getFloat4(ProblemData.inputData.smallSd, index);
//				Float4 mediumSd = Float4.getFloat4(ProblemData.inputData.mediumSd, index);
//				Float4 largeSd = Float4.getFloat4(ProblemData.inputData.largeSd, index);
//
//				DataInstance instance = new DataInstance(input,
//						smallAvg, mediumAvg, largeAvg,
//						smallSd, mediumSd, largeSd,
//						label);
//				instances.add(instance);
//				trainingPoints.add(new Integer(index));
//			} // end-if
//
//		} // end-while
//
//		// Create a CudaData instance to hold the same training points
//		CudaData trainingData = new CudaData(instances);
//		ProblemData.trainingData = trainingData;
//	}

	@Override
	public void evaluate(final EvolutionState state, final Individual ind, final int subpopulation, final int threadnum)
	{

//		if (!ind.evaluated) // don't bother reevaluating
//		{
//			int tp = 0, tn = 0;
//			int fp = 0, fn = 0;
//
//			for (DataInstance instance : trainingInstances) {
//				// set the instance as the input
//				input.instance = instance;
//
//				((GPIndividual) ind).trees[0].child.eval(state, threadnum, input, stack, ((GPIndividual) ind), this);
//
//				boolean obtained = input.value > 0;
//				boolean expected = instance.label == 1;
//
//				if (obtained && expected)
//					tp++;
//				else if (!obtained && !expected)
//					tn++;
//				else if (obtained && !expected)
//					fp++;
//				else
//					fn++;
//			}
//
//			//			final double ALPHA =  0.6;
//			//			
//			//			double fitness = fp + fn * (Math.exp(((double)fn / ProblemData.positiveExamples) - ALPHA));
//			//			((KozaFitness)ind.fitness).setStandardizedFitness(state, (float)fitness);
//			//			//((SimpleFitness) ind.fitness).setFitness(state, (float) fitness, fitness == 65456);
//			float fitness = ((float) (tp + tn)) / (ProblemData.positiveExamples + ProblemData.negativeExamples) * 100;
//			((SimpleFitness) ind.fitness).setFitness(state, fitness, fitness == 100);
//			ind.evaluated = true;
//		}
	}

	@Override
	public void describe(EvolutionState state, Individual ind, int subpopulation, int threadnum, int log)
	{
		state.output.message("Beginning the describe phase...");
		//		cudaDescribe(state, ind, subpopulation, threadnum, log);
		//		pixelDescribe(state, ind, subpopulation, threadnum, log);
	}

//	private void cudaDescribe(EvolutionState state, Individual ind, int subpopulation, int threadnum, int log) {
//		byte[] expression = ((CudaNode) (((GPIndividual) ind).trees[0].child)).makePostfixExpression();
//
//		PreciseTimer timer = new PreciseTimer();
//		timer.start();
//		ByteImage by = cuState.getCudaInterop().describeIndividual(expression, ProblemData.testingData);
//		timer.stopAndLog(state.output, "Describing");
//
//		try {
//			dumpFilteredImage(by.getBufferedImage(), ProblemData.testImagePath, ProblemData.ext, "[CU describe]");
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
//	}

	// Runs the best individual on all pixels of the image (this is obviously slow)
	public void pixelDescribe(EvolutionState state, Individual ind, int subpopulation, int threadnum, int log)
	{
		//		state.output.message("Beginning the [pixel] describe phase...");
		//		ByteImage pixelFilteredImage = ProblemData.byteTestImage.clone();
		//		
		//		// For statistics about the testing/training performance
		//		int totalTp = 0, totalTn = 0, totalPixels = 0;
		//		int testTp = 0, testTn = 0, totalTestPixels = 0;
		//		
		//		int windowSize = ProblemData.blockWindowSize;
		//		int imageWidth = ProblemData.byteTestImage.getWidth(), imageHeight = ProblemData.byteTestImage.getHeight(); 
		//		int iUpperBound = windowSize, jUpperBound = windowSize;
		//		int progressTotal = imageWidth * imageHeight, progressCurrent = 0, lastProgress = 0, currentProgress = 0;
		//		String progress = "";
		//		
		//		for (int index = 0 ; index < ProblemData.byteTestImage.getSize(); index++) {
		//			Color gtColor = ProblemData.byteTestGt.getColor(index);
		//			boolean expected = gtColor.equals(Color.green);
		//			
		//			Float3 inputColor = Float3.getFloat3(ProblemData.testingData.input, index);
		//			
		//			Float3 smallAvg = Float3.getFloat3(ProblemData.testingData.smallAvg, index);
		//			Float3 mediumAvg = Float3.getFloat3(ProblemData.testingData.mediumAvg, index);
		//			Float3 largeAvg = Float3.getFloat3(ProblemData.testingData.largeAvg, index);
		//			
		//			
		//			Float3 smallSd = Float3.getFloat3(ProblemData.testingData.smallSd, index);
		//			Float3 mediumSd = Float3.getFloat3(ProblemData.testingData.mediumSd, index);
		//			Float3 largeSd = Float3.getFloat3(ProblemData.testingData.largeSd, index);
		//			
		//			DataInstance instance = new DataInstance(inputColor,
		//					smallAvg, mediumAvg, largeAvg,
		//					smallSd, mediumSd, largeSd,
		//					0);
		//			
		//			input.instance = instance;
		//			
		//			((GPIndividual) ind).trees[0].child.eval(state, threadnum, input, stack, ((GPIndividual) ind), this);
		//			
		//			
		//			boolean obtained = input.value > 0;
		//			if (obtained) {
		//				pixelFilteredImage.setColor(index, Color.green);
		//			}
		//			
		//		}
		//		
		//		// write output to file
		//		try {
		//			dumpFilteredImage(pixelFilteredImage.toBufferedImage(), ProblemData.testImagePath, ProblemData.ext, "");
		//		} catch (IOException e) {
		//			// TODO Auto-generated catch block
		//			e.printStackTrace();
		//		}
	}

	private void dumpRunStats(EvolutionState state, String label, int totalTp, int totalTn, int totalPixels, int testTp, int testTn, int testTotalPixels, String path) throws IOException {
		File f = new File(path + "/run" + label + ".csv");
		StringBuilder sb = null;
		int jobNumber = ((Integer) (state.job[0])).intValue() + 1; // Get the current job number

		if (f.exists()) { // Should open the existing file and start appending to the end
			FileInputStream fs = new FileInputStream(f);
			sb = new StringBuilder(IOUtils.toString(fs));
			fs.close();
		}
		else { // Should create a new file and start from scratch
			f.createNewFile();
			sb = new StringBuilder("run,training-tp,training-tn,test-tp,test-tn,tot-tp,tot-tn");
		}

		System.out.println("\nDumping this run's stats...");

		//File file = new File(path + " " + date + ".txt");
		sb.append(System.lineSeparator() + String.format("%d,%d,%d,%d,%d,%d,%d", jobNumber, totalTp - testTp, totalTn - testTn, testTp, testTn, totalTp, totalTn));
		FileOutputStream fo = new FileOutputStream(f);
		IOUtils.write(sb.toString(), fo);
		fo.close();
	}

	private void dumpFilteredImage(BufferedImage image, String path, String ext, String label) throws IOException {
		SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HHmmss");
		Date d = new Date(System.currentTimeMillis());
		String date = df.format(d) + (label.equals("") || label == null ? "" : "--" + label);
		System.out.println("\nWriting file : " + path + " " + date + ext);
		ImageIO.write(image, "png", new File(path + " " + date + ext));
	}

	public static void main(String[] args) {

		System.out.println("Beginning GP");

		long startTime = System.nanoTime();
		if (args.length == 0)
			Evolve.main(new String[] { "-file", "bin/m2xfilter/m2xfilter.params" });
		else
			Evolve.main(args);
		long endTime = System.nanoTime();

		System.out.println();
		System.out.println("=======================================================");
		System.out.printf("Accumulative time: %5.3fs\n", (endTime - startTime) / 1e9);
	}
}