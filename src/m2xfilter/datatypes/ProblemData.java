package m2xfilter.datatypes;

import java.awt.Color;

import cuda.CudaInterop;
import ec.gp.GPData;
import ec.util.MersenneTwisterFast;
import utils.cuda.datatypes.*;

public class ProblemData extends GPData
{
	public static final String DESCRIBE_PIXEL = "pixel";
	public static final String DESCRIBE_BLOCK = "block";
	
	public float value; // node value
	public DataInstance instance; // the test/training data associated with this ProblemData
	
	// these data are shared among all instances of ProblemData, therefore they are declared as static and they don't need to be cloned
	public static String path = null;
	public static String ext = null; // file extension
	public static String positiveImagePath = null;
	public static String groundTruthPath = null;
	public static String testImagePath = null;
	public static String testGtPath = null;
	
	public static int smallWindowSize;	// small window size for mask filters
	public static int mediumWindowSize;	// medium window size for mask filters
	public static int largeWindowSize;	// large window size for mask filters
	
	public static int blockWindowSize = 32;	// thresholding window size to use when calling "describe"
	
	public static int positiveExamples; // number of positive examples
	public static int negativeExamples; // number of negative examples
	
	public static ByteImage byteInputImage;
	public static ByteImage byteInputGt;
	public static ByteImage byteTestImage;
	public static ByteImage byteTestGt;
	
	/**
	 * CUDA ZODE 
	 */	
	public static CudaData inputData = null;
	public static CudaData trainingData = null;
	public static CudaData testingData = null;
	
	@Override
	public void copyTo(final GPData gpd) // copy my stuff to another ProblemData
	{
		((ProblemData) gpd).value = value;
		((ProblemData) gpd).instance = (DataInstance) instance.clone();
	}
}