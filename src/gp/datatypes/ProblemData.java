package gp.datatypes;


import ec.gp.GPData;

public class ProblemData extends GPData
{
	public static final String DESCRIBE_PIXEL = "pixel";
	public static final String DESCRIBE_BLOCK = "block";
	
	public float value; // node value
	
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
}