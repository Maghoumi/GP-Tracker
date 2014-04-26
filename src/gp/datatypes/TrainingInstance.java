package gp.datatypes;

/**
 * Represents a single training instance that is used to train the GP system.
 * Objects of this class represent the feature vectors of the GP system. Each
 * instance has some features as well as an expected label for that instance.
 * A list of TrainingInstance objects defines all training examples in the system. 
 * 
 * NOTE: The training instance doesn't know nor it cares how the pixels are internally
 * represented (eg. ABGR etc.). The order of the storage depends on whatever has been
 * passed to the class as the data source.
 * 
 * @author Mehran Maghoumi
 *
 */
public class TrainingInstance {
	/** Constant for the target of input field */
	public static final int TARGET_INPUT = 0;
	
	/** Constant for the target of smallAvg field */
	public static final int TARGET_SMALL_AVG = 1;
	
	/** Constant for the target of mediumAvg field */
	public static final int TARGET_MEDIUM_AVG = 2;
	
	/** Constant for the target of largeAvg field */
	public static final int TARGET_LARGE_AVG = 3;
	
	/** Constant for the target of smallSd field */
	public static final int TARGET_SMALL_SD = 4;
	
	/** Constant for the target of mediumSd field */
	public static final int TARGET_MEDIUM_SD = 5;
	
	/** Constant for the target of largeSd field */
	public static final int TARGET_LARGE_SD = 6;
	
	
	/** The number of channels in each pixel */
	protected int numChannels;
	
	/** The RGB value of the pixel */
	protected float[] input;
	
	/** The RGB value of the small average of the pixel */ 
	protected float[] smallAvg;
	
	/** The RGB value of the medium average of the pixel */
	protected float[] mediumAvg;
	
	/** The RGB value of the large average of the pixel */
	protected float[] largeAvg;
	
	/** The RGB value of the small standard deviation of the pixel */
	protected float[] smallSd;
	
	/** The RGB value of the small standard deviation of the pixel */
	protected float[] mediumSd;
	
	/** The RGB value of the small standard deviation of the pixel */
	protected float[] largeSd;
	
	/** The label of this instance */
	protected int label;
	
	/**
	 * Initializes an object of this class using the specified number of channels
	 * per pixel and the specified label
	 * 
	 * @param numChannels
	 * @param label
	 */
	public TrainingInstance(int numChannels, int label) {
		this.numChannels = numChannels;
		this.label = label;
		
		// Initialize instance placeholders
		this.input = new float[numChannels];
		this.smallAvg = new float[numChannels];
		this.mediumAvg = new float[numChannels];
		this.largeAvg = new float[numChannels];
		this.smallSd = new float[numChannels];
		this.mediumSd = new float[numChannels];
		this.largeSd = new float[numChannels];
	}
	
	/**
	 * Initializes an instance of this class and fills the instance using the data of the
	 * provided FilteredImage object at the specified offset 
	 * 
	 * @param image
	 * @param offset
	 * @param label
	 */
	public TrainingInstance(FilteredImage image, int offset, int label) {
		this(image.getNumChannels(), label);
		
		fillTarget(TrainingInstance.TARGET_INPUT, image.getInput().getUnclonedArray(), offset);
		
		fillTarget(TrainingInstance.TARGET_SMALL_AVG, image.getSmallAvg().getUnclonedArray(), offset);
		fillTarget(TrainingInstance.TARGET_MEDIUM_AVG, image.getMediumAvg().getUnclonedArray(), offset);
		fillTarget(TrainingInstance.TARGET_LARGE_AVG, image.getLargeAvg().getUnclonedArray(), offset);
		
		fillTarget(TrainingInstance.TARGET_SMALL_SD, image.getSmallSd().getUnclonedArray(), offset);
		fillTarget(TrainingInstance.TARGET_MEDIUM_SD, image.getMediumSd().getUnclonedArray(), offset);
		fillTarget(TrainingInstance.TARGET_LARGE_SD, image.getLargeSd().getUnclonedArray(), offset);
	}
	
	/**
	 * Copies the value of the channels from the source to the specified target field
	 * 
	 * @param target	A constant signifying the target of the copy
	 * @param source	The source of pixel values
	 * @param offset	The offset to begin in the source
	 */
	public void fillTarget(int target, float[] source, int offset) {
		float[] targetField = null;
		
		// Select the target
		switch (target) {
		case TARGET_INPUT:
			targetField = input;
			break;
			
		case TARGET_SMALL_AVG:
			targetField = smallAvg;
			break;
			
		case TARGET_MEDIUM_AVG:
			targetField = mediumAvg;
			break;
			
		case TARGET_LARGE_AVG:
			targetField = largeAvg;
			break;
			
		case TARGET_SMALL_SD:
			targetField = smallSd;
			break;
			
		case TARGET_MEDIUM_SD:
			targetField = mediumSd;
			break;
			
		case TARGET_LARGE_SD:
			targetField = largeSd;
			break;

		default:
			throw new RuntimeException("Unknown target specified");
		}
		
		// Copy pixel values from source to the target
		System.arraycopy(source, offset, targetField, 0, numChannels);		
	}
	
}
