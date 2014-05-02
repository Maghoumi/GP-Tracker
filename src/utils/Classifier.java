package utils;

import java.awt.Color;
import java.util.ArrayList;
import java.util.List;

import cuda.gp.CudaNode;
import ec.gp.GPIndividual;

/**
 * Represents a single GP classifier program. The objects of this class
 * are run on the input data using CUDA and are used for object tracking.
 * Each classifier is assigned a color at the moment of instantiation which
 * remains constant during the run. The color is like a "primary key" for the
 * objects of this class. Therefore, the Comparable interface has been implemented
 * for the objects of this class so that classifiers can be compared against
 * one another.
 * Furthermore, each classifier can be disabled. If disabled, that classifier is not
 * executed on the final image.
 * 
 * @author Mehran Maghoumi
 */
public class Classifier implements Comparable<Classifier>{
	
	/** Constant field for the training method that uses separate positive/negative examples */
	public final static int TYPE_POS_NEG = 0;
	
	/** Constant field for the training method that uses a training image and a ground truth */
	public final static int TYPE_GT = 1;
	
	private static ColorUtils colorList = new ColorUtils();
	
	/** The GP expression tree of the individual. */
	private byte[] expression;
	
	/** The unique color of the individual that distinguishes this individual
	 * from other individuals.
	 */
	private Color color;
	
	/** The GP individual of this classifier. The individual is preserved for population seeding purposes */
	private GPIndividual individual;
	
	/** A flag indicating whether this classifier is enabled or not */
	private boolean enabled = true;
	
	/** Used in the toString method */
	private String colorName;
	
	/** A flag indicating whether this classifier should be used to seed the initial population */
	private boolean shouldSeed = false;
	
	/** The list of all the segments that this classifier has claimed */
	private List<Segment> claimedSegments = new ArrayList<Segment>();
	
	/** The list of the positive examples that are used to train this classifier */
	private List<ByteImage> positiveExamples = null;
	
	/** The list of negative examples that are used to train this classifier */
	private List<ByteImage> negativeExamples = null;
	
	/** The training image used to train this classifier */
	private ByteImage trainingImage = null;
	
	/** The ground truth used to train this classifier */
	private ByteImage gtImage = null;
	
	/** The training type for this classifier: pos/neg or image/gt? @see Job.java */
	private int trainingType;
	
	
	/**
	 * Initializes a classifier object with a random color and the specified positive and negative
	 * training instances. The object maintains its own list of positive/negative examples, thus the
	 * original lists are left intact.
	 */
	public Classifier(List<ByteImage> positiveExamples, List<ByteImage> negativeExamples) {
		this();
		this.trainingType = Classifier.TYPE_POS_NEG;
		
		this.positiveExamples = new ArrayList<ByteImage>();		
		for(ByteImage image : positiveExamples) {
			this.positiveExamples.add(image.clone());
		}
		
		this.negativeExamples = new ArrayList<ByteImage>();		
		for(ByteImage image : negativeExamples) {
			this.negativeExamples.add(image.clone());
		}
		
	}
	
	/**
	 * Initializes a classifier object with a random color and the specified training image 
	 * and the ground truth. The object maintains its own instance of the images, thus the
	 * original lists are left intact.
	 */
	public Classifier(ByteImage trainingImage, ByteImage gtImage) {
		this();
		this.trainingType = Classifier.TYPE_GT;
		this.trainingImage = trainingImage.clone();
		this.gtImage = gtImage.clone();
	}
	
	/**
	 * Initializes a classifier object with the given training type
	 * @param trainingMode
	 */
	public Classifier(int trainingType) {
		this();
		
		this.trainingType = trainingType;
		
		switch (this.trainingType) {
		case Classifier.TYPE_POS_NEG:
			this.positiveExamples = new ArrayList<ByteImage>();
			this.negativeExamples = new ArrayList<ByteImage>();
			break;

		case Classifier.TYPE_GT: 
			// Nothing needs to be done
			break;
			
		default:
			throw new RuntimeException("Unsupported training type");
		}
	}
	
	/**
	 * Chooses a color for this classifier instance.
	 */
	private Classifier() {
		// Select a random color for the classifier
		this.color = colorList.seizeColor();
		this.colorName = colorList.getColorNameFromColor(this.color);
	}
	
	/**
	 * Must be called before this classifier is completely removed form the system.
	 * This will release the color of this classifier and makes it available to other
	 * classifiers.
	 */
	public void destroy() {
		colorList.releaseColor(color);
	}
	
	/**
	 * @return	The list of positive examples
	 */
	public List<ByteImage> getPositiveExamples() {
		return this.positiveExamples;
	}
	
	/**
	 * @return The list of negative examples
	 */
	public List<ByteImage> getNegativeExamples() {
		return this.negativeExamples;
	}
	
	/**
	 * Adds a positive example to the list of positive examples of this classifier.
	 * This method, clones the provided image.
	 * @param example
	 */
	public void addPositiveExample(ByteImage example) {
		if (this.trainingType != Classifier.TYPE_POS_NEG)
			throw new RuntimeException("The training type of this instance does not allow this operation!");
		
		this.positiveExamples.add(example.clone());
	}
	
	/**
	 * Adds a negative example to the list of negative examples of this classifier.
	 * This method, clones the provided image.
	 * @param example
	 */
	public void addNegativeExample(ByteImage example) {
		if (this.trainingType != Classifier.TYPE_POS_NEG)
			throw new RuntimeException("The training type of this instance does not allow this operation!");
		
		this.negativeExamples.add(example.clone());
	}
	
	/**
	 * Set the training image of this classifier.
	 * This method clones the provided image.
	 * @param image
	 */
	public void setTrainingImage(ByteImage image) {
		if (this.trainingType != Classifier.TYPE_GT)
			throw new RuntimeException("The training type of this instance does not allow this operation!");
		
		this.trainingImage = image.clone();
	}
	
	/**
	 * @return	The training image
	 */
	public ByteImage getTrainingImage() {
		return this.trainingImage;
	}
	
	/**
	 * Set the ground truth of this classifier.
	 * This method clones the provided image.
	 * @param image
	 */
	public void setGtImage(ByteImage image) {
		if (this.trainingType != Classifier.TYPE_GT)
			throw new RuntimeException("The training type of this instance does not allow this operation!");
		
		this.gtImage = image.clone();
	}
	
	/**
	 * @return	The ground truth
	 */
	public ByteImage getGtImage() {
		return this.gtImage;
	}
	
	/**
	 * Set the GP individual of this individual. This method was declared synchronized as there
	 * was the possibility of the best individual being set by multiple classifiers.
	 * @param individual
	 */
	public synchronized void setIndividual(GPIndividual individual) {
		this.individual = individual;
		// Update the expression tree
		this.expression = ((CudaNode)individual.trees[0].child).makePostfixExpression();
	}
	
	/** 
	 * @return	Returns the GP Individual of this classifier
	 */
	public GPIndividual getIndividual() {
		return this.individual;
	}
	
	/**
	 * Returns the GP-expression of this individual.
	 * @return
	 */
	public byte[] getExpression() {
		return this.expression;
	}
	
	/**
	 * @return	Returns the unique classification color of this classifier
	 */
	public Color getColor() {
		return this.color;
	}
	
	/**
	 * @return Check whether the classifier is enabled or not.
	 */
	public boolean isEnabled() {
		return this.enabled;
	}
	
	/**
	 * Disable or enable this classifier
	 * @param enabled
	 */
	public void setEnabled(boolean enabled) {
		if (this.enabled != enabled) {
			this.enabled = enabled;
		}
	}
	
	/**
	 * Indicate whether this classifier should be used to seed the initial population.
	 * Sometimes when we need to improve a classifier, it is better to use that individual
	 * to seed the initial population so that the GP system can evolve classifiers that are
	 * similar to this classifier.
	 * 
	 * @param shouldSeed
	 */
	public void setShouldSeed(boolean shouldSeed) {
		this.shouldSeed = shouldSeed;
	}
	
	/**
	 * @return	Whether this classifier should be used to seed the initial population 
	 */
	public boolean shouldSeed() {
		return this.shouldSeed;
	}
	
	/**
	 * @return	The training mode of this classifier (pos/neg vs. image/gt)
	 */
	public int getTrainingType() {
		return this.trainingType;
	}
	
	/**
	 * Adds a segment as a claimed segment by this classifier
	 * Note: a claim is when a classifier "claims" a texture
	 */
	public void addClaim(Segment segment) {
		this.claimedSegments.add(segment);
	}
	
	/**
	 * Reset the claims this classifier has over the segments
	 * Note: a claim is when a classifier "claims" a texture
	 */
	public void resetClaims() {
		this.claimedSegments.clear();
	}
	
	/**
	 * @return	The number of claims that this classifier has
	 */
	public int getClaimsCount() {
		return this.claimedSegments.size();
	}
	
	/**
	 * @return	The list of claimed segments by this classifier
	 */
	public List<Segment> getClaims() {
		return this.claimedSegments;
	}
	
	/**
	 * Determines if this classifier has claimed the specified segment 
	 * @param segment
	 * @return
	 */
	public boolean hasClaimed(Segment segment) {
		return this.claimedSegments.contains(segment);
	}
	
	/**
	 * Two classifiers are the same (same, meaning they are meant for the 
	 * same texture) if their colors are equal.
	 */
	@Override
	public boolean equals(Object obj) {
		Classifier other = (Classifier) obj;
		
		return this.color.equals(other.color);
	}
	
	@Override
	public int hashCode() {
		return this.color.hashCode();
	}
	
//	/**
//	 * A utility class for assigning a random color to the classifier. 
//	 */
//	static class ColorSelector {
//		/** Index of the current predifined color that has not been used yet */
//		private static int index = 0;
//		
//		/**
//		 * Returns the next available predefined color in the list.
//		 * This method will throw an exception if no available
//		 * color exists.
//		 */
//		public static Color getNextColor() {
//			if (index >= colorList.getColorList().size())
//				throw new RuntimeException("Out of colors for the classifiers");
//			
//			return colorList.getColorList().get(index++).getColor();
//		}
//	}

	@Override
	public int compareTo(Classifier o) {
		return colorList.compare(this.color, o.color);
	}
	
	@Override
	public String toString() {
		return "Classifier: " + colorName;
	}
}