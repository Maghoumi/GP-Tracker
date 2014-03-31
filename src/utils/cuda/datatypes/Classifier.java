package utils.cuda.datatypes;

import java.awt.Color;
import java.lang.reflect.Field;
import java.util.ArrayList;

import m2xfilter.GPSystem;
import m2xfilter.datatypes.Job;
import cuda.gp.CudaNode;
import ec.gp.GPIndividual;
import ec.gp.GPNode;

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
	
	/** The ClassifierSet that this classifier belongs to. Used for notifying of changes in disability of this classifier */
	private ClassifierSet owner = null;
	
	/** Used in the toString method */
	private String colorName;
	
	/** A flag indicating whether this classifier should be used to seed the initial population */
	private boolean shouldSeed = false;
	
	/** The list of the positive examples that are used to train this classifier */
	private ArrayList<ByteImage> positiveExamples = null;
	
	/** The list of negative examples that are used to train this classifier */
	private ArrayList<ByteImage> negativeExamples = null;
	
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
	public Classifier(ArrayList<ByteImage> positiveExamples, ArrayList<ByteImage> negativeExamples) {
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
		this.color = ColorSelector.getNextColor();
		this.colorName = colorList.getColorNameFromColor(this.color);
	}
	
	/**
	 * @return	The list of positive examples
	 */
	public ArrayList<ByteImage> getPositiveExamples() {
		return this.positiveExamples;
	}
	
	/**
	 * @return The list of negative examples
	 */
	public ArrayList<ByteImage> getNegativeExamples() {
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
		this.expression = ((CudaNode)individual.trees[0].child).byteTraverse();
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
			owner.notifyEnabilityChanged(this);
		}
	}
	
	/**
	 * Sets the owner of this classifier
	 * @param owner
	 */
	public void setOwner(ClassifierSet owner) {
		this.owner = owner;
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
	 * Retrains this classifier
	 * 
	 * @param system	The GPSystem object to use for retraining this classifier 
	 * @param shouldSeed	Should the existing GPTree be used as the initial population seed?  
	 */
	public void retrain(GPSystem system, boolean shouldSeed) {
		this.shouldSeed = shouldSeed;
		system.queueJob(new Job(this));
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
	
	/**
	 * A utility class for assigning a random color to the classifier. 
	 */
	static class ColorSelector {
		/** A list of colors to use. This list is predefined. */
		private static Color[] colors = new Color[] {
			Color.GREEN,
			Color.ORANGE,
			Color.BLUE,
			Color.YELLOW,
			Color.CYAN,
			Color.MAGENTA,
			Color.PINK,
			Color.RED
		};
		
		/** Index of the current predifined color that has not been used yet */
		private static int index = 0;
		
		/**
		 * Returns the next available predefined color in the list.
		 * This method will throw an exception if no available
		 * color exists.
		 */
		public static Color getNextColor() {
			if (index >=colors.length)
				throw new RuntimeException("Out of colors for the classifiers.");
			
			return colors[index++];
		}
		
		public static int compareTo(Color c1, Color c2) {
			int i = 0, j = 0;
			for (; i < colors.length ; i++)
				if (c1.equals(colors[i]))
					break;
			
			for (; j < colors.length ; j++)
				if (c2.equals(colors[j]))
					break;
			if (i < j)
				return -1;
			else if (i > j)
				return 1;
			
			return 0;
			
		}
	}

	@Override
	public int compareTo(Classifier o) {
		return ColorSelector.compareTo(this.color, o.getColor());
	}
	
	@Override
	public String toString() {
		return "Classifier: " + colorName;
	}
}