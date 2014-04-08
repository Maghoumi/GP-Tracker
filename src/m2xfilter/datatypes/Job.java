package m2xfilter.datatypes;

import java.util.ArrayList;
import java.util.List;

import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Classifier;

/**
 * Represents a GP classifier evolution job. Objects of this class have
 * the necessary elements that are required by the GP system for the evolution of a
 * classifier. These elements are a set of training examples as well
 * as a classifier container (either a new classifier or a classifier that was
 * evolved before). These jobs will be queued in GPSystem's job queue and will
 * be used during the evolutionary run.
 * 
 * @author Mehran Maghoumi
 *
 */
public class Job {
	/** Constant field for jobs that deal with separate positive/negative examples */
	public final static int TYPE_POS_NEG = 0;
	
	/** Constant field for jobs that deal with a training image and a ground truth */
	public final static int TYPE_GT = 1;
	
	/** The type of this job (either pos/neg or image/gt) */
	private int jobType = -1;
	
	/** The list of positive examples required by this job */
	private List<ByteImage> positiveExamples;
	
	/** The list of negative examples required by this job */
	private List<ByteImage> negativeExamples;
	
	/** The training image required by this job */
	private ByteImage trainingImage;
	
	/** The ground truth required by this job */
	private ByteImage gtImage;
	
	/** The classifier container used for this job */
	private Classifier classfier;
	
	/**
	 * Creates a job for positive/negative training scheme.
	 * Note: The items in the lists are copied, thus the Job object will have its own 
	 * training instances.
	 * @deprecated Use the classifier to provide the positive and negative examples.
	 * 
	 * @param positiveExamples	The list of positive examples
	 * @param negativeExamples	The list of negative examples
	 * @param classifier	The classifier to be evolved for this job
	 */
	public Job(List<ByteImage> positiveExamples, List<ByteImage> negativeExamples, Classifier classifier) {
		this.jobType = Job.TYPE_POS_NEG;
		
		this.positiveExamples = new ArrayList<ByteImage>();		
		for(ByteImage image : positiveExamples) {
			this.positiveExamples.add(image.clone());
		}
		
		this.negativeExamples = new ArrayList<ByteImage>();		
		for(ByteImage image : negativeExamples) {
			this.negativeExamples.add(image.clone());
		}
		
		this.classfier = classifier;
	}
	
	/**
	 * Creates a job for image/ground-truth training scheme.
	 * Note: The items in the lists are copied, thus the Job object will have its own 
	 * training instances.
	 * @deprecated Use the classifier to provide the training image and the ground truth.
	 * 
	 * @param trainingImage
	 * @param gtImage
	 * @param classifier
	 */
	public Job(ByteImage trainingImage, ByteImage gtImage, Classifier classifier) {
		this.jobType = Job.TYPE_GT;
		
		this.trainingImage = trainingImage.clone();
		this.gtImage = gtImage.clone();
		
		this.classfier = classifier;
	}
	
	
	public Job(Classifier classifier) {
		this.jobType = classifier.getTrainingType();
		this.classfier = classifier;
		
		switch (this.jobType) {
		case TYPE_POS_NEG:
			
			this.positiveExamples = new ArrayList<ByteImage>();		
			for(ByteImage image : classifier.getPositiveExamples()) {
				this.positiveExamples.add(image.clone());
			}
			
			this.negativeExamples = new ArrayList<ByteImage>();		
			for(ByteImage image : classifier.getNegativeExamples()) {
				this.negativeExamples.add(image.clone());
			}
			break;

		case TYPE_GT:
			this.trainingImage = classifier.getTrainingImage().clone();
			this.gtImage = classifier.getGtImage().clone();
			break;
			
		default:
			throw new RuntimeException("The classifier's training mode is not supported!");
		}
	}
	
	/**
	 * @return	Returns the job type of this job (pos/neg vs. image/gt)
	 */
	public int getJobType() {
		return this.jobType;
	}

	/**
	 * @return the positiveExamples
	 */
	public List<ByteImage> getPositiveExamples() {
		return positiveExamples;
	}

	/**
	 * @param positiveExamples the positiveExamples to set
	 */
	public void setPositiveExamples(List<ByteImage> positiveExamples) {
		this.positiveExamples = positiveExamples;
	}

	/**
	 * @return the negativeExamples
	 */
	public List<ByteImage> getNegativeExamples() {
		return negativeExamples;
	}

	/**
	 * @param negativeExamples the negativeExamples to set
	 */
	public void setNegativeExamples(List<ByteImage> negativeExamples) {
		this.negativeExamples = negativeExamples;
	}

	/**
	 * @return the trainingImage
	 */
	public ByteImage getTrainingImage() {
		return trainingImage;
	}

	/**
	 * @param trainingImage the trainingImage to set
	 */
	public void setTrainingImage(ByteImage trainingImage) {
		this.trainingImage = trainingImage;
	}

	/**
	 * @return the gtImage
	 */
	public ByteImage getGtImage() {
		return gtImage;
	}

	/**
	 * @param gtImage the gtImage to set
	 */
	public void setGtImage(ByteImage gtImage) {
		this.gtImage = gtImage;
	}

	/**
	 * @return the classfier
	 */
	public Classifier getClassfier() {
		return classfier;
	}

	/**
	 * @param classfier the classfier to set
	 */
	public void setClassfier(Classifier classfier) {
		this.classfier = classfier;
	}

	/**
	 * Compares this Job with another Job object. Two jobs are equal if
	 * they are working with the same classifier.
	 */
	@Override
	public boolean equals(Object obj) {
		Classifier otherClassifier = ((Job)obj).classfier;
		return this.classfier.equals(otherClassifier);
	}
	
	
	
	
}
