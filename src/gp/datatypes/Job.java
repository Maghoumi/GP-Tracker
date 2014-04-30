package gp.datatypes;

import java.util.ArrayList;
import java.util.List;

import utils.ByteImage;
import utils.Classifier;
import utils.FilteredImage;

/**
 * Represents a GP classifier evolution job. A job is just a wrapper around the
 * classifier that needs to be evolved. Jobs will be queued in GPEngine's job
 * queue and will be used during the evolutionary run. Also, Job objects have
 * the required data that are needed to evaluate an individual
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
	protected int jobType = -1;
	
	/** The classifier container used for this job */
	protected Classifier classifier;
	
	/** An ID to distinguish this job from other jobs in the GP system -- used for stat purposes */
	protected String id;
	
	/** Timestamp of the moment this job was created */
	protected long timeStamp;
	
	/** The list of positive examples of the underlying classifier which have been filtered */
	protected List<FilteredImage> filteredPositiveImages = new ArrayList<FilteredImage>();
	
	/** The list of negative examples of the underlying classifier which have been filtered */
	protected List<FilteredImage> filteredNegativeImages = new ArrayList<FilteredImage>();
	
	/** Underlying classifier's training image which has been filtered */
	protected FilteredImage filteredTrainingImage;
	
	/** The training instances of this job */
	protected CudaTrainingInstance trainingInstances;
	
	public Job(Classifier classifier, String id) {
		this.jobType = classifier.getTrainingType();
		this.classifier = classifier;
		this.id = id;
		this.timeStamp = System.currentTimeMillis();
	}
	
	/**
	 * @return	Returns the job type of this job (pos/neg vs. image/gt)
	 */
	public int getJobType() {
		return this.jobType;
	}

		/**
	 * @return the classifier
	 */
	public Classifier getClassifier() {
		return classifier;
	}
	
	/**
	 * @return	The ID string of this job which distinguishes this job from other
	 * 			jobs added to the system. Should be primarily used for stat purposes
	 */
	public String getId() {
		return this.id;
	}
	
	/**
	 * @return	The timestamp of the creation of this job
	 */
	public long getTimestamp() {
		return this.timeStamp;
	}
	
	/**
	 * @param classifier the classifier to set
	 */
	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
	}
	
	/**
	 * @return	The list of positive examples of the underlying classifier
	 */
	public List<ByteImage> getPositiveExamples() {
		return this.classifier.getPositiveExamples();
	}
	
	/**
	 * @return	The list of negative examples of the underlying classifier
	 */
	public List<ByteImage> getNegativeExamples() {
		return this.classifier.getNegativeExamples();
	}
	
	/**
	 * @return	The training image of the underlying classifier
	 */
	public ByteImage getTrainingImage() {
		return this.classifier.getTrainingImage();
	}
	
	/**
	 * @return	The groundtruth of the underlying classifier
	 */
	public ByteImage getGtImage() {
		return this.classifier.getGtImage();
	}
	
	/**
	 * Set the filtered training image
	 * @param image
	 */
	public void setFilteredTrainingImage(FilteredImage image) {
		this.filteredTrainingImage = image;
	}
	
	/**
	 * @return	Filtered training image
	 */
	public FilteredImage getFilteredTrainingImage() {
		return this.filteredTrainingImage;
	}
	
	/**
	 * Add a positive filtered image to the list of filtered positive images
	 * @param image
	 */
	public void addFilteredPositiveImage(FilteredImage image) {
		this.filteredPositiveImages.add(image);
	}
	
	/**
	 * Add a negative filtered image to the list of filtered negative images
	 * @param image
	 */
	public void addFilteredNegativeImage(FilteredImage image) {
		this.filteredNegativeImages.add(image);
	}
	
	/**
	 * @return	The list of filtered positive images
	 */
	public List<FilteredImage> getFilteredPositiveImages() {
		return this.filteredPositiveImages;
	}
	
	/**
	 * @return	The list of filtered negative images
	 */
	public List<FilteredImage> getFilteredNegativeImages() {
		return this.filteredNegativeImages;
	}	
	
	/**
	 * @return	The CUDA training instances of this job 
	 */
	public CudaTrainingInstance getTrainingInstances() {
		return this.trainingInstances;
	}
	
	public void setCudaTrainingInstances(CudaTrainingInstance trainingInstances) {
		this.trainingInstances = trainingInstances;
	}

	/**
	 * Compares this Job with another Job object. Two jobs are equal if
	 * they are working with the same classifier.
	 */
	@Override
	public boolean equals(Object obj) {
		Classifier otherClassifier = ((Job)obj).classifier;
		return this.classifier.equals(otherClassifier);
	}
	
	@Override
	public String toString() {
		return this.id + " -- " + this.timeStamp;
	}
	
	public void freeAll() {
		// Free FilteredImages
		if (filteredPositiveImages != null) {
			for (FilteredImage image : filteredPositiveImages)
				image.freeAll();
		}
		
		if (filteredNegativeImages != null) {
			for (FilteredImage image : filteredNegativeImages)
				image.freeAll();
		}
		
		if (filteredTrainingImage != null)
			filteredTrainingImage.freeAll();
		// Free training instances
		trainingInstances.freeAll();
	}
	
	
	
}