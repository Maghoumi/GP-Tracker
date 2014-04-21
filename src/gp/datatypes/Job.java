package gp.datatypes;

import utils.cuda.datatypes.Classifier;

/**
 * Represents a GP classifier evolution job. A job is just a wrapper around the
 * classifier that needs to be evolved. Jobs will be queued in GPSystem's job
 * queue and will be used during the evolutionary run.
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
	
	/** The classifier container used for this job */
	private Classifier classifier;
	
	public Job(Classifier classifier) {
		this.jobType = classifier.getTrainingType();
		this.classifier = classifier;
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
	 * @param classifier the classifier to set
	 */
	public void setClassifier(Classifier classifier) {
		this.classifier = classifier;
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
	
	
	
	
}
