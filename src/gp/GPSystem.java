package gp;

import java.util.List;

import utils.UniqueBlockingQueue;
import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Classifier;
import utils.cuda.datatypes.Segment;
import cuda.CudaInterop;
import cuda.gp.CudaEvolutionState;
import cuda.gp.CudaSimpleStatistics;
import ec.Evolve;
import ec.gp.GPIndividual;
import gp.datatypes.EvolutionListener;
import gp.datatypes.Job;

/**
 * And finally... My complete GP system! Using an object of this class, you can
 * invoke the GP system for filtering. Three things are required to invoke a GP
 * system: 1) An ECJ parameter file 2) A set of positive images 3) A set of
 * negative images
 * 
 * When an object of this class is instantiated, the ECJ parameter file should
 * be provided. Using that parameter file, ECJ is setup for the required
 * GP-Language and functionality. Any subsequent invocations will essentially
 * use the GP system that was setup initially, therefore speeding up the
 * process!
 * 
 * 
 * @author Mehran Maghoumi
 * 
 */
public class GPSystem extends Evolve implements Runnable {

	/**
	 * Number of jobs that can be queued on this GPSystem without blocking the
	 * calling thread
	 */
	private static final int JOB_CAPACITY = 1;
	
	/** The job queue for the GP system */
	private UniqueBlockingQueue<Job> jobs = new UniqueBlockingQueue<Job>(JOB_CAPACITY);

	/** A flag indicating that this GPSystem should no longer wait for jobs */
	private volatile boolean threadAlive = false;

	/** The worker thread associated with this GPSystem */
	private Thread runThread = null;

	/**
	 * The state object that is initialized using the parameters only once but
	 * is used multiple times. Everytime GP has to evolve a classifier, this
	 * instance of the state will be used.
	 */
	private CudaEvolutionState state = null;

	/**
	 * Defines the current run type of this GPSystem. (Fresh? Checkpoint?
	 * Already started?)
	 */
	private int runType;

	/** ECJ's commandline arguments */
	private String[] args;

	/**
	 * Initializes a new GPSystem using the given parameter file.
	 * If <b>startThread</b> is true, the worker thread will start
	 * working immediately.
	 * 
	 * @param args
	 * @param startThread
	 */
	public GPSystem(String[] args, boolean startThread) {
		this.args = args; 

		state = (CudaEvolutionState) possiblyRestoreFromCheckpoint(this.args);
		if (state != null) { // loaded from checkpoint 
			runType = CudaEvolutionState.C_STARTED_FROM_CHECKPOINT;
			state.startFromCheckpoint();
		} else {
			state = (CudaEvolutionState) initialize(loadParameterDatabase(this.args), 0);
			state.job = new Object[1];
			state.job[0] = new Integer(0);
			runType = CudaEvolutionState.C_STARTED_FRESH;
			state.startFresh();
		}

		// Create the worker thread associated with this GPSystem
		if (startThread) {
			this.runThread = new Thread(this);
			this.runThread.start();
		}
	}
	
	/**
	 * Add an EvolutionListener to the list of this system's listeners
	 * @param listener
	 */
	public void addEvolutionListener(EvolutionListener listener) {
		this.state.addEvolutionListener(listener);
	}
	
	/**
	 * Remove an EvolutionListener from the list of this system's listeners
	 * @param listener
	 */
	public void removeEvolutionListener(EvolutionListener listener) {
		this.state.removeEvolutionListener(listener);
	}

	/**
	 * Starts the worker thread that processes the Job queue.
	 */
	public void startWorkerThread() {
		if (this.threadAlive)
			return;
		
		this.threadAlive = true;
		this.runThread = new Thread(this);
		this.runThread.start();
	}

	/**
	 * A call to this function will kill the GPSystem's thread and will prepare
	 * the system for a shutdown.
	 */
	public void stopWorkerThread() {
		this.threadAlive = false;
		if (this.runThread != null)
			runThread.interrupt();
	}

	/**
	 * Queues a job for the GPSystem. This method will block the calling thread
	 * if the GPSystem's queue is already full.
	 * 
	 * @param job
	 *            The job to be queued
	 */
	public boolean queueJob(Job job) {
		try {
			
			this.jobs.put(job);
			return true;
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		return false;
	}
	
	/**
	 * @return	True if the job queue is empty, False otherwise.
	 */
	public boolean isQueueEmpty() {
		return this.jobs.size() == 0;
	}
	
	/**
	 * Runs a GP run on the calling thread using the passed Job instance.
	 * Obviously, this call is synchronous.
	 * @param job
	 */
	public void runJob(Job job) {
		// Set the CUDA context to the calling thread
		getCudaInterop().switchContext();
		
		Classifier passedClassifier = job.getClassifier();
		state.setWorkingClassifier(passedClassifier); // set the working classifier of the EvolutionState object

		// Here, I already have a job. Decide which training scheme to adopt:
		switch (job.getJobType()) {

		case Job.TYPE_POS_NEG:
			passedClassifier.setIndividual(this.call(job.getClassifier().getPositiveExamples(), job.getClassifier().getNegativeExamples()));
			break;

		case Job.TYPE_GT:
			passedClassifier.setIndividual(this.call(job.getClassifier().getTrainingImage(), job.getClassifier().getGtImage()));
			break;
			
		default:
			throw new RuntimeException("Wow! Unknown job type! Shutting down...");
		}
		
		
	}

	/**
	 * @return Returns the CudaInterop object associated with this GP system.
	 */
	public CudaInterop getCudaInterop() {
		return this.state.getCudaInterop();
	}

	/**
	 * Calls the GP system using the specified positive and negative examples,
	 * waits for the GP system to finish and returns the single best individual
	 * of the whole run.
	 * 
	 * 
	 * @param args
	 *            The command-line argument (usually should specify the location
	 *            of the ECJ's parameter file)
	 * @param positives
	 *            A list of positive examples
	 * @param negatives
	 *            A list of negative examples
	 * @return The best individual of the whole run
	 */
	private GPIndividual call(List<ByteImage> positives, List<ByteImage> negatives) {
		// Switch context
		getCudaInterop().switchContext();
		state.setExamples(positives, negatives);asdasd

		// Ask the State to be started
		state.resumeStart();
		
		// Run the GP system
		state.run(runType);

		// Set the run type for the next time
		this.runType = CudaEvolutionState.C_STARTED_AGAIN;

		// Determine and return the best individual of the run
		GPIndividual bestIndividual = (GPIndividual) ((CudaSimpleStatistics) state.statistics).best_of_run[0];
		return bestIndividual;
	}

	/**
	 * Calls the GP system using the specified training image and its
	 * corresponding ground-truth and waits for the system to finish and returns
	 * the single best individual of the whole run.
	 * 
	 * @param trainingImage
	 *            The image to train on
	 * @param gtImage
	 *            The ground truth of the training image
	 * @return The best individual of the whole run
	 */
	private GPIndividual call(ByteImage trainingImage, ByteImage gtImage) {
		// Switch context
		getCudaInterop().switchContext();

		state.setTrainingImages(trainingImage, gtImage);asdasd

		// Ask the State to be started
		state.resumeStart();
		
		// Run the GP system
		state.run(runType);

		// Set the run type for the next time
		this.runType = CudaEvolutionState.C_STARTED_AGAIN;

		// Determine and return the best individual of the run
		GPIndividual bestIndividual = (GPIndividual) ((CudaSimpleStatistics) state.statistics).best_of_run[0];
		return bestIndividual;
	}

	@Override
	public void run() {
		// Switch the CUDA context
		getCudaInterop().switchContext();

		while (threadAlive) {
			Job newJob = null;
			
			try {
				// Wait for a new job however, don't remove it!
				// Necessary for the auto-retrain function to work! A job should only be queued 
				// if it is not already being processed! By not removing the job from the job queue,
				// we can ensure that only new (and of course useful) jobs are successfully queued 
				// and duplicated jobs never get a chance of being queued.
				newJob = jobs.blockPeek();
			} catch (InterruptedException e) {
				// If this thread is interrupted, then we should probably go for a shutdown
				// Therefore, we will check the threadAlive flag
				break;
			}

			getCudaInterop().switchContext(); // Safety measure
			state.setActiveJob(newJob);

			// Here, I already have a job. Decide which training scheme to adopt:
			GPIndividual evolvedIndividual = null;
			
			switch (newJob.getJobType()) {

			case Job.TYPE_POS_NEG:
				evolvedIndividual = this.call(newJob.getClassifier().getPositiveExamples(), newJob.getClassifier().getNegativeExamples());
				break;

			case Job.TYPE_GT:
				evolvedIndividual = this.call(newJob.getClassifier().getTrainingImage(), newJob.getClassifier().getGtImage());
				break;
			default:
				throw new RuntimeException("Wow! Unknown job type! Shutting down...");
			}
			
			passedClassifier.setIndividual(evolvedIndividual);
			state.reportIndividual(evolvedIndividual);
			jobs.poll(); // remove this job from the queue permenantly!
			System.err.println("Finished processing " + newJob.getClassifier().toString());
		} // end-while
	}
}
