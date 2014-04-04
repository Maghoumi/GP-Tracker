package m2xfilter;

import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import m2xfilter.datatypes.EvolutionListener;
import m2xfilter.datatypes.Job;
import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Classifier;
import visualizer.Visualizer;
import cuda.CudaInterop;
import cuda.gp.CudaEvolutionState;
import cuda.gp.CudaSimpleStatistics;
import ec.Evolve;
import ec.gp.GPIndividual;

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

	/** The job queue for the GP system */
	private BlockingQueue<Job> jobs;

	/**
	 * Number of jobs that can be queued on this GPSystem without blocking the
	 * calling thread
	 */
	private static final int JOB_CAPACITY = 10;

	/** A flag indicating that this GPSystem should no longer wait for jobs */
	private volatile boolean isFinalized = false;

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
		this.jobs = new ArrayBlockingQueue<Job>(JOB_CAPACITY);

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
		this.runThread = new Thread(this);
		if (startThread) {
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
		if (!runThread.isAlive())
			this.runThread.start();
	}

	/**
	 * A call to this function will kill the GPSystem's thread and will prepare
	 * the system for a shutdown.
	 */
	public void finalize() {
		this.isFinalized = true;
		runThread.interrupt();
	}

	/**
	 * Queues a job for the GPSystem. This method will block the calling thread
	 * if the GPSystem's queue is already full.
	 * 
	 * @param job
	 *            The job to be queued
	 */
	public void queueJob(Job job) {
		try {
			this.jobs.put(job);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Runs a GP run on the calling thread using the passed Job instance.
	 * Obviously, this call is synchronous.
	 * @param job
	 */
	public void runJob(Job job) {
		// Set the CUDA context to the calling thread
		getCudaInterop().switchContext();
		
		Classifier passedClassifier = job.getClassfier();
		state.setWorkingClassifier(passedClassifier); // set the working classifier of the EvolutionState object

		// Here, I already have a job. Decide which training scheme to adopt:
		switch (job.getJobType()) {

		case Job.TYPE_POS_NEG:
			passedClassifier.setIndividual(this.call(job.getPositiveExamples(), job.getNegativeExamples()));
			break;

		case Job.TYPE_GT:
			passedClassifier.setIndividual(this.call(job.getTrainingImage(), job.getGtImage()));
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
		state.setExamples(positives, negatives);

		if (runType == CudaEvolutionState.C_STARTED_AGAIN) {
			state.startAgain();
		}

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

		state.setTrainingImages(trainingImage, gtImage);

		if (runType == CudaEvolutionState.C_STARTED_AGAIN) {
			state.startAgain();
		}

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

		while (!isFinalized) {
			Job newJob = null;

			try {
				// Wait for a new job
				newJob = jobs.take();
			} catch (InterruptedException e) {
				// If this thread is interrupted, then we should probably go for a shutdown
				// Therefore, we will check the isFinalized flag
				continue;
			}

			getCudaInterop().switchContext(); // Safety measure
			Classifier passedClassifier = newJob.getClassfier();
			state.setWorkingClassifier(passedClassifier); // set the working classifier of the EvolutionState object

			// Here, I already have a job. Decide which training scheme to adopt:
			switch (newJob.getJobType()) {

			case Job.TYPE_POS_NEG:
				passedClassifier.setIndividual(this.call(newJob.getPositiveExamples(), newJob.getNegativeExamples()));
				break;

			case Job.TYPE_GT:
				passedClassifier.setIndividual(this.call(newJob.getTrainingImage(), newJob.getGtImage()));
				break;
			default:
				throw new RuntimeException("Wow! Unknown job type! Shutting down...");
			}
		} // end-while
	}

}
