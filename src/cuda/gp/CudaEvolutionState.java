package cuda.gp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import m2xfilter.datatypes.DataInstance;
import m2xfilter.datatypes.EvolutionListener;
import cuda.CudaInterop;
import utils.PreciseTimer;
import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Classifier;
import utils.cuda.datatypes.CudaData;
import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.simple.SimpleEvolutionState;
import ec.util.Checkpoint;
import ec.util.Parameter;

/**
 * A generational evolution state which will also contain required CUDA objects
 * (such as handles to kernels as well as lists of individuals who are not
 * evaluated yet)
 * 
 * @author Mehran Maghoumi
 * 
 */
public class CudaEvolutionState extends SimpleEvolutionState {

	/** The population started using new bunch of data only */
	public final static int C_STARTED_AGAIN = 2;
	
	/** Training mode for this state is to use separate positive and negative samples */
	public final static int TRAINING_MODE_POS_NEG = 0;
	
	/** Training mode for this state is to use ground-truth image */
	public final static int TRAINING_MODE_GT = 1;
	
	/** Determines the training mode of this evolutionary run */
	private int trainingMode = -1;
	
	/** The list of all EvolutionListeners that are registered with this EvolutionState object */
	private Set<EvolutionListener> evolutionListeners = new HashSet<EvolutionListener>();
	
	/** Interop instance for CUDA communications*/
	private CudaInterop cudaInterop = null;
	
	/**
	 * Indicates whether this state was started fresh. This flag should be used
	 * to decide whether the kernel should be compiled or not, whether the
	 * variables should be loaded or not and so on.
	 */
	private boolean isStartFresh = false;
	
	/** Stores the training instances that the GP system needs for training */
	public List<DataInstance> trainingInstances = new ArrayList<DataInstance>();
	
	/** The training instances that are transferred to the GPU */
	public CudaData devTrainingInstances;
	
	/** Represents a list of positive examples that the system uses for training */
	private List<ByteImage> positiveExamples = new ArrayList<ByteImage>();
	
	/** Represents a list of negative examples that the system uses for training */
	private List<ByteImage> negativeExamples = new ArrayList<ByteImage>();
	
	/** The list of positive examples on the GPU */
	private List<CudaData> devPositiveExamples;
	
	/**The list of negative examples on the GPU */
	private List<CudaData> devNegativeExamples;
	
	/** The image to train on (should be supplemented with a ground-truth image as well) */
	private ByteImage trainingImage;
	
	/** The prepared (ie. filtered) training image on the GPU */
	private CudaData devTrainingImage;
	
	/** The ground truth to train on (should be supplied with a training image) */
	private ByteImage gtImage;
	
	/** The classifier that this state is currently working on */
	private Classifier classifier;

	@Override
	public void setup(EvolutionState state, Parameter base) {
		// Initialize the non-evaluated individuals' list
		if (evalthreads != breedthreads)
			output.fatal("Number of evaluation and breeding threads are different! This may cause problems for CUDA evaluation.");

		// Setup the languages, problems and everything else
		if (this.cudaInterop == null)
			this.cudaInterop = new CudaInterop(false);
		
		super.setup(state, base);
		
		cudaInterop.setup(this, null);
	}
	
	/**
	 * Set the image and the ground-truth image for training
	 * @param image	The image
	 * @param gtImage	The ground truth image
	 */
	public void setTrainingImages(ByteImage image, ByteImage gtImage) {
		this.trainingMode = TRAINING_MODE_GT;	// Set the training mode
		this.trainingImage = image;
		this.gtImage = gtImage;
	}
	
	/**
	 * Sets the lists of examples for this evolution state. Note that the
	 * passed lists are not cloned.
	 * 
	 * @param positives
	 * 		A list containing all positive examples
	 * @param negatives
	 * 		A list containing all negative examples
	 */
	public void setExamples(List<ByteImage> positives, List<ByteImage> negatives) {
		this.trainingMode = TRAINING_MODE_POS_NEG;
		
		for (ByteImage example : positives) {
			this.positiveExamples.add(example.clone());
		}
		
		for (ByteImage example : negatives) {
			this.negativeExamples.add(example.clone());
		}
	}
	
	/**
	 * Adds a positive example to the positives list of this state
	 * 
	 * @param positiveImage
	 * 		A positive example to add
	 */
	public void addPositiveExample(ByteImage positiveImage) {
		positiveExamples.add(positiveImage);
	}
	
	/**
	 * Adds a negative example to the negativess list of this state
	 * 
	 * @param negativeImage
	 * 		A negative example to add
	 */
	public void addNegativeExample(ByteImage negativeImage) {
		negativeExamples.add(negativeImage);
	}
	
	/**
	 * Returns the positive examples list of this state
	 * @return
	 */
	public List<ByteImage> getPositiveExamples() {
		return this.positiveExamples;
	}
	
	/**
	 * Returns the negative examples list of this state
	 * @return
	 */
	public List<ByteImage> getNegativeExamples() {
		return this.negativeExamples;
	}
	
	/**
	 * Resets the examples list for this state object. Will deallocate CUDA memories
	 * as well. This function should be called by the finisher after the evolutionary run.
	 */
	public void resetExamples() {
		//FIXME TODO check and see if extra operations are needed here
		trainingInstances.clear();
		devTrainingInstances.freeAll();
		
		if (trainingMode == TRAINING_MODE_GT) {
			trainingImage = null;
			devTrainingImage.freeAll();
			gtImage = null;
		}
		else if (trainingMode == TRAINING_MODE_POS_NEG) {
			positiveExamples.clear();
			negativeExamples.clear();
			
			for (CudaData data : devPositiveExamples)
				data.freeAll();
			devPositiveExamples.clear();
			
			for (CudaData data : devNegativeExamples)
				data.freeAll();
			devNegativeExamples.clear();
		}
	}
	
	
	/**
	 * Adds a GPU positive example to the positives list of this state
	 * 
	 * @param positiveImage
	 * 		A positive example to add
	 */
	public void addDevPositiveExample(CudaData positiveImage) {
		if (devPositiveExamples == null)
			devPositiveExamples = new ArrayList<CudaData>();
		
		devPositiveExamples.add(positiveImage);
	}
	
	/**
	 * Adds a GPU negative example to the positives list of this state
	 * 
	 * @param positiveImage
	 * 		A positive example to add
	 */
	public void addDevNegativeExample(CudaData negativeImage) {
		if (devNegativeExamples == null)
			devNegativeExamples = new ArrayList<CudaData>();
		
		devNegativeExamples.add(negativeImage);
	}
	
	/**
	 * Returns the GPU positive examples list of this state
	 * @return
	 */
	public List<CudaData> getDevPositiveExamples() {
		return this.devPositiveExamples;
	}
	
	/**
	 * Returns the GPU negative examples list of this state
	 * @return
	 */
	public List<CudaData> getDevNegativeExamples() {
		return this.devNegativeExamples;
	}
	
	/**
	 * Returns the training mode of the current evolution state
	 * @return
	 */
	public int getTrainingMode() {
		return this.trainingMode;
	}
	
	/**
	 * Returns the original training image
	 * @return
	 */
	public ByteImage getTrainingImage() {
		return this.trainingImage;
	}
	
	/**
	 * Returns the GPU training image of this state
	 * @return
	 */
	public CudaData getDevTrainingImage() {
		return this.devTrainingImage;
	}
	
	/**
	 * Set the GPU training image for this state
	 * @param data
	 */
	public void setDevTrainingImage(CudaData data) {
		this.devTrainingImage = data;
	}
	
	/**
	 * Returns the ground truth of the training image
	 * @return
	 */
	public ByteImage getGtImage() {
		return this.gtImage;
	}
	
	/**
	 * Accessor for the CudaInterop instance associated with this instance of the
	 * evolution state
	 * @return	The CudaInterop instance
	 */
	public CudaInterop getCudaInterop() {
		return this.cudaInterop;
	}
	
	/**
	 * Add an EvolutionListener to the list of this object's listeners
	 * @param listener
	 */
	public void addEvolutionListener(EvolutionListener listener) {
		this.evolutionListeners.add(listener);
	}
	
	/**
	 * Remove an EvolutionListener from the list of this object's listeners
	 * @param listener
	 */
	public void removeEvolutionListener(EvolutionListener listener) {
		this.evolutionListeners.remove(listener);
	}
	
	/**
	 * Sets the working classifier of this EvolutionState
	 * @param classifier	The classifier that this EvolutionState should work on
	 */
	public void setWorkingClassifier(Classifier classifier) {
		this.classifier = classifier;
	}
	
	/** 
	 * @return	Whether the previous classifier's individual should be used to seed the population.
	 */
	public boolean shouldSeed() {
		if (this.classifier != null)
			return this.classifier.shouldSeed();
		
		return false;
	}
	
	public Individual getPopulationSeed() {
		return this.classifier.getIndividual();
	}
	
	/**
	 * Report the evolution of a new individual for visualization purposes.
	 * 
	 * @param classifier
	 */
	public void reportIndividual(GPIndividual individual) {
		this.classifier.setIndividual(individual);
		
		for (EvolutionListener listener : this.evolutionListeners) {
			int indReportFrequency = listener.getIndReportingFrequency();
			
			if (generation % indReportFrequency == 0 /*&& !classifier.getIndividual().equals(individual)*/) {	//TODO comparing individuals may have a performance hit!
				listener.reportClassifier(classifier);	//FIXME concurrent access warning!
				//FIXME probably won't even need to add this guy again because we have already added the pointer
			}
		}
	}


	@Override
	public void startFresh() {
		output.message("Setting up");
		setup(this, null); // a garbage Parameter
		start();
	}
	
	public void startAgain() {
		output.message("Restarting the previous state");
		generation = 0;	// reset the generation number
		start();
	}
	
	public void start() {
		// POPULATION INITIALIZATION
		output.message("Initializing Generation 0");
		statistics.preInitializationStatistics(this);
		population = initializer.initialPopulation(this, 0); // unthreaded
		statistics.postInitializationStatistics(this);

		// INITIALIZE CONTACTS -- done after initialization to allow
		// a hook for the user to do things in Initializer before
		// an attempt is made to connect to island models etc.
		exchanger.initializeContacts(this);
		evaluator.initializeContacts(this);
	}

	/**
	 * @return
	 * @throws InternalError
	 */
	public int evolve() {
		long t1 = 0, t2 = 0;
		if (generation > 0)
			/*if (generation % 20 == 0)*/ output.message("Generation " + generation);

		// EVALUATION
		statistics.preEvaluationStatistics(this);

		PreciseTimer timer = new PreciseTimer();
		timer.start();
		evaluator.evaluatePopulation(this);
		timer.stop();
		/*if (generation %20 == 0)*/ timer.log(output, "Evaluation");

		statistics.postEvaluationStatistics(this);

		// SHOULD WE QUIT?
		if (evaluator.runComplete(this) && quitOnRunComplete) {
			output.message("Found Ideal Individual");
			return R_SUCCESS;
		}

		// SHOULD WE QUIT?
		if (generation == numGenerations - 1) {
			return R_FAILURE;
		}

		// PRE-BREEDING EXCHANGING
		statistics.prePreBreedingExchangeStatistics(this);
		population = exchanger.preBreedingExchangePopulation(this);
		statistics.postPreBreedingExchangeStatistics(this);

		String exchangerWantsToShutdown = exchanger.runComplete(this);
		if (exchangerWantsToShutdown != null) {
			output.message(exchangerWantsToShutdown);
			/*
			 * Don't really know what to return here. The only place I could
			 * find where runComplete ever returns non-null is IslandExchange.
			 * However, that can return non-null whether or not the ideal
			 * individual was found (for example, if there was a communication
			 * error with the server).
			 * 
			 * Since the original version of this code didn't care, and the
			 * result was initialized to R_SUCCESS before the while loop, I'm
			 * just going to return R_SUCCESS here.
			 */

			return R_SUCCESS;
		}

		// BREEDING
		statistics.preBreedingStatistics(this);

		population = breeder.breedPopulation(this);

		// POST-BREEDING EXCHANGING
		statistics.postBreedingStatistics(this);

		// POST-BREEDING EXCHANGING
		statistics.prePostBreedingExchangeStatistics(this);
		population = exchanger.postBreedingExchangePopulation(this);
		statistics.postPostBreedingExchangeStatistics(this);

		// INCREMENT GENERATION AND CHECKPOINT
		generation++;
		if (checkpoint && generation % checkpointModulo == 0) {
			output.message("Checkpointing");
			statistics.preCheckpointStatistics(this);
			Checkpoint.setCheckpoint(this);
			statistics.postCheckpointStatistics(this);
		}

		return R_NOTDONE;
	}

	@Override
	public void run(int condition) {
		
		if (this.trainingMode == TRAINING_MODE_POS_NEG) {
		
			if (positiveExamples == null || negativeExamples == null)
				output.fatal("The list of positive/negative examples is not initialized");
			
			if (positiveExamples.size() == 0 || negativeExamples.size() == 0)
				output.fatal("No positive/negative examples were provided to the GP system");
		}
		else if (this.trainingMode == TRAINING_MODE_GT) {
			//TODO what?? :D
		}
		
		// Prepare evolution data
		cudaInterop.prepareDataForRun(this);

		/* the big loop */
		int result = R_NOTDONE;
		while (result == R_NOTDONE) {
			result = evolve();
		}

		finish(result);
	}

}

