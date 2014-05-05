package cuda.gp;

import java.util.HashSet;
import java.util.Set;

import utils.Classifier;
import utils.EvolutionListener;
import utils.PreciseTimer;
import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.simple.SimpleEvolutionState;
import ec.util.Checkpoint;
import ec.util.Parameter;
import gp.datatypes.Job;

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
	
	/** The job that the GP system is currently processing */ 
	private Job activeJob;
	
	/** The prefix of the current session */
	private String sessionPrefix;

	@Override
	public void setup(EvolutionState state, Parameter base) {
		// Initialize the non-evaluated individuals' list
		if (evalthreads != breedthreads)
			output.fatal("Number of evaluation and breeding threads are different! This may cause problems for CUDA evaluation.");

		// Setup the languages, problems and everything else
		if (this.cudaInterop == null)
			this.cudaInterop = new CudaInterop();
		
		super.setup(state, base);
		
		cudaInterop.setup(this, null);
	}	
	
	
	/**
	 * Returns the training mode of the current evolution state
	 * @return
	 */
	public int getTrainingMode() {
		return this.trainingMode;
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
	 * Set the active job that the system should work on
	 * @param job
	 */
	public void setActiveJob(Job job) {
		this.activeJob = job;
	}
	
	/**
	 * @return	Return the active job that the system is currently working on
	 */
	public Job getActiveJob() {
		return this.activeJob;
	}
	
	/**
	 * Set the prefix string for this GP session
	 * @param sessionPrefix
	 */
	public void setSessionPrefix(String sessionPrefix) {
		this.sessionPrefix = sessionPrefix;
	}
	
	/**
	 * @return	The prefix string of the current session
	 */
	public String getSessionPrefix() {
		return this.sessionPrefix;
	}
	
	/** 
	 * @return	Whether the previous classifier's individual should be used to seed the population.
	 */
	public boolean shouldSeed() {
		return this.activeJob.getClassifier().shouldSeed();
	}
	
	public Individual getPopulationSeed() {
		return this.activeJob.getClassifier().getIndividual();
	}
	
	/**
	 * Report the evolution of a new individual for visualization purposes.
	 * 
	 * @param classifier
	 */
	public void reportIndividual(GPIndividual individual) {
		Classifier classifier = activeJob.getClassifier(); 
		classifier.setIndividual(individual);
		
		for (EvolutionListener listener : this.evolutionListeners) {
			int indReportFrequency = listener.getIndReportingFrequency();
			
			if (generation % indReportFrequency == 0 /*&& !classifier.getIndividual().equals(individual)*/) {	//TODO comparing individuals may have a performance hit!
				listener.reportClassifier(classifier);	//FIXME concurrent access warning!
				//FIXME probably won't even need to add this guy again because we have already added the pointer
			}
		}
	}


	/**
	 * The function's name is a little bit misleading, I had to override it but this
	 * actually sets up the EvolutionState and should be called if no GP run has been
	 * done yet. To start the whole thing, "resumeStart" should be called 
	 */
	@Override
	public void startFresh() {
		output.message("Setting up");
		setup(this, null); // a garbage Parameter
	}
	
	/**
	 * Sets the generation number to zero and calls the start function
	 */
	public void resumeStart() {
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
		if (generation > 0)
			output.message("Generation " + generation);

		// EVALUATION
		statistics.preEvaluationStatistics(this);

		PreciseTimer timer = new PreciseTimer();
		timer.start();
		evaluator.evaluatePopulation(this);
		timer.stop();
		timer.log(output, "Evaluation");

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
		// Prepare evolution data
		cudaInterop.prepareDataForRun(this, activeJob);

		/* the big loop */
		int result = R_NOTDONE;
		while (result == R_NOTDONE) {
			result = evolve();
		}

		finish(result);
	}

}

