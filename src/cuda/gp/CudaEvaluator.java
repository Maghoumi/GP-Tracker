package cuda.gp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import m2xfilter.M2XFilter;
import m2xfilter.datatypes.ProblemData;
import ec.Evaluator;
import ec.EvolutionState;
import ec.Subpopulation;
import ec.gp.GPIndividual;
import ec.simple.SimpleEvaluator;
import ec.simple.SimpleFitness;
import ec.simple.SimpleProblemForm;
import ec.util.Parameter;
import gnu.trove.list.array.TByteArrayList;

/**
 * An evaluator which works exactly like SimpleEvaluator but performs the
 * evaluations in CUDA. Also relies on the CudaEvolutionState because forces the
 * threads to evaluate their portions of the unevaluated individuals.
 * 
 * @author Mehran Maghoumi
 * 
 */
public class CudaEvaluator extends SimpleEvaluator {
	
	/** Keeps the expression list for each thread */
	protected List<List<TByteArrayList>> threadExpList;
	private boolean useCuda = false;
	
	private CudaEvolutionState cuState;
	

	// checks to make sure that the Problem implements SimpleProblemForm
	@Override
	public void setup(final EvolutionState state, final Parameter base) {
		super.setup(state, base);
		if (!(p_problem instanceof SimpleProblemForm))
			state.output.fatal("" + this.getClass()
					+ " used, but the Problem is not of SimpleProblemForm",
					base.push(P_PROBLEM));
		
		// Initialize the list of list of expressions
		
		threadExpList = new ArrayList<List<TByteArrayList>>(state.evalthreads);
		
		useCuda = state.parameters.getString(base.push("evaluationMethod"), null).toLowerCase().equals("gpu"); 
		cuState = (CudaEvolutionState) state;
		
		
		String cudaMessage
						= "#########################\n"
				+ 		  "#   Using NVIDIA CUDA   #\n"
				+ 		  "#########################";
		
		String cpuMessage
						= "#################\n"
				+ 		  "#   Using CPU   #\n"
				+ 		  "#################";
		
		state.output.message(useCuda ? cudaMessage : cpuMessage);
		
		for(int i = 0 ; i < state.evalthreads ; i++) {
			threadExpList.add(new ArrayList<TByteArrayList>());
		}
		
	}

	/**
	 * A simple evaluator that doesn't do any coevolutionary evaluation.
	 * Basically it applies evaluation pipelines, one per thread, to various
	 * subchunks of a new population. Each thread is responsible for converting
	 * its own subchunk to a postfix expression.
	 */
	@Override
	public void evaluatePopulation(final EvolutionState state) {
		
		int count = 0;
		for(int x = 0;x<state.population.subpops.length;x++)
            for(int y=0;y<state.population.subpops[x].individuals.length;y++)
                if (!state.population.subpops[x].individuals[y].evaluated) {
                	count++;
                }
		// Not using CUDA? What a shame... :-<
		if (!useCuda) {
			super.evaluatePopulation(state);
			return;
		}
		

		int[] from = new int[state.evalthreads]; // starting index of this thread
		int[] to = new int[state.evalthreads];	// ending index of this thread
		
		int offset = 0;
		
		
		// These stuff should be done per subpopulation.
		for (Subpopulation sp : state.population.subpops) {
			CudaSubpopulation subPop = (CudaSubpopulation) sp;
			
			// Determine the working scope of each thread
			for (int i = 0 ; i < state.evalthreads ; i++) {
				List<GPIndividual> listOfInd = subPop.needEval.get(i);
				
				from[i] = offset;
				to[i] = from[i] + listOfInd.size() - 1;
				offset += listOfInd.size();
			}
			
			if (state.evalthreads == 1)
				traversePopChunk(state, subPop, 0);
			else {
				Thread[] t = new Thread[state.evalthreads];
	
				// start up the threads
				for (int y = 0; y < state.evalthreads; y++) {
					ByteTraverseThread r = new ByteTraverseThread();
					r.threadnum = y;
					r.me = this;
					r.state = state;
					r.subPop = subPop;
					t[y] = new Thread(r);
					t[y].start();
				}
	
				// gather the threads
				for (int y = 0; y < state.evalthreads; y++)
					try {
						t[y].join();
					} catch (InterruptedException e) {
						state.output
								.fatal("Whoa! The main evaluation thread got interrupted!  Dying...");
					}
	
			}
			
			// Call the CUDA kernel
			float[] fitnesses = cuState.getCudaInterop().evaluatePopulation(threadExpList, cuState.devTrainingInstances);
			
			if (fitnesses.length != count) {
				System.out.println("Mismatched count");
				System.exit(0);
			}
				
			
			// Clean the expressions! Will clean it bellow!
			
			// call the assignFitness and assign fitnesses to each individual
			if (state.evalthreads == 1) {
				threadExpList.get(0).clear();
				assignFitness(state, subPop, 0, fitnesses, from, to);
			}
			else {
				Thread[] t = new Thread[state.evalthreads];
				
				// start up the threads
				for (int y = 0; y < state.evalthreads; y++) {
					// first clean this threads expressions
					
					threadExpList.get(y).clear();
					
					FitnessAssignmentThread r = new FitnessAssignmentThread();
					r.threadnum = y;
					r.me = this;
					r.state = state;
					r.subPop = subPop;
					r.fitnesses = fitnesses;
					r.from = from;
					r.to = to;
					t[y] = new Thread(r);
					t[y].start();
				}
	
				// gather the threads
				for (int y = 0; y < state.evalthreads; y++)
					try {
						t[y].join();
					} catch (InterruptedException e) {
						state.output
								.fatal("Whoa! The main evaluation thread got interrupted!  Dying...");
					}
			}
		}
		//Finished! :-)
	}

	protected void traversePopChunk(EvolutionState state, CudaSubpopulation subPop, int threadnum) {
		// Get the unevaluateds for the current thread
		List<GPIndividual> myNeedEvals = subPop.needEval.get(threadnum);
		List<TByteArrayList> myExpList = threadExpList.get(threadnum);
		
		// Walk through my individuals and convert them to byte[] expressions
		// and store them in myExpList
		for(GPIndividual ind : myNeedEvals) {
			CudaNode root = (CudaNode) ind.trees[0].child;
			byte[] exp = root.makePostfixExpression();
			// Add this expression to the list
			myExpList.add(new TByteArrayList(exp));
		}
	}
	
	/**
	 * Assigns the calculated CUDA fitness to the corresponding individual
	 * 
	 * @param state
	 * @param threadnum
	 * @param startIndex
	 * @param endIndex
	 */
	protected void assignFitness(EvolutionState state, CudaSubpopulation subPop, int threadnum, float[] fitnesses, int[] from, int[] to) {
		List<GPIndividual> myUnevals = subPop.needEval.get(threadnum); // get my unevaluated individuals
		
		int indIndex = 0;	// hold the index to my individuals
		for (int i = from[threadnum] ; i <= to[threadnum] ; i++) {
			GPIndividual currentInd = myUnevals.get(indIndex++);
			
			((SimpleFitness) currentInd.fitness).setFitness(state, fitnesses[i] , fitnesses[i] >= 0.985);
			
			/** DEBUG */
//			float testFitness = ProblemData.cudaCaller.cpuEvaluate(state, currentInd, ((M2XFilter)p_problem.clone()).trainingInstances);if (Math.abs(testFitness -fitnesses[i]) > 0.1) {
//				System.out.println("Different -> CUDA: " + fitnesses[i] + " Actual: " + testFitness); System.out.println(currentInd.trees[0].child.makeLatexTree());}
			
			
			currentInd.evaluated = true; // Current individual is now evaluated :-)
		}
		
		// Clean my unevaluateds
		myUnevals.clear();
	}

	/** A private helper class for implementing multithreaded byte traversal */
	private class ByteTraverseThread implements Runnable {
		public CudaEvaluator me;
		public EvolutionState state;
		public int threadnum;
		public CudaSubpopulation subPop;
		
		public synchronized void run() {
			me.traversePopChunk(state, subPop, threadnum);
		}
	}
	
	/** A private helper class for implementing multithreaded fitness assignment */
	private class FitnessAssignmentThread implements Runnable {
		public CudaEvaluator me;
		public EvolutionState state;
		public int threadnum;
		public int[] from;
		public int[] to;
		public float[] fitnesses;
		public CudaSubpopulation subPop;
		
		public synchronized void run() {
			me.assignFitness(state, subPop, threadnum, fitnesses, from, to);
		}
	}
}
