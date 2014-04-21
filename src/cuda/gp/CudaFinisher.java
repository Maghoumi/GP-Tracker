package cuda.gp;

import ec.EvolutionState;
import ec.simple.SimpleFinisher;
import gp.datatypes.ProblemData;

/**
 * A finisher which will make sure to deallocate all CUDA allocations and 
 * clean up after itself!
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaFinisher extends SimpleFinisher {

	@Override
	public void finishPopulation(EvolutionState state, int result) {
		state.output.message("Executing finisher: a little housekeeping...");
//		ProblemData.inputData.freeAll();
//		ProblemData.trainingData.freeAll();
//		ProblemData.testingData.freeAll();
		
		CudaEvolutionState cuState = (CudaEvolutionState) state;
		cuState.resetExamples();
		//FIXME the below method should be called
//		cuState.cudaCaller.cleanUp();
//		cuState.cudaCaller.destroy();
	}
	
}
