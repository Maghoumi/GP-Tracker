package cuda.gp;

import ec.EvolutionState;
import ec.simple.SimpleShortStatistics;

/**
 * A statistics hook which is an extension of ECJ's SimpleShortStatistics class
 * but provides additional separators and information about the segment being retrained
 * in its dump file 
 * 
 * @author Mehran Maghoumi
 */
public class CudaSimpleShortStatistics extends SimpleShortStatistics {
	
	public static final String TERMINATING_SEQUENCE = " ";	// alt + 255
	
	@Override
	public void preInitializationStatistics(EvolutionState state) {
		super.preInitializationStatistics(state);
		
		boolean output = (state.generation % modulus == 0);
		
		// Make the job id and put it as the first line of the stat
		CudaEvolutionState cuState = (CudaEvolutionState) state;
		String jobId = cuState.jobId + " -- " + System.currentTimeMillis();
		
		if (output)
			if (output) state.output.println(jobId, statisticslog);
			
	}
	
	/**
	 * Would simply append a terminating sequence at the end of the log file
	 */
	@Override
	public void finalStatistics(EvolutionState state, int result) {
		super.finalStatistics(state, result);
		
		boolean output = (state.generation % modulus == 0);
		
		if (output)
			state.output.println(TERMINATING_SEQUENCE, statisticslog);
	}
}
