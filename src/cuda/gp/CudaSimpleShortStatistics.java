package cuda.gp;

import java.io.File;
import java.io.IOException;

import ec.EvolutionState;
import ec.Statistics;
import ec.simple.SimpleShortStatistics;
import ec.util.Output;
import ec.util.Parameter;

/**
 * A statistics hook which is an extension of ECJ's SimpleShortStatistics class
 * but provides additional separators and information about the segment being
 * retrained in its dump file
 * 
 * @author Mehran Maghoumi
 */
public class CudaSimpleShortStatistics extends SimpleShortStatistics {

	public static final String TERMINATING_SEQUENCE = " "; // alt + 255

	public void setup(final EvolutionState state, final Parameter base)
	{
		int t = state.parameters.getIntWithDefault(base.push(P_NUMCHILDREN), null, 0);
		if (t < 0)
			state.output.fatal("A Statistics object cannot have negative number of children",
					base.push(P_NUMCHILDREN));

		// load the trees
		children = new Statistics[t];

		for (int x = 0; x < t; x++)
		{
			Parameter p = base.push(P_CHILD).push("" + x);
			children[x] = (Statistics) (state.parameters.getInstanceForParameterEq(p, null, Statistics.class));
			((Statistics) children[x]).setup(state, p);
		}
		
		File statisticsFile = state.parameters.getFile(base.push(P_STATISTICS_FILE), null);
		// Prepend the session prefix! :D
		CudaEvolutionState cuState = (CudaEvolutionState) state;
		statisticsFile = new File(statisticsFile.getParentFile().getAbsolutePath() + "\\" + cuState.getSessionPrefix() + statisticsFile.getName());		

		modulus = state.parameters.getIntWithDefault(base.push(P_STATISTICS_MODULUS), null, 1);
		muzzle = state.parameters.getBoolean(base.push(P_MUZZLE), null, false);

		if (muzzle)
		{
			statisticslog = Output.NO_LOGS;
		}
		else if (statisticsFile != null)
			try
			{
				statisticslog = state.output.addLog(statisticsFile,
						!state.parameters.getBoolean(base.push(P_COMPRESS), null, false),
						state.parameters.getBoolean(base.push(P_COMPRESS), null, false));
			} catch (IOException i)
			{
				state.output.fatal("An IOException occurred while trying to create the log " + statisticsFile + ":\n" + i);
			}
		doSize = state.parameters.getBoolean(base.push(P_DO_SIZE), null, false);
		doTime = state.parameters.getBoolean(base.push(P_DO_TIME), null, false);
		if (state.parameters.exists(base.push(P_FULL), null))
		{
			state.output.warning(P_FULL + " is deprecated.  Use " + P_DO_SIZE + " and " + P_DO_TIME + " instead.  Also be warned that the table columns have been reorganized. ", base.push(P_FULL), null);
			boolean gather = state.parameters.getBoolean(base.push(P_FULL), null, false);
			doSize = doSize || gather;
			doTime = doTime || gather;
		}
		doSubpops = state.parameters.getBoolean(base.push(P_DO_SUBPOPS), null, false);
	}

	@Override
	public void preInitializationStatistics(EvolutionState state) {
		super.preInitializationStatistics(state);

		boolean output = (state.generation % modulus == 0);

		// Make the job id and put it as the first line of the stat
		CudaEvolutionState cuState = (CudaEvolutionState) state;

		if (output)
			if (output)
				state.output.println(cuState.getActiveJob().toString(), statisticslog);

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
