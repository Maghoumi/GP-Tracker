package cuda.gp;

import java.io.File;
import java.io.IOException;

import ec.EvolutionState;
import ec.Individual;
import ec.Statistics;
import ec.gp.GPIndividual;
import ec.simple.SimpleProblemForm;
import ec.simple.SimpleStatistics;
import ec.util.Output;
import ec.util.Parameter;

/**
 * An extension of the ECJ's SimpleStatistics class with the only difference
 * begin the passing of the best individual tree to the visualizer.
 * 
 * In case the visualizer is enabled, the best individual is flattened and
 * passed to the visualizer for visualization. *
 * 
 * @author Mehran Maghoumi
 */
public class CudaSimpleStatistics extends SimpleStatistics {

	private CudaEvolutionState cuState;

	@Override
	public void setup(EvolutionState state, Parameter base) {
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

		compress = state.parameters.getBoolean(base.push(P_COMPRESS), null, false);

		File statisticsFile = state.parameters.getFile(base.push(P_STATISTICS_FILE), null);
		this.cuState = (CudaEvolutionState) state;
		statisticsFile = new File(statisticsFile.getParentFile().getAbsolutePath() + "\\" + cuState.getSessionPrefix() + statisticsFile.getName());

		doFinal = state.parameters.getBoolean(base.push(P_DO_FINAL), null, true);
		doGeneration = state.parameters.getBoolean(base.push(P_DO_GENERATION), null, true);
		doMessage = state.parameters.getBoolean(base.push(P_DO_MESSAGE), null, true);
		doDescription = state.parameters.getBoolean(base.push(P_DO_DESCRIPTION), null, true);
		doPerGenerationDescription = state.parameters.getBoolean(base.push(P_DO_PER_GENERATION_DESCRIPTION), null, false);

		muzzle = state.parameters.getBoolean(base.push(P_MUZZLE), null, false);

		if (muzzle)
		{
			statisticslog = Output.NO_LOGS;
		}
		else if (statisticsFile != null)
			try
			{
				statisticslog = state.output.addLog(statisticsFile, !compress, compress);
			} catch (IOException i)
			{
				state.output.fatal("An IOException occurred while trying to create the log " + statisticsFile + ":\n" + i);
			}
	}

	// The exact same as super but the best individual is flattened and passed to the visualizer 
	public void postEvaluationStatistics(final EvolutionState state) {
		// super.super.postEvaluationStatistics
		for (int x = 0; x < children.length; x++)
			children[x].postEvaluationStatistics(state);

		// for now we just print the best fitness per subpopulation.
		Individual[] best_i = new Individual[state.population.subpops.length]; // quiets
																				// compiler
																				// complaints
		for (int x = 0; x < state.population.subpops.length; x++) {
			best_i[x] = state.population.subpops[x].individuals[0];
			for (int y = 1; y < state.population.subpops[x].individuals.length; y++)
				if (state.population.subpops[x].individuals[y].fitness.betterThan(best_i[x].fitness))
					best_i[x] = state.population.subpops[x].individuals[y];

			// now test to see if it's the new best_of_run
			if (best_of_run[x] == null || best_i[x].fitness.betterThan(best_of_run[x].fitness)) {
				best_of_run[x] = (Individual) (best_i[x].clone());

				GPIndividual gpInd = (GPIndividual) best_of_run[x];
				// Report the best individual to the CudaEvolutionState object
				cuState.reportIndividual(gpInd);
			}
		}

		// print the best-of-generation individual
		if (doGeneration)
			state.output.println("\nGeneration: " + state.generation, statisticslog);
		if (doGeneration)
			state.output.println("Best Individual:", statisticslog);
		for (int x = 0; x < state.population.subpops.length; x++) {
			if (doGeneration)
				state.output.println("Subpopulation " + x + ":", statisticslog);
			if (doGeneration)
				best_i[x].printIndividualForHumans(state, statisticslog);
			if (doMessage)
				state.output.message("Subpop " + x + " best fitness of generation" + (best_i[x].evaluated ? " " : " (evaluated flag not set): ") + best_i[x].fitness.fitnessToStringForHumans());

			// describe the winner if there is a description
			if (doGeneration && doPerGenerationDescription) {
				if (state.evaluator.p_problem instanceof SimpleProblemForm)
					((SimpleProblemForm) (state.evaluator.p_problem.clone())).describe(state, best_i[x], x, 0, statisticslog);
			}
		}
	}

}
