package cuda.gp;

import ec.EvolutionState;
import ec.Individual;
import ec.gp.GPIndividual;
import ec.simple.SimpleProblemForm;
import ec.simple.SimpleStatistics;
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
		super.setup(state, base);

		this.cuState = (CudaEvolutionState) state;
	}

	// The exact same as super but the best individual is flattened and passed to the visualizer 
	public void postEvaluationStatistics(final EvolutionState state) {
		// super.super.postEvaluationStatistics
		for(int x=0;x<children.length;x++)
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
