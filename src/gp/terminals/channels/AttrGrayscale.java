package gp.terminals.channels;

import cuda.gp.CudaNode;
import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import ec.util.Parameter;
import gp.datatypes.ProblemData;

public class AttrGrayscale extends CudaNode
{
	
	@Override
	public String toString()
	{
		return "BW";
	}

	@Override
	public void checkConstraints(final EvolutionState state, final int tree, final GPIndividual typicalIndividual, final Parameter individualBase)
	{
		super.checkConstraints(state, tree, typicalIndividual, individualBase);
		if (children.length != 0)
			state.output.error("Incorrect number of children for node " + toStringForError() + " at " + individualBase);
	}

	@Override
	public void eval(final EvolutionState state, final int thread, final GPData input, final ADFStack stack, final GPIndividual individual,
			final Problem problem)
	{
		ProblemData rd = ((ProblemData) (input));
		rd.value = 3;
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}


	@Override
	public String getCudaAction() {
		return "push(3.0);";
	}
}