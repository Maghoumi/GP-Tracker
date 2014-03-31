package m2xfilter.terminals.channels;

import cuda.gp.CudaNode;
import m2xfilter.datatypes.ProblemData;
import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.util.Parameter;

public class AttrGreen extends CudaNode
{
	
	@Override
	public String toString()
	{
		return "G";
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
		rd.value = 1;
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}

	@Override
	public String getCudaAction() {
		return "push(1.0);";
	}
}