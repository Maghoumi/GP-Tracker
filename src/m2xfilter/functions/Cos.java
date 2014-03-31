package m2xfilter.functions;

import cuda.gp.CudaNode;
import m2xfilter.datatypes.ProblemData;
import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import ec.util.Parameter;

public class Cos extends CudaNode
{
	@Override
	public String toString()
	{
		return "cos";
	}

	@Override
	public void checkConstraints(final EvolutionState state, final int tree, final GPIndividual typicalIndividual, final Parameter individualBase)
	{
		super.checkConstraints(state, tree, typicalIndividual, individualBase);
		if (children.length != 1)
			state.output.error("Incorrect number of children for node " + toStringForError() + " at " + individualBase);
	}

	@Override
	public void eval(final EvolutionState state, final int thread, final GPData input, final ADFStack stack, final GPIndividual individual,
			final Problem problem)
	{
		ProblemData rd = ((ProblemData) (input));

		children[0].eval(state, thread, input, stack, individual, problem);
		rd.value = (float) Math.cos(rd.value);
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"push(cos(top));";
	}
}
