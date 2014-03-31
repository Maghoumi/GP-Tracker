package m2xfilter.functions;

import cuda.gp.CudaNode;
import m2xfilter.datatypes.ProblemData;
import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.util.Parameter;

public class Add extends CudaNode
{
	@Override
	public String toString()
	{
		return "+";
	}

	@Override
	public void checkConstraints(final EvolutionState state, final int tree, final GPIndividual typicalIndividual, final Parameter individualBase)
	{
		super.checkConstraints(state, tree, typicalIndividual, individualBase);
		if (children.length != 2)
			state.output.error("Incorrect number of children for node " + toStringForError() + " at " + individualBase);
	}

	@Override
	public void eval(final EvolutionState state, final int thread, final GPData input, final ADFStack stack, final GPIndividual individual,
			final Problem problem)
	{
		double result;
		
		ProblemData rd = ((ProblemData) (input));

		children[0].eval(state, thread, input, stack, individual, problem);
		result = rd.value;

		children[1].eval(state, thread, input, stack, individual, problem);
		
		rd.value += result;
	}

	@Override
	public int getNumberOfChildren() {
		return 2;
	}

	@Override
	public String getCudaAction() {
		return 	"float second; pop(second);" +
				"float first; pop(first);" +
				"push(first + second);";
	}
}
