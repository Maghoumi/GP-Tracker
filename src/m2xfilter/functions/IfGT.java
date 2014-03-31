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

public class IfGT extends CudaNode
{
	@Override
	public String toString()
	{
		return "IfGT";
	}

	@Override
	public void checkConstraints(final EvolutionState state, final int tree, final GPIndividual typicalIndividual, final Parameter individualBase)
	{
		super.checkConstraints(state, tree, typicalIndividual, individualBase);
		if (children.length != 4)
			state.output.error("Incorrect number of children for node " + toStringForError() + " at " + individualBase);
	}

	@Override
	public void eval(final EvolutionState state, final int thread, final GPData input, final ADFStack stack, final GPIndividual individual,
			final Problem problem)
	{
		float result0, result1;
		ProblemData rd = ((ProblemData) (input));

		children[0].eval(state, thread, input, stack, individual, problem);
		result0 = rd.value;

		children[1].eval(state, thread, input, stack, individual, problem);
		result1 = rd.value;

		if (result0 > result1)
			children[2].eval(state, thread, input, stack, individual, problem);
		else
			children[3].eval(state, thread, input, stack, individual, problem);
	}

	@Override
	public int getNumberOfChildren() {
		return 4;
	}

	@Override
	public String getCudaAction() {
		return 	"float forth; pop(forth);" +
				"float third; pop(third);" +
				"float second; pop(second);" +
				"float first; pop(first);" +
				"if (first > second) push(third); else push(forth);";
	}
}
