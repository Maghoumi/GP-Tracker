package gp.functions;

import cuda.gp.CudaNode;
import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import ec.util.Parameter;
import gp.datatypes.ProblemData;

public class Max extends CudaNode
{
	@Override
	public String toString()
	{
		return "max";
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
		float result;
		
		ProblemData rd = ((ProblemData) (input));

		children[0].eval(state, thread, input, stack, individual, problem);
		result = rd.value;

		children[1].eval(state, thread, input, stack, individual, problem);
		
		rd.value = Math.max(rd.value, result);
	}

	@Override
	public int getNumberOfChildren() {
		return 2;
	}

	@Override
	public String getCudaAction() {
		return 	"float second; pop(second);" +
				"float first; pop(first);" +
				"if (first > second) push(first); else push(second);";
	}
}
