package gp.terminals.center_pixel;

import java.awt.Color;

import cuda.gp.CudaNode;
import utils.cuda.datatypes.Float4;
import ec.EvolutionState;
import ec.Problem;
import ec.gp.ADFStack;
import ec.gp.GPData;
import ec.gp.GPIndividual;
import ec.gp.GPNode;
import ec.util.Parameter;
import gp.datatypes.ProblemData;

public class MediumAvg extends CudaNode
{
	@Override
	public String toString()
	{
		return "MediumAvg";
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
		float result;
		
		ProblemData rd = ((ProblemData) (input));

		children[0].eval(state, thread, input, stack, individual, problem);
		result = rd.value;
		
		Float4 feature = rd.instance.mediumAvg;
		
		switch((int)result) {
			case 0:
				rd.value = feature.y / 255.0f;
				break;
			
			case 1:
				rd.value = feature.z / 255.0f;
				break;
				
			case 2:
				rd.value = feature.w / 255.0f;
				break;
				
			case 3:
				rd.value = (feature.y + feature.z + feature.w) / (3*255.0f);
				break;
			
			default:
				state.output.fatal("MAVGValue was " + ((int) result) + " Original: " + result + " Constraints not met :(");
		}
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}


	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"if (top == 0.0) push(mediumAvg[tid].y/255.0);" +
				"else if (top == 1.0) push (mediumAvg[tid].z/255.0);" +
				"else if (top == 2.0) push (mediumAvg[tid].w/255.0);" +
				"else if (top == 3.0) {float4 value = mediumAvg[tid];" +
				"push ((value.y + value.z + value.w)/( 3 * 255.0));}" ;
	}
}
