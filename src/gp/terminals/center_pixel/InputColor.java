package gp.terminals.center_pixel;

import cuda.gp.CudaNode;

public class InputColor extends CudaNode
{
	@Override
	public String toString()
	{
		return "InputColor";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	//FIXME should define a constant somewhere so that I would not have to write
	// these craps here.
	// a separate file is required that would define all possible CUDA operations.
	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"if (top == 0.0) push(input[tid].y/255.0);" +
				"else if (top == 1.0) push (input[tid].z/255.0);" +
				"else if (top == 2.0) push (input[tid].w/255.0);" +
				"else if (top == 3.0) {float4 value = input[tid];" +
				"push ((value.y + value.z + value.w)/( 3 * 255.0));}" ;
	}
}
