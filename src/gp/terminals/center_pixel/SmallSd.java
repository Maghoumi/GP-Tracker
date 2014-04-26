package gp.terminals.center_pixel;

import cuda.gp.CudaNode;

public class SmallSd extends CudaNode
{
	@Override
	public String toString()
	{
		return "SmallSd";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);\n" +
				"if (top == 0.0) push(smallSd[tid].y/255.0);\n" +
				"else if (top == 1.0) push (smallSd[tid].z/255.0);\n" +
				"else if (top == 2.0) push (smallSd[tid].w/255.0);\n" +
				"else if (top == 3.0) {float4 value = smallSd[tid];\n" +
				"push ((value.y + value.z + value.w)/( 3 * 255.0));}\n" ;
	}
}
