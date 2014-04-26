package gp.terminals.center_pixel;


import cuda.gp.CudaNode;

public class MediumAvg extends CudaNode
{
	@Override
	public String toString()
	{
		return "MediumAvg";
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
