package gp.functions;

import cuda.gp.CudaNode;

public class Mul extends CudaNode
{
	@Override
	public String toString()
	{
		return "*";
	}

	@Override
	public int getNumberOfChildren() {
		return 2;
	}


	@Override
	public String getCudaAction() {
		return 	"float second; pop(second);" +
				"float first; pop(first);" +
				"push(first * second);";
	}
}
