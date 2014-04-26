package gp.functions;

import cuda.gp.CudaNode;

public class Neg extends CudaNode
{
	@Override
	public String toString()
	{
		return "-";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float first; pop(first);" +
				"push(-first);";
	}
}
