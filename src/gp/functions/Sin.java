package gp.functions;

import cuda.gp.CudaNode;

public class Sin extends CudaNode
{
	@Override
	public String toString()
	{
		return "sin";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"push(sin(top));";
	}
}
