package gp.functions;

import cuda.gp.CudaNode;

public class Cos extends CudaNode
{
	@Override
	public String toString()
	{
		return "cos";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"push(cos(top));";
	}
}
