package gp.functions;

import cuda.gp.CudaNode;

public class Exp extends CudaNode
{
	@Override
	public String toString()
	{
		return "exp";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"push(exp(top));";
	}
}
