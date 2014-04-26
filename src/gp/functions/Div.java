package gp.functions;

import cuda.gp.CudaNode;

public class Div extends CudaNode
{
	@Override
	public String toString()
	{
		return "/";
	}

	@Override
	public int getNumberOfChildren() {
		return 2;
	}

	@Override
	public String getCudaAction() {
		return 	"float second; pop(second);" +
				"float first; pop(first);" +
				"if (second == 0.0) push(1.0);" +
				"else push(first / second);";
	}
}
