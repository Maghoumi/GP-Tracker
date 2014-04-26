package gp.functions;

import cuda.gp.CudaNode;

public class Max extends CudaNode
{
	@Override
	public String toString()
	{
		return "max";
	}

	@Override
	public int getNumberOfChildren() {
		return 2;
	}

	@Override
	public String getCudaAction() {
		return 	"float second; pop(second);" +
				"float first; pop(first);" +
				"if (first > second) push(first); else push(second);";
	}
}
