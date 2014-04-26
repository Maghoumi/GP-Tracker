package gp.functions;

import cuda.gp.CudaNode;

public class IfGT extends CudaNode
{
	@Override
	public String toString()
	{
		return "IfGT";
	}

	@Override
	public int getNumberOfChildren() {
		return 4;
	}

	@Override
	public String getCudaAction() {
		return 	"float forth; pop(forth);" +
				"float third; pop(third);" +
				"float second; pop(second);" +
				"float first; pop(first);" +
				"if (first > second) push(third); else push(forth);";
	}
}
