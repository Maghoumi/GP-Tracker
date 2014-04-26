package gp.functions;

import cuda.gp.CudaNode;

public class Log extends CudaNode
{
	@Override
	public String toString()
	{
		return "Log";
	}

	@Override
	public int getNumberOfChildren() {
		return 1;
	}

	@Override
	public String getCudaAction() {
		return 	"float top; pop(top);" +
				"push(logf(fabs(top)));";
	}
}
