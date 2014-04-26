package gp.terminals.channels;

import cuda.gp.CudaNode;

public class AttrBlue extends CudaNode
{
	
	@Override
	public String toString()
	{
		return "B";
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}

	@Override
	public String getCudaAction() {
		return "push(2.0);";
	}
}