package gp.terminals.channels;

import cuda.gp.CudaNode;

public class AttrGreen extends CudaNode
{
	
	@Override
	public String toString()
	{
		return "G";
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}

	@Override
	public String getCudaAction() {
		return "push(1.0);";
	}
}