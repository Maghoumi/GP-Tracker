package gp.terminals.channels;

import cuda.gp.CudaNode;

public class AttrRed extends CudaNode
{
	
	@Override
	public String toString()
	{
		return "R";
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}

	@Override
	public String getCudaAction() {
		return "push(0.0);";
	}
}