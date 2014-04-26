package gp.terminals.channels;

import cuda.gp.CudaNode;

public class AttrGrayscale extends CudaNode
{
	
	@Override
	public String toString()
	{
		return "BW";
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}


	@Override
	public String getCudaAction() {
		return "push(3.0);";
	}
}