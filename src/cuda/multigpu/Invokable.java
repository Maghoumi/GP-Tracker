package cuda.multigpu;

import jcuda.driver.CUcontext;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

/**
 * Defines the elements that are required to make a kernel function
 * invokable in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class Invokable { 
	/** The context that contains the function to be invoked */
	public CUcontext context;
	
	/** The module that this function is being invoked on */
	public CUmodule module;
	
	/** The CUDA kernel that could be invoked */
	public CUfunction function;
	
	public Invokable(CUcontext context, CUmodule module, CUfunction function) {
		this.context = context;
		this.module = module;
		this.function = function;
	}
}
