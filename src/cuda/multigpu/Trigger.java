package cuda.multigpu;

import jcuda.driver.CUmodule;

/**
 * Defines the operations that should be done before or after a CUDA kernel
 * call.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface Trigger {
	/**
	 * Called when a CUDA kernel is about to happen or has already happened
	 * The framework that calls this function must also provide some CUDA related
	 * environment object. Such as the module that the code is working on so that in case
	 * the clients want to make API calls that require specific CUDA environment objects 
	 * they can. 
	 * @param module	The CUmodule object that this trigger is called on
	 */
	public void doTask(CUmodule module);
}
