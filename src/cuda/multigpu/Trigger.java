package cuda.multigpu;

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
	 */
	public void doTask();
}
