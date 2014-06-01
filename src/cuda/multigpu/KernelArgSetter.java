package cuda.multigpu;

import jcuda.Pointer;

/**
 * Defines the interface for setting kernel arguments on the
 * GPU interop thread.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface KernelArgSetter {
	public Pointer getArgs();
}
