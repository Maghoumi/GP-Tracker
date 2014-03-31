package utils.cuda.datatypes.pointers;

import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

/**
 * A wrapper for JCuda's CUdeviceptr class. The objects of this class represent 1D 
 * CUDA pointers and offer methods for obtaining their values or copying new values
 * to the device memory.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaPointer extends CUdeviceptr {
	
	/**
	 * Frees the allocated memory
	 */
	public void free() {
		cuMemFree(this);
	}
	
}
