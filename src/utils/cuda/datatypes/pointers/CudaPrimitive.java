package utils.cuda.datatypes.pointers;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import static jcuda.driver.JCudaDriver.*;

/**
 * An extension of CUdeviceptr that represents a primitive variable in CUDA with
 * an equivalent variable in Java. The values are synched between the GPU and the
 * CPU.  
 * 
 * @author Mehran Maghoumi
 *
 */
public abstract class CudaPrimitive extends CUdeviceptr implements Cloneable {
	
	/**
	 * A flag indicating that this pointer has been freed and subsequent operations
	 * cannot be performed unless this pointer has been reallocated on the GPU memory
	 */
	protected boolean freed = false;
	
	/**
	 * @return	The size of this pointer in bytes
	 */
	public abstract int getSizeInBytes();
	
	/**
	 * Fetch the current value of this pointer from the GPU memory
	 * @return	The error code returned by JCuda
	 */
	public abstract int refresh();
	
	/** Reallocates this pointer if it was freed */
	public abstract int reallocate();
	
	/**
	 * @return	True if this pointer is freed from the GPU memory,
	 * 			False otherwise
	 */
	public boolean isFreed() {
		return this.isFreed();
	}
	
	/**
	 * Frees the allocated memory but keeps the host data intact
	 * WARNING: Do not use cuMemFree on this object directly!
	 */
	public void free() {
		if (!freed)
			cuMemFree(this);
		
		freed = true;
	}
	
	/**
	 * Creates a pointer to the GPU pointer of this object
	 * @return The native pointer to pointer to GPU memory space
	 */
	public Pointer toPointer() {
		return Pointer.to(this);
	}
	
	/**
	 * Convert's the underlying host data of this object to a Pointer
	 */
	public abstract Pointer hostDataToPointer();
	
	@Override
	protected abstract Object clone();
	
}
