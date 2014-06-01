package utils.cuda.pointers;

import jcuda.Pointer;
import jcuda.Sizeof;
import static jcuda.driver.JCudaDriver.*;

/**
 * Represents a primitive long that is synchronized with a long pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaLong extends CudaPrimitive {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected long longValue = 0;
	
	public CudaLong() {
		this(0);
	}
	
	public CudaLong (long initialValue) {
		this.longValue = initialValue;
		cuMemAlloc(this, getSizeInBytes());
	}
	
	/**
	 * @return	The cached value of the variable pointed by this pointer.
	 * 			Again, note that this is a cached value!
	 */
	public long getValue() {
		return this.longValue;
	}
	
	/**
	 * Set the value of this pointer to a new value
	 * @param newValue	The new value to be represented by the memory space of this pointer
	 * @return	JCuda's error code
	 */
	public int setValue(long newValue) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		this.longValue = newValue;
		return cuMemcpyHtoD(this, Pointer.to(new long[] {longValue}), getSizeInBytes());
	}

	@Override
	public int getSizeInBytes() {
		return Sizeof.LONG;
	}

	@Override
	public int refresh() {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		long[] newValue = new long[1];
		int errCode = cuMemcpyDtoH(Pointer.to(newValue), this, getSizeInBytes());  
		this.longValue = newValue[0];
		return errCode;
	}
	
	@Override
	public int reallocate() {
		if (!freed)
			return 0;
		
		freed = false;		
		cuMemAlloc(this, getSizeInBytes());
		return setValue(this.longValue);
	}

	@Override
	public Object clone() {
		return new CudaLong(longValue);
	}

	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(new long[] {this.longValue});
	}
}
