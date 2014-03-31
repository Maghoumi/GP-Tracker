package utils.cuda.datatypes;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmemorytype;
import static jcuda.driver.JCudaDriver.*;

/**
 * Represents a 2D device pointer in JCuda. This pointer uses pitched
 * memory for the allocations.
 * 
 * @author Mehran Maghoumi
 * 
 */
public class CUdeviceptr2D extends CUdeviceptr {
	
	/** The pitch of the memory pointed by this pointer */
	private long[] pitch = new long[] {0};
	
	/**
	 * The number of fields in each of the memory locations that this pointer points to.
	 * For storing structures such as color, image etc. each memory can have multiple fields.
	 */
	private int numFields;
	
	/**
	 * Defines the size of the most basic element that this pointer points to. In other words,
	 * are the elements byte, int, float or something else?
	 */
	private int elemSize; 
	
	/** The width of the 2D array pointed by this pointer*/
	private int width;
	
	/** The height of the 2D array pointed by this pointer */
	private int height;
	
	/**
	 * Creates and instance of this class using the specified number of fields and the 
	 * specified element size.
	 * 
	 * @param width	The width of the 2D array
	 * @param height	The height of the 2D array
	 * @param numFields	How many fields are there per allocation?
	 * @param elemSize	How large is each field (in bytes)
	 */
	public CUdeviceptr2D(int width, int height, int numFields, int elemSize) {
		super();
		this.width = width;
		this.height = height;
		this.numFields = numFields;
		this.elemSize = elemSize;		
	}	
	
	/**
	 * Allocates and transfers a byte array to the GPU
	 * 
	 * @param input	The byte array to be allocated
	 */
	public void allocTransByte(byte[] input) {
		int elementSizeBytes = elemSize * numFields;
		// Check to see if we can allocate using pitched memory
		if (elementSizeBytes == 4 || elementSizeBytes == 8 || elementSizeBytes == 16) {
			allocTransBytePitch(input);
		}
		else {
			allocTransByteNonPitch(input);
		}
	}
	
	/**
	 * Allocates and transfers a byte array on the GPU memory using a non-pitched
	 * memory. This is a helper function for the cases that the data structure at
	 * hand is not compatible with cuMemAllocPitch (eg. does not meet alignment 
	 * requirements etc.)
	 * 
	 * @param input	The byte array to be allocated on the device memory
	 */
	private void allocTransByteNonPitch(byte[] input) {
		int byteCount = width * numFields * height * elemSize;
		cuMemAlloc(this, byteCount);
		cuMemcpyHtoD(this, Pointer.to(input), byteCount);
		this.pitch[0] = width * numFields * elemSize; // Fake pitch to preserve compatibility
	}

	/**
	 * Allocates and transfers a byte array on the GPU memory using pitched memory.
	 * This function should only be called if the data structure at hand meets
	 * cuMemAllocPitch's alignment requirements.
	 * 
	 * @param input	The bye array to be allocated on the device memory
	 */
	private void allocTransBytePitch(byte[] input) {
		cuMemAllocPitch(this, this.pitch, width * numFields * elemSize, height, elemSize * numFields);
		
		CUDA_MEMCPY2D copyParam = new CUDA_MEMCPY2D();
		copyParam.srcHost = Pointer.to(input);
        copyParam.srcPitch = width * numFields * elemSize;
        copyParam.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
                
        copyParam.dstDevice = this;
        copyParam.dstPitch = pitch[0];
        copyParam.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        
        copyParam.WidthInBytes = width * numFields * elemSize;
        copyParam.Height = height;
        
        cuMemcpy2D(copyParam);		
	}

	/**
	 * Retrieves the 2D memory pointed by this pointer
	 * @return
	 */
	public byte[] retrieveByte() {
		if (true)
			throw new RuntimeException("Yo buddy! This is shit! Not implemented properly");
		byte[] result = new byte[width * numFields * height];
		
        CUDA_MEMCPY2D copyParam = new CUDA_MEMCPY2D();
        copyParam.srcDevice = this;
        copyParam.srcPitch = this.pitch[0];
        copyParam.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        
        copyParam.dstHost = Pointer.to(result);
        copyParam.dstPitch = width * numFields * elemSize;
        copyParam.WidthInBytes = width * numFields * 4;
        copyParam.Height = height;
        copyParam.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        
        cuMemcpy2D(copyParam);
		
		return result;
	}
	
	/**
	 * The original pitch of the allocated memory as returned by cuMemAllocPitch()
	 * @return
	 */
	public long[] getPitch() {
		return this.pitch;
	}
	
	/**
	 * The width of the 2D array pointed by this pointer
	 * @return
	 */
	public int getWidth() {
		return this.width;
	}
	
	/**
	 * The height of the 2D array pointed by this pointer
	 * @return
	 */
	public int getHeight() {
		return this.height;
	}
	
	/**
	 * @return Returns a convenient pitch value to use for kernel calls. This pitch is the
	 * original pitch divided by the element size; i.e. (pitch / (elemSize * numFields)).
	 * Without this pitch, you'd have to use the cumbersome pointer casting in the CUDA kernel
	 * to access memory locations pointed by this pointer.
	 */
	public long[] getPitchInElements() {
		return new long[] {this.pitch[0] / (numFields * elemSize)};
	}
	
	/**
	 * Frees the allocated space that this pointer points to
	 */
	public void free() {
		cuMemFree(this);
	}
}
