package utils.cuda.datatypes.pointers;

import static jcuda.driver.JCudaDriver.*;
import jcuda.Pointer;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUmemorytype;


public abstract class CudaPrimitive2D extends CudaPrimitive {
	
	/** The pitch of the memory pointed by this pointer */
	protected long[] pitch = new long[] {0};
	
	/**
	 * The number of fields in each of the memory locations that this pointer points to.
	 * For storing structures such as color, image etc. each memory can have multiple fields.
	 */
	protected int numFields;
	
	/** The width of the 2D array pointed by this pointer*/
	protected int width;
	
	/** The height of the 2D array pointed by this pointer */
	protected int height;
	
	/**
	 * Creates and instance of this class for the specified number of fields and
	 * the specified width and height
	 * 
	 * @param width	The width of the 2D array
	 * @param height	The height of the 2D array
	 * @param numFields	How many fields are there per allocation?
	 */
	public CudaPrimitive2D (int width, int height, int numFields) {
		this.width = width;
		this.height = height;
		this.numFields = numFields;		
	}
	
	/**
	 * Creates and instance of this class for the specified width and height.
	 * The number of fields will default to 1.
	 * 
	 * @param width	The width of the 2D array
	 * @param height	The height of the 2D array
	 * @param numFields	How many fields are there per allocation?
	 */
	public CudaPrimitive2D (int width, int height) {
		this(width, height, 1);
	}
	
	/**
	 * Allocates the GPU memory required for this Primitive2D object. This method
	 * will decide if the allocation needs to be pitched or non-pitched
	 * 
	 * @return JCuda's error code
	 */
	protected int allocate() {
		int recordSizeInBytes = getElementSizeInBytes() * numFields;
		
		// Check to see if we can allocate using pitched memory
		if (recordSizeInBytes == 4 || recordSizeInBytes == 8 || recordSizeInBytes == 16)
			return allocatePitched();
		else
			return allocateNonPitched();
	}
	
	/**
	 * Allocates GPU pitched memory required for this Primitive2D object 
	 * @return JCuda's error code
	 */
	protected int allocatePitched() {
		return cuMemAllocPitch(this, this.pitch, width * numFields * getElementSizeInBytes(), height, getElementSizeInBytes() * numFields);
	}
	
	/**
	 * Allocates GPU non-pitched memory required for this Primitive2D object
	 * @return JCuda's error code
	 */
	protected int allocateNonPitched() {
		int byteCount = width * numFields * height * getElementSizeInBytes();
		int errCode = cuMemAlloc(this, byteCount);
		this.pitch[0] = width * numFields * getElementSizeInBytes(); // Fake pitch to preserve compatibility
		
		return errCode;
	}
	
	/**
	 * Transfers the values of the array variable of this object to GPU memory.
	 * This method will decide to do a pitched or non-pitched transfer.
	 * @return JCuda's error code 
	 */
	protected int upload() {
		int recordSizeInBytes = getElementSizeInBytes() * numFields;
		
		// Check to see if we can transfer using pitched memory
		if (recordSizeInBytes == 4 || recordSizeInBytes == 8 || recordSizeInBytes == 16)
			return uploadPitched();
		else
			return uploadNonPitched();
	}
	
	/**
	 * Transfer's this object's array to the GPU pitched memory
	 * @return JCuda's error code
	 */
	protected int uploadPitched() {
		CUDA_MEMCPY2D copyParam = new CUDA_MEMCPY2D();
		copyParam.srcHost = hostDataToPointer();
        copyParam.srcPitch = width * numFields * getElementSizeInBytes();
        copyParam.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        
        copyParam.dstDevice = this;
        copyParam.dstPitch = pitch[0];
        copyParam.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        
        copyParam.WidthInBytes = width * numFields * getElementSizeInBytes();
        copyParam.Height = height;
        
        return cuMemcpy2D(copyParam);
	}
	
	/**
	 * Transfer's this object's array to the GPU non-pitched memory
	 * @return JCuda's error code
	 */
	protected int uploadNonPitched() {
		int byteCount = width * numFields * height * getElementSizeInBytes();
		return cuMemcpyHtoD(this, hostDataToPointer(), byteCount);
	}
	
	/**
	 * Transfers the values of the GPU memory to the array of this object.
	 * This method will decide to do a pitched or non-pitched transfer.
	 * @return JCuda's error code 
	 */
	protected int download() {
		int recordSizeInBytes = getElementSizeInBytes() * numFields;
		
		// Check to see if we can transfer using pitched memory
		if (recordSizeInBytes == 4 || recordSizeInBytes == 8 || recordSizeInBytes == 16)
			return downloadPitched();
		else
			return downloadNonPitched();
	}
	
	/**
	 * Transfer's the contents of GPU pitched memory to this object's array
	 * @return JCuda's error code
	 */
	protected int downloadPitched() {
		CUDA_MEMCPY2D copyParam = new CUDA_MEMCPY2D();
		copyParam.srcDevice = this;
        copyParam.srcPitch = getPitch()[0];
        copyParam.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        
        copyParam.dstHost = hostDataToPointer();
        copyParam.dstPitch = width * numFields * getElementSizeInBytes();
        copyParam.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        
        copyParam.WidthInBytes = width * numFields * getElementSizeInBytes();
        copyParam.Height = height;
        
        return cuMemcpy2D(copyParam);
	}
	
	/**
	 * Transfer's the contents of GPU non-pitched memory to this object's array
	 * @return JCuda's error code
	 */
	protected int downloadNonPitched() {
		int byteCount = width * numFields * height * getElementSizeInBytes();
		return cuMemcpyDtoH(hostDataToPointer(), this, byteCount);
	}
	
	@Override
	public int refresh() {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		return download();
	}
	
	@Override
	public int reallocate() {
		if (!freed)
			return 0;
		
		freed = false;		
		int allocResult = allocate();
		int uploadResult = upload();
		
		return Math.max(allocResult, uploadResult);
	}
	
	/**
	 * @return	The pitch of the allocated memory as returned by cuMemAllocPitch()
	 */
	public long[] getPitch() {
		return this.pitch.clone();
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
		return new long[] {this.pitch[0] / (numFields * getElementSizeInBytes())};
	}
	
	/**
	 * @return	The size of the building blocks of this array in bytes
	 * 			i.e. if this is an array of integers, the element size
	 * 			has to be Sizeof.INTEGER
	 */
	public abstract int getElementSizeInBytes();
}
