package utils.cuda.pointers;

import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Represents a primitive array of longs which is synchronized with a long pointer
 * in CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaLong2D extends CudaPrimitive2D {
	
	/** The CPU value of this GPU pointer. This is the cached value */
	protected long[] array;
	
	public CudaLong2D (int width, int height) {
		this(width, height, 1);
	}
	
	public CudaLong2D (int width, int height, int numFields) {
		this(width, height, numFields, null);
	}
	
	/**
	 * Initializes an object of this class using the specified width, height, numFileds
	 * and the passed initialValues. If initialValues is null, an array will be created.
	 * Note that the passed initialValues is cloned and a separate copy is held for internal
	 * use of this object.
	 * 
	 * @param width
	 * @param height
	 * @param numFields
	 * @param initialValues
	 */
	public CudaLong2D (int width, int height, int numFields, long[] initialValues) {
		super(width, height, numFields);
		
		// Initialize the host array
		if (initialValues == null)
			this.array = new long[width * height * numFields];
		else {
			if (initialValues.length != width * height * numFields)
				throw new RuntimeException("Given array's length is different than specified specifications");
			
			this.array = initialValues.clone();
		}
		
		allocate();
		upload();
	}
	
	/**
	 * Obtains the cached value that resides in the specified coordinates of the 
	 * memory pointed by this pointer. Since an array can have more than 1 field,
	 * this method returns an array with the length equal to the size of numFileds.
	 * Again, note that this is a cached value! To obtain a fresh value, call refresh()
	 * before calling this method. 
	 *  
	 * @param x	The column index of the matrix
	 * @param y	The row index of the matrix
	 * @return
	 */
	public long[] getValueAt (int x, int y) {
		if (x >= width)
			throw new IndexOutOfBoundsException("Column index out of bounds");
		
		if (y >= height)
			throw new IndexOutOfBoundsException("Row index out of bounds");
		
		long[] result = new long[numFields]; 
		// Determine the start index
		int startIndex = y * width * numFields + x * numFields;
		// Perform copy
		System.arraycopy(array, startIndex, result, 0, numFields);
		
		return result;
	}
	
	/**
	 * @return	A copy (i.e. a clone) of the underlying array of this Long2D object
	 */
	public long[] getArray() {
		return this.array.clone();
	}
	
	/**
	 * @return	The underlying array of this Long2D object
	 * 			WARNING: Do not modify this array directly! Use getArray()
	 * 					 if you need to modify the returned array!
	 * 
	 */
	public long[] getUnclonedArray() {
		return this.array;
	}
	
	/**
	 * Sets the array of this Long2D object to the specified array.
	 * The new array must meet the original specifications (i.e. same width, height etc.)
	 * After the array is set, the new values are automatically written back to the GPU
	 * memory.
	 * Note that the passed array is cloned and a separate copy of the passed array is
	 * maintained for internal use.
	 * 
	 * @param newArray
	 * @return JCuda's error code
	 */
	public int setArray(long[] newArray) {
		if (freed)
			throw new RuntimeException("The pointer has already been freed");
		
		if (newArray.length != width * height * numFields)
			throw new RuntimeException("Given array's length is different than specified specifications");
		
		this.array = newArray.clone();
		return upload();
	}

	@Override
	public int getElementSizeInBytes() {
		return Sizeof.LONG;
	}

	@Override
	public int getSizeInBytes() {
		return width * height * numFields * getElementSizeInBytes();
	}
	
	@Override
	public Pointer hostDataToPointer() {
		return Pointer.to(this.array);
	}

	@Override
	protected Object clone() {
		return new CudaLong2D(width, height, numFields, array);
	}
}
