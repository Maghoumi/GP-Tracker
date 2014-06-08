package utils;

import com.sir_m2x.transscale.pointers.*;

/**
 * Represents classes that are able to filter images using CUDA. This
 * interface encapsulates CUDA image filtering. We don't care how our
 * images are filtered, we only need to be able to give it the necessary
 * data and the filter sizes and the filtering should be done automatically.
 * 
 * @author Mehran Maghoumi
 * 
 */
public interface ImageFilterProvider {
	
	/**
	 * Performs the averge and standard deviation filters on the provided input and
	 * stores it in the passed CudaPrimitive2D objects.
	 * 
	 * @param byteInput	The input image to be filtered
	 * @param smallAvg	Placeholder to store the result of small average filter
	 * @param mediumAvg	Placeholder to store the result of medium average filter
	 * @param largeAvg	Placeholder to store the result of large average filter
	 * @param smallSd	Placeholder to store the result of small standard deviation filter
	 * @param mediumSd	Placeholder to store the result of medium standard deviation filter
	 * @param largeSd	Placeholder to store the result of large standard deviation filter
	 */
	public void performFilters(CudaByte2D byteInput,
								CudaFloat2D smallAvg, CudaFloat2D mediumAvg, CudaFloat2D largeAvg,
								CudaFloat2D smallSd, CudaFloat2D mediumSd, CudaFloat2D largeSd);
	
	/**
	 * @return	The size of the small filter
	 */
	public int getSmallFilterSize();
	
	/**
	 * @return	The size of the medium filter
	 */
	public int getMediumFilterSize();
	
	/**
	 * @return	The size of the large filter
	 */
	public int getLargeFilterSize();
}
