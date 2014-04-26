package utils;

import utils.cuda.pointers.*;

/**
 * Represents a ByteImage that has been prepared and has been filtered
 * using CUDA's image filters. A prepared image has all filters performed on it
 * and can be easily used in GPU code  
 * 
 * @author Mehran Maghoumi
 */
public class FilteredImage {
	
	/** The block size to use for the filter function */
	protected static final int FILTER_BLOCK_SIZE = 16;
	
	/** The original ByteImage converted to a pitched byte array and copied to GPU */
	protected CudaByte2D byteInput;
	
	/** The ByteImage converted to a pitched float array and copied to GPU */
	protected CudaFloat2D input;
	
	/** The ByteImage passed through the small average filter, converted to a pitched float array and copied to GPU */
	protected CudaFloat2D smallAvg;
	
	/** The ByteImage passed through the medium average filter, converted to a pitched float array and copied to GPU */
	protected CudaFloat2D mediumAvg;
	
	/** The ByteImage passed through the large average filter, converted to a pitched float array and copied to GPU */
	protected CudaFloat2D largeAvg;
	
	/** The ByteImage passed through the small standard deviation filter, converted to a pitched float array and copied to GPU */
	protected CudaFloat2D smallSd;
	
	/** The ByteImage passed through the medium standard deviation filter, converted to a pitched float array and copied to GPU */
	protected CudaFloat2D mediumSd;
	
	/** The ByteImage passed through the large standard deviation filter, converted to a pitched float array and copied to GPU */
	protected CudaFloat2D largeSd;
	
	/** The width of the image */
	protected int imageWidth;
	
	/** The height of this image */
	protected int imageHeight;
	
	/** The number of channels per pixel in this image */
	protected int numChannels;
	
	/**
	 * Filters the provided ByteImage using the specified filter function handle and the provided filter sizes
	 * and stores the results in the underlying data placeholders of this class.
	 * 
	 * @param image	The ByteImage to be filtered
	 * @param filterProvider	An ImageFilterProvider implementation to use for performing filters
	 */
	public FilteredImage(ByteImage image, ImageFilterProvider filterProvider) {
		this.imageWidth = image.getWidth();
		this.imageHeight = image.getHeight();
		this.numChannels = image.getNumChannels();
		
		this.byteInput = new CudaByte2D(imageWidth, imageHeight, numChannels, image.getByteData());
		this.input = new CudaFloat2D(imageWidth, imageHeight, numChannels, image.getFloatData());
		
		this.smallAvg = new CudaFloat2D(imageWidth, imageHeight, numChannels);
		this.mediumAvg = new CudaFloat2D(imageWidth, imageHeight, numChannels);
		this.largeAvg = new CudaFloat2D(imageWidth, imageHeight, numChannels);
		
		this.smallSd = new CudaFloat2D(imageWidth, imageHeight, numChannels);
		this.mediumSd = new CudaFloat2D(imageWidth, imageHeight, numChannels);
		this.largeSd = new CudaFloat2D(imageWidth, imageHeight, numChannels);
		
		filterProvider.performFilters(byteInput, smallAvg, mediumAvg, largeAvg, smallSd, mediumSd, largeSd);
	}

		/**
	 * @return the byteInput
	 */
	public CudaByte2D getByteInput() {
		return byteInput;
	}

	/**
	 * @return the input
	 */
	public CudaFloat2D getInput() {
		return input;
	}

	/**
	 * @return the smallAvg
	 */
	public CudaFloat2D getSmallAvg() {
		return smallAvg;
	}

	/**
	 * @return the mediumAvg
	 */
	public CudaFloat2D getMediumAvg() {
		return mediumAvg;
	}

	/**
	 * @return the largeAvg
	 */
	public CudaFloat2D getLargeAvg() {
		return largeAvg;
	}

	/**
	 * @return the smallSd
	 */
	public CudaFloat2D getSmallSd() {
		return smallSd;
	}

	/**
	 * @return the mediumSd
	 */
	public CudaFloat2D getMediumSd() {
		return mediumSd;
	}

	/**
	 * @return the largeSd
	 */
	public CudaFloat2D getLargeSd() {
		return largeSd;
	}

	/**
	 * @return the imageWidth
	 */
	public int getImageWidth() {
		return imageWidth;
	}

	/**
	 * @return the imageHeight
	 */
	public int getImageHeight() {
		return imageHeight;
	}

	/**
	 * @return the numChannels
	 */
	public int getNumChannels() {
		return numChannels;
	}
	
	/**
	 * Frees all CUDA memories that were allocated for the filtered images
	 */
	public void freeAll() {
		byteInput.free();
		input.free();
		smallAvg.free();
		mediumAvg.free();
		largeAvg.free();
		smallSd.free();
		mediumSd.free();
		largeSd.free();
	}
}
