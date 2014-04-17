package utils.cuda.datatypes;

import java.awt.Rectangle;

import org.apache.commons.lang3.builder.HashCodeBuilder;

import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

/**
 * Represents a segment of an image. The objects of this class must be
 * instantiated by the segmentation algorithm that we are planning to
 * use in the future.
 * 
 * @author Mehran Maghoumi
 *
 */
public class Segment implements Cloneable {
	
	private NewCudaData data;
	private ByteImage image;
	
	public int x;
	public int y;
	public int width;
	public int height;
	
	public Segment(ByteImage image, int x, int y, int width, int height) {
		this.data = new NewCudaData(image);
		this.image = image;
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
	}
	
	/**
	 * Set the CUDA objects that are necessary for the CudaData that this segment has 
	 * @param module	The cuda module
	 * @param fncFilter The handle to the filter functions
	 */
	public void setCudaObjects(CUmodule module, CUfunction fncFilter) {
		this.data.setCudaObjects(module, fncFilter);
	}
	
	/**
	 * @return The underlying CudaData that this segment has 
	 */
	public NewCudaData getImageData() {
		return this.data;
	}
	
	/**
	 * @return	The image data of this segment as a ByteImage object
	 */
	public ByteImage getByteImage() {
		return this.image;
	}
	
	public Rectangle getRectangle() {
		return new Rectangle(this.x, this.y, this.width, this.height);
	}
	
	@Override
	public boolean equals(Object obj) {
		Segment other = (Segment) obj;
		
		if(this.x == other.x &&	this.y == other.y &&
			this.width == other.width && this.height == other.height)
			return true;
					
		return false;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#hashCode()
	 */
	@Override
	public int hashCode() {
		return new HashCodeBuilder()
			.append(this.x)
			.append(this.y)
			.append(this.width)
			.append(this.height)
			.toHashCode();
	}
	
	/**
	 * Performs a shallow clone of this object. Note that the image data and CudaData
	 * of this object are not cloned!
	 */
	@Override
	public Object clone() {
		return new Segment(image, x, y, width, height);
	}
	
}
