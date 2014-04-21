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
	
	/** The CudaData instance that holds the GPU data of this segment */
	private NewCudaData data;
	
	/** The image data of this segment */
	private ByteImage image;
	
	/** An arbitrary unique ID that is associated to this segment to distinguish it from other segments */
	protected String id;
	
	/** The boundaries of this segment */
	protected Rectangle bounds;
	
	public Segment(ByteImage image, int x, int y, int width, int height, String id) {
		this(image, new Rectangle(x, y, width, height), id);
	}
	
	public Segment(ByteImage image, Rectangle bounds, String id) {
		this.data = new NewCudaData(image);
		this.image = image;
		this.bounds = new Rectangle(bounds);
		this.id = id;
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
	
	/**
	 * @return The boundaries of this segment as a rectangle
	 */
	public Rectangle getBounds() {
		return this.bounds;
	}
	
	@Override
	public boolean equals(Object obj) {
		Segment other = (Segment) obj;
		
		if(this.bounds.x == other.bounds.x &&	this.bounds.y == other.bounds.y &&
			this.bounds.width == other.bounds.width && this.bounds.height == other.bounds.height)
			return true;
					
		return false;
	}

	@Override
	public int hashCode() {
		return new HashCodeBuilder()
			.append(this.bounds.x)
			.append(this.bounds.y)
			.append(this.bounds.width)
			.append(this.bounds.height)
			.toHashCode();
	}
	
	/**
	 * Performs a shallow clone of this object. Note that the image data and CudaData
	 * of this object are not cloned!
	 */
	@Override
	public Object clone() {
		return new Segment(image, bounds.x, bounds.y, bounds.width, bounds.height, id + "_cloned");
	}
	
	@Override
	public String toString() {
		return this.id;
	}
	
}
