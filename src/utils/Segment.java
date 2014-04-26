package utils;

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
	
	/** The image data of this segment */
	protected ByteImage image;
	
	/** The FilteredImage equivalent of this segment */
	protected FilteredImage filteredImage;
	
	/** An arbitrary unique ID that is associated to this segment to distinguish it from other segments */
	protected String id;
	
	/** The boundaries of this segment */
	protected Rectangle bounds;
	
	/**
	 * Initializes a new segment object using the provided boundaries and ID
	 * @param image
	 * @param x
	 * @param y
	 * @param width
	 * @param height
	 * @param id
	 */
	public Segment(ByteImage image, int x, int y, int width, int height, String id) {
		this(image, new Rectangle(x, y, width, height), id);
	}
	
	/**
	 * Initializes a new segment object using the provided boundaries and ID
	 * 
	 * @param image
	 * @param bounds
	 * @param id
	 */
	public Segment(ByteImage image, Rectangle bounds, String id) {
		this.image = image;
		this.bounds = new Rectangle(bounds);
		this.id = id;
	}
	
	/**
	 * @return The filtered images 
	 */
	public FilteredImage getFilteredImage() {
		return this.filteredImage;
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
	
	/**
	 * Filters the image using the provided ImageFilterProvider.
	 * Note that the filter is done on the calling thread thus the clients
	 * must take care of any context switching that may be required
	 * @param filterProvider
	 */
	public void filterImage(ImageFilterProvider filterProvider) {
		this.filteredImage = new FilteredImage(image, filterProvider);
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
	 * Performs a shallow clone of this object. Note that the image data and filtered images
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