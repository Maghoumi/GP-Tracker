package utils;

import java.awt.Rectangle;
import java.util.*;

import org.apache.commons.lang3.builder.HashCodeBuilder;

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
	
	/** Flag indicating that this segment is orphan */
	protected boolean orphan = false;
	
	/** The list of the classifiers that have claimed this segment */
	protected Set<Classifier> claimers = new HashSet<>();
	
	/**
	 * Flag indicating whether this segment is permanent orphan.
	 * If a segment is permanent orphan, it means that it has exceeded the
	 * maximum number of GP retrain requests
	 */
	protected boolean permanentOrphan = false;
	
	/**
	 * Initializes a new segment object using the provided boundaries and ID
	 * @param image
	 * @param x
	 * @param y
	 * @param width
	 * @param height
	 * @param id
	 */
	public Segment(ByteImage image, int x, int y, int width, int height, String id, Set<Classifier> claimers) {
		this(image, new Rectangle(x, y, width, height), id);
		this.claimers.addAll(claimers);
	}
	
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
	 * @return orphan status of this segment
	 */
	public boolean isOrphan() {
		return orphan;
	}

	/**
	 * @param orphan the orphan to set
	 */
	public void setOrphan(boolean orphan) {
		this.orphan = orphan;
	}
	
	/**
	 * @return the permanentOrphan
	 */
	public boolean isPermanentOrphan() {
		return permanentOrphan;
	}

	/**
	 * @param permanentOrphan the permanentOrphan to set
	 */
	public void setPermanentOrphan(boolean permanentOrphan) {
		this.permanentOrphan = permanentOrphan;
	}
	
	/**
	 * Add a classifier to the list of the classifiers that have claimes this segment
	 * @param c
	 */
	public void addClaimer(Classifier c) {
		this.claimers.add(c);
	}
	
	/**
	 * @return	The list of classifiers that have claimed this segment
	 */
	public Set<Classifier> getClaimers() {
		return this.claimers;
	}
	
	/**
	 * @return	The number of classifiers that have claimed this segment
	 */
	public int getClaimersCount() {
		return this.claimers.size();
	}
	
	/**
	 * Clear the list of classifiers that have claimed this segment
	 */
	public void resetClaimers() {
		this.claimers.clear();
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
		return new Segment(image, bounds.x, bounds.y, bounds.width, bounds.height, id + "_cloned", this.claimers);
	}
	
	@Override
	public String toString() {
		return this.id;
	}
	
}
