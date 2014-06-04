package utils;

import java.util.*;


/**
 * Represents a video frame along with a set of segments that were
 * found in that video frame. Instances of this class are usually created
 * using an implementation of the VideoFeeder interface. These instances
 * are passed to the Visualizer object for visualization using CUDA.
 * 
 * @author Mehran Maghoumi
 *
 */
public class SegmentedVideoFrame implements Iterable<Segment> {
	/** The image data of the frame represented by this object */
	private ByteImage frame;
	
	/** The set of segments that were found in the frame */
	private List<Segment> segments = new ArrayList<Segment>();
	
	/** The background segment of this SegmentedVideoFrame */ 
	private Segment background;
	
	/**
	 * Instantiates an object of this class using the provided video frame and
	 * the set of found segments on that frame. Note that the segments inside
	 * the passed segment set are cloned.
	 * 
	 * @param frame	The video frame
	 * @param segments	The set of segments that were found in this video frame
	 */
	public SegmentedVideoFrame(ByteImage frame, Set<Segment> segments, Segment background) {
		this.frame = frame;
		
		for (Segment s : segments) {
			this.segments.add((Segment) s.clone());
		}
		
		this.background = (Segment) background.clone();
	}

	@Override
	public Iterator<Segment> iterator() {
		return this.segments.iterator();
	}
	
	/**
	 * @return	The actual video frame represented by this object
	 */
	public ByteImage getFrame() {
		return this.frame;
	}
	
	/**
	 * Returns the segment at the specified position in this frame.
	 * @param index
	 * @return
	 */
	public Segment get(int index) {
		return this.segments.get(index);
	}
	
	/**
	 * @return	The number of segments in this video frame
	 */
	public int size() {
		return this.segments.size();
	}
	
	/**
	 * @return	True if there are any segments in this frame, False otherwise
	 */
	public boolean hasSegments() {
		return size() != 0;
	}
	
	/** 
	 * @return	The background of this SegmentedVideoFrame object. The background
	 * segment should be used for training as an instance that is always negative
	 */
	public Segment getBackground() {
		return this.background;
	}
	
	/**
	 * Shuffles the segments in this frame so that the order is random
	 * This would be useful for training the GP system. The GP system will
	 * not get stuck on a single segment
	 */
	public void shuffle() {
		Collections.shuffle(segments);
	}
}
