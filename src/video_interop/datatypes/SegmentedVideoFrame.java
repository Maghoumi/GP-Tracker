package video_interop.datatypes;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Segment;

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
	private Set<Segment> segments = new HashSet<Segment>();
	
	/**
	 * Instantiates an object of this class using the provided video frame and
	 * the set of found segments on that frame. Note that the segments inside
	 * the passed segment set are cloned.
	 * 
	 * @param frame	The video frame
	 * @param segments	The set of segments that were found in this video frame
	 */
	public SegmentedVideoFrame(ByteImage frame, Set<Segment> segments) {
		this.frame = frame;
		
		for (Segment s : segments) {
			this.segments.add((Segment) s.clone());
		}
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
}
