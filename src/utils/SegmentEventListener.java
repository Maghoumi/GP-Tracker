package utils;

import java.util.List;

/**
 * Defines the interface for various events regarding the segments in the system
 * 
 * @author Mehran Maghoumi
 *
 */
public interface SegmentEventListener {
	/**
	 * Will be called when a new segment has been added to the system
	 * @param segmentCount	The total number of segments in the current video frame
	 * @param orphansCount	The current number of orphans in the system
	 * @param permanentOrphansCount	The current number of permanent orphans in the system
	 */
	public void segmentAdded(int segmentCount, int orphansCount, int permanentOrphansCount);
}
