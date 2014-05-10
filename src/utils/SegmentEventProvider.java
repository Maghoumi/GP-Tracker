package utils;

import java.util.List;

/**
 * Defines the interface that the classes will have when they want to
 * provide information about segments
 * 
 * @author Mehran Maghoumi
 *
 */
public interface SegmentEventProvider {
	/** Add a SegmentEventListener object to the list of the listeners that this class has */
	public void addSegmentEventListener(SegmentEventListener listener);
	
	/** Remove a SegmentEventListener object from the list of the listeners that this class has */
	public void removeSegmentEventListener(SegmentEventListener listener);
	
	/** Notify the listeners that a new segment has been added */
	public void notifySegmentAdded(int segmentCount, int orphansCount, int permanentOrphansCount);
}
