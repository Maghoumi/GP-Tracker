package utils;

import visualizer.Visualizer;

/**
 * Defines the interface for any object that is interested in knowin that a
 * GP run session has concluded successfully. A session has concluded successfully
 * if all segments are classified correctly and no GP calls are necessary.
 * 
 * @author Mehran Maghoumi
 * 
 */
public interface SuccessListener {
	/**
	 * Notify the listener of the success of the current session
	 * @param visualizer	The Visualizer object that was successful
	 */
	public void notifySuccess(Visualizer visualizer);
	
	/**
	 * Notify the listener of the failure of the current session
	 * @param reason	The reason of failure
	 */
	public void notifyFailure(String reason);
}
