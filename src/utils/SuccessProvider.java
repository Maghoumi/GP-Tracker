package utils;

/**
 * Defines the interface for classes that can provide information
 * regarding a successful GP session.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface SuccessProvider {
	/**
	 * Add a listener to the list of this object's listeners
	 * @param listener
	 */
	public void addSuccessListener(SuccessListener listener);
	
	/**
	 * Remove a listener from the list of this object's listeners
	 * @param listener
	 */
	public void removeSuccessListener(SuccessListener listener);
	
	/**
	 * Notify the listeners of the success of the GP session
	 */
	public void notifySuccess();
	
	/**
	 * Notify the listeners of the failure of the GP session. Could
	 * be because the maximum number of GP calls have reached or something
	 * else
	 */
	public void notifyFailure(String reason);
}
