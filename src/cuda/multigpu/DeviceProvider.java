package cuda.multigpu;

/**
 * Defines the interface that other classes which want to provide
 * CUDA devices to the load balancer must adhere to.
 * 
 * Implementations of this interface provide information such as the number
 * of devices, the compute caability of those devices and etc. Furthermore, using
 * these classes, one could send CUDA commands over the network.
 * 
 * @author Mehran Maghoumi
 *
 */
public interface DeviceProvider {
	/**
	 * @return	The number of CUDA devices that we could work with
	 */
	public int getNumberOfDevices();	
}
