package cuda.multigpu;

import static jcuda.driver.JCudaDriver.*;

import java.util.concurrent.*;

/**
 * Manages the CUDA interop for multiple GPUs. For each available device, a 
 * CPU thread is created. Each thread will monitor a job queue (a job consists of 
 * a video frame and a set of classifiers). When a new job becomes available, one of
 * the threads will dequeue that job and starts processing it on its own GPU. At some
 * point the thread in question will call cuCtxSynchronize and obtain the results from
 * its GPU. When the results are obtained, they are stored in another queue: a queue
 * that is monitored by the OpenGL thread (the thread that schedules the jobs on the
 * job queue). Whenever a new result is available, that thread will obtain it and visualize
 * it!
 * 
 * Note:
 * 		This class uses the singleton pattern. We want a bunch of modules loaded at the beginning
 * 		and we usually don't add modules on the fly.
 * 
 * @author Mehran Maghoumi
 *
 */
public class LoadBalancer implements DeviceProvider {
	
	/** The number of CUDA devices that are available */
	protected int numDevices = 0;
	
	/** Array containing all the threads that are responsible for each device */
	protected DeviceThread[] daemonThreads = null;	
	
	/** The singleton instance */
	protected static LoadBalancer instance = null;
	
	/** A queue of the devices that are not busy and are available */
	protected BlockingQueue<DeviceThread> availableDevs = null;
	
	protected Object jobMutex = new Object();
	
	/**
	 * @return	The singleton instance (if exists), otherwise instantiates a
	 * 			new instance and returns it.
	 */
	public static LoadBalancer getInstance() {
		if (instance == null)
			instance = new LoadBalancer();
		
		return instance;
	}
	
	private LoadBalancer() {
		cuInit(0);
		// Get the number of devices 
		int[] numDevicesArray = { 0 };
		cuDeviceGetCount(numDevicesArray);
		this.numDevices = numDevicesArray[0];
		
		this.availableDevs = new ArrayBlockingQueue<>(numDevices);
		this.daemonThreads = new DeviceThread[numDevices];
		
		// Fill arrays and start the daemon thread
		for (int i = 0 ; i < numDevices ; i++) {
			this.daemonThreads[i] = new DeviceThread(this, i);
			availableDevs.add(this.daemonThreads[i]);		// Add to available queue
			this.daemonThreads[i].start();
		}
	}
	
	/**
	 * Add a new kernel to the list of kernels that the GPUs in the system can invoke
	 * @param job
	 */
	public synchronized void addKernel(KernelAddJob job) {
		synchronized (jobMutex) {
			// Wait for all devices to become available
			while (availableDevs.size() != numDevices) {
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			for (DeviceThread th : this.daemonThreads) {
				th.addKernel(job);
			}
		}
	}
	
	public synchronized void queueJob(KernelInvoke job) {
		synchronized (jobMutex) {
			//Select an available device
			DeviceThread th = null;
			try {
				th = availableDevs.take();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			th.queueJob(job);
		}
	}
	
	/**
	 * Notify this balancer of the availability of the specified device
	 * @param dev
	 */
	public synchronized void notifyAvailable(DeviceThread dev) {
		synchronized (this.availableDevs) {
			try {
				this.availableDevs.put(dev);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Notify this balancer that the specified device is busy
	 * @param dev
	 */
	public synchronized void notifyBusy(DeviceThread dev) {
		synchronized (this.availableDevs) {
			availableDevs.remove(dev);
		}
	}
	

	@Override
	public int getNumberOfDevices() {
		return numDevices;
	}
	
	public static void main(String[] args) {
		LoadBalancer.getInstance();
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
		
		
		
}
