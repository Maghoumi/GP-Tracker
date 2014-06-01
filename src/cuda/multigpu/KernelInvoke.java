package cuda.multigpu;

/**
 * Defines the elements that are required for invoking a kernel in CUDA.
 * This class defines the operations that should be done before and after 
 * a CUDA kernel call
 * 
 * @author Mehran Maghoumi
 *
 */
public class KernelInvoke {
	
	/** The ID of the kernel to call */
	public String functionId;
	
	/** Pre-call triggers */
	public Trigger preTrigger;
	
	/** Post-call triggers */
	public Trigger postTrigger;
	
	/** A function that returns the pointer to the kernel arguments */
	public KernelArgSetter argSetter;
	
	/** Grid size in X */
	public int gridDimX;
	
	/** Grid size in Y */
	public int gridDimY;
	
	/** Grid size in Z */
	public int gridDimZ = 1;
	
	/** Block size in X */
	public int blockDimX;
	
	/** Block size in Y */
	public int blockDimY;
	
	/** Block size in Z */
	public int blockDimZ;
	
	/** The size of the shared memory to pass to the kernel function */
	public int sharedMemorySize = 0;
	
	/** Can be used for debugging purposes */
	public String id;
	
	/** Flag indicating that this job has completed */
	protected volatile boolean jobComplete = false;
	
	/** For waiting for this job to complete */
	protected Object mutex = new Object();
	
	protected Thread waitingThread = null;
	
	/**
	 * Wait for this job to complete. Blocks the calling thread until
	 * this job has been completed on the graphics card.
	 */
	public void waitFor() {
		
		synchronized(mutex) {
			while (!jobComplete) {
				try {
					waitingThread = Thread.currentThread();
					mutex.wait();
				} catch (InterruptedException e) {}
			}
		}
	}
	
	/**
	 * Notify the threads that are waiting for this job to finish
	 */
	public void notifyComplete() {
		
		synchronized (mutex) {
			jobComplete = true;
			
			if (waitingThread != null)
				waitingThread.interrupt();			
		}
	}
}
