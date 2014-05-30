package cuda.multigpu;

import jcuda.Pointer;

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
	
	/** Pointer to pointer to all the arguments the kernel needs */
	public Pointer pointerToArguments;
	
	/** Pre-call triggers */
	public Trigger preTrigger;
	
	/** Post-call triggers */
	public Trigger postTrigger;
	
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
	
	/** For waiting for this job to complete */
	protected Object waitMutex = new Object();
	
	private volatile boolean jobComplete = false;
	
	/**
	 * Wait for this job to complete. Blocks the calling thread until
	 * this job has been completed on the graphics card.
	 */
	public void waitFor() {
		while (!jobComplete) {
			try {
				Thread.sleep(1);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
		}
//		synchronized (waitMutex) {
//			try {
//				waitMutex.wait();
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
//		}
	}
	
	public void notifyComplete() {
		jobComplete = true;
//		synchronized (waitMutex) {
//			waitMutex.notify();
//		}
	}
}
