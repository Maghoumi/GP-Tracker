package cuda.multigpu;

import static jcuda.driver.JCudaDriver.*;

import java.util.*;
import java.util.concurrent.*;

import jcuda.driver.*;

/**
 * Defines the thread that handles the interop for the selected device.
 * TODO: multiple addKernel jobs, (addKernelQueue)
 * job queue
 * invoke pre and post methods
 * 
 * @author Mehran Maghoumi
 *
 */
public class DeviceThread extends Thread {
	
	/** The capacity of the jobs queue */
	public static final int JOB_CAPACITY = 1;
	
	/** The LoadBalancer instance that has instantiated this object */
	protected TransScale parent = null;
	
	/** The device that this thread interops with */
	protected CUdevice device = null;
	
	/** The ordinal of the device that this thread interops with */
	protected int deviceNumber = -1;
	
	/** Invokables indexed by their ID */
	protected Map<String, Invokable> invokables = new HashMap<>();
	
	/** The queue that manages the module load requests */
	protected BlockingQueue<KernelAddJob> kernelJobsQueue = new ArrayBlockingQueue<>(JOB_CAPACITY);
	
	/** Queue for all the kernel calls that this thread should make on the CUDA device */
	protected BlockingQueue<KernelInvoke> invocationJobs = new ArrayBlockingQueue<>(JOB_CAPACITY);
	
	/** Barrier to inform that something is available */
	protected Object barrier = new Object();
	
	
	/**
	 * Initializes a thread that interops with the specified device
	 * @param deviceNumber	The ordinal of the device that this thread should interop with
	 */
	public DeviceThread(TransScale parent, int deviceNumber) {
		this.parent = parent;
		this.deviceNumber = deviceNumber;
		this.device = new CUdevice();
		setDaemon(true);
		cuDeviceGet(device, deviceNumber);
	}
	
	/**
	 * Add a new kernel to the list of kernels that this thread can invoke
	 * @param job
	 */
	public synchronized void addKernel(KernelAddJob job) {
		
		synchronized(barrier) {
			synchronized(kernelJobsQueue) {
				try {
					kernelJobsQueue.put(job);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			interrupt();
		}
	}
	
	/**
	 * Queue a kernel invocation job on this device
	 * @param job
	 */
	public synchronized void queueJob(KernelInvoke job) {
		
		synchronized(barrier) {
			synchronized(invocationJobs) {
				try {
					invocationJobs.put(job);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			interrupt();
		}
	}
	
	
	/**
	 * Creates a new CUDA context and loads the supplied PTX module and obtains
	 * function handles to all the kernels that should be later invoked.
	 * 
	 * @param job
	 */
	protected void loadKernel(KernelAddJob job) {
		// Notify parent that I am busy
		parent.notifyBusy(this);
		
		// Create context for the new kernel
		CUcontext newContext = new CUcontext();
		cuCtxCreate(newContext, 0, device);
		
		// Create and load the module for the new kernel
		CUmodule newModule = new CUmodule();		
		cuModuleLoad(newModule, job.ptxFile.getPath());
		
		// Store and index all function handlers
		for (String id : job.functionMapping.keySet()) {
			CUfunction newFunction = new CUfunction();			
			cuModuleGetFunction(newFunction, newModule, job.functionMapping.get(id));
			
			invokables.put(id, new Invokable(newContext, newModule, newFunction));
		}
		
		// Notify parent that I am available
		parent.notifyAvailable(this);
	}
	
	/** 
	 * A helper function to invoke a kernel using the specified job
	 * object.
	 * @param job
	 */
	private void invoke(KernelInvoke job) {
		// Notify parent that I am busy
		parent.notifyBusy(this);
		
		Invokable invokable = invokables.get(job.functionId);
		CUcontext context = invokable.context;
		CUmodule module = invokable.module;
		CUfunction function = invokable.function;
		
		// Switch the context
		cuCtxSetCurrent(context);
		
		job.preTrigger.doTask(module);
		
		cuLaunchKernel(function, job.gridDimX, job.gridDimY, job.gridDimZ,
				job.blockDimX, job.blockDimY, job.blockDimZ,
				job.sharedMemorySize, null,
				job.pointerToArguments, null);
		cuCtxSynchronize();
		
		job.postTrigger.doTask(module);
		job.notifyComplete();
		
		// Notify parent that I am available
		parent.notifyAvailable(this);
	}	
	
	@Override
	public void run() {
		while (true) {
			
			// Wait for something to become availble
			synchronized(barrier) {
				try {
					barrier.wait();
				} catch (InterruptedException e) {
				}
			}
			
			synchronized(kernelJobsQueue) {
				if (!kernelJobsQueue.isEmpty())
					loadKernel(kernelJobsQueue.poll());
			}
			
			synchronized(invocationJobs) {
				if (!invocationJobs.isEmpty())
					invoke(invocationJobs.poll());
			}
		}
		
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
