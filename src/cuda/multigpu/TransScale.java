package cuda.multigpu;

import static jcuda.driver.JCudaDriver.*;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.*;

import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

/**
 * This class provides transparent scalability for using JCuda with multiple GPUs. 
 * It manages the CUDA interop for multiple GPUs. For each available device, a 
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
public class TransScale implements DeviceProvider {
	
	/** The number of CUDA devices that are available */
	protected int numDevices = 0;
	
	/** Array containing all the threads that are responsible for each device */
	protected DeviceThread[] daemonThreads = null;	
	
	/** The singleton instance */
	protected static TransScale instance = null;
	
	/** A queue of the devices that are not busy and are available */
	protected BlockingQueue<DeviceThread> availableDevs = null;
	
	protected Object jobMutex = new Object();
	
	/**
	 * @return	The singleton instance (if exists), otherwise instantiates a
	 * 			new instance and returns it.
	 */
	public static synchronized TransScale getInstance() {
		if (instance == null)
			instance = new TransScale();
		
		return instance;
	}
	
	private TransScale() {
		JCudaDriver.setExceptionsEnabled(true);
		JCuda.setExceptionsEnabled(true);
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
			this.daemonThreads[i].setName("DeviceThread#" + i);
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
	
	/**
	 * Queue a job and give it to the next available device
	 * @param job
	 */
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
	public void notifyAvailable(DeviceThread dev) {
		synchronized (this.availableDevs) {
			if (!this.availableDevs.offer(dev))
				throw new RuntimeException("The queue can't be full but it is!!");
		}
	}
	
	/**
	 * Notify this balancer that the specified device is busy
	 * @param dev
	 */
	public void notifyBusy(DeviceThread dev) {
		synchronized (this.availableDevs) {
			availableDevs.remove(dev);
		}
	}
	

	@Override
	public int getNumberOfDevices() {
		return numDevices;
	}
	
	/**
	 * The extension of the given file name is replaced with "ptx". If the file
	 * with the resulting name does not exist, it is compiled from the given
	 * file using NVCC. The name of the PTX file is returned.
	 * 
	 * @param cuFileName
	 *            The name of the .CU file
	 * @param recompile
	 * 				Flag indicating whether the sourcefile should be recompiled or not
	 * @param genDebug
	 * 				Flag indicating whether debugging information should be generated or not
	 * @return The name of the PTX file
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	public static String preparePtxFile(String cuFileName, boolean recompile, boolean generateDebug) throws IOException {
		int endIndex = cuFileName.lastIndexOf('.');
		if (endIndex == -1) {
			endIndex = cuFileName.length() - 1;
		}
		String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
		File ptxFile = new File(ptxFileName);

		if (ptxFile.exists() && !recompile) {
			return ptxFileName;
		}

		File cuFile = new File(cuFileName);
		if (!cuFile.exists()) {
			throw new IOException("Input file not found: " + cuFileName);
		}
		String modelString = "-m" + System.getProperty("sun.arch.data.model");
		String command = "nvcc " + modelString + " -arch compute_20 -code sm_30 " + (generateDebug ? " -G" : "") +" -ptx " + cuFile.getPath() + " -o " + ptxFileName;

		System.out.println("Executing\n" + command);
		Process process = Runtime.getRuntime().exec(command);

		String errorMessage = new String(toByteArray(process.getErrorStream()));
		String outputMessage = new String(toByteArray(process.getInputStream()));
		int exitValue = 0;
		try {
			exitValue = process.waitFor();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new IOException("Interrupted while waiting for nvcc output", e);
		}

		if (exitValue != 0) {
			System.out.println("nvcc process exitValue " + exitValue);
			System.out.println("errorMessage:\n" + errorMessage);
			System.out.println("outputMessage:\n" + outputMessage);
			throw new IOException("Could not create .ptx file: " + errorMessage);
		}

		System.out.println("Finished creating PTX file");
		return ptxFileName;
	}

	/**
	 * Fully reads the given InputStream and returns it as a byte array
	 * 
	 * @param inputStream
	 *            The input stream to read
	 * @return The byte array containing the data from the input stream
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	private static byte[] toByteArray(InputStream inputStream) throws IOException {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		byte buffer[] = new byte[8192];
		while (true) {
			int read = inputStream.read(buffer);
			if (read == -1) {
				break;
			}
			baos.write(buffer, 0, read);
		}
		return baos.toByteArray();
	}
	
	
}
