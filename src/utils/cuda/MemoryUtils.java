package utils.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.utils.KernelLauncher;

//TODO do some profiling and figure out if keeping the first level pointers will increase the performance
// right now, since I am not keeping them, I keep generating them
public class MemoryUtils {

	/**
	 * Allocates and copies a 2D array in CUDA memory and return its pointer
	 * NOTE: JCuda engine must be initialized before a call to this function
	 * 
	 * @param input
	 * @return
	 */
	public static OldCUdeviceptr2D allocate2DArray(int[][] input) {
		//TODO only works for int
		int width = input[0].length, height = input.length;
		CUdeviceptr[] firstLevelPointers = new CUdeviceptr[height];

		for (int i = 0; i < height; i++) {
			firstLevelPointers[i] = new CUdeviceptr();
			JCudaDriver.cuMemAlloc(firstLevelPointers[i], width * Sizeof.INT);
			JCudaDriver.cuMemcpyHtoD(firstLevelPointers[i], Pointer.to(input[i]), width * Sizeof.INT);
		}

		// create a pointer to pointer and allocate and copy it
		CUdeviceptr basePointer = new CUdeviceptr();
		JCudaDriver.cuMemAlloc(basePointer, height * Sizeof.POINTER);
		JCudaDriver.cuMemcpyHtoD(basePointer, Pointer.to(firstLevelPointers), height * Sizeof.POINTER);

		return new OldCUdeviceptr2D(basePointer, firstLevelPointers);
	}

	/**
	 * Fetches a 2D array from CUDA memory by its pointer and copies it in the
	 * output array
	 * 
	 * @param pointer
	 *            CUDA pointer that points to a 2D array in CUDA memory
	 * @param output
	 *            CUDA memory is copied in this array
	 */
	public static void fetch2DArray(OldCUdeviceptr2D pointer, int[][] output) {
		// fetch the first level pointers
		int width = output[0].length; /* , height = output.length; */
		//		CUdeviceptr[] firstLevelPointers = new CUdeviceptr[height];
		//		
		//		for (int i = 0; i < height; i++) {
		//			firstLevelPointers[i] = new CUdeviceptr();
		//		}
		//		
		//		JCudaDriver.cuMemcpyDtoH(Pointer.to(firstLevelPointers), pointer, height * Sizeof.POINTER);

		// now that we have the first level pointers, it's time to fetch the actual results

		for (int i = 0; i < pointer.getHeight(); i++) {
			JCudaDriver.cuMemcpyDtoH(Pointer.to(output[i]), pointer.levelPointer[i], width * Sizeof.INT);
		}
	}

	/**
	 * Frees a 2D array allocated in CUDA memory.
	 * 
	 * @param pointer
	 *            A 2D pointer that points to the array in memory
	 */
	public static void free2DArray(OldCUdeviceptr2D pointer) {
		// free level pointers
		for (int i = 0; i < pointer.getHeight(); i++) {
			JCudaDriver.cuMemFree(pointer.levelPointer[i]);
		}

		// free base pointer
		JCudaDriver.cuMemFree(pointer);
	}

	public static void compareHostArrays(int[][] a, int[][] b) throws Exception {
		for (int i = 0 ; i < a.length ; i++) {
			for (int j = 0 ; j < a[i].length ; j++) {
				if (a[i][j] != b[i][j])
					throw new Exception("Difference at " + i + ", " + j + " : " + a[i][j] + " | " + b[i][j]);
			}
		}
		
		System.out.println("The arrays are equal");
	}
	
	/**
	 * Prints a 2D array in standard output
	 * @param input
	 * 		Input array
	 * @param separator
	 * 		Separator to use to separate elements in the console window
	 */
	public static void print2DArray(int[][] input, String separator) {
		for (int i = 0 ; i < input.length ; i++) {
			for (int j = 0 ; j < input[i].length ; j++) {
				System.out.print(input[i][j] + separator);
			}
			System.out.println();
		}
	}

	public static void main(String[] args) {
		// do CUDA stuff

		KernelLauncher kernel;
		int imageWidth = 12001, imageHeight = 10003;
		int blockWidth = 32, blockHeight = 32;
		
		
		int[][] input = new int[imageHeight][imageWidth];
		int[][] output = new int[imageHeight][imageWidth];
		OldCUdeviceptr2D _input, _output;
		
		for(int i = 0 ; i < imageHeight ; i++)
			for (int j = 0 ; j < imageWidth ; j++) {
				input[i][j] = i * j;
				output[i][j] = -1;
			}
		
		
		
		JCudaDriver.setExceptionsEnabled(true);
		kernel = KernelLauncher.create("MemoryUtils.cu", "evaluate", true, "-arch=sm_21");
		

		int gridSizeX = (int) Math.ceil((double) imageWidth / blockWidth);
		int gridSizeY = (int) Math.ceil((double) imageHeight / blockHeight);
		kernel.setGridSize(gridSizeX, gridSizeY);
		kernel.setBlockSize(blockWidth, blockHeight, 1);
		
		_input = allocate2DArray(input);
		_output = allocate2DArray(output);
		
		kernel.call(_input, _output, imageWidth, imageHeight);
		
		fetch2DArray(_input, input);
		fetch2DArray(_output, output);
		
		
		//print2DArray(output, "\t");
		
		try {
			compareHostArrays(input, output);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		//System.out.println("Grid(x): " + gridSizeX + ", Grid(y): " + gridSizeY);
//		System.out.println("BlockWidth: " + blockWidth + ", BlockHeight: " + blockHeight);
	}

}
