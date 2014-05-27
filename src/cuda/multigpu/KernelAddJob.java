package cuda.multigpu;

import java.io.File;
import java.util.Map;

/**
 * Defines the elements that are required to load a kernel into all devices.
 * 
 * @author Mehran Maghoumi
 *
 */
public class KernelAddJob {
	
	/** The PTX file containing the CUDA code that we want to load */
	public File ptxFile;
	
	/**
	 * To solve name collision problem among different kernels in different PTX files, I 
	 * decided to make function calls possible using their ID. If there are two kernels
	 * both named "add", to discern which one should be called, we assign them different IDs.
	 * The thread that wants to call the function will resolve this collision.
	 * This map does the following:
	 * 	Add ==> Add
	 * 	Add2 ==> Add (later in another PTX file)
	 */
	public Map<String, String> functionMapping;
	
}
