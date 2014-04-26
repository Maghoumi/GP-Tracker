package gp.terminals;

import java.nio.ByteBuffer;

import cuda.gp.CudaERC;
import jcuda.Sizeof;
import ec.EvolutionState;
import ec.gp.GPNode;
import ec.util.Code;

public class DoubleErc extends CudaERC {
	public float value;

	public void resetNode(final EvolutionState state, final int thread) {
		value = state.random[thread].nextFloat();
	}

	@Override
	public void mutateERC(EvolutionState state, int thread) {
		float v;
		do v = (float) (value + state.random[thread].nextGaussian() * 0.01);
		while( v < 0.0 || v >= 1.0 );
		value = v;
	}

	public String toStringForHumans() {
		return String.format("%.2f", value);
	}

	@Override
	public boolean nodeEquals(GPNode node) {
		return false;
	}

	@Override
	public String encode() {
		return Code.encode(value);
	}

	@Override
	public int getNumberOfChildren() {
		return 0;
	}
	
	
	@Override
	public byte[] getOpcode() {
		// This is a double ERC. A double is 8 bytes.
		// I will convert this ERC to a byte code using the
		// constant value of this ERC.
		
		byte[] bytes = new byte[Sizeof.FLOAT + 1];	
		
		ByteBuffer.wrap(bytes).putFloat(0, value);
		bytes[bytes.length - 1] = this.opcode;
		
		return bytes;
	}

	@Override
	public String getCudaAction() {
		//TODO increase the performance of this calculation
		
		/*return "unsigned char* args = (unsigned char*)&(expression[++k]);\n" +
				"double* ercVal = (double*) args;\n" +
				"push (*ercVal);\n" +
				"k += 7;\n";*/
		
		/**
		 * Note to self: The above code should work properly, however it doesn't.
		 * The reason is CUDA pointer alignment requirements. A pointer must be a multiple of
		 * the types it wants to read from the global memory (address wise I mean!)
		 * Therefore, you cannot randomly read from a location in the global memory. That
		 * address must be aligned to the size of the data type you want to read.
		 * Here I am simply copying the data from the global memory to another space. 
		 * Since the size of this space is a multiple of 8 bytes (the size of the double), the 
		 * pointer can be easily casted to a double pointer and the value could be used directly.  
		 * C is a very interesting language in deed!  
		 */
		return "char args[4];"+
				"for (int i = 0 ; i < 4 ; i++)" +
					"args[i] = (char)(expression[++k]);" +
				"push (*((float*) args));";
	}
}