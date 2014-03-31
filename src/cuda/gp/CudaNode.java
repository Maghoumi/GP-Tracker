package cuda.gp;

import java.util.ArrayList;
import java.util.Stack;

import ec.gp.GPNode;

public abstract class CudaNode extends GPNode {
	
	/**
	 * The OPCODE of this class. CUDA interpreter uses this OPCODE
	 * to know the number of arguments of this node and also the type
	 * of the arguments.
	 * You should not modify this value and the CUDAInitializer will assign a
	 * generated value to this field.
	 * Note that each sub-class has a single OPCODE.
	 */
	protected byte opcode;
	
	/**
	 * Sets the OPCODE for the prototype of this class
	 * You should not call this function!
	 * 
	 * @param opcode
	 */
	public void setOpcode(byte opcode) {
		this.opcode = opcode;
	}
	
	/**
	 * Returns the OPCODE assosicated with this prototype.
	 * Will be later used to convert the tree to a RPN notation
	 * for transfer to CUDA memory. It was designed to return an array
	 * of bytes for those operators that can have constant values in front of 
	 * them (Eg. ERCs or random values)
	 *  
	 * @return
	 */
	public byte[] getOpcode() {
		return new byte[] {this.opcode};
	}
	
	/**
	 * Creates a postfix representation of this node and its child nodes
	 */
	public StringBuilder traverse() {
    	StringBuilder result = new StringBuilder();
    	Stack<CudaNode> stack = new Stack<CudaNode>();
    	stack.push(this);
    	CudaNode node;
    	
    	while (!stack.isEmpty()) {
    		node = stack.pop();
    		result.insert(0, node.toStringForHumans());
    		
    		for (int i = 0 ; i < node.children.length ; i++ ) {
    			stack.push((CudaNode) node.children[i]);
    		}
    	}
    	
    	return result.append((char)0);
	}
	
	/**
	 * Creates a postfix representation of this node and its child nodes
	 */
	public byte[] byteTraverse() {
		ArrayList<Byte> result = new ArrayList<Byte>();
		
    	Stack<CudaNode> stack = new Stack<CudaNode>();
    	stack.push(this);
    	CudaNode node;
    	
    	while (!stack.isEmpty()) {
    		node = stack.pop();
    		byte[] byteCodes = node.getOpcode();
    		for (int i = 0 ; i < byteCodes.length ; i++)
    			result.add(byteCodes[i]);
    		
    		for (int i = 0 ; i < node.children.length ; i++ ) {
    			stack.push((CudaNode) node.children[i]);
    		}
    	}
    	
    	// reverse the thing
    	int length = result.size();
    	byte[] byted = new byte[length+1];

    	
    	for (int i = 0 ; i < length ; i++)
    		byted[i] = result.get(length - i - 1);
    	
    	// append terminating sequence
    	byted[length] = 0;
    	
    	return byted;
	}
	
	
	/**
	 * This method which should be implemented by the user, specifies the
	 * number of child nodes that this node can have.
	 * 
	 * @return
	 * 		The number of children that can go under this node.
	 */
	public abstract int getNumberOfChildren();
	
	
	/**
	 * Returns a C-like source code that will tell us the micro-operations
	 * that are required in order to evaluate this node.
	 * 
	 * @return		A C-like code, detailing the implementation of this tree
	 * 	node. (eg. float second = pop(); float first = pop(); push(first + second) )
	 */
	public abstract String getCudaAction();
}
