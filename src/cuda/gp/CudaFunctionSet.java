package cuda.gp;

import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import ec.EvolutionState;
import ec.gp.GPFunctionSet;
import ec.gp.GPInitializer;
import ec.gp.GPNode;
import ec.gp.GPType;
import ec.util.Parameter;

import cuda.gp.*;

/**
 * Basically the same as ec.gp.GPFunctionSet with the only difference being the assignment
 * of OPCODE to each node based on the number of child nodes and their return types.
 * 
 * 
 * @author Mehran Maghoumi
 *
 */
public class CudaFunctionSet extends GPFunctionSet {
	
	/** For opcode generation */
	private byte currentOpcode = 0;
	
	private byte getNextOpcode(EvolutionState state) {
		currentOpcode++;
		
		if ((currentOpcode & 0xFF) == 255)
			state.output.fatal("You are out of OPCODEs! A byte can only hold 255 distinct values...");
		
		return (byte) (currentOpcode & 0xFF);
	}
	
	 /** Must be done <i>after</i> GPType and GPNodeConstraints have been set up */

    public void setup(final EvolutionState state, final Parameter base)
        {
    	CudaEvolutionState cuState = (CudaEvolutionState) state;
    	
        // What's my name?
        name = state.parameters.getString(base.push(P_NAME),null);
        if (name==null)
            state.output.fatal("No name was given for this function set.",
                base.push(P_NAME));
        // Register me
        GPFunctionSet old_functionset = (GPFunctionSet)(((GPInitializer)state.initializer).functionSetRepository.put(name,this));
        if (old_functionset != null)
            state.output.fatal("The GPFunctionSet \"" + name + "\" has been defined multiple times.", base.push(P_NAME));

        // How many functions do I have?
        int numFuncs = state.parameters.getInt(base.push(P_SIZE),null,1);
        if (numFuncs < 1)
            state.output.error("The GPFunctionSet \"" + name + "\" has no functions.",
                base.push(P_SIZE));
        
        nodesByName = new Hashtable();

        Parameter p = base.push(P_FUNC);
        Vector tmp = new Vector();
        
        StringBuilder actionsCode = new StringBuilder();
        
        for(int x = 0; x < numFuncs; x++)
            {
            // load
            Parameter pp = p.push(""+x);
            CudaNode gpfi = (CudaNode)(state.parameters.getInstanceForParameter(
                    pp, null, CudaNode.class));
            gpfi.setup(state,pp);

            // Assign an OPCODE to this prototype
            gpfi.setOpcode(getNextOpcode(state));
            
            // create the ACTION in CUDA. Remember that OPCODE can be an array of bytes.
            // Only the beginning element is used as the "case" labels
            byte[] opcode = gpfi.getOpcode();
            actionsCode.append("case " + opcode[opcode.length - 1] + ": {" + gpfi.getCudaAction() + "}break;" + System.lineSeparator());

            // add to my collection
            tmp.addElement(gpfi);
                        
            // Load into the nodesByName hashtable
            GPNode[] nodes = (GPNode[])(nodesByName.get(gpfi.name()));
            if (nodes == null)
                nodesByName.put(gpfi.name(), new GPNode[] { gpfi });
            else
                {
                // O(n^2) but uncommon so what the heck.
                GPNode[] nodes2 = new GPNode[nodes.length + 1];
                System.arraycopy(nodes, 0, nodes2, 0, nodes.length);
                nodes2[nodes2.length - 1] = gpfi;
                nodesByName.put(gpfi.name(), nodes2);
                }
            }
        
        actionsCode.append("default:printf(\"Unrecognized OPCODE in the expression tree!\");break;");
        cuState.getCudaInterop().setKernelCode(actionsCode.toString());

        // Make my hash tables
        nodes_h = new Hashtable();
        terminals_h = new Hashtable();
        nonterminals_h = new Hashtable();

        // Now set 'em up according to the types in GPType

        Enumeration e = ((GPInitializer)state.initializer).typeRepository.elements();
        GPInitializer initializer = ((GPInitializer)state.initializer);
        while(e.hasMoreElements())
            {
            GPType typ = (GPType)(e.nextElement());
            
            // make vectors for the type.
            Vector nodes_v = new Vector();
            Vector terminals_v = new Vector();
            Vector nonterminals_v = new Vector();

            // add GPNodes as appropriate to each vector
            Enumeration v = tmp.elements();
            while (v.hasMoreElements())
                {
                GPNode i = (GPNode)(v.nextElement());
                if (typ.compatibleWith(initializer,i.constraints(initializer).returntype))
                    {
                    nodes_v.addElement(i);
                    if (i.children.length == 0)
                        terminals_v.addElement(i);
                    else nonterminals_v.addElement(i);
                    }
                }

            // turn nodes_h' vectors into arrays
            GPNode[] ii = new GPNode[nodes_v.size()];
            nodes_v.copyInto(ii);
            nodes_h.put(typ,ii);

            // turn terminals_h' vectors into arrays
            ii = new GPNode[terminals_v.size()];
            terminals_v.copyInto(ii);
            terminals_h.put(typ,ii);

            // turn nonterminals_h' vectors into arrays
            ii = new GPNode[nonterminals_v.size()];
            nonterminals_v.copyInto(ii);
            nonterminals_h.put(typ,ii);
            }

        // I don't check to see if the generation mechanism will be valid here
        // -- I check that in GPTreeConstraints, where I can do the weaker check
        // of going top-down through functions rather than making sure that every
        // single function has a compatible argument function (an unneccessary check)

        state.output.exitIfErrors();  // because I promised when I called n.setup(...)

        // postprocess the function set
        postProcessFunctionSet();
        }
}
