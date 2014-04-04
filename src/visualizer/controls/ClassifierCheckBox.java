package visualizer.controls;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JCheckBox;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import utils.cuda.datatypes.Classifier;

/**
 * A JCheckBox that has bindings to a classifier. Checkboxes of this type
 * are able to enable or disable a bounded classifier.
 * 
 * @author Mehran Maghoumi
 *
 */
public class ClassifierCheckBox extends JCheckBox implements ChangeListener {
	/** The classifier that this checkbox is bound to */
	private Classifier boundedClassifier = null;
	
	/**
	 * Instantiates a new checkbox and binds it to the given
	 * classifier object.
	 * @param classifier
	 */
	public ClassifierCheckBox(Classifier classifier) {
		super(classifier.toString(), classifier.isEnabled());
		this.boundedClassifier = classifier;
		addChangeListener(this);
	}
	

	@Override
	public void stateChanged(ChangeEvent e) {
		boundedClassifier.setEnabled(isSelected());		
	}
	
	/**
	 * @return	The actual classifier object that this checkbox is bound to
	 */
	public Classifier getBoundedClassifier() {
		return this.boundedClassifier;
	}
	
	
}