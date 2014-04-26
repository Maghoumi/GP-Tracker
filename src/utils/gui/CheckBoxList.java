package utils.gui;

import javax.swing.*;
import javax.swing.border.*;

import utils.Classifier;

import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

public class CheckBoxList extends JList {
	protected static Border noFocusBorder = new EmptyBorder(1, 1, 1, 1);
	
	/** The list of the items of this list */
	private List<ClassifierCheckBox> items = new ArrayList<ClassifierCheckBox>();

	public CheckBoxList() {
		setCellRenderer(new CellRenderer());

		addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				int index = locationToIndex(e.getPoint());

				if (index != -1) {
					JCheckBox checkbox = (JCheckBox) getModel().getElementAt(index);
					if (e.getX() >= 0 && e.getX() <= 15)	// If the X of the clicked spot is in the boundaries of the checkbox mark ==> we should flip state :D
						checkbox.setSelected(!checkbox.isSelected());	// smart condition, eh? :D
					repaint();
				}
			}
		});

		setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		setBorder(new BevelBorder(BevelBorder.LOWERED, null, null, null, null));
	}
	
	public void addItem(ClassifierCheckBox item) {
		
		for (ClassifierCheckBox x : items) {
			if (x.getBoundedClassifier().equals(item.getBoundedClassifier()))
				return;
		}
		
		this.items.add(item);
		refreshListData();
	}
	
	public void removeItem (ClassifierCheckBox item) {
		this.items.remove(item);
		refreshListData();
	}
	
	public void addItem(Classifier classifier) {
		ClassifierCheckBox checkbox = new ClassifierCheckBox(classifier);
		this.addItem(checkbox);
	}
	
	public void removeItem (Classifier classifier) {
		ClassifierCheckBox toBeRemoved = null;
		
		for (ClassifierCheckBox control : this.items)
			if (control.getBoundedClassifier().equals(classifier)) {
				toBeRemoved = control;
				break;
			}
		
		removeItem(toBeRemoved);
	}
	
	public boolean contains(ClassifierCheckBox item) {
		return this.items.contains(item);
	}
	
	protected void refreshListData() {
		setListData(this.items.toArray());
		repaint();
	}

	protected class CellRenderer implements ListCellRenderer {
		public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
			JCheckBox checkbox = (JCheckBox) value;
			checkbox.setBackground(isSelected ? getSelectionBackground() : getBackground());
			checkbox.setForeground(isSelected ? getSelectionForeground() : getForeground());
			checkbox.setEnabled(isEnabled());
			checkbox.setFont(getFont());
			checkbox.setFocusPainted(false);
			checkbox.setBorderPainted(true);
			checkbox.setBorder(isSelected ? UIManager.getBorder("List.focusCellHighlightBorder") : noFocusBorder);
			return checkbox;
		}
	}
}