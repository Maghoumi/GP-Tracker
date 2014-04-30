package utils.gui;

import javax.swing.*;
import javax.swing.border.*;

import utils.Classifier;

import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;

public class CheckBoxList extends JScrollPane {
	protected static Border noFocusBorder = new EmptyBorder(1, 1, 1, 1);
	
	protected JList<ClassifierCheckBox> list = new JList<>();
	
	/** The list of the items of this list */
	private List<ClassifierCheckBox> items = new ArrayList<ClassifierCheckBox>();

	public CheckBoxList() {
		setViewportView(list);
		list.setCellRenderer(new CellRenderer());

		list.addMouseListener(new MouseAdapter() {
			public void mousePressed(MouseEvent e) {
				int index = list.locationToIndex(e.getPoint());

				if (index != -1) {
					JCheckBox checkbox = (JCheckBox) list.getModel().getElementAt(index);
					if (e.getX() >= 0 && e.getX() <= 15)	// If the X of the clicked spot is in the boundaries of the checkbox mark ==> we should flip state :D
						checkbox.setSelected(!checkbox.isSelected());	// smart condition, eh? :D
					repaint();
				}
			}
		});

		list.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
//		setBorder(new BevelBorder(BevelBorder.LOWERED, null, null, null, null));
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
		ClassifierCheckBox[] arrayed = new ClassifierCheckBox[items.size()];
		this.items.toArray(arrayed);
		list.setListData(arrayed);
		repaint();
	}
	
	public Object getSelectedValue() {
		return list.getSelectedValue();
	}

	protected class CellRenderer implements ListCellRenderer {
		public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
			JCheckBox checkbox = (JCheckBox) value;
			checkbox.setBackground(isSelected ? list.getSelectionBackground() : list.getBackground());
			checkbox.setForeground(isSelected ? list.getSelectionForeground() : list.getForeground());
			checkbox.setEnabled(isEnabled());
			checkbox.setFont(getFont());
			checkbox.setFocusPainted(false);
			checkbox.setBorderPainted(true);
			checkbox.setBorder(isSelected ? UIManager.getBorder("List.focusCellHighlightBorder") : noFocusBorder);
			return checkbox;
		}
	}

	public void selectAll() {
		modifyAll(true);
	}
	
	public void selectNone() {
		modifyAll(false);
	}
	
	protected void modifyAll(boolean selected) {
		for (ClassifierCheckBox item : this.items) {
			item.setSelected(selected);
		}
		
		repaint();
	}
}