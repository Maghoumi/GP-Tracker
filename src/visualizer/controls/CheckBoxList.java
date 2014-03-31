package visualizer.controls;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;

public class CheckBoxList extends JList {
	protected static Border noFocusBorder = new EmptyBorder(1, 1, 1, 1);
	
	/** The list of the items of this list */
	private ArrayList<ClassifierCheckBox> items;

	public CheckBoxList() {
		items = new ArrayList<ClassifierCheckBox>();
		
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
		this.items.add(item);
		refreshListData();
	}
	
	public void removeItem(int index) {
		this.items.remove(index);
		refreshListData();
	}
	
	public void removeItem (ClassifierCheckBox item) {
		this.items.remove(item);
		refreshListData();
	}
	
	public boolean contains(ClassifierCheckBox item) {
		return this.items.contains(item);
	}
	
	protected void refreshListData() {
		setListData(this.items.toArray());
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