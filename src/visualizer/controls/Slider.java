package visualizer.controls;

import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import javax.swing.JSlider;
import javax.swing.plaf.basic.BasicSliderUI;

/**
 * An extension of Swing's JSlider. This slider does not have the mouse ticking
 * problem of the JSlider class. Nevertheless, it still does not support tracker
 * dragging.
 * 
 * @author Mehran Maghoumi (code taken from StackOverflow)
 *
 */
public class Slider extends JSlider {

	public Slider() {
		super();
		MouseListener[] listeners = getMouseListeners();
		for (MouseListener l : listeners)
			removeMouseListener(l); // remove UI-installed TrackListener
		final BasicSliderUI ui = (BasicSliderUI) getUI();
		BasicSliderUI.TrackListener tl = ui.new TrackListener() {

			@Override
			public void mousePressed(MouseEvent e) {
				Point p = e.getPoint();
				int value = ui.valueForXPosition(p.x);

				setValue(value);
			}

			// this is where we jump to absolute value of click
			@Override
			public void mouseReleased(MouseEvent e) {
				Point p = e.getPoint();
				int value = ui.valueForXPosition(p.x);

				setValue(value);
			}

			// disable check that will invoke scrollDueToClickInTrack
			@Override
			public boolean shouldScroll(int dir) {
				return false;
			}
		};
		addMouseListener(tl);
	}
}
