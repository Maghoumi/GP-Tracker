package feeder;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JPanel;

import feeder.datatypes.SegmentedVideoFrame;
import utils.ImageFilterProvider;
import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Segment;
import utils.opengl.OpenGLUtils;

public class GLFeeder extends JFrame implements VideoFeeder {
	//TODO document me
	protected final static int NUM_CHANNELS = 4;
	protected final static int LENGTH_IN_FRAMES = 170;
	protected final static int FRAME_WIDTH = 854;
	protected final static int FRAME_HEIGHT = 640;
	protected final static int VERTICAL_MARGIN = 10;
	protected final static int OBJECT_HEIGHT = 64;
	
	protected byte[] buffer = new byte[FRAME_WIDTH * FRAME_HEIGHT * NUM_CHANNELS];
	
	protected volatile boolean paused = false;
	protected volatile int position = 0;
	
	/** the list of the objects that should be painted on the canvas */
	protected Set<Segment> objects = new HashSet<Segment>();
	
	/** The background segment (used for negative example when there are no other segments in the image) */
	protected Segment background = null;
	
	protected JFileChooser dlgOpen = new JFileChooser("textures/part-small");	
	
	
	protected final JPanel panel = new JPanel();
	protected final JButton btnBrowse = new JButton("Browse and add image...");
	
	public GLFeeder() {
		setupUI();
				
		try {
			ByteImage bgImage = ByteImage.loadFromFile("textures/part/background.png");
			this.background = new Segment(bgImage, 0, 0, 128, 128, "background");
		}
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		setVisible(true);
	}
	
	protected void setupUI() {
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		setBounds(1100, 100, 300, 150);
		getContentPane().setLayout(new BorderLayout(0, 0));
		
		getContentPane().add(this.panel, BorderLayout.CENTER);
		this.panel.setLayout(null);
		this.btnBrowse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				dlgOpen.setMultiSelectionEnabled(true);
				
				if (dlgOpen.showDialog(GLFeeder.this, "Open") != JFileChooser.APPROVE_OPTION)
					return;
				
				try {
					
					for (File f : dlgOpen.getSelectedFiles()) {
						ByteImage image = ByteImage.loadFromFile(f);
						// Add this image as a segment to the list of objects
						GLFeeder.this.objects.add(new Segment(image, 0, getNextYPosition(), image.getWidth(), image.getHeight(), f.getName()));
					}
					
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				
			}
		});
		this.btnBrowse.setBounds(10, 42, 173, 23);
		
		this.panel.add(this.btnBrowse);
	}
	
	protected void drawObjectsOnCanvas() {
		// Clear the canvas
		Arrays.fill(buffer, (byte)255);
		
		for (Segment object : this.objects) {
			// reset the position of the objects if necessary
			if (position == 0)
				object.getBounds().x = 0;
			
			// Draw this object on canvas
			OpenGLUtils.drawOnBuffer(buffer, object.getByteImage(), FRAME_WIDTH, FRAME_HEIGHT, object.getBounds());
			
			// Move the objects!
			if (!this.paused)
				object.getBounds().x += 3;
		}
	}
	
	@Override
	public ByteImage getNextFrame() {
		if (position == LENGTH_IN_FRAMES)
			position = 0;
		
		drawObjectsOnCanvas();
		
		return new ByteImage(buffer, FRAME_WIDTH, FRAME_HEIGHT);
	}
	
	@Override
	public SegmentedVideoFrame getNextSegmentedFrame() {
		ByteImage newFrame = getNextFrame();
		
		if (!this.paused)
			position++;
		
		return new SegmentedVideoFrame(newFrame, objects, background);
	}
	
	@Override
	public int getLengthInFrames() {
		return LENGTH_IN_FRAMES;
	}
	
	@Override
	public void restart() {
		this.position = 0;
	}
	
	@Override
	public int getCurrentPosition() {
		return this.position;
	}
	
	@Override
	public void setCurrentPosition(int position) {
		this.position = position;
		for (Segment s : this.objects) {
			s.getBounds().x = position * 3;
		}
	}
	
	@Override
	public int getFrameWidth() {
		return FRAME_WIDTH;
	}

	@Override
	public int getFrameHeight() {
		return FRAME_HEIGHT;
	}

	@Override
	public void pause() {
		this.paused = true;
	}

	@Override
	public void resume() {
		this.paused = false;
	}

	@Override
	public boolean togglePaused() {
		this.paused = !this.paused;
		return this.paused;
	}

	@Override
	public boolean isPaused() {
		return this.paused;
	}
	
	protected static int y = 0;
	
	protected static int getNextYPosition() {
		int result = y;
		y += OBJECT_HEIGHT + VERTICAL_MARGIN;
		return result;
	}
}
