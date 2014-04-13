package video_interop;

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

import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Segment;
import utils.opengl.OpenGLUtils;
import video_interop.datatypes.SegmentedVideoFrame;

public class OpenGLFeeder extends JFrame implements VideoFeeder {
	private final static int NUM_CHANNELS = 4;
	private final static int LENGTH_IN_FRAMES = 170;
	private final static int FRAME_WIDTH = 854;
	private final static int FRAME_HEIGHT = 640;
	private final static int VERTICAL_MARGIN = 10;
	private final static int OBJECT_HEIGHT = 128;
	
	private byte[] buffer = new byte[FRAME_WIDTH * FRAME_HEIGHT * NUM_CHANNELS];
	
	private volatile boolean paused = false;
	private volatile int position = 0;
	
	/** the list of the objects that should be painted on the canvas */
	private Set<Segment> objects = new HashSet<Segment>();
	
	/** The background segment (used for negative example when there are no other segments in the image) */
	private Segment background = null;
	
	private JFileChooser dlgOpen = new JFileChooser("textures/part");	
	
	
	private final JPanel panel = new JPanel();
	private final JButton btnBrowse = new JButton("Browse and add image...");
	private final JButton btnAdd = new JButton("Add to canvas");
	
	public OpenGLFeeder() {
		setupUI();
		
		try {
			ByteImage bgImage = ByteImage.loadFromFile("textures/part/background.png");
			this.background = new Segment(bgImage, 0, 0, 128, 128);
		}
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		setVisible(true);
	}
	
	private void setupUI() {
		
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		setBounds(600, 100, FRAME_WIDTH, FRAME_HEIGHT);
		getContentPane().setLayout(new BorderLayout(0, 0));
		
		getContentPane().add(this.panel, BorderLayout.CENTER);
		this.panel.setLayout(null);
		this.btnBrowse.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				dlgOpen.setMultiSelectionEnabled(true);
				
				if (dlgOpen.showDialog(OpenGLFeeder.this, "Open") != JFileChooser.APPROVE_OPTION)
					return;
				
				try {
					
					for (File f : dlgOpen.getSelectedFiles()) {
						ByteImage image = ByteImage.loadFromFile(f);
						// Add this image as a segment to the list of objects
						OpenGLFeeder.this.objects.add(new Segment(image, 0, getNextYPosition(), image.getWidth(), image.getHeight()));
					}
					
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				
			}
		});
		this.btnBrowse.setBounds(58, 98, 173, 23);
		
		this.panel.add(this.btnBrowse);
		this.btnAdd.setBounds(306, 199, 118, 23);
		
		this.panel.add(this.btnAdd);
		this.btnRunOneFrame.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				ByteImage frame = getNextFrame();
				
				try {
					ImageIO.write(frame.getBufferedImage(), "png", new File("D:\\frame-dump\\frame-" + position + ".png"));
				} catch (IOException e1) {
					e1.printStackTrace();
				}
			}
		});
		this.btnRunOneFrame.setBounds(58, 199, 105, 23);
		
		this.panel.add(this.btnRunOneFrame);
		this.btnRunAll.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				
				for (int i = 0 ; i < LENGTH_IN_FRAMES ; i++) {
					ByteImage frame = getNextFrame();
					try {
						ImageIO.write(frame.getBufferedImage(), "png", new File("D:\\frame-dump\\frame-" + position + ".png"));
					} catch (IOException e1) {
						e1.printStackTrace();
					}
				}
				
			}
		});
		this.btnRunAll.setBounds(58, 247, 89, 23);
		
		this.panel.add(this.btnRunAll);
	}
	
	private void drawObjectsOnCanvas() {
		// Clear the canvas
		Arrays.fill(buffer, (byte)255);
		
		for (Segment object : this.objects) {
			// reset the position of the objects if necessary
			if (position == 0)
				object.x = 0;
			
			// Draw this object on canvas
			OpenGLUtils.drawOnBuffer(buffer, object.getByteImage(), FRAME_WIDTH, FRAME_HEIGHT, object.getRectangle());
			
			// Move the objects!
			if (!this.paused)
				object.x += 3;
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
			s.x = position * 3;
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
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	private static int y = 0;
	private final JButton btnRunOneFrame = new JButton("Run one Frame");
	private final JButton btnRunAll = new JButton("Run all");
	
	private static int getNextYPosition() {
		int result = y;
		y += OBJECT_HEIGHT + VERTICAL_MARGIN;
		return result;
	}
}
