package video_interop;

import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.UIManager;

import java.awt.BorderLayout;

import javax.swing.JPanel;
import javax.swing.JButton;

import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Segment;
import utils.opengl.OpenGLUtils;
import video_interop.datatypes.SegmentedVideoFrame;
import visualizer.Visualizer;

import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashSet;

public class OpenGLFeeder extends JFrame implements VideoFeeder {
	
	private final int NUM_CHANNELS = 4;
	private final int LENGTH_IN_FRAMES = 170;
	private final int FRAME_WIDTH = 640;
	private final int FRAME_HEIGHT = 480;
	
	private byte[] buffer = new byte[FRAME_WIDTH * FRAME_HEIGHT * NUM_CHANNELS];
	
	private volatile boolean paused = false;
	private volatile boolean alive = false;
	private volatile int position = 0;
	
	private Thread workerThread = null;
	private Visualizer owner;
	
	/** the list of the objects that should be painted on the canvas */
	private HashSet<Segment> objects = new HashSet<Segment>();
	
	private JFileChooser dlgOpen = new JFileChooser("textures/part");	
	
	
	private final JPanel panel = new JPanel();
	private final JButton btnBrowse = new JButton("Browse and add image...");
	private final JButton btnAdd = new JButton("Add to canvas");
	
	public OpenGLFeeder() {
		setupUI();
		this.workerThread = new Thread(this);
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
			OpenGLUtils.drawOnBuffer(buffer, object.getByteImage(), this.FRAME_WIDTH, this.FRAME_HEIGHT, object.getRectangle());
			
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
	public void getAndPassNextFrame() {
		this.pause();
		
		ByteImage newFrame = getNextFrame();
		this.owner.setCurrentFrame(this.position);
		
		this.owner.passNewFrame(new SegmentedVideoFrame(newFrame, objects));		
	}
	
	@Override
	public int getLengthInFrames() {
		return this.LENGTH_IN_FRAMES;
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
	public void setOwner(Visualizer visualizer) {
		this.owner = visualizer;
		this.owner.setVideoLength(LENGTH_IN_FRAMES);
	}

	@Override
	public int getFrameWidth() {
		return this.FRAME_WIDTH;
	}

	@Override
	public int getFrameHeight() {
		return this.FRAME_HEIGHT;
	}

	@Override
	public void run() {
		try {
			
			while (this.alive) {
				
				//Run one frame, pass whatever to the owner :-)
				ByteImage newFrame = getNextFrame();
				this.owner.setCurrentFrame(this.position);
				this.owner.passNewFrame(new SegmentedVideoFrame(newFrame, objects));
				
				if (!this.paused)
					position++;
			}
			
		}
		catch (Throwable e) {
			e.printStackTrace();
			System.exit(1);
		}	
	}

	@Override
	public void start() {
		this.alive = this.workerThread.isAlive();
		
		if (!this.alive) {
			this.workerThread.start();
			this.alive = true;
		}
	}

	@Override
	public void stop() {
		this.alive = false;
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
		y += 128 + 10;
		return result;
	}
}
