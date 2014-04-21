package visualizer;

import feeder.datatypes.SegmentedVideoFrame;
import invoker.Invoker;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.net.URL;
import java.nio.Buffer;
import java.nio.IntBuffer;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

import javax.media.opengl.GL;
import javax.media.opengl.GL2;
import javax.media.opengl.GLAutoDrawable;
import javax.media.opengl.GLCapabilities;
import javax.media.opengl.GLEventListener;
import javax.media.opengl.GLProfile;
import javax.media.opengl.awt.GLCanvas;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JToggleButton;
import javax.swing.border.EmptyBorder;
import javax.swing.border.EtchedBorder;
import javax.swing.border.TitledBorder;

import net.miginfocom.swing.MigLayout;
import utils.cuda.datatypes.ByteImage;
import utils.cuda.datatypes.Classifier;
import utils.cuda.datatypes.ClassifierSet;
import utils.cuda.datatypes.ClassifierSet.ClassifierAllocationResult;
import utils.cuda.datatypes.Segment;
import utils.opengl.OpenGLUtils;
import visualizer.controls.CheckBoxList;
import visualizer.controls.ClassifierCheckBox;
import visualizer.controls.Slider;

import com.jogamp.opengl.util.Animator;

import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

/**
 * And finally... The JCuda/OpenGL visualizer. This class is basically a JFrame designed with WindowBuilder that has an OpenGL Canvas on it. This
 * class is able to run the visualizers evolved by ECJ on CUDA and paint their results using OpenGL. It also provides other functionalities which are
 * yet to be determined!
 * 
 * @author Mehran Maghoumi
 * 
 */
public class GLVisualizer extends JFrame implements GLEventListener, Visualizer {

	public static final URL ICON_PLAY = GLVisualizer.class.getResource("/icons/play-icon.png");
	public static final URL ICON_PAUSE = GLVisualizer.class.getResource("/icons/pause-icon.png");
	public static final URL ICON_FFW = GLVisualizer.class.getResource("/icons/ff-icon.png");
	public static final URL ICON_RW = GLVisualizer.class.getResource("/icons/rw-icon.png");
	public static final URL KERNEL_PATH = GLVisualizer.class.getResource("/cuda/kernels/visualizer/visualizer-kernel.cu");

	private static final int REPORT_FREQUENCY = 1;

	/** The kernel wrapper object to use for invoking CUDA */
	private GLVisualizerKernel kernel;

	/** The width of the image (frame) to be displayed */
	private int imageWidth;

	/** The height of the image (frame) to be displayed */
	private int imageHeight;

	/** The shared OpenGL/CUDA buffer */
	private int glBuffer;

	/** OpenGL texture handle */
	private int glTexture;

	/** OpenGL animator that constantly calls the Display() function */
	private Animator animator;

	/** A flag to indicate that this visualizer is loaded and ready for operation */
	private boolean isReady = false;

	/** The frame and individual queue capacity */
	private final int QUEUE_CAPACITY = 1;

	/** The frame queue which holds the frames and the discovered segments that we want to display using OpenGL */
	private BlockingQueue<SegmentedVideoFrame> segmentedFrameQueue = new ArrayBlockingQueue<>(QUEUE_CAPACITY);

	/**
	 * A set of the evolved classifiers. When a new classifier is passed to this object, that classifier is stored in this set. Also, when the CUDA
	 * kernel needs to be called, the classifiers in this set are passed to the kernel function as GP individuals
	 */
	private ClassifierSet classifiers = new ClassifierSet();

	/** The associated invoker reference */
	private Invoker invoker;

	/** A flag indicating whether filtering is enabled or not */
	private boolean filterEnabled = true;

	/** FPS helper variable */
	private long prevTime = -1;
	/** FPS helper variable */
	private int step = 0;

	private JPanel contentPane;
	private JPanel pnlContent;
	private JPanel pnlOpenGL;
	private GLCanvas glComponent;
	private JPanel panel;
	private JButton btnResetSize;
	private JPanel pnlRight;
	private CheckBoxList checkBoxList;
	private JButton btnSeedRetrain;
	private JPanel pnlCenter;
	private JPanel pnlVideoControls;
	private Slider sliderVidPosition;
	private JButton btnRew;
	private JButton btnPlayPause;
	private JButton btnFF;
	private JCheckBox chckbxShowConflicts;
	private JButton btnRetrain;
	private JCheckBox chckbxThreshold;
	private JSpinner spnThreshold;
	private JPanel panel_3;
	private JLabel lblNewLabel_1;
	private JPanel panel_4;
	private JLabel lblOpacity;
	private JSpinner spnOpacity;
	private JToggleButton tglbtnGpSystem;

	/**
	 * Initializes the visualizer with the given image width and height. Use this constructor if you want to dynamically pass data to the visualizer.
	 * 
	 * @param imageWidth
	 *            The width of the image to be visualized
	 * @param imageHeight
	 *            The height of the image to be visualized
	 * 
	 */
	public GLVisualizer(int imageWidth, int imageHeight) {
		this.imageWidth = imageWidth;
		this.imageHeight = imageHeight;

		setup();
	}

	public GLVisualizer(Invoker invoker) {
		this(invoker.getFrameWidth(), invoker.getFrameHeight());
		this.invoker = invoker;
	}

	/**
	 * Fake constructor! Only here for the purposes of Eclipse's WindowBuilder plugin.
	 * 
	 * @wbp.parser.constructor
	 */
	private GLVisualizer() {
		setPreferredSize(new Dimension(500, 550));

		//		setBounds(100, 100, 467, 338);
		setup();
	}

	/**
	 * Sets up the UI elements. Should be used by WindowBuilder. Try not to touch!
	 */
	private void setupUIElements() {
		setTitle("JCuda/JOGL Visualizer");
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		contentPane = new JPanel();
		contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
		setContentPane(contentPane);
		contentPane.setLayout(new BorderLayout(0, 0));
		this.panel = new JPanel();
		this.contentPane.add(this.panel, BorderLayout.CENTER);
		this.panel.setLayout(new BorderLayout(5, 5));

		pnlContent = new JPanel();
		this.pnlContent.setMinimumSize(new Dimension(200, 200));
		this.panel.add(this.pnlContent, BorderLayout.CENTER);
		this.pnlContent.setLayout(new BorderLayout(0, 0));

		pnlRight = new JPanel();
		pnlRight.setBorder(new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null), "Options", TitledBorder.LEADING, TitledBorder.TOP,
				null, null));
		pnlContent.add(pnlRight, BorderLayout.EAST);
		pnlRight.setLayout(new MigLayout("", "[grow]", "[150px:n,fill][center][][][][][][][][][]"));

		checkBoxList = new CheckBoxList();
		pnlRight.add(checkBoxList, "cell 0 0,grow");

		btnSeedRetrain = new JButton("Seed Retrain");
		btnSeedRetrain.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Object selected = checkBoxList.getSelectedValue();
				if (selected == null)
					return;

				Classifier classifier = ((ClassifierCheckBox) selected).getBoundedClassifier();
				invoker.retrain(classifier, true);
			}
		});

		chckbxShowConflicts = new JCheckBox("Show conflicts");
		pnlRight.add(chckbxShowConflicts, "cell 0 1,growx");

		chckbxThreshold = new JCheckBox("Do thresholding");
		chckbxThreshold.setSelected(true);
		pnlRight.add(chckbxThreshold, "cell 0 2,growx");

		panel_3 = new JPanel();
		pnlRight.add(panel_3, "cell 0 3,growx");
		panel_3.setLayout(new BorderLayout(0, 0));

		lblNewLabel_1 = new JLabel(" Threshold:");
		panel_3.add(lblNewLabel_1, BorderLayout.WEST);

		spnThreshold = new JSpinner();
		spnThreshold.setPreferredSize(new Dimension(35, 20));
		panel_3.add(spnThreshold, BorderLayout.EAST);
		spnThreshold.setModel(new SpinnerNumberModel(50, 0, 100, 1));

		this.panel_4 = new JPanel();
		this.pnlRight.add(this.panel_4, "cell 0 4,growx");
		this.panel_4.setLayout(new BorderLayout(0, 0));

		this.lblOpacity = new JLabel(" Opacity:");
		this.panel_4.add(this.lblOpacity, BorderLayout.WEST);

		this.spnOpacity = new JSpinner();
		this.spnOpacity.setModel(new SpinnerNumberModel(50, 0, 100, 1));
		this.spnOpacity.setPreferredSize(new Dimension(35, 20));
		this.panel_4.add(this.spnOpacity, BorderLayout.EAST);
		pnlRight.add(btnSeedRetrain, "cell 0 6,growx");

		btnRetrain = new JButton("Retrain");
		btnRetrain.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Object selected = checkBoxList.getSelectedValue();
				if (selected == null)
					return;

				Classifier classifier = ((ClassifierCheckBox) selected).getBoundedClassifier();
				invoker.retrain(classifier, false);
			}
		});
		pnlRight.add(btnRetrain, "cell 0 7,growx");

		btnResetSize = new JButton("Reset Canvas");
		this.pnlRight.add(this.btnResetSize, "cell 0 8,growx");

		tglbtnGpSystem = new JToggleButton("GP [Enabled]", true);
		tglbtnGpSystem.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (tglbtnGpSystem.isSelected())
					tglbtnGpSystem.setText("GP [Enabled]");
				else
					tglbtnGpSystem.setText("GP [Disabled]");

				invoker.setGPStatus(tglbtnGpSystem.isSelected());
			}
		});
		pnlRight.add(tglbtnGpSystem, "cell 0 10,growx");
		btnResetSize.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				glComponent.setPreferredSize(new Dimension(imageWidth, imageHeight));
				pack();
			}
		});
	}

	/**
	 * Used to pass a new individual to the visualizer. Usually, when a fitter individual is found in the run, this fncDescribe should be called. Note
	 * that the calling thread will be blocked if the individual queue of this instance is full.
	 * 
	 * @param classifier
	 *            The new classifier that is found during the GP run
	 */
	public void passNewClassifier(Classifier classifier) {
		synchronized (this.classifiers) {
			if (this.classifiers.update(classifier))
				this.checkBoxList.addItem(classifier);
		}
	}

	public void removeClassifier(Collection<Classifier> classifiers) {
		synchronized (this.classifiers) {
			for (Classifier c : classifiers) {
				this.checkBoxList.removeItem(c);
				this.classifiers.remove(c);
			}
		}
	}

	/**
	 * Used to pass a new video frame (image) to the visualizer. Usually should be called after grabbing a frame of the video. Note that the calling
	 * thread will be blocked if the frame queue of the visualizer is full
	 * 
	 * @param frame
	 *            The new frame that is obtained from the video/image
	 */
	public void passNewFrame(SegmentedVideoFrame frame, int currentPosition) {
		try {
			segmentedFrameQueue.put(frame);
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}
		setCurrentFrame(currentPosition);
	}

	/**
	 * Initializes the OpenGL canvas and window. Usually this method must be called after the constructor. Note that the calling thread will be
	 * blocked until the canvas is ready and OpenGL thread has done its initialization magic.
	 */
	private void setup() {

		setupUIElements();

		GLProfile profile = GLProfile.get(GLProfile.GL2);
		final GLCapabilities capabilities = new GLCapabilities(profile);
		KeyboardControl keyListener = new KeyboardControl();

		// Create the main frame 
		this.addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				animator.stop();
			}
		});

		pnlCenter = new JPanel();
		pnlContent.add(pnlCenter, BorderLayout.CENTER);
		pnlCenter.setLayout(new BorderLayout(0, 0));
		this.pnlOpenGL = new JPanel();
		pnlCenter.add(pnlOpenGL);
		this.pnlOpenGL.setBorder(new TitledBorder(new EtchedBorder(EtchedBorder.LOWERED, null, null), "OpenGL", TitledBorder.LEADING,
				TitledBorder.TOP, null, null));

		// Initialize the GL component and the animator
		glComponent = new GLCanvas(capabilities);
		glComponent.setFocusable(true);
		glComponent.addGLEventListener(this);
		glComponent.addKeyListener(keyListener);
		this.pnlOpenGL.setLayout(new BorderLayout(0, 0));

		glComponent.setPreferredSize(new Dimension(imageWidth, imageHeight));

		this.pnlOpenGL.add(glComponent);

		animator = new Animator(glComponent);

		pnlVideoControls = new JPanel();
		this.pnlContent.add(this.pnlVideoControls, BorderLayout.SOUTH);
		pnlVideoControls.setLayout(new MigLayout("", "[][][][grow]", "[]"));

		btnRew = new JButton("");
		btnRew.setIcon(new ImageIcon(ICON_RW));
		pnlVideoControls.add(btnRew, "cell 0 0");

		btnPlayPause = new JButton("");
		this.btnPlayPause.setIcon(new ImageIcon(ICON_PAUSE));
		btnPlayPause.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if (GLVisualizer.this.invoker.togglePaused())
					btnPlayPause.setIcon(new ImageIcon(ICON_PLAY));
				else
					btnPlayPause.setIcon(new ImageIcon(ICON_PAUSE));
			}
		});
		btnPlayPause.setFont(new Font("Tahoma", Font.PLAIN, 11));
		pnlVideoControls.add(btnPlayPause, "cell 1 0");

		btnFF = new JButton();
		btnFF.setIcon(new ImageIcon(ICON_FFW));
		this.btnFF.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				throw new RuntimeException("Not implemented yet");
				//												GLVisualizer.this.videoFeeder.pause();
				//												try {
				//													Thread.sleep(1000);
				//												} catch (InterruptedException e1) {
				//													e1.printStackTrace();
				//												}
				//												
				//												GLVisualizer.this.segmentedFrameQueue.clear();
				//												GLVisualizer.this.videoFeeder.getAndPassNextFrame();
			}
		});
		pnlVideoControls.add(btnFF, "cell 2 0");
		sliderVidPosition = new Slider();
		this.sliderVidPosition.addMouseListener(new MouseAdapter() {
			@Override
			public void mousePressed(MouseEvent e) {
				//												GLVisualizer.this.videoFeeder.pause();
				throw new RuntimeException("Not implemented yet");
			}

			@Override
			public void mouseReleased(MouseEvent e) {
				//												GLVisualizer.this.segmentedFrameQueue.clear();
				//												GLVisualizer.this.videoFeeder.setCurrentPosition(sliderVidPosition.getValue());
				//												GLVisualizer.this.videoFeeder.resume();
				throw new RuntimeException("Not implemented yet");
			}
		});
		this.sliderVidPosition.setValue(0);
		this.sliderVidPosition.setMajorTickSpacing(1);
		this.sliderVidPosition.setMinorTickSpacing(1);
		sliderVidPosition.setMinimum(1);
		pnlVideoControls.add(sliderVidPosition, "cell 3 0,growx");
		animator.setUpdateFPSFrames(30, null); // animator will use 30 FPS
		animator.start();

		this.pack();
		this.setVisible(true);
		// Block the calling thread until OpenGL is ready and fully loaded
		waitReady();
	}

	@Override
	public void init(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		// set erase color to white
		gl.glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		gl.glDisable(GL.GL_DEPTH_TEST);

		// Initialize CUDA
		this.kernel = new GLVisualizerKernel();

		initBuffers(drawable);
		initTexture(drawable);
		this.isReady = true; // Indicate that I am ready an loaded! Gimme data!
	}

	public void initBuffers(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();

		int[] buffer = new int[1];

		// Generate buffer
		gl.glGenBuffers(1, IntBuffer.wrap(buffer));
		glBuffer = buffer[0];

		/*
		 * Bind the generated buffer. Note that I am using the pixel unpack buffer here. I think the reason is that I want to bind the data to a
		 * texture later. Per OpenGL's documentation, glTexSubImage2D will use pixel unpack buffer data if a buffer is bound to pixel unpack buffer.
		 */
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);
		// Specify the size of the buffer (no data is pre-loaded in this buffer)
		gl.glBufferData(GL2.GL_PIXEL_UNPACK_BUFFER, imageWidth * imageHeight * 4, (Buffer) null, GL2.GL_DYNAMIC_COPY);
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0); // Unbind pixel unpack buffer

		this.kernel.registerGLBuffer(glBuffer);
	}

	public void initTexture(GLAutoDrawable drawable) {
		GL2 gl = drawable.getGL().getGL2();
		int[] texture = new int[1];

		gl.glGenTextures(1, IntBuffer.wrap(texture));
		glTexture = texture[0];

		gl.glBindTexture(GL2.GL_TEXTURE_2D, glTexture);

		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MIN_FILTER, GL2.GL_LINEAR);
		gl.glTexParameteri(GL2.GL_TEXTURE_2D, GL2.GL_TEXTURE_MAG_FILTER, GL2.GL_LINEAR);

		gl.glTexImage2D(GL2.GL_TEXTURE_2D, 0, GL2.GL_RGBA8, imageWidth, imageHeight, 0, GL2.GL_BGRA, GL2.GL_UNSIGNED_BYTE, (Buffer) null);

		gl.glBindTexture(GL2.GL_TEXTURE_2D, 0); // I put it here!

	}

	@Override
	public void display(GLAutoDrawable drawable) {

		// Will invoke CUDA if necessary
		if (this.invoker == null)
			return;

		if (filterEnabled)
			runCuda(drawable);
		else
			copyCat(drawable);

		GL2 gl = drawable.getGL().getGL2();

		// Bind current buffer to the pixel unpack buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, glBuffer);

		// Bind texture
		gl.glBindTexture(GL2.GL_TEXTURE_2D, glTexture);

		// Ask OpenGL to use the data which exist in the pixel unpack buffer as the texture
		/**
		 * Also note that RGBA was specified earlier, however right now, I have asked OpenGL to do the pixel format conversion
		 */
		gl.glTexSubImage2D(GL2.GL_TEXTURE_2D, 0, 0, 0, imageWidth, imageHeight, GL2.GL_ABGR_EXT, GL2.GL_UNSIGNED_BYTE, 0); //ZERO! NOT NULL! :-) because I want the content of the PIXEL_UNPACK_BUFFER to be used

		// Unbind pixel unpack buffer
		gl.glBindBuffer(GL2.GL_PIXEL_UNPACK_BUFFER, 0);

		gl.glEnable(GL2.GL_TEXTURE_2D);
		gl.glDisable(GL2.GL_DEPTH_TEST);
		gl.glDisable(GL2.GL_LIGHTING);
		gl.glTexEnvf(GL2.GL_TEXTURE_ENV, GL2.GL_TEXTURE_ENV_MODE, GL2.GL_REPLACE);

		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glPushMatrix();
		gl.glLoadIdentity();
		gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();

		gl.glBegin(GL2.GL_QUADS);
		gl.glTexCoord2f(0.0f, 1.0f);
		gl.glVertex2f(-1.0f, -1.0f);

		gl.glTexCoord2f(1.0f, 1.0f);
		gl.glVertex2f(1.0f, -1.0f);

		gl.glTexCoord2f(1.0f, 0.0f);
		gl.glVertex2f(1.0f, 1.0f);

		gl.glTexCoord2f(0.0f, 0.0f);
		gl.glVertex2f(-1.0f, 1.0f);
		gl.glEnd();

		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glPopMatrix();

		gl.glDisable(GL2.GL_TEXTURE_2D);
		gl.glBindTexture(GL2.GL_TEXTURE_2D, 0); // I put it here

		calculateFramerate();

	}

	/**
	 * Helper function that invokes CUDA if necessary. Will also determine if classifiers
	 * have issues and need to be retrained/recreated
	 * 
	 * @param drawable
	 *            OpenGL drawable
	 */
	public void runCuda(GLAutoDrawable drawable) {

		/**
		 * When CUDA is to be called, the classifiers must be locked so that
		 * CUDA can finish processing and then GP can interfere
		 * Otherwise, in the middle of processing, a classifier may be updated
		 * or added to the system and we may not be able to see that it has actually
		 * changed
		 */		
		synchronized (this.classifiers) {

			// This fncDescribe only needs to invoke CUDA kernel if:
			// 		1) There is a new frame in the frame queue 
			//		2) There is a new individual in the individual queue **AND** we
			//		   have had at least one frame in our frame queue
			//
			// If neither of these conditions are true, the OpenGL buffer
			// should not be touched and CUDA should not be invoked

			// Check my queues and see what I should do
			SegmentedVideoFrame frame = segmentedFrameQueue.poll();

			// Either way we need to copy the frame in the queue to the OpenGL buffer
			copyCat(drawable);

			if (frame == null)
				return;

			// We obtain the pointer to all, if there are no classifiers, we have null here
			ClassifierAllocationResult pointerToAll = classifiers.getPointerToAll();

			for (Segment s : frame) {
				float threshold = Float.valueOf(spnThreshold.getValue().toString()).floatValue() / 100f;
				float opacity = Float.valueOf(spnOpacity.getValue().toString()).floatValue() / 100f;

				// If there are no classifiers, then there are no claimers for this texture!
				int claimers = pointerToAll == null ? 0 : kernel.call(invoker, drawable, pointerToAll, s,
						chckbxThreshold.isSelected(), threshold, opacity,
						chckbxShowConflicts.isSelected(), imageWidth, imageHeight);

				// If no classifier has claimed this texture, we should ask the Invoker to train a classifier for this dude
				if (claimers == 0 && invoker.isQueueEmpty()) {
					invoker.evolveClassifier(frame, s);

					if (pointerToAll != null)
						pointerToAll.freeAll();
					
					return;	// No need to do the rest!
				}
			}

			if (pointerToAll != null)
				pointerToAll.freeAll();
			
			if (!invoker.isQueueEmpty())
				return;

			// Resolve retraining issues:
			Set<Classifier> toBeDestroyed = new HashSet<Classifier>(); // A set of classifiers that have problems and need to be destroyed

			inspect: for (Classifier c : classifiers) {
				if (c.getClaimsCount() == 1)
					continue;

				boolean somethingHappened = false;

				for (Segment claimed : c.getClaims()) {

					for (Classifier other : classifiers) {
						if (other == c)
							continue;

						if (other.hasClaimed(claimed) && other.getClaimsCount() == 1 && invoker.isQueueEmpty()) { // claimed by another one as well, the other is very certain about his claim
							toBeDestroyed.add(c);
							somethingHappened = true;
							break inspect;
						}
						else if (other.hasClaimed(claimed)) { // claimed by both!
							toBeDestroyed.add(other);
							toBeDestroyed.add(c);
							somethingHappened = true;
							break inspect;
						}
					}

				}

				if (!somethingHappened && c.getClaimsCount() > 1) {
					// If here ==> this classifier has claimed multiple textures!!
					// Destroy the sucker
					toBeDestroyed.add(c);
				}
			} // end-for (inspect)

			// Get rid of problematic classifiers
			removeClassifier(toBeDestroyed);

		}// end-synchronized

	}

	/**
	 * Bypasses CUDA and will directly copy the frame in the frame queue to the OpenGL buffer.
	 * 
	 * @param drawable
	 *            OpenGL drawable
	 */
	private void copyCat(GLAutoDrawable drawable) {
		// Grab the next frame from the frame queue
		ByteImage frame = null;

		try {
			frame = segmentedFrameQueue.take().getFrame();
		} catch (InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}

		OpenGLUtils.copyBuffer(frame, glBuffer, drawable);
	}

	private void calculateFramerate() {
		step++;
		long currentTime = System.nanoTime();
		if (prevTime == -1) {
			prevTime = currentTime;
		}
		long diff = currentTime - prevTime;
		if (diff > 1e9) {
			double fps = (diff / 1e9) * step;
			String t = String.format("%.2f", fps);
			((TitledBorder) this.pnlOpenGL.getBorder()).setTitle("OpenGL - " + t + "fps");
			this.pnlOpenGL.repaint();

			prevTime = currentTime;
			step = 0;
		}
	}

	@Override
	public void reshape(GLAutoDrawable drawable, int x, int y, int width, int height) {
		GL2 gl = drawable.getGL().getGL2();
		gl.glMatrixMode(GL2.GL_PROJECTION);
		gl.glLoadIdentity();
		gl.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
		gl.glMatrixMode(GL2.GL_MODELVIEW);
		gl.glLoadIdentity();
	}

	@Override
	public void dispose(GLAutoDrawable drawable) {/* Do nothing */
	}

	/**
	 * Blocks the calling thread until this instance of the visualizer is loaded and ready to accept data.
	 */
	public void waitReady() {
		System.out.println(System.lineSeparator() + "Waiting for the visualizer to get ready...");

		while (!isReady)
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
	}

	/**
	 * A key adapter for JOGL key even functionalities
	 * 
	 * @author Mehran Maghoumi
	 */
	public class KeyboardControl extends KeyAdapter {
		public void keyTyped(KeyEvent e) {
			char c = e.getKeyChar();
			if (c == 'f') {
				filterEnabled = !filterEnabled;
			}
			if (c == ' ') {
				invoker.togglePaused();
			}
		}
	}

	/**
	 * Set the length of the video that this visualizer shows. This method will set the maximum value for the slider that shows the video position.
	 * 
	 * @param lengthInFrames
	 *            The video length in frames
	 */
	public void setVideoLength(int lengthInFrames) {
		this.sliderVidPosition.setMaximum(lengthInFrames);
	}

	/**
	 * Set the current position in the video. This will update the slider
	 * 
	 * @param frameNumber
	 */
	public synchronized void setCurrentFrame(Integer frameNumber) {
		this.sliderVidPosition.setValue(frameNumber);
	}

	@Override
	public void setVideoDimensions(int width, int height) {
		this.imageWidth = width;
		this.imageHeight = height;
		setPreferredSize(new Dimension(width, height));
		pack();
	}
}
