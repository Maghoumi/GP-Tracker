package invoker;

import java.io.File;
import java.io.IOException;
import java.lang.Thread.UncaughtExceptionHandler;

import javax.swing.UIManager;

import org.apache.commons.io.FileUtils;

import feeder.GLOneByOneFeeder;
import utils.SuccessListener;
import visualizer.GLVisualizer;
import visualizer.Visualizer;

public class GLOneByOneInvoker extends Invoker implements SuccessListener {

	public GLOneByOneInvoker(String sessionPrefix, int initialNumTextures, int numTextures, String path) {
		super(new String[] { "-file", "bin/gp/object-tracker.params" }, sessionPrefix);
		this.feeder = new GLOneByOneFeeder(initialNumTextures, numTextures, path);
		this.visualizer = new GLVisualizer(this);
		visualizer.addSuccessListener(this);
	}
	
	public static void main(String[] args) throws Throwable {
		UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		Thread.setDefaultUncaughtExceptionHandler(new UncaughtExceptionHandler() {
			
			@Override
			public void uncaughtException(Thread arg0, Throwable arg1) {
				arg1.printStackTrace();
				System.exit(1); 
			}
		});
		
		String prefix;
		if (args.length == 0)
			prefix = null;
		else
			prefix = args[0];
		
		int initialNumTextures = 4;
		if (args.length != 0)
			initialNumTextures = Integer.parseInt(args[1]);
		
		int numTextures = 5;
		
		if (args.length != 0)
			numTextures = Integer.parseInt(args[2]);
		
		String path = "textures/gecco-textures/easy";
		if (args.length != 0)
			path = args[3];
		
		new GLOneByOneInvoker(prefix, initialNumTextures, numTextures, path).start();
	}
	
	DelayedExitThread th = null;	// Will delay System.exit() so that the visualizer will achieve its peek FPS
	boolean justBeenNotified = false;	// Will prevent getting too many notifications of success! Yup! Crappy but oh well

	@Override
	public void notifySuccess(Visualizer visualizer) {
		if (justBeenNotified)
			return;
		
		if (th != null)
			return;
		
		justBeenNotified = true;
		
		new Thread(new Runnable() {
			
			@Override
			public void run() {
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				
				justBeenNotified = false;				
			}
		}).start();
		
		// Attempt to add the next texture to the canvas
		boolean addResult = ((GLOneByOneFeeder)this.feeder).addNextTexture();
		
		if (addResult)
			return;
		
		th = new DelayedExitThread();
		th.visualizer = visualizer;
		th.start();
	}
	
	private class DelayedExitThread extends Thread {
		
		public Visualizer visualizer;

		@Override
		public void run() {
			try {
				Thread.sleep(5000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			System.out.println("===================================");
			System.out.println(visualizer.getFramerate());
			
			File success = new File("stat-dump/success.log");
			long timeStamp = System.currentTimeMillis();
			
			try {
				FileUtils.writeStringToFile(success, timeStamp + "\t" + visualizer.getFramerate() + System.lineSeparator(), true);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			System.exit(0);			
		}
		
	}

}
