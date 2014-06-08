package invoker;

import java.io.File;
import java.io.IOException;
import java.lang.Thread.UncaughtExceptionHandler;

import javax.swing.UIManager;

import org.apache.commons.io.FileUtils;

import feeder.GLAllFeeder;
import utils.SuccessListener;
import visualizer.GLVisualizer;
import visualizer.Visualizer;

public class GLAllInvoker extends Invoker {
	
	public GLAllInvoker(String sessionPrefix, int numTextures, String path) {
		super(new String[] { "-file", "bin/gp/object-tracker.params" }, sessionPrefix);
		this.feeder = new GLAllFeeder(numTextures, path);
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
		
		int numTextures = 16;
		
		if (args.length != 0)
			numTextures = Integer.parseInt(args[1]);
		
		String path = "textures/gecco-textures/hard";
		if (args.length != 0)
			path = args[2];
		
		new GLAllInvoker(prefix, numTextures, path).start();
	}
	
	DelayedExitThread th = null;

	@Override
	public void notifySuccess(Visualizer visualizer) {
		if (th != null)
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
			
			try {
				File success = new File("stat-dump/success.log");
				long timeStamp = System.currentTimeMillis();
				GLVisualizer glvis = (GLVisualizer) visualizer;
				int numPermanentOrphans = glvis.getPermanentOrphans().size();
				String output = String.format("%d\t%.2f\t%d" + System.lineSeparator(), timeStamp, visualizer.getFramerate(), numPermanentOrphans);
				FileUtils.writeStringToFile(success, output, true);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			try {
				File orphans = new File("stat-dump/" + (sessionPrefix == null ? "" : sessionPrefix + ".") +"orphans.log");
				GLVisualizer glvis = (GLVisualizer) visualizer;
				StringBuilder output = new StringBuilder();
				
				for (String s : glvis.getPermanentOrphans())
					output.append(s + System.lineSeparator());
				
				FileUtils.writeStringToFile(orphans, output.toString(), false);
			} catch (IOException e) {
				e.printStackTrace();
			}
			
			
			System.exit(10);			
		}		
	}
	
	@Override
	public void notifyFailure(String reason) {
		File failure = new File("stat-dump/success.log");
		long timeStamp = System.currentTimeMillis();
		
		try {
			FileUtils.writeStringToFile(failure, timeStamp + "\t" + reason + System.lineSeparator(), true);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.exit(10);		
	}
	
	
}
