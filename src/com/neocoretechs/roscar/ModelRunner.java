/**
 * COMPILE_OPTIONS --add-modules=jdk.incubator.vector <br>
 * RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 <br>
 * <p>
 * Practical inference in a single Java file. <o>
 * Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models. <p>
 * Multi-threaded matrix vector multiplication routines implemented using Java's Vector API. <p>
 * Accepts commands from RosJavaLite bus topics, including sensors and status, and fuses those
 * into coherent responses to perform embodied field robotics. Derived from Oracle model runner.
 * Uses LSH indexing and semantic retrieval to provide virtually unlimited context with semantic augmentation.<p>
 * Remember: Llama models use GPT2 vocabulary while non-Llama models use Llama vocabulary!
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2025
 */
package com.neocoretechs.roscar;

import jdk.incubator.vector.*;
import stereo_msgs.StereoImage;
import trajectory_msgs.ComeToHeadingStamped;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.Externalizable;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.Serializable;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Iterator;
import java.util.OptionalInt;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.LongConsumer;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.select.Elements;

import org.ros.concurrent.CancellableLoop;
import org.ros.concurrent.CircularBlockingDeque;
import org.ros.message.MessageListener;
import org.ros.namespace.GraphName;
import org.ros.node.AbstractNodeMain;
import org.ros.node.ConnectedNode;
import org.ros.node.topic.Publisher;
import org.ros.node.topic.Subscriber;

import org.json.JSONObject;

import com.neocoretechs.relatrix.client.asynch.AsynchRelatrixClientTransaction;
import com.neocoretechs.relatrix.key.NoIndex;
import com.neocoretechs.relatrix.parallel.SynchronizedThreadManager;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.relatrix.Relation;
import com.neocoretechs.rocksack.TransactionId;

import com.neocoretechs.roscar.GGUF.GGMLTensorEntry;
import com.neocoretechs.roscar.chatformat.ChatFormat;
import com.neocoretechs.roscar.chatformat.ChatFormatInterface;
import com.neocoretechs.roscar.chatformat.MistralChatFormat;
import com.neocoretechs.roscar.chatformat.PromptFrame;
import com.neocoretechs.roscar.relatrix.RelatrixLSH;
import com.neocoretechs.roscar.tokenizer.Tokenizer;
import com.neocoretechs.roscar.tokenizer.TokenizerInterface;

import diagnostic_msgs.DiagnosticStatus;
import diagnostic_msgs.KeyValue;

import com.neocoretechs.rocksack.Alias;
import com.neocoretechs.relatrix.DuplicateKeyException;

public class ModelRunner extends AbstractNodeMain {
	private static final Log log = LogFactory.getLog(ModelRunner.class);
	// Batch-size used in prompt evaluation.
	private static int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);
	public final static boolean DEBUG = false;
	public static boolean DISPLAY_METADATA = false;
	AsynchRelatrixClientTransaction dbClient = null;
	//static RelatrixTransaction dbClient = null;
	TransactionId xid = null;
	Alias tensorAlias = null;
	// metadata dump
	public static BufferedWriter outputStream = null;
	public static PrintWriter output = null;
	public static FileWriter fileWriter = null;
	Llama model = null;
	Sampler sampler = null;
	Options options = null;
	PromptFrame promptFrame = null;
	public static final String SYSTEM_PROMPT = "/system_prompt";
	public static final String USER_PROMPT = "/user_prompt";
	public static final String ASSIST_PROMPT = "/assist_prompt";
	public static final String LLM = "/model";

	CircularBlockingDeque<String> messageQueue = new CircularBlockingDeque<String>(1024);

	protected Object mutex = new Object();
	protected CountDownLatch modelLatch = new CountDownLatch(1);
	protected CountDownLatch dbLatch = new CountDownLatch(1);

	static long MESSAGE_THRESHOLD = 5000; // ms minimum between subscribed message reception
	static long lastImageTime = System.currentTimeMillis();

	static RelatrixLSH relatrixLSH = null;
	ChatFormatInterface chatFormat;

	static class EulerTime {
		sensor_msgs.Imu euler;
		long eulerTime = 0L;
	}
	EulerTime euler = new EulerTime();

	static class RangeTime {
		std_msgs.String range;
		long rangeTime = 0L;
		public String toJSON() {
			return String.format("{range=%s}", range.getData());
		}
	}
	RangeTime ranges = new RangeTime();

	static Sampler selectSampler(int vocabularySize, float temperature, float topp, long rngSeed) {
		Sampler sampler;
		if (temperature == 0.0f) {
			// greedy argmax sampling: take the token with the highest probability
			sampler = Sampler.ARGMAX;
		} else {
			// we sample from this distribution to get the next token
			RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
			Sampler innerSampler;
			if (topp <= 0 || topp >= 1) {
				// simply sample from the predicted probability distribution
				innerSampler = new CategoricalSampler(rng);
			} else {
				// top-p (nucleus) sampling, clamping the least likely tokens to zero
				innerSampler = new ToppSampler(vocabularySize, topp, rng);
			}
			sampler = logits -> {
				// apply the temperature to the logits
				logits.divideInPlace(0, logits.size(), temperature);
				// apply softmax to the logits to get the probabilities for next token
				logits.softmaxInPlace(0, logits.size());
				return innerSampler.sampleToken(logits);
			};
		}
		return sampler;
	}

	public static String getContextInfo(Llama model, List<Integer> conversationTokens) {
		return String.format("%d out of %d context tokens used (%d tokens remaining)%n",
				conversationTokens.size(),
				model.configuration().contextLength,
				model.configuration().contextLength - conversationTokens.size());
	}
	/**
	 * After model is loaded we can preserve special tokens extracted
	 * @param model
	 */
	public void extractSpecialTokens(Llama model) {
		try {
			FileOutputStream fileWriter = new FileOutputStream(options.modelPath().toString()+"-specialTokens.ser", false);
			ObjectOutputStream outputStream = new ObjectOutputStream(fileWriter);
			outputStream.writeObject(model.tokenizer().getSpecialTokens());
			outputStream.flush();
			outputStream.close();
			fileWriter.close();
		} catch(IOException e) {
			log.error("Could not open file " + options.modelPath().toString()+"-specialTokens.ser\r\n"+e);
		}
	}
	/**
	 * After model is loaded, we can preserve Merges.
	 * @param model
	 */
	public void extractMerges(Llama model) {
		try {
			FileOutputStream fileWriter = new FileOutputStream(options.modelPath().toString()+"-merges.ser", false);
			ObjectOutputStream outputStream = new ObjectOutputStream(fileWriter);
			outputStream.writeObject(model.tokenizer().getMerges());
			outputStream.flush();
			outputStream.close();
			fileWriter.close();
		} catch(IOException e) {
			log.error("Could not open file " + options.modelPath().toString()+"-merges.ser\r\n"+e);
		}
	}
	/**
	 * After model is loaded, we can preserve Merges.
	 * @param model
	 */
	public static Map<Pair<Integer,Integer>,Integer> loadMerges() {
		try {
			FileInputStream fileReader = new FileInputStream("merges.ser");
			ObjectInputStream inputStream = new ObjectInputStream(fileReader);
			Map<Pair<Integer,Integer>,Integer> merges = (Map<Pair<Integer,Integer>,Integer>) inputStream.readObject();
			fileReader.close();
			return merges;
		} catch(IOException e) {
			log.error("Could not open file merges.ser\r\n"+e);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return null;
	}
	/**
	 * Parse the command line for url and xpath directive
	 * @param urlc array of cmdl args, link at 0
	 * @return The Element that matches directive
	 */
	private static Element parseLinks(String[] urlc) {
		//try {	
		Document doc = null;
		if(urlc == null || urlc.length < 2)
			return null;
		try {
			doc = Jsoup.connect(urlc[0])
					.userAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
					.get();
		} catch(IOException ioe) {
			ioe.printStackTrace();
			return null;
		}
		Element result = null;
		Elements results = null;
		//for(int i = 1; i < urlc.length; i++) {
		//	results = doc.select(urlc[i]);
		//}
		results = doc.selectXpath(urlc[1]);
		if(results == null)
			return null;
		result = results.first();
		if(result == null)
			return null;
		if(result.is("a"))
			return parseLinks(new String[] {result.attr("href"),"//a"});
		return result;
		//System.out.printf("toString:%s text:%s wholeText:%s%n", result.toString(),result.text(),result.wholeText());
		//System.out.printf("result is a:%b result is a[href]:%b%n",result.is("a"),result.is("a[href]"));
		//} catch(MalformedURLException e) {
		//	e.printStackTrace();
		//}
		//return null;
	}

	/**
	 * element 0 is command <br> /recalltime 
	 * arg day time to end day time
	 * @param query the command line with command times
	 * @return String of Result instances from db that contain 2 elements of question/answer string in time range
	 */
	private static String parseTime(String[] query) {
		CompletableFuture<Stream> s;
		String tq,tqe;
		LocalDateTime localDateTime;
		long millis,millise;
		if(query == null)
			return null;
		if(query.length == 5) {
			// day time to end day time
			tq = String.format("%s %s", query[1], query[2]);
			tqe = String.format("%s %s", query[3], query[4]);
			localDateTime = LocalDateTime.parse(tq, DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss") );
			millis = localDateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
			localDateTime = LocalDateTime.parse(tqe, DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss") );
			millise = localDateTime.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
			//s = dbClient.findSubStream(xid,'*','?','?',millis,millise,String.class,String.class);
			StringBuilder sb = new StringBuilder();
			/*
    		try {
    			s.get().forEach(e->{
    				sb.append(((Result)e).get(0));
    				sb.append(((Result)e).get(1));
    			});
    		} catch(InterruptedException | ExecutionException ie) {}
			 */
			return sb.toString();
		}
		return null;
	}

	@Override
	public GraphName getDefaultNodeName() {
		return GraphName.of("llm");
	}

	@Override
	public void onStart(final ConnectedNode connectedNode) {
		SynchronizedThreadManager.getInstance().init(new String[] {"LLM","DB"});
		try {
			dbClient = connectedNode.getRelatrixClient();
			//dbClient.setTablespace("D:/etc/Relatrix/db/test/ai");
			//try {
			xid = dbClient.getTransactionId();
			//} catch (IllegalAccessException | ClassNotFoundException e) {}
			//tensorAlias = new Alias("Tensors");
			//try {
			//	if(dbClient.getAlias(tensorAlias).get() == null)
			//		dbClient.setRelativeAlias(tensorAlias);
			//} catch(ExecutionException | InterruptedException ie) {}
			if(DEBUG)
				log.info("Relatrix transaction Id:"+xid);
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}

		//
		// Extract the command line options and parse them into the model options class
		//
		List<String> nodeArgs = connectedNode.getNodeConfiguration().getCommandLineLoader().getNodeArguments();
		//System.out.println("Args:"+Arrays.toString(nodeArgs.toArray(new String[nodeArgs.size()])));
		options = Options.parseOptions(nodeArgs);

		//
		// NOTE: dont use options.maxTokens() from here on out after we parse metadata, as the value may be -1 indicating metadata
		// contextLength is used for maximum context size. Instead make sure to use model.configuration().contextLength, which
		// has the parsed metadata value thats the official value.
		//
		SynchronizedThreadManager.getInstance().spin(new Runnable() {
			@Override
			public void run() {
				try {
					model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
					if(model == null) {
						model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
						if(DISPLAY_METADATA) {
							extractMerges(model);
						}
					}
					modelLatch.countDown();
				} catch(IOException e) {
					log.error("Could not load model " + options.modelPath().toString()+e);
					System.exit(1);
				}		
			}	
		},"LLM");
		//
		// Start new thread for balance of model
		//
		SynchronizedThreadManager.getInstance().spin(new Runnable() {
			@Override
			public void run() {
				try {
					modelLatch.await();
				} catch (InterruptedException e) { return; }
				relatrixLSH = new RelatrixLSH(dbClient, model.configuration().contextLength);
				sampler = selectSampler(model.configuration().vocabularySize, options.temperature(), options.topp(), options.seed());
				// Chat format seems solely based on individual model, so we extract a name in model loader from Metada general.name
				if(ModelLoader.name.equals("mistral")) {
					chatFormat = new MistralChatFormat(model.tokenizer());
				} else {
					if(ModelLoader.name.equals("llama")) {
						chatFormat = new ChatFormat(model.tokenizer());
					} else {
						throw new IllegalArgumentException("expected metadata general.name containing mistral or llama but found "+ModelLoader.name);
					}
				}
				// set up the preamble system directives
				promptFrame = new PromptFrame(chatFormat);
				List<Integer> promptTokens = new ArrayList<>();
				promptTokens.add(chatFormat.getBeginOfText());
				List<ChatFormat.Message> prompts = ROSCARSystemPrompts.getSystemMessages();
				State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
				promptTokens.addAll(chatFormat.encodeDialogPrompt(true, prompts));
				Optional<String> response = processMessage(model, options, sampler, state, chatFormat, promptTokens);
				if(response.isPresent()) {
					if(DEBUG)
						log.info("***Queueing from system preamble:"+response.get());
					ChatFormat.Message responseMessage = new ChatFormat.Message(ChatFormat.Role.ASSISTANT, response.get());
					PromptFrame responseFrame = new PromptFrame(chatFormat);
					responseFrame.setMessage(responseMessage);
					List<Integer> responseTokens = (List<Integer>)responseFrame.getRawTokens();
					relatrixLSH.addInteraction(System.currentTimeMillis(), ChatFormat.Role.SYSTEM, promptTokens, responseTokens);
					try {
						messageQueue.addLastWait(response.get());
					} catch(InterruptedException ie) {}
				}
				// See if we preload DB with interactions
				if(options.preload()) {
					try {
						String fileName = options.modelPath().getFileName().toString();
						int dotIndex = fileName.lastIndexOf('.');     
						fileName = (dotIndex == -1) ? fileName : fileName.substring(0, dotIndex);
						ROSCARSystemPrompts.frontloadDb(relatrixLSH, chatFormat, fileName+".txt");
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
				dbLatch.countDown();
			}
		},"DB");
		//
		// Set up publisher
		//final Log log = connectedNode.getLog();
		final Publisher<std_msgs.String> pubmodel = connectedNode.newPublisher(LLM, std_msgs.String._TYPE);
		// Subscribers
		final Subscriber<std_msgs.String> subsystem = connectedNode.newSubscriber(SYSTEM_PROMPT, std_msgs.String._TYPE);
		final Subscriber<std_msgs.String> subsuser = connectedNode.newSubscriber(USER_PROMPT, std_msgs.String._TYPE);
		final Subscriber<stereo_msgs.StereoImage> subsobjd = connectedNode.newSubscriber("/stereo_msgs/ObjectDetect", stereo_msgs.StereoImage._TYPE);
		final Subscriber<sensor_msgs.Imu> subsimu = connectedNode.newSubscriber("/sensor_msgs/Imu", sensor_msgs.Imu._TYPE);
		final Subscriber<std_msgs.String> subsrange = connectedNode.newSubscriber("/sensor_msgs/range",std_msgs.String._TYPE);
		final Subscriber<diagnostic_msgs.DiagnosticStatus> subsbat = connectedNode.newSubscriber("robocore/status", diagnostic_msgs.DiagnosticStatus._TYPE);
		//
		// set up subscriber callback for object detection messages
		//
		subsobjd.addMessageListener(new MessageListener<stereo_msgs.StereoImage>() {
			@Override
			public void onNewMessage(StereoImage message) {
				try {
					dbLatch.await();
				} catch (InterruptedException e) { return; }
				ByteBuffer buf = message.getData();
				String sbuf = new String(buf.array(), buf.position(), buf.remaining(), StandardCharsets.UTF_8);
				long imageTime = System.currentTimeMillis();
				if((imageTime-lastImageTime) >= MESSAGE_THRESHOLD) {
					lastImageTime = imageTime;
					processRole(sbuf, ChatFormat.Role.USER);
				}
			}
		});

		subsuser.addMessageListener(new MessageListener<std_msgs.String>() {
			@Override
			public void onNewMessage(std_msgs.String message) {
				try {
					dbLatch.await();
				} catch (InterruptedException e) { return; }
				processRole(message.getData(), ChatFormat.Role.USER);
			}
		});

		subsystem.addMessageListener(new MessageListener<std_msgs.String>() {
			@Override
			public void onNewMessage(std_msgs.String message) {
				try {
					dbLatch.await();
				} catch (InterruptedException e) { return; }
				processRole(message.getData(), ChatFormat.Role.SYSTEM);
			}
		});
		//
		// update all TimedImage in the queue that match the current timestamp to 1 ms with the current
		// IMU reading
		//
		subsimu.addMessageListener(new MessageListener<sensor_msgs.Imu>() {
			@Override
			public void onNewMessage(sensor_msgs.Imu message) {
				try {
					dbLatch.await();
				} catch (InterruptedException e) { return; }
				synchronized(euler) {
					if(euler.euler == null) {
						euler.euler = message;
						euler.eulerTime = System.currentTimeMillis();
						processRole("IMU update:\n"+euler.euler.toJSON(), ChatFormat.Role.USER);
					} else
						if(euler.euler.getCompassHeadingDegrees() != message.getCompassHeadingDegrees() ||
						euler.euler.getRoll() != message.getRoll() ||
						euler.euler.getPitch() != message.getPitch()) {
							euler.euler = message;
							if((System.currentTimeMillis() - euler.eulerTime) >= MESSAGE_THRESHOLD) {
								euler.eulerTime = System.currentTimeMillis();
								processRole("IMU update:\n"+euler.euler.toJSON(), ChatFormat.Role.USER);
							}
						}
				}
			}
		});
		//
		// Ultrasonic distance sensor also timestamped and correlated
		//
		subsrange.addMessageListener(new MessageListener<std_msgs.String>() {
			@Override
			public void onNewMessage(std_msgs.String message) {
				try {
					dbLatch.await();
				} catch (InterruptedException e) { return; }
				synchronized(ranges) {
					if(ranges.range == null) {
						ranges.range = message; //Float.parseFloat(message.getData());
						ranges.rangeTime = System.currentTimeMillis();
						processRole("Nearest distance update:\n"+ranges.toJSON(), ChatFormat.Role.USER);
					} else
						if(ranges.range.getData() != message.getData()) {
							ranges.range = message; //Float.parseFloat(message.getData());
							if((System.currentTimeMillis() - ranges.rangeTime) >= MESSAGE_THRESHOLD) {
								ranges.rangeTime = System.currentTimeMillis();
								processRole("Nearest distance update:\n"+ranges.toJSON(), ChatFormat.Role.USER);
							}
						}
				}
			}
		});

		subsbat.addMessageListener(new MessageListener<diagnostic_msgs.DiagnosticStatus>() {
			@Override
			public void onNewMessage(DiagnosticStatus message) {
				//System.out.println(message.getHardwareId()+" Status "+message.getMessage());
				StringBuilder sb = new StringBuilder();
				sb.append("MessageName:");
				sb.append(message.getName());
				sb.append(" Level:");
				sb.append(message.getLevel()+"\r\n");
				sb.append(message.getMessage()+"\r\n");
				sb.append(message.getHardwareId()+"\r\n");
				List<KeyValue> diagMsgs = message.getValues();
				if( diagMsgs != null ) {
					for( KeyValue msg : diagMsgs) {
						sb.append(msg.getKey()+" ");
						if( msg.getValue() != null ) {
							sb.append(msg.getValue()+"\r\n");
						}
					}
					if(DEBUG)
						System.out.println(sb.toString());
					processRole(sb.toString(), ChatFormat.Role.USER);
				}
			}
		});

		/**
		 * Main publishing loop. Essentially we are publishing the data in whatever state its in.
		 * This CancellableLoop will be canceled automatically when the node shuts down
		 */
		connectedNode.executeCancellableLoop(new CancellableLoop() {
			private int sequenceNumber;
			@Override
			protected void setup() {
				sequenceNumber = 0;
			}
			@Override
			protected void loop() throws InterruptedException {
				//log.info(connectedNode.getName()+" "+sequenceNumber);		
				//std_msgs.Header imghead = connectedNode.getTopicMessageFactory().newFromType(std_msgs.Header._TYPE);
				//imghead.setSeq(sequenceNumber);
				//org.ros.message.Time tst = connectedNode.getCurrentTime();
				//imghead.setStamp(tst);
				//imghead.setFrameId(tst.toString());
				// block until we have a message, take from head of queue
				String responseData = messageQueue.takeFirstNotify();
				std_msgs.String pubmess = pubmodel.newMessage();
				pubmess.setData(responseData);
				//pubmess.setHeader(imghead);
				//log.info("Publishing "+responseData);
				pubmodel.publish(pubmess);
				sequenceNumber++;
			}
		});
	} // onStart

	public static Optional<String> processMessage(Llama model, Options options, Sampler sampler, State state, ChatFormatInterface chatFormat, List<Integer> promptTokens ) {
		Set<Integer> stopTokens = chatFormat.getStopTokens();
		List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, model.configuration().contextLength, sampler, options.echo(), null);
		if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
			responseTokens.removeLast();
		}
		return Optional.ofNullable(model.tokenizer().decode(responseTokens));
	}
	/**
	 * Process the given interaction using the role provided, beginning with model.CreateNewState
	 * and ending with a check for response.isPresent and if so, relatrixLSH.addInteraction, then messageQueue.addLastWait(response).
	 * @param message The message to process, the response is in ChatFormat.Role.ASSISTANT
	 * @param role the role context. role is ChatFromat.Role.USER, ChatFromat.Role.SYSTEM, ChatFromat.Role.ASSISTANT
	 */
	private void processRole(String message, ChatFormat.Role role) {
		State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
		List<Integer> promptTokens = new ArrayList<>();
		promptTokens.add(chatFormat.getBeginOfText());
		ChatFormat.Message chatMessage = new ChatFormat.Message(role, message);
		promptFrame.setMessage(chatMessage);
		List<Integer> userMessage = new ArrayList<Integer>(promptFrame.getRawTokens());
		List<ChatFormat.Message> responses = null;
		try {
			responses = relatrixLSH.findNearest(promptFrame, model.tokenizer());
		} catch (IllegalArgumentException | ClassNotFoundException | IllegalAccessException | IOException | InterruptedException | ExecutionException e) {
			e.printStackTrace();
			responses = new ArrayList<ChatFormat.Message>();
		}
		promptTokens.addAll(chatFormat.encodeDialogPrompt(true, responses));
		if(DEBUG)
			log.info("***User FindNearest returned:"+ model.tokenizer().decode(promptTokens));
		Optional<String> response = processMessage(model, options, sampler, state, chatFormat, promptTokens);
		if(response.isPresent()) {
			if(DEBUG)
				log.info("***Queueing from role USER:"+response.get());
			ChatFormat.Message responseMessage = new ChatFormat.Message(ChatFormat.Role.ASSISTANT, response.get());
			PromptFrame responseFrame = new PromptFrame(chatFormat);
			responseFrame.setMessage(responseMessage);
			List<Integer> responseTokens = (List<Integer>)responseFrame.getRawTokens();
			relatrixLSH.addInteraction(System.currentTimeMillis(), role, userMessage, responseTokens);
			try {
				messageQueue.addLastWait(response.get());
			} catch(InterruptedException ie) {}
		}
	}
	/**
	 * Intercept the model output and generate a movement command to be published to the {@link com.neocoretechs.robocore.propulsion.MotionController}
	 * @param modelOutput
	 * @param currentHeading
	 * @return
	 */
	public Optional<ComeToHeadingStamped> intercept(String modelOutput, float currentHeading) {
		try {
			JSONObject obj = new JSONObject(modelOutput);
			String actionStr = obj.getString("action");
			ComeToHeadingStamped.action act = ComeToHeadingStamped.action.valueOf(actionStr);
			int distance = obj.optInt("distance", 0);
			float heading = obj.optFloat("heading", currentHeading);
			long timestamp = obj.optLong("timestamp", System.currentTimeMillis());
			ComeToHeadingStamped cths = new ComeToHeadingStamped();
			cths.fromJSON(actionStr);
			std_msgs.Int32 mdist = new std_msgs.Int32();
			mdist.setData(distance);
			std_msgs.Float32 mhead = new std_msgs.Float32();
			mhead.setData(heading);
			std_msgs.UInt64 mtime = new std_msgs.UInt64();
			mtime.setData(timestamp);
			return Optional.of(new ComeToHeadingStamped(act,mdist,mhead,mtime));
		} catch (Exception e) {
			log.error("Failed to parse model output: " + e.getMessage());
			return Optional.empty();
		}
	}

}

final class MemoryUtils {
	public static MemorySegment allocateSegmentFromBytes(byte[] bytes, Arena arena) {
		MemorySegment segment = arena.allocate(bytes.length, 1); // allocate off-heap with alignment
		segment.copyFrom(MemorySegment.ofArray(bytes)); // copy contents
		return segment;
	}
}

final class RoPE {
	/**
	 * For GPT2 vocab
	 * @param contextLength
	 * @param headSize
	 * @param theta
	 * @param ropeScaling
	 * @param scaleFactor
	 * @param loFreqFactor
	 * @param hiFreqFactor
	 * @param oldContextLength
	 * @return
	 */
	public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta,
			boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
		assert headSize % 2 == 0;
		float[] cr = new float[contextLength * (headSize / 2)];
		float[] ci = new float[contextLength * (headSize / 2)];
		int n = 0;
		for (int pos = 0; pos < contextLength; ++pos) {
			for (int i = 0; i < headSize; i += 2) {
				float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
				if (ropeScaling) {
					// Llama 3.1 scaling
					float loFreqWavelen = oldContextLength / loFreqFactor;
					float hiFreqWavelen = oldContextLength / hiFreqFactor;
					float wavelen = (float) (2.0 * Math.PI / freq);
					if (wavelen < hiFreqWavelen) {
						freq = freq;
					} else if (wavelen > loFreqWavelen) {
						freq = freq / scaleFactor;
					} else {
						float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
						freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
					}
				}
				float val = pos * freq;
				cr[n] = (float) Math.cos(val);
				ci[n] = (float) Math.sin(val);
				n++;
			}
		}
		assert contextLength * (headSize / 2) == n;
		return new Pair<>(cr, ci);
	}
	/**
	 * for Llama vocab
	 * @param contextLength
	 * @param headSize
	 * @param theta
	 * @return
	 */
	public static Pair<float[], float[]> precomputeFreqsCis(int contextLength, int headSize, double theta) {
		assert headSize % 2 == 0;
		float[] cr = new float[contextLength * (headSize / 2)];
		float[] ci = new float[contextLength * (headSize / 2)];
		int n = 0;
		for (int pos = 0; pos < contextLength; ++pos) {
			for (int i = 0; i < headSize; i += 2) {
				float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
				float val = pos * freq;
				cr[n] = (float) Math.cos(val);
				ci[n] = (float) Math.sin(val);
				n++;
			}
		}
		assert contextLength * (headSize / 2) == n;
		return new Pair<>(cr, ci);
	}

}

@FunctionalInterface
interface Sampler {
	int sampleToken(FloatTensor logits);
	Sampler ARGMAX = FloatTensor::argmax;
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {
	@Override
	public int sampleToken(FloatTensor logits) {
		// sample index from probabilities (they must sum to 1!)
		float random0to1 = rng.nextFloat(1f);
		float cdf = 0.0f;
		for (int i = 0; i < logits.size(); i++) {
			cdf += logits.getFloat(i);
			if (random0to1 < cdf) {
				return i;
			}
		}
		return logits.size() - 1; // in case of rounding errors
	}
}

final class ToppSampler implements Sampler {

	final int[] indices;
	final float topp;
	final RandomGenerator rng;

	public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
		this.indices = new int[maxNumberOfElements];
		this.topp = topp;
		this.rng = rng;
	}

	static void swap(int[] array, int from, int to) {
		int tmp = array[from];
		array[from] = array[to];
		array[to] = tmp;
	}

	static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
		int prev = from, next;
		while ((next = 2 * prev + 1) < n) {
			int r = 2 * prev + 2;
			if (r < n && comparator.compare(array[r], array[next]) < 0) {
				next = r;
			}
			if (comparator.compare(array[next], array[prev]) < 0) {
				swap(array, prev, next);
				prev = next;
			} else {
				break;
			}
		}
	}

	@Override
	public int sampleToken(FloatTensor logits) {
		// top-p sampling (or "nucleus sampling") samples from the smallest set of
		// tokens that exceed probability topp. This way we never sample tokens that
		// have very low probabilities and are less likely to go "off the rails".
		Comparator<Integer> comparator = Comparator.comparingDouble(logits::getFloat).reversed();

		int n = logits.size();
		int head = 0;
		int tail = n - 1;
		// values smaller than (1 - topp) / (n - 1) cannot be part of the result
		// so for efficiency we crop these out as candidates before sorting
		float cutoff = (1.0f - topp) / (n - 1);
		for (int i = 0; i < indices.length; i++) {
			if (logits.getFloat(i) >= cutoff) {
				indices[head++] = i;
			} else {
				indices[tail--] = i;
			}
		}

		int n0 = head;
		// build heap O(n0)
		for (int i = n0 / 2 - 1; i >= 0; --i) {
			siftDown(indices, i, n0, comparator);
		}

		// truncate the list where cumulative probability of the largest k elements exceeds topp
		// O(k lg n0)
		float cumulativeProb = 0.0f;
		int lastIndex = 0;
		for (int i = n0 - 1; i >= 0; i--) {
			swap(indices, 0, i);
			cumulativeProb += logits.getFloat(indices[i]);
			if (cumulativeProb > topp) {
				lastIndex = i;
				break; // we've exceeded topp by including lastIndex
			}
			siftDown(indices, 0, i - 1, comparator);
		}

		// sample from the truncated list
		float r = rng.nextFloat(1f) * cumulativeProb;
		float cdf = 0.0f;
		for (int i = n0 - 1; i >= lastIndex; i--) {
			cdf += logits.getFloat(indices[i]);
			if (r < cdf) {
				return indices[i];
			}
		}

		return indices[lastIndex]; // in case of rounding errors
	}
}

/**
 * Support for AOT preloading of GGUF metadata with GraalVM's Native Image.
 *
 * <p>
 * To preload a model at build time, pass {@code -Dllama.PreloadGGUF=/path/to/model.gguf}
 * to the native-image builder command. At runtime, the preloaded model will be used
 * iff the specified and preloaded file names (base name) match.
 */
final class AOT {
	record PartialModel(String modelFileName, Llama model, long tensorDataOffset, Map<String, GGUF.GGUFTensorInfo> tensorInfos) {}

	private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("llama.PreloadGGUF"));

	private static PartialModel preLoadGGUF(String modelPath) {
		if (modelPath == null || modelPath.isEmpty()) {
			return null;
		}
		try {
			Path path = Path.of(modelPath);
			if (!Files.exists(path) || !Files.isRegularFile(path)) {
				throw new IllegalArgumentException("Cannot pre-load model: " + path);
			}
			GGUF gguf = GGUF.loadModel(path);
			try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
				return new PartialModel(
						path.getFileName().toString(),
						ModelLoader.loadModel(fileChannel, gguf, Options.DEFAULT_MAX_TOKENS, false),
						gguf.getTensorDataOffset(),
						gguf.getTensorInfos()
						);
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Tries to reuse a compatible AOT preloaded model.
	 * The file name (base name) must match with the preloaded file name.
	 * No checksum/hash is checked for performance reasons.
	 */
	public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
		AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
		if (preLoaded == null) {
			return null; // no pre-loaded model stored
		}
		String optionsModel = modelPath.getFileName().toString();
		String preLoadedModel = preLoaded.modelFileName();
		if (!Objects.equals(optionsModel, preLoadedModel)) {
			// Preloaded and specified model file names didn't match.
			return null;
		}
		Llama baseModel = preLoaded.model();
		try (var timer = Timer.log("Load tensors from pre-loaded model");
				var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
			// Load only the tensors (mmap slices).
			Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensorInfos());
			Weights weights = ModelLoader.loadGPT2Weights(tensorEntries, baseModel.configuration());
			return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
		}
	}
}

