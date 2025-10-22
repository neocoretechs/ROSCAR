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

import com.neocoretechs.roscar.ChatFormat.Message;
import com.neocoretechs.roscar.ChatFormat.Role;
import com.neocoretechs.roscar.GGUF.GGMLTensorEntry;
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
 * Serves as morphism map for relating LSH index to token vector
 */
class TimestampRole implements Serializable, Comparable {
	private static final long serialVersionUID = 1L;
	private Long timestamp;
	private ChatFormat.Role role;
	public TimestampRole() {}
	public TimestampRole(Long timestamp, ChatFormat.Role role) {
		this.timestamp = timestamp;
		this.role = role;
	}
	public Long getTimestamp() {
		return timestamp;
	}
	public void setTimestamp(Long timestamp) {
		this.timestamp = timestamp;
	}
	public ChatFormat.Role getRole() {
		return role;
	}
	public void setRole(ChatFormat.Role role) {
		this.role = role;
	}
	@Override
	public int hashCode() {
		return Objects.hash(role, timestamp);
	}
	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!(obj instanceof TimestampRole)) {
			return false;
		}
		TimestampRole other = (TimestampRole) obj;
		return role == other.role && Objects.equals(timestamp, other.timestamp);
	}
	@Override
	public String toString() {
		return LocalDateTime.ofInstant(Instant.ofEpochMilli(timestamp), ZoneId.systemDefault()).toString()+" "+role.getRole();
	}
	@Override
	public int compareTo(Object o) {
		int res;
		res = timestamp.compareTo(((TimestampRole)o).timestamp);
		if(res != 0)
			return res;
		return role.compareTo(((TimestampRole)o).role);
	}	
}
/**
 * Contract for all ChatFormat tokenizers
 */
interface ChatFormatInterface {
	public TokenizerInterface getTokenizer();
	public List<Integer> encodeMessage(Message message, List<Integer> tokenList);
	public Set<Integer> getStopTokens();
	public List<Integer> encodeHeader(ChatFormat.Message message);
	public List<Integer> encodeMessage(ChatFormat.Message message);
	/**
	 * Encode beginOfText, then follow with list of supplied messages
	 * @param appendAssistantTurn true to add a blank ASSISTANT header at the end of the list of prompts
	 * @param dialog List of messages to tokenize
	 * @return the tokenized list of appended messages
	 */
	public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog);
	public int getBeginOfText();
	public String stripFormatting(String input);
}
/**
 * Encapsulates ChatFormatInterface tokenizer and manages raw and formatted token lists
 */
final class PromptFrame {
	private ChatFormat.Message message;
	private final ChatFormatInterface chatFormat;
	private Collection<? extends Integer> rawTokens;
	private List<Integer> formattedTokens;

	public PromptFrame(ChatFormatInterface format) {
		this.chatFormat = format;
	}
	public void setMessage(ChatFormat.Message message) {
		this.message = message;
		this.rawTokens = chatFormat.getTokenizer().encode(chatFormat.stripFormatting(message.content()));
		this.formattedTokens = chatFormat.encodeMessage(message); // Includes headers + role
	}
	public Collection<? extends Integer> getRawTokens() {
		return rawTokens;
	}
	public List<Integer> getFormattedTokens() {
		return formattedTokens;
	}
	public int getBeginOfTextToken() {
		return chatFormat.getBeginOfText();
	}
	public Set<Integer> getStopTokens() {
		return chatFormat.getStopTokens();
	}
	public Message getMessage() {
		return message;
	}
}

/**
 * Utility tailored for Llama 3 instruct prompt format.
 */
class ChatFormat implements ChatFormatInterface {

	final Tokenizer tokenizer;
	final int beginOfText;
	final int endHeader;
	final int startHeader;
	final int endOfTurn;
	final int endOfText;
	final int endOfMessage;
	final Set<Integer> stopTokens;

	public ChatFormat(TokenizerInterface tokenizer) {
		this.tokenizer = (Tokenizer)tokenizer;
		Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
		this.beginOfText = specialTokens.get("<|begin_of_text|>");
		this.startHeader = specialTokens.get("<|start_header_id|>");
		this.endHeader = specialTokens.get("<|end_header_id|>");
		this.endOfTurn = specialTokens.get("<|eot_id|>");
		this.endOfText = specialTokens.get("<|end_of_text|>");
		this.endOfMessage = specialTokens.getOrDefault("<|eom_id|>", -1); // only in 3.1
		this.stopTokens = Set.of(endOfText, endOfTurn);
	}
	@Override
	public TokenizerInterface getTokenizer() {
		return tokenizer;
	}
	@Override
	public Set<Integer> getStopTokens() {
		return stopTokens;
	}
	@Override
	public int getBeginOfText() {
		return beginOfText;
	}
	@Override
	public List<Integer> encodeHeader(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(startHeader);
		tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
		tokens.add(endHeader);
		tokens.addAll(this.tokenizer.encodeAsList("\n"));
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message) {
		List<Integer> tokens = this.encodeHeader(message);
		tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
		tokens.add(endOfTurn);
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message, List<Integer> tokenList) {
		List<Integer> tokens = this.encodeHeader(message);
		tokens.addAll(tokenList);
		tokens.add(endOfTurn);
		return tokens;
	}
	/**
	 * Encode beginOfText, then follow with list of supplied messages
	 * @param appendAssistantTurn true to add a blank ASSISTANT header at the end of the list of prompts
	 * @param dialog List of messages to tokenize
	 * @return the tokenized list of appended messages
	 */
	@Override
	public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(beginOfText);
		for (ChatFormat.Message message : dialog) {
			tokens.addAll(this.encodeMessage(message));
		}
		if (appendAssistantTurn) {
			// Add the start of an assistant message for the model to complete.
			tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
		}
		return tokens;
	}
	@Override
	public String stripFormatting(String input) {
		return input.replaceAll("<\\|.*?\\|>", "")
				.replaceAll("\\*+", "")
				.replaceAll("(?m)^USER:|AI:", "")
				.trim();
	}

	public record Message(ChatFormat.Role role, String content) {
	}

	public enum Role {
		SYSTEM("SYSTEM"),
		USER("USER"),
		ASSISTANT("ASSISTANT");
		private final String role;
		Role(String role) {
			this.role = role;
		}
		public String getRole() {
			return role;
		}
		@Override
		public String toString() {
			return role;
		}
	}
}

/**
 * Utility tailored for Mistral v0.3 instruct prompt format.
 */
final class MistralChatFormat implements ChatFormatInterface {

	protected final TokenizerInterface tokenizer;
	protected final int unknownToken;
	protected final int beginOfText;
	protected final int endOfText;
	protected final int beginOfInstruction;
	protected final int endOfInstruction;
	protected final int toolCalls;
	protected final int beginOfAvailableTools;
	protected final int endOfAvailableTools;
	protected final int beginOfToolResults;
	protected final int endOfToolResults;
	protected final int prefix;
	protected final int middle;
	protected final int suffix;

	public MistralChatFormat(TokenizerInterface tokenizer) {
		this.tokenizer = tokenizer;
		Map<String, Integer> specialTokens = this.tokenizer.getSpecialTokens();
		this.unknownToken = specialTokens.get("<unk>");
		this.beginOfText = specialTokens.get("<s>");
		this.endOfText = specialTokens.get("</s>");
		this.beginOfInstruction = specialTokens.get("[INST]");
		this.endOfInstruction = specialTokens.get("[/INST]");
		this.toolCalls = specialTokens.get("[TOOL_CALLS]");
		this.beginOfAvailableTools = specialTokens.get("[AVAILABLE_TOOLS]");
		this.endOfAvailableTools = specialTokens.get("[/AVAILABLE_TOOLS]");
		this.beginOfToolResults = specialTokens.get("[TOOL_RESULTS]");
		this.endOfToolResults = specialTokens.get("[/TOOL_RESULTS]");
		// Only Codestral supports FIM tokens.
		this.prefix = specialTokens.getOrDefault("[PREFIX]", unknownToken);
		this.suffix = specialTokens.getOrDefault("[SUFFIX]", unknownToken);
		this.middle = specialTokens.getOrDefault("[MIDDLE]", unknownToken);
	}
	@Override
	public TokenizerInterface getTokenizer() {
		return tokenizer;
	}
	@Override
	public Set<Integer> getStopTokens() {
		return Set.of(endOfText);
	}
	@Override
	public int getBeginOfText() {
		return beginOfText;
	}

	public List<Integer> encodeMessage(String userMessage, boolean addHeader, boolean addFooter) {
		List<Integer> tokens = new ArrayList<>();
		if (addHeader) {
			tokens.add(this.beginOfInstruction);
		}
		if (userMessage != null) {
			tokens.addAll(this.tokenizer.encodeAsList(userMessage.strip()));
		}
		if (addFooter) {
			tokens.add(endOfInstruction);
		}
		return tokens;
	}

	public List<Integer> encodeFillInTheMiddle(String prefix, String suffix) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.suffix);
		tokens.addAll(tokenizer.encode(suffix));
		tokens.add(this.prefix);
		tokens.addAll(tokenizer.encode(prefix));
		return tokens;
	}
	@Override
	public List<Integer> encodeHeader(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.beginOfInstruction);
		tokens.addAll(this.tokenizer.encodeAsList(message.role().name()));
		tokens.add(endOfInstruction);
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.beginOfInstruction);
		tokens.addAll(this.tokenizer.encodeAsList(message.content().strip()));
		tokens.add(endOfInstruction);
		return tokens;
	}
	@Override
	public List<Integer> encodeMessage(ChatFormat.Message message, List<Integer> tokenList) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(this.beginOfInstruction);
		tokens.addAll(tokenList);
		tokens.add(endOfInstruction);
		return tokens;
	}
	@Override
	public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog) {
		List<Integer> tokens = new ArrayList<>();
		tokens.add(beginOfText);
		for (ChatFormat.Message message : dialog) {
			tokens.addAll(this.encodeMessage(message));
		}
		//if (appendAssistantTurn) {
		//    // Add the start of an assistant message for the model to complete.
		//    tokens.addAll(this.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
		//}
		tokens.add(endOfText);
		return tokens;
	}
	@Override
	public String stripFormatting(String input) {
		return input
				.replaceAll("\\[/?[A-Z_]+\\]", "")
				.replaceAll("\\[(PREFIX|SUFFIX|MIDDLE)\\]", "")
				.replaceAll("<\\|.*?\\|>", "")
				.replaceAll("\\*+", "")
				.replaceAll("(?m)^USER:|AI:|ASSISTANT:", "")
				.trim();
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
/**
 * <h3>LSH (Locality-Sensitive Hashing) is a technique used for efficient similarity search and clustering of high-dimensional data. 
 * Cosine hashing is one type of LSH.</h3>
 * In cosine hashing, the goal is to map similar vectors (in terms of cosine similarity) to the same or nearby hash buckets 
 * with high probability. The hash function is designed such that the probability of two vectors being mapped to the same 
 * bucket is proportional to their cosine similarity.<p>
 * Given two vectors x and y, the cosine similarity is defined as:<br>
 * cos(x, y) = (x · y) / (||x|| ||y||) <br>
 * where x · y is the dot product of x and y, and ||x|| and ||y|| are the magnitudes (norms) of x and y, respectively. <p>
 * In cosine hashing, we use a random projection vector w to compute the hash value. Specifically, the hash function is defined as:<br>
 * h(x) = sign(w · x) <br>
 * where sign is a function that returns 1 if the dot product is positive and 0 otherwise. <p>
 * Relation of distance to number of hashes: <br>
 * The key idea behind LSH is that if two vectors are similar (i.e., have high cosine similarity), 
 * they are more likely to be mapped to the same bucket. The probability of two vectors being mapped to the same bucket is given by: <br>
 * P(h(x) = h(y)) = 1 - (θ(x, y) / π) <br>
 * where θ(x, y) is the angle between x and y.<p>
 * To increase the accuracy of the similarity search, we use multiple hash functions (i.e., multiple random projection vectors w).<p>
 * The number of hashes required to achieve a certain level of accuracy depends on the desired similarity threshold and the 
 * dimensionality of the data. <p>
 * In general, the more hashes we use, the more accurate the similarity search will be. However, using too many hashes can lead to 
 * increased computational cost and storage requirements.<p>
 * Trade-off:<br>
 * There is a trade-off between the number of hashes and the accuracy of the similarity search.<p> 
 * Increasing the number of hashes improves the accuracy but also increases the computational cost and storage requirements.<br>
 * In practice, the number of hashes is typically chosen based on the specific requirements of the application, such as the 
 * desired level of accuracy, the size of the dataset, and the available computational resources.<p>
 * By using multiple hash functions and combining the results, LSH can efficiently identify similar vectors in high-dimensional space, 
 * making it a powerful technique for similarity search and clustering applications.
 *
 */
final class CosineHash implements Serializable, Comparable {
	private static final long serialVersionUID = 778951747630668248L;
	FloatTensor randomProjection;

	public CosineHash() {}

	public CosineHash(int dimensions){
		ThreadLocalRandom rand  = ThreadLocalRandom.current();
		float[] randomp = new float[dimensions];
		if(dimensions > 1000) {
			Parallel.parallelFor(0, dimensions, d -> {
				double val = rand.nextGaussian();
				//randomProjection.setFloat(d, (float) val);
				randomp[d] = (float)val;
			});
		} else {
			for(int d=0; d < dimensions; d++) {
				double val = rand.nextGaussian();
				//randomProjection.setFloat(d, (float) val);
				randomp[d] = (float)val;
			}
		}
		MemorySegment segment = MemorySegment.ofArray(randomp);
		randomProjection = new F32FloatTensor(dimensions, segment);
	}

	public static Integer combine(int[] hashes) {
		//Treat the hashes as a series of bits.
		//They are either zero or one, the index 
		//represents the value.
		int result = 0;
		//factor holds the power of two.
		int factor = 1;
		for(int i = 0 ; i < hashes.length ; i++){
			result += hashes[i] == 0 ? 0 : factor;
			factor *= 2;
		}
		return result;
	}

	public int hash(FloatTensor vector) {
		//calculate the dot product.
		double result;
		if(vector.size() < randomProjection.size())
			result = randomProjection.dot(0, vector, 0, vector.size());
		else
			result = vector.dot(0, randomProjection, 0, randomProjection.size());
		//returns a 'bit' encoded as an integer.
		//1 when positive or zero, 0 otherwise.
		return result > 0 ? 1 : 0;
	}

	@Override
	public String toString(){
		//return String.format("%s\nrandomProjection:%s",this.getClass().getName(), randomProjection);
		return String.format("%s randomProjectionSize=%d",this.getClass().getName(), randomProjection.size());
	}

	@Override
	public int compareTo(Object arg0) {
		return randomProjection.compareTo(arg0);
	}
}
/**
 * An {@link Index} contains one or more locality sensitive hash tables. These hash
 * tables contain the mapping between a combination of a number of hashes
 * (encoded using an integer) and a list of possible nearest neighbors.
 *
 * A hash function can hash a vector of arbitrary dimensions to an integer
 * representation. The hash function needs to be locality sensitive to work in
 * the locality sensitive hash scheme, meaning that vectors that are 'close'
 * according to some metric have a high probability to end up with the same
 * hash.<p>
 * In the context of Locality-Sensitive Hashing (LSH), w represents the bucket width or window size.<p>
 * When we compute the hash value for a vector using a random projection. Here's what each component does:<p>
 * vector.dot(randomProjection): Computes the dot product of the input vector and a random projection vector. <br>
 * This projects the input vector onto a random direction.<br>
 * offset: Adds a random offset to the projected value. <br>
 * This helps to shift the projected values and create a more uniform distribution.<p>
 * w: The bucket width or window size. This value determines the granularity of the hash function.<br>
 * By dividing the projected value (plus offset) by w, you're essentially:<br>
 * Quantizing the projected values into discrete buckets.<br>
 * Assigning each bucket a unique hash value. <br>
 * The choice of w affects the trade-off between:<br>
 * Precision: Smaller w values result in more precise hashing, but may lead to more collisions.
 * Larger w values result in fewer collisions, but may reduce precision.<p>
 * In general, w is a hyperparameter that needs to be tuned for specific applications and datasets. 
 * A good choice of w can significantly impact the performance of the LSH algorithm.
 * 
 */
final class HashTable implements Serializable {
	private static final Log log = LogFactory.getLog(HashTable.class);
	private static final long serialVersionUID = -5410017645908038641L;
	private static boolean DEBUG = true;
	/**
	 * Contains the mapping between a combination of a number of hashes (encoded
	 * using an integer) and a list of possible nearest neighbours
	 */
	private HashMap<Integer,List<FloatTensor>> hashTable;
	private CosineHash[] hashFunctions;
	private CosineHash family;
	private int index;

	/**
	 * Initialize a new hash table, it needs the number of hash
	 * functions that should be used and projection vector size. Its organized by index up to number of hash tables.
	 * @param index The projection vector and cosine hash number index.
	 * @param numberOfHashes The number of hash functions that should be used.
	 * @param projectionVectorSize The number of elements in the random projection vector
	 */
	public HashTable(int index, int numberOfHashes, int projectionVectorSize) {
		this.index = index;
		this.hashTable = new HashMap<Integer, List<FloatTensor>>();
		this.hashFunctions = new CosineHash[numberOfHashes];
		if(numberOfHashes > 64)
			Parallel.parallelFor(0, numberOfHashes, i -> {
				hashFunctions[i] = new CosineHash(projectionVectorSize);
			});
		else
			for(int i = 0; i < numberOfHashes; i++)
				hashFunctions[i] = new CosineHash(projectionVectorSize);
	}

	/**
	 * Query the hash table for a vector. It calculates the hash for the vector,
	 * and does a lookup in the hash table. If no candidates are found, an empty
	 * list is returned, otherwise, the list of candidates is returned.
	 * 
	 * @param query The query vector.
	 * @return Does a lookup in the tables for a query using its combined hash based on passed tensor. If no
	 *         candidates are found, an empty list is returned, otherwise, the
	 *         list of candidates is returned.
	 */
	public List<FloatTensor> query(FloatTensor query) {
		Integer combinedHash = hash(query);
		if(DEBUG)
			log.info("Combined hash for query:"+combinedHash);
		if(hashTable.containsKey(combinedHash))
			return hashTable.get(combinedHash);
		else
			return new ArrayList<FloatTensor>();
	}

	/**
	 * Add the vector to the map of hash tables based on key of combined hash generated from passed vector
	 * @param vector
	 */
	public void add(FloatTensor vector) {
		Integer combinedHash = hash(vector);
		if(!hashTable.containsKey(combinedHash)){
			hashTable.put(combinedHash, new ArrayList<FloatTensor>());
		}
		hashTable.get(combinedHash).add(vector);
	}

	/**
	 * Calculate the combined hash for a vector.
	 * @param vector The vector to calculate the combined hash for.
	 * @return An integer representing a combined hash.
	 */
	private Integer hash(FloatTensor vector){
		int hashes[] = new int[hashFunctions.length];
		for(int i = 0 ; i < hashFunctions.length ; i++){
			hashes[i] = hashFunctions[i].hash(vector);
		}
		Integer combinedHash = CosineHash.combine(hashes);
		return combinedHash;
	}

	/**
	 * Return the number of hash functions used in the hash table.
	 * @return The number of hash functions used in the hash table.
	 */
	public int getNumberOfHashes() {
		return hashFunctions.length;
	}

	@Override
	public String toString() {
		return String.format("%s index=%d family=%s hashes=%s tableSize=%d",this.getClass().getName(), index, family, Arrays.toString(hashFunctions), hashTable.size());
	}
}
/**
 * An {@link Index} contains one or more locality sensitive hash tables. These hash
 * tables contain the mapping between a combination of a number of hashes
 * (encoded using an integer) and a list of possible nearest neighbors.<p>
 *
 * A hash function can hash a vector of arbitrary dimensions to an integer
 * representation. The hash function needs to be locality sensitive to work in
 * the locality sensitive hash scheme. Meaning that vectors that are 'close'
 * according to some metric have a high probability to end up with the same
 * hash.<p>
 * In the context of Locality-Sensitive Hashing (LSH), w represents the bucket width or window size.<p>
 * When we compute the hash value for a vector using a random projection. Here's what each component does:<p>
 * vector.dot(randomProjection): Computes the dot product of the input vector and a random projection vector. <br>
 * This projects the input vector onto a random direction.<br>
 * offset: Adds a random offset to the projected value. <br>
 * This helps to shift the projected values and create a more uniform distribution.<p>
 * w: The bucket width or window size. This value determines the granularity of the hash function.<br>
 * By dividing the projected value (plus offset) by w, you're essentially:<br>
 * Quantizing the projected values into discrete buckets.<br>
 * Assigning each bucket a unique hash value. <br>
 * The choice of w affects the trade-off between:<br>
 * Precision: Smaller w values result in more precise hashing, but may lead to more collisions.
 * Larger w values result in fewer collisions, but may reduce precision.<p>
 * In general, w is a hyperparameter that needs to be tuned for specific applications and datasets. 
 * A good choice of w can significantly impact the performance of the LSH algorithm.<p>
 * This class is designed to be stored in the Relatrix database to serve as a template for encoding and retrieving
 * a given set of floating point tensors.<p>
 * add(normalize(tokenList));
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2025
 */
final class RelatrixLSH implements Serializable, Comparable {
	private static final Log log = LogFactory.getLog(RelatrixLSH.class);
	private static final long serialVersionUID = -5410017645908038641L;
	private static boolean DEBUG = true;
	public int numberOfHashTables = 16;
	public int numberOfHashes = 12;
	public AsynchRelatrixClientTransaction dbClient;
	private TransactionId xid;
	private int maxTokens;

	/**
	 * Contains the mapping between a combination of a number of hashes (encoded
	 * using an integer) and a list of possible nearest neighbours
	 */
	private List<CosineHash[]> hashTable;
	private UUID key;

	public RelatrixLSH() {}

	public RelatrixLSH(AsynchRelatrixClientTransaction dbClient, int maxTokens) {
		this(dbClient, 12, 16, 50, maxTokens);
	}
	/**
	 * Initialize a new hash table, uses Cosine hash family.
	 * @param dbClient the Relatrix client from connected node
	 * @param numberOfHashes The number of hash functions that should be used.
	 * @param numberOfhashTables the number of tables each containing number of hashes
	 * @param projectionVectorSize The number of elements in the vector projected into high dimensional space
	 */
	public RelatrixLSH(AsynchRelatrixClientTransaction dbClient, int numberOfHashes, int numberOfHashTables, int projectionVectorSize, int maxTokens) {
		this.dbClient = dbClient;
		this.numberOfHashes = numberOfHashes;
		this.numberOfHashTables = numberOfHashTables;
		this.key = UUID.randomUUID();
		this.hashTable = new ArrayList<CosineHash[]>();
		this.maxTokens = maxTokens;
		for(int i = 0; i < numberOfHashTables; i++) {
			final CosineHash[] cHash = new CosineHash[numberOfHashes];
			this.hashTable.add(cHash);
			if(numberOfHashes > 64)
				Parallel.parallelFor(0, numberOfHashes, j -> {
					cHash[j] = new CosineHash(projectionVectorSize);
				});
			else
				for(int j = 0; j < numberOfHashes; j++)
					cHash[j] = new CosineHash(projectionVectorSize);
		}
		xid = dbClient.getTransactionId();
	}


	public UUID getKey() {
		return key;
	}

	/**
	 * Query the hash table for a vector. It calculates the hash for the vector,
	 * and does a lookup in the hash table. If no candidates are found, an empty
	 * list is returned, otherwise, the list of candidates is returned.
	 * 
	 * @param query The query vector.
	 * @return Does a lookup in the table for a query using its hash. If no
	 *         candidates are found, an empty list is returned, otherwise, the
	 *         list of candidates is returned as List<Result> where Result contains timetamp, NoIndex with vector
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public List<Result> query(List<Integer> query) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException, InterruptedException, ExecutionException {
		ArrayList<Result> res = new ArrayList<Result>();
		for(int i = 0; i < hashTable.size(); i++) {
			Integer combinedHash = hash(hashTable.get(i), normalize(query));
			if(DEBUG)
				log.info("Querying combined hash for query "+i+" of "+hashTable.size()+":"+combinedHash);
			CompletableFuture<Iterator> cit = dbClient.findSet(xid, combinedHash, '?', '?');
			Iterator<?> it = cit.get();
			//int cnt = 0;
			while(it.hasNext()) {
				Result r = (Result) it.next();
				// should be NoIndex values
				res.add(r);
				//System.out.print(++cnt+"\r");
			}
			//System.out.println();
		}
		return res;
	}
	/**
	 * Query the hash table in parallel for a vector of tokens. It calculates the hash for the vector,
	 * and does a lookup in the hash table. If no candidates are found, an empty
	 * list is returned, otherwise, the list of candidates is returned.
	 * 
	 * @param query The query vector.
	 * @param normalizedQuery TODO
	 * @return Does a lookup in the table for a query using its hash. If no
	 *         candidates are found, an empty list is returned, otherwise, the
	 *         list of candidates is returned as List<Result> where Result contains TimestampRole, vector
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public List<Result> queryParallel(List<Integer> query, FloatTensor normalizedQuery) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException, InterruptedException, ExecutionException {
		List<Result> res = new ArrayList<Result>();
		ArrayList<Object> iq = new ArrayList<Object>();
		for(int i = 0; i < hashTable.size(); i++) {
			Integer combinedHash = hash(hashTable.get(i), normalizedQuery);
			iq.add(combinedHash);
		}
		try (var timer = Timer.log("Querying combined hash for table of "+hashTable.size())) {
			CompletableFuture<List> cres = dbClient.findSetParallel(xid, iq, '?', '?');
			res = cres.get();
			//for(Result r: res) {
			// should be NoIndex values
			//}
		}
		return res;
	}
	/**
	 * Query the hash table in parallel for a vector of LSH indexes. The query has the indexes,
	 * and uses these directly to do a lookup. If no candidates are found, an empty
	 * list is returned, otherwise, the list of candidates is returned.
	 * 
	 * @param query The query vector.
	 * @return Does a lookup in the table for a query using its hash. If no
	 *         candidates are found, an empty list is returned, otherwise, the
	 *         list of candidates is returned as List<Result> where Result contains TimestampRole, vector
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public List<Result> queryParallel2(List<Object> query) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException, InterruptedException, ExecutionException {
		List<Result> res = null;
		try (var timer = Timer.log("Querying combined hash for List of "+query.size())) {
			CompletableFuture<List> cres = dbClient.findSetParallel(xid, query, '?', '?');
			res = cres.get();
		}
		return res;
	}
	/**
	 * Normalizes integer tokens into a zero-centered, unit-length float tensor
	 * for cosine similarity use with Gaussian random projection.
	 * @param tokens List of tokenized values
	 * @return FloatTensor normalized to unit length zero-centered mean
	 */
	public static FloatTensor normalize(List<Integer> tokens) {
		int size = tokens.size();
		float[] floats = new float[size];
		// Cast tokens to float and compute mean
		float mean = 0.0f;
		for (int i = 0; i < size; i++) {
			float value = (float) tokens.get(i);
			floats[i] = value;
			mean += value;
		}
		mean /= size;
		// Zero-center
		for (int i = 0; i < size; i++) {
			floats[i] -= mean;
		}
		// Unit-length normalization
		float norm = 0.0f;
		for (float f : floats) {
			norm += f * f;
		}
		norm = (float) Math.sqrt(norm);
		if (norm != 0f) {
			for (int i = 0; i < size; i++) {
				floats[i] /= norm;
			}
		}
		return new F32FloatTensor(size, MemorySegment.ofArray(floats));
	}
	/**
	 * Add the user/assistant interaction
	 * @param ts the timestamp
	 * @param initiator the initiator of the interaction; either USER or SYSTEM
	 * @param userMessage tokenized initiator message
	 * @param responseTokens tokenized response
	 */
	public void addInteraction(Long ts, ChatFormat.Role initiator, List<Integer> userMessage, List<Integer> responseTokens) {
		TimestampRole tr_assistant = new TimestampRole(ts, ChatFormat.Role.ASSISTANT);
		TimestampRole tr_user = new TimestampRole(ts, initiator);
		try(Timer t = Timer.log("SaveState of reponse:"+responseTokens.size()+" initiator:"+tr_user.toString())) {
			try {
				add(tr_user, userMessage);
				add(tr_assistant, responseTokens);
			} catch (IllegalAccessException | ClassNotFoundException | IOException | InterruptedException | ExecutionException e) {
				log.error(e);
				dbClient.rollback(xid);
				return;
			}
			dbClient.commit(xid); // Only after both store ops succeed
		}
	}	
	/**
	 * Add a vector to the index. Create a UUID and store the vector in a K/V datastore, use the UUID to
	 * reference the vector in the Relatrix relationship.
	 * @param timestampRole The map of the morphism to store LSH->TimestampRole->NoIndex key contains timestamp and role
	 * @param vector the list of tokens
	 * @throws DuplicateKeyException 
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public void add(TimestampRole timestampRole, List<Integer> vector) throws IllegalAccessException, ClassNotFoundException, IOException, InterruptedException, ExecutionException {
		FloatTensor fvec = normalize(vector);
		NoIndex noIndex = NoIndex.create(vector);
		for(int i = 0; i < hashTable.size(); i++) {
			Integer combinedHash = hash(hashTable.get(i), fvec);
			CompletableFuture<Relation> res = dbClient.store(xid, combinedHash, timestampRole, noIndex);
			res.get();
		}
	}
	/**
	 * Find the nearest candidates using cosine similarity. If none are found get the lset timestsamp
	 * retrieve that vector, then get the other vectors with the LSH index and obtain
	 * the most relevant.
	 * @param promptFrame list of messages to populate starting with initial request
	 * @param tokenizer decoder for String from tokens, needed to create new Messages
	 * @return List of retrieved messages
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public List<Message> findNearest(PromptFrame promptFrame, TokenizerInterface tokenizer) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException, InterruptedException, ExecutionException {
		List<Result> nearest = null;
		List<Integer> results = (List<Integer>)promptFrame.getRawTokens();
		if(DEBUG)
			log.info("User query has "+results.size()+" tokens");
		List<ChatFormat.Message> returns = new ArrayList<ChatFormat.Message>();

		FloatTensor fmessage = normalize(results);
		nearest = queryParallel(results, fmessage);
		if(DEBUG)
			log.info("Retrieved "+nearest.size()+" entries from LSH index query.");
		// If we retrieved nothing from semantic query of initial message, try getting last timestamp
		if(nearest.isEmpty()) {
			//return results;
			List<Result> resByTime = primeByTime();
			// if we have a list of the timestamped results, get the index from them and
			// retrieve identical indexes that indicate relevance to last timestamped messages
			if(resByTime != null && !resByTime.isEmpty()) {
				ArrayList<Object> lshQuery = new ArrayList<Object>();
				// each timestamp entry
				for(int i = 0; i < resByTime.size(); i++) {
					Result result = resByTime.get(i);
					// LSH at Result.get(0)
					// re-form the nearest list by getting all the LSH for the given timestamp
					lshQuery.add(result.get(0));
				}
				// now query the matching LSH indexes we got from each timestamp
				nearest = queryParallel2(lshQuery);
			}
			// we could have come up index and timestamp empty
			if(nearest == null || nearest.isEmpty()) {
				// put most recent user query last
				returns.add(promptFrame.getMessage());
				if(DEBUG)
					log.info("Returning from empty index and timestamp query with original prompt");
				return returns;
			}
		}
		// nearest has Result(s) from the last series of TimestampRole query, TimestampRole LSH index, and/or original message
		// fmessage is our original message, mormalized as FloatTensor
		// organize our current Results and find similar relevant entries via cosine similarity
		// and theta similarity
		// organize their indexes in a TreeMap in descending order of cosDist, index in Result
		double[] cossim = new double[nearest.size()];
		int cnt = 0;
		TreeMap<Double, Integer> tm = new TreeMap<Double, Integer>();
		for(int i = 0; i < nearest.size(); i++) {
			Result result = nearest.get(i);
			// TimestampRole at Result.get(0)
			NoIndex noIndex = (NoIndex) result.get(1);
			List<Integer> restensor = (List<Integer>)noIndex.getInstance();
			if(DEBUG)
				log.info("retrieved dialog:"+result.get(0)+" "+tokenizer.decode(restensor));
			double cosDist;
			FloatTensor cantensor = normalize(restensor);
			if(cantensor.size() < fmessage.size())
				cosDist = fmessage.dot(0, cantensor, 0, cantensor.size());
			else
				cosDist = cantensor.dot(0, fmessage, 0, fmessage.size());
			cosDist = Math.acos(cosDist); // radians
			cossim[i] = cosDist;
			tm.put(cosDist, i);
		}
		// Now we need another TreeMap with the entries in TimestampRole descending order. As we
		// walk the cosine tree we check the timestamp tree for proper order of time/role
		// and if we dont see the order we superimpose it for that entry in the returns
		TreeMap<TimestampRole, Integer> tr = new TreeMap<TimestampRole, Integer>();
		for(int i = 0; i < nearest.size(); i++) {
			Result result = nearest.get(i);
			TimestampRole trr = (TimestampRole) result.get(0);
			tr.put(trr, i);
		}
		// tr is sorted by timestamp ASSISTANT, timestamp SYSTEM, timestamp USER timestamp ascending
		// Walk the TreeMap in ascending theta similarity order, fill our
		// context tokens until we get to maximum context length -30% for response overhead.
		// At each entry, check the role and retrieve the proper counterpart; If assistant, 
		// get user with same timestamp if it wasnt the previous entry. if user, get assistant
		// with same timestamp if its not the next entry. In essence we want question/answer pairs
		// that are relevant. If we are missing an answer, get it, if we are missing a question, get that
		// if one of the pairs was picked for semantic relevance and the other wasnt.
		// We have the entries retrieved via sematic relevance or recent timestamp in these TreeMaps.
		// One we walk in order of theta ascending, the other we have sorted by TimestampRole that we
		// check if we see an out of order pair, and if that map doesnt have the missing pair member we need
		// with our sorted check; TimestampRole key being either higherEntry or lowerEntry, we go out
		// to the database and get it by constructing a TimestampRole with same timestamp, complimentary role,
		// insert it to results, and up our context token count toward max.
		Iterator<Integer> it = tm.values().iterator();
		boolean wasUser = false; // we need a user first, our user query that initiated this interaction goes in last
		while(it.hasNext() && results.size() < (maxTokens - (((float)maxTokens) * .3))) {
			int i = it.next();
			Result result = nearest.get(i);
			TimestampRole trr =  (TimestampRole) result.get(0);
			if(trr.getRole().equals(ChatFormat.Role.ASSISTANT)) {
				if(!wasUser) {
					Map.Entry<TimestampRole, Integer> trn = tr.higherEntry(trr); // system or user same timestamp?
					if(trn != null && (trn.getKey().getTimestamp() == trr.getTimestamp())) { 		
						if(trn.getKey().getRole() == ChatFormat.Role.USER) {
							Result result2 = nearest.get(trn.getValue()); // index to nearest list
							addRetrievedMessage(result2, results, returns, tokenizer);
						} else {
							if(trn.getKey().getRole() == ChatFormat.Role.SYSTEM) {
								trn = tr.higherEntry(tr.higherKey(trr));
								if(trn != null && trn.getKey().getRole() == ChatFormat.Role.USER) {
									Result result2 = nearest.get(trn.getValue()); // index to nearest list
									addRetrievedMessage(result2, results, returns, tokenizer);
								} else {
									TimestampRole tr2 = new TimestampRole(trr.getTimestamp(), ChatFormat.Role.USER);
									getTimestampRole(results, returns, tr2, tokenizer);
								}
							} else {
								TimestampRole tr2 = new TimestampRole(trr.getTimestamp(), ChatFormat.Role.USER);
								getTimestampRole(results, returns, tr2, tokenizer);
							}
						}
					} else {
						TimestampRole tr2 = new TimestampRole(trr.getTimestamp(), ChatFormat.Role.USER);
						getTimestampRole(results, returns, tr2, tokenizer);
					}
				} else {
					wasUser = false; // fall through
				}
			} else {
				if(trr.getRole().equals(ChatFormat.Role.USER)) {
					if(wasUser) { // 2 users
						Map.Entry<TimestampRole, Integer> trn = tr.lowerEntry(trr); // system or assistant same timestamp?
						if(trn != null && (trn.getKey().getTimestamp() == trr.getTimestamp())) {
							if(trn.getKey().getRole() == ChatFormat.Role.ASSISTANT) {
								Result result2 = nearest.get(trn.getValue()); // index to nearest list
								addRetrievedMessage(result2, results, returns, tokenizer);
							} else {
								if(trn.getKey().getRole() == ChatFormat.Role.SYSTEM) {
									trn = tr.lowerEntry(tr.lowerKey(trr)); 
									// lower key of user is assistant same timestamp?
									if(trn != null && trn.getKey().getRole() == ChatFormat.Role.ASSISTANT) {
										Result result2 = nearest.get(trn.getValue()); // index to nearest list
										addRetrievedMessage(result2, results, returns, tokenizer);
									} else {
										TimestampRole tr2 = new TimestampRole(trr.getTimestamp(), ChatFormat.Role.ASSISTANT);
										getTimestampRole(results, returns, tr2, tokenizer);
									}
								} else {
									TimestampRole tr2 = new TimestampRole(trr.getTimestamp(), ChatFormat.Role.ASSISTANT);
									getTimestampRole(results, returns, tr2, tokenizer);
								}
							}					
						} else {
							TimestampRole tr2 = new TimestampRole(trr.getTimestamp(), ChatFormat.Role.ASSISTANT);
							getTimestampRole(results, returns, tr2, tokenizer);
						}
					} else {
						wasUser = true;
					}
				}
			}
			addRetrievedMessage(result, results, returns, tokenizer);
			++cnt;
		}
		if(DEBUG)
			log.info(cnt+" results inserted into context. Similarities:"+Arrays.toString(cossim));
		// if no interactions played out, lets build a system level series of responses with what we have
		if(cnt == 0) {
			if(DEBUG)
				log.info(tm.values().size()+" context entries, current results:"+results.size()+" max:"+(maxTokens - (((float)maxTokens) * .3)));
			it = tm.values().iterator();
			while(it.hasNext() && results.size() < (maxTokens - (((float)maxTokens) * .3))) {
				int i = it.next();
				Result result = nearest.get(i);
				TimestampRole trr =  (TimestampRole) result.get(0);
				trr.setRole(Role.SYSTEM);
				addRetrievedMessage(result, trr, results, returns, tokenizer);
			}
		}
		// put most recent user query last
		returns.add(promptFrame.getMessage());
		return returns;
	}
	/**
	 * Add the retrieved Result of 0 - TimestampRole 1 - NoIndex vector of List<Integer> to returns list
	 * as ChatFormat.Message
	 * @param result
	 * @param results
	 * @param returns
	 * @param tokenizer
	 */
	private void addRetrievedMessage(Result result, List<Integer> results, List<Message> returns, TokenizerInterface tokenizer) {
		NoIndex noIndex = (NoIndex) result.get(1);
		List<Integer> restensor = (List<Integer>)noIndex.getInstance();
		results.addAll(restensor); // to keep track of max context size
		if(DEBUG)
			log.info("addRetrievedMessage:"+(TimestampRole)result.get(0));
		returns.add(new ChatFormat.Message(((TimestampRole)result.get(0)).getRole(), tokenizer.decode(restensor)));
	}
	/**
	 * Permutation for existing TimestampRole, Result 0 - NoIndex vector
	 * add List<Integer> to returns list as ChatFormat.Message
	 * @param result
	 * @param trr
	 * @param results
	 * @param returns
	 * @param tokenizer
	 */
	private void addRetrievedMessage(Result result, TimestampRole trr, List<Integer> results, List<Message> returns, TokenizerInterface tokenizer) {
		NoIndex noIndex = (NoIndex) result.get(0);
		List<Integer> restensor = (List<Integer>)noIndex.getInstance();
		results.addAll(restensor); // to keep track of max context size
		if(DEBUG)
			log.info("addRetrievedMessage:"+trr);
		returns.add(new ChatFormat.Message(trr.getRole(), tokenizer.decode(restensor)));
	}
	/**
	 * Get the NoIndex vector of List<Integer> for existing TimestampRole
	 * @param results
	 * @param returns
	 * @param trr
	 * @param tokenizer
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	private void getTimestampRole(List<Integer> results, List<Message> returns, TimestampRole trr, TokenizerInterface tokenizer) throws InterruptedException, ExecutionException {
		if(DEBUG)
			log.info("getTimestampRole for "+trr);
		//CompletableFuture<Stream> cit = dbClient.findStream(xid, '*', trr, '?');
		//cit.get().forEach(e->{
		CompletableFuture<Iterator> cit = dbClient.findSet(xid, '*', trr, '?');
		Iterator<?> it = cit.get();
		// get one instance
		if(it.hasNext()) {
			if(DEBUG)
				log.info("getTimeStampRole result");
			//addRetrievedMessage((Result)e, trr, results, returns, tokenizer);
			addRetrievedMessage((Result)it.next(), trr, results, returns, tokenizer);
			//});
		}
	}

	/**
	 * Prime the semantic pump by retrieving last time value, then the relations with that value, later, feed the
	 * vectors for that time into the prompt, then retrieve any other indexes that match the retrieved indexes.
	 * So should pick up at least the 2 indexes for a USER/ASSISTANT request/response for a given timestamp
	 * @return The Results with index at element 0
	 * @throws InterruptedException
	 * @throws ExecutionException
	 */
	private List<Result> primeByTime() throws InterruptedException, ExecutionException {
		ArrayList<Result> res = new ArrayList<Result>();
		TimestampRole lastTime = (TimestampRole) dbClient.last(xid, TimestampRole.class).get();
		if(lastTime != null) {
			try (var timer = Timer.log("Querying by time "+ LocalDateTime.ofInstant(Instant.ofEpochMilli(lastTime.getTimestamp()), ZoneId.systemDefault()))) {
				CompletableFuture<Iterator> cres = dbClient.findSet(xid, '?', lastTime, '*');
				Iterator<?> it = cres.get();
				while(it.hasNext()) {
					// should be LSH index, NoIndex List<Integer> vector values
					res.add((Result) it.next());
				}
			}
		}
		if(DEBUG)
			log.info("primeByTime returned "+res.size()+" results.");
		return res;
	}
	/**
	 * Calculate the combined hash for a vector.
	 * @param hash one of numberOfHashes
	 * @param vector The vector to calculate the combined hash for.
	 * @return An integer representing a combined hash.
	 */
	private Integer hash(CosineHash[] hash, FloatTensor vector){
		int hashes[] = new int[hash.length];
		for(int i = 0 ; i < hash.length ; i++){
			hashes[i] = hash[i].hash(vector);
		}
		Integer combinedHash = CosineHash.combine(hashes);
		return combinedHash;
	}

	/**
	 * Return the number of hash functions used in the hash table.
	 * @return The number of hash functions used in the hash table.
	 */
	public int getNumberOfHashes() {
		return hashTable.get(0).length;
	}

	@Override
	public String toString() {
		return String.format("%s key=%s tables=%d hashes=%d",this.getClass().getName(), key, numberOfHashTables, numberOfHashes);
	}

	@Override
	public int compareTo(Object o) {
		int key0 = key.compareTo(((RelatrixLSH)o).key);
		if(key0 != 0)
			return key0;
		for(int i = 0; i < hashTable.size(); i++) {
			CosineHash[] cos0 = hashTable.get(i);
			CosineHash[] cos1 = (((RelatrixLSH)o).hashTable.get(i));
			for(int j = 0; j < cos0.length; j++) {
				if(j >= cos1.length)
					return 1;
				int key1 = cos0[j].compareTo(cos1[j]);
				if(key1 != 0)
					return key1;
			}
		}
		return 0;
	}
}


