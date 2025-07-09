/**
 * COMPILE_OPTIONS --add-modules=jdk.incubator.vector
 * RUNTIME_OPTIONS --add-modules=jdk.incubator.vector -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0
 *
 * Practical inference in a single Java file
 * Supports llama.cpp's GGUF format, restricted to Q4_0 and Q8_0 quantized models
 * Multi-threaded matrix vector multiplication routines implemented using Java's Vector API
 * Simple CLI with --chat and --instruct mode
 *
 *
 * Remember: Llama models use GPT2 vocabulary while non-Llama models use Llama vocabulary!
 */
package com.neocoretechs.roscar;

import jdk.incubator.vector.*;

import java.io.BufferedWriter;
import java.io.Externalizable;
import java.io.Serializable;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
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

import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

import java.util.*;
import java.util.concurrent.CompletableFuture;
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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.neocoretechs.relatrix.client.asynch.AsynchRelatrixClientTransaction;
import com.neocoretechs.relatrix.key.NoIndex;
import com.neocoretechs.relatrix.Result;
import com.neocoretechs.relatrix.Relation;
import com.neocoretechs.rocksack.TransactionId;
import com.neocoretechs.rocksack.Alias;
import com.neocoretechs.relatrix.DuplicateKeyException;

public class ModelRunner extends AbstractNodeMain {
	private static final Log log = LogFactory.getLog(ModelRunner.class);
    // Batch-size used in prompt evaluation.
    private static int BATCH_SIZE = Integer.getInteger("llama.BatchSize", 16);
    public final static boolean DEBUG = false;
    public static boolean DISPLAY_METADATA = false;
    public static AsynchRelatrixClientTransaction dbClient = null;
    //static RelatrixTransaction dbClient = null;
    public static TransactionId xid = null;
    public static Alias tensorAlias = null;
    // metadata dump
	public static BufferedWriter outputStream = null;
	public static PrintWriter output = null;
	public static FileWriter fileWriter = null;
	Llama model = null;
	Sampler sampler = null;
	Options options = null;
	public static final String SYSTEM_PROMPT = "/system_prompt";
	public static final String USER_PROMPT = "/user_prompt";
	public static final String ASSIST_PROMPT = "/assist_prompt";
	public static final String LLM = "/model";
	
	public static CircularBlockingDeque<String> messageQueue = new CircularBlockingDeque<String>(1024);
	
	public static RelatrixLSH relatrixLSH = null;

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
    
    public void displayMetadata(Llama model) {
        if(DISPLAY_METADATA) {
    		try {
    			ModelRunner.fileWriter = new FileWriter(options.modelPath.toString()+".metadata", false);
    			ModelRunner.outputStream = new BufferedWriter(fileWriter);
    			ModelRunner.output = new PrintWriter(outputStream);
    		} catch(IOException e) {
    			log.error("Could not open file " + options.modelPath.toString()+".metadata\r\n"+e);
    		}
        	ModelRunner.output.println("Begin Special tokens:");
        	ModelRunner.output.println(model.tokenizer().getSpecialTokens());
        	ModelRunner.output.println("End Special tokens.\r\n");
        	try {
        		ModelRunner.outputStream.flush();
        		ModelRunner.output.close();
        	} catch (final IOException e) {
        		log.error("Could not flush metadata file "+e);
        	} finally {
        		try {
        			if (ModelRunner.outputStream != null) {
        				ModelRunner.outputStream.close();
        			}
        			if (ModelRunner.output != null) {
        				ModelRunner.output.close();
        			}
        		} catch (final IOException e) {
        			log.error("Failed to close file: "+e);
        		}
        	}
        }
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
    /**
     * Element 0 is command /recallwords
     * @param query the command line with command keywords
     * @return the string of question/answer containing keywords
     */
    private static String parseKeywords(String[] query) {
      	if(query == null || query.length < 2)
    		return null;
     	StringBuilder sb = new StringBuilder();
      	/*CompletableFuture<Stream> s = dbClient.findStream(xid, '*', '?', '?');
      	try {
      		s.get().forEach(e->{
      			String s1 = (String)((Result)e).get(0);
      			for(int i = 1; i < query.length; i++) {
      				if(s1.contains(query[i])) {
      					sb.append(s1);
      					break;
      				}
      			}
      			s1 = (String)((Result)e).get(1);
      			for(int i = 1; i < query.length; i++) {
      				if(s1.contains(query[i])) {
      					sb.append(s1);
      					break;
      				}
      			}
      		});
      	} catch(InterruptedException | ExecutionException ie) {}
      	*/
      	return sb.toString();
    }
    
    record Options(Path modelPath, String prompt, String systemPrompt, boolean interactive,
                   float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo) {

        static final int DEFAULT_MAX_TOKENS = 512;

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        }

        static void require(boolean condition, String messageFormat, Object... args) {
            if (!condition) {
                log.error("ERROR " + messageFormat.formatted(args));
                log.info(printUsage());
                System.exit(-1);
            }
        }

        static String printUsage() {
            return 
            "Options:\r\n"+
            "  --model, -m <path>            required, path to .gguf file\r\n"+
            "  --system-prompt, -sp <string> (optional) system prompt\r\n"+
            "  --temperature, -temp <float>  temperature in [0,inf], default 0.1\r\n"+
            "  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95\r\n"+
            "  --seed <long>                 random seed, default System.nanoTime()\r\n"+
            "  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS+"\r\n"+
            "  --echo <boolean>              print ALL tokens to stderr, if true, default false\r\n"+
            "  --metadata                    write metadata file of <model file>.metadata\r\n\r\n"+
            "Examples:\r\n"+
            " --system-prompt \"Reply concisely, in French\"\r\n"+
            " --system-prompt \"Answer concisely\"\r\n";
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String systemPrompt = null;
            float temperature = 0.1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
            // Keep max context length small for low-memory devices.
            int maxTokens = DEFAULT_MAX_TOKENS;
            boolean interactive = false;
            boolean stream = true;
            boolean echo = false;

            for (int i = 0; i < args.length; i++) {
                String optionName = args[i];
                require(optionName.startsWith("-"), "Invalid option %s", optionName);
                switch (optionName) {
                    case "--interactive", "--chat", "-i" -> interactive = true;
                    case "--instruct" -> interactive = false;
                    case "--help", "-h" -> {
                        log.info(printUsage());
                        System.exit(0);
                    }
                    default -> {
                        String nextArg;
                        if (optionName.contains("=")) {
                            String[] parts = optionName.split("=", 2);
                            optionName = parts[0];
                            nextArg = parts[1];
                        } else {
                            require(i + 1 < args.length, "Missing argument for option %s", optionName);
                            nextArg = args[i + 1];
                            i += 1; // skip arg
                        }
                        switch (optionName) {
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Paths.get(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                            case "--metadata" -> DISPLAY_METADATA = true;
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            return new Options(modelPath, prompt, systemPrompt, interactive, temperature, topp, seed, maxTokens, stream, echo);
        }
    }
    
	@Override
	public GraphName getDefaultNodeName() {
		return GraphName.of("llm");
	}
	
	@Override
	public void onStart(final ConnectedNode connectedNode) {
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
			relatrixLSH = new RelatrixLSH(dbClient, options.maxTokens());
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}
		List<String> nodeArgs = connectedNode.getNodeConfiguration().getCommandLineLoader().getNodeArguments();
		options = Options.parseOptions(nodeArgs.toArray(new String[nodeArgs.size()]));
		ChatFormatInterface chatFormat;

		try {
			model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
			if(model == null)
				model = ModelLoader.loadModel(options.modelPath(), options.maxTokens(), true);
		} catch(IOException e) {
			log.error("Could not load model " + options.modelPath.toString()+e);
			System.exit(1);
		}
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
		//
		// Set up publisher
		//final Log log = connectedNode.getLog();
		final Publisher<std_msgs.String> pubmodel = connectedNode.newPublisher(LLM, std_msgs.String._TYPE);
		// Subscribers
		Subscriber<std_msgs.String> subsystem = connectedNode.newSubscriber(SYSTEM_PROMPT, std_msgs.String._TYPE);
		Subscriber<std_msgs.String> subsuser = connectedNode.newSubscriber(USER_PROMPT, std_msgs.String._TYPE);
		Subscriber<std_msgs.String> subsassist = connectedNode.newSubscriber(ASSIST_PROMPT, std_msgs.String._TYPE);
		//
		// set up subscriber callback
		//
		subsuser.addMessageListener(new MessageListener<std_msgs.String>() {
			@Override
			public void onNewMessage(std_msgs.String message) {
		        Llama.State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
		        List<Integer> promptTokens = new ArrayList<>();
		        List<Integer> memoryTokens = new ArrayList<>();
		        memoryTokens.add(chatFormat.getBeginOfText());
		        memoryTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, message.getData())));
		        try {
					memoryTokens = relatrixLSH.findNearest(memoryTokens);
				} catch (IllegalArgumentException | ClassNotFoundException | IllegalAccessException | IOException | InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
		        promptTokens.addAll(memoryTokens);
		        Optional<String> response = processMessage(model, options, sampler, state, chatFormat, promptTokens);
		        if(response.isPresent()) {
		        	//log.info("Queueing from role USER:"+response.get());
		        	try {
		        		messageQueue.addLastWait(response.get());
		        	} catch(InterruptedException ie) {}
		        }
			}
		});
		subsystem.addMessageListener(new MessageListener<std_msgs.String>() {
			@Override
			public void onNewMessage(std_msgs.String message) {
		        Llama.State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
		        List<Integer> promptTokens = new ArrayList<>();
		        List<Integer> memoryTokens = new ArrayList<>();
		        memoryTokens.add(chatFormat.getBeginOfText());
		        memoryTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, message.getData())));
		        try {
					memoryTokens = relatrixLSH.findNearest(memoryTokens);
				} catch (IllegalArgumentException | ClassNotFoundException | IllegalAccessException | IOException | InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
		        promptTokens.addAll(memoryTokens);
		        Optional<String> response = processMessage(model, options, sampler, state, chatFormat, promptTokens);
		        if(response.isPresent()) {
		        	//log.info("Queueing from role SYSTEM:"+response.get());
		        	try {
		        		messageQueue.addLastWait(response.get());
        			} catch(InterruptedException ie) {}
		        }
			}
		});
		subsassist.addMessageListener(new MessageListener<std_msgs.String>() {
			@Override
			public void onNewMessage(std_msgs.String message) {
		        Llama.State state = model.createNewState(BATCH_SIZE, chatFormat.getBeginOfText());
		        List<Integer> promptTokens = new ArrayList<>();
		        List<Integer> memoryTokens = new ArrayList<>();
		        memoryTokens.add(chatFormat.getBeginOfText());
		        memoryTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, message.getData())));
		        try {
					memoryTokens = relatrixLSH.findNearest(memoryTokens);
				} catch (IllegalArgumentException | ClassNotFoundException | IllegalAccessException | IOException | InterruptedException | ExecutionException e) {
					e.printStackTrace();
				}
		        promptTokens.addAll(memoryTokens);
		        Optional<String> response = processMessage(model, options, sampler, state, chatFormat, promptTokens);
		        if(response.isPresent()) {
		        	//log.info("Queueing from role ASSIST:"+response.get());
		        	try {
		        		messageQueue.addLastWait(response.get());
		        	} catch(InterruptedException ie) {}
		        }
			}
		});
		
		/**
		 * Main publishing loop. Essentially we are publishing the data in whatever state its in, using the
		 * mutex appropriate to establish critical sections. A sleep follows each publication to keep the bus arbitrated
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
	}

	public static Optional<String> processMessage(Llama model, Options options, Sampler sampler, Llama.State state, ChatFormatInterface chatFormat, List<Integer> promptTokens ) {
        Set<Integer> stopTokens = chatFormat.getStopTokens();
        List<Integer> responseTokens = Llama.generateTokens(model, state, 0, promptTokens, stopTokens, options.maxTokens(), sampler, options.echo(), null);
        if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
            responseTokens.removeLast();
        }
		try(Timer t = Timer.log("SaveState")) {
			relatrixLSH.add(responseTokens);
		} catch (IllegalAccessException | ClassNotFoundException | IOException | InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
        return Optional.ofNullable(model.tokenizer().decode(responseTokens));
	}
	
	public static final int HISTORY_DEPTH = 3; // last 3 messages
	public static List<Integer> buildPrompt(ChatFormatInterface format,
			List<ChatFormat.Message> recentMessages,
			FloatTensor queryVec,
			Llama model,
			Options options) {
		List<Integer> promptTokens = new ArrayList<>();
		// System prompt
		promptTokens.add(format.getBeginOfText());
		if (options.systemPrompt() != null) {
			promptTokens.addAll(format.encodeMessage(new ChatFormat.Message(
					ChatFormat.Role.SYSTEM, options.systemPrompt())));
		}
		// Semantic memory via Relatrix
		//List<Result> retrieved = MemoryAugmentor.augmentState(queryVec);
		//for (Result r : retrieved) {
		//	promptTokens.addAll(format.encodeMessage(new ChatFormat.Message(
		//			ChatFormat.Role.SYSTEM, "Relevant: " + r.word())));
		//}
		// Most recent messages
		int start = Math.max(0, recentMessages.size() - HISTORY_DEPTH);
		for (int i = start; i < recentMessages.size(); i++) {
			promptTokens.addAll(format.encodeMessage(recentMessages.get(i)));
		}
		// Assistant header
		promptTokens.addAll(format.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
		return promptTokens;
	}

}

final class GGUF {
	private static final Log log = LogFactory.getLog(GGUF.class);
    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private Map<String, GGUFTensorInfo> tensorInfos;

    private long tensorDataOffset;

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    private final ByteBuffer BB_1 = ByteBuffer.allocate(Byte.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_2 = ByteBuffer.allocate(Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_4 = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath);
            var ignored = Timer.log("Parse " + modelPath)) {
            GGUF gguf = new GGUF();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }

    enum MetadataValueType {
        // The value is a 8-bit unsigned integer.
        UINT8(1),
        // The value is a 8-bit signed integer.
        INT8(1),
        // The value is a 16-bit unsigned little-endian integer.
        UINT16(2),
        // The value is a 16-bit signed little-endian integer.
        INT16(2),
        // The value is a 32-bit unsigned little-endian integer.
        UINT32(4),
        // The value is a 32-bit signed little-endian integer.
        INT32(4),
        // The value is a 32-bit IEEE754 floating point number.
        FLOAT32(4),
        // The value is a boolean.
        // 1-byte value where 0 is false and 1 is true.
        // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
        BOOL(1),
        // The value is a UTF-8 non-null-terminated string, with length prepended.
        STRING(-8),
        // The value is an array of other values, with the length and type prepended.
        // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
        ARRAY(-8),
        // The value is a 64-bit unsigned little-endian integer.
        UINT64(8),
        // The value is a 64-bit signed little-endian integer.
        INT64(8),
        // The value is a 64-bit IEEE754 floating point number.
        FLOAT64(8);
        private final int byteSize;

        MetadataValueType(int byteSize) {
            this.byteSize = byteSize;
        }

        private static final MetadataValueType[] VALUES = values();

        public static MetadataValueType fromIndex(int index) {
            return VALUES[index];
        }

        public int byteSize() {
            return byteSize;
        }
    }

    private void loadModelImpl(FileChannel fileChannel) throws IOException {
        // The header of the file.
        readHeader(fileChannel); // gguf_header_t header;
        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUF.GGUFTensorInfo ti = readTensorInfo(fileChannel);
            assert !tensorInfos.containsKey(ti.name);
            tensorInfos.put(ti.name, ti);
        }
        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        //long _padding = -fileChannel.position() & (ALIGNMENT - 1);
        long _padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + _padding);
        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be close
        // or identical to the data in the original model file, but may be different due to quantization or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = fileChannel.position();
    }

    public static Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset, Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        Arena arena = Arena.ofAuto();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset, fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        if(ModelRunner.DISPLAY_METADATA)
        	ModelRunner.output.println("Begin Tensors:");
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            int numberOfElements = FloatTensor.numberOfElements(ti.dimensions());
            int sizeInBytes = Math.toIntExact(ti.ggmlType().byteSizeFor(numberOfElements));
            if(ModelRunner.DISPLAY_METADATA)
            	ModelRunner.output.println("Tensor:"+entry.getKey()+"="+ti.name+" offset:"+ti.offset+" dims:"+Arrays.toString(ti.dimensions)+" number elems:"+numberOfElements+" size:"+sizeInBytes);
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(), new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        if(ModelRunner.DISPLAY_METADATA)
        	ModelRunner.output.println("End Tensors.\r\n");
        return tensorEntries;
    }

    public record GGUFTensorInfo(String name, int[] dimensions, GGMLType ggmlType, long offset) {
    }

    private GGMLType readGGMLType(FileChannel fileChannel) throws IOException {
        int ggmlTypeId = readInt(fileChannel); // ggml_type type;
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUF.GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(fileChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(fileChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(fileChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(fileChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUF.GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(FileChannel fileChannel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        byte[] bytes = new byte[len]; // char string[len];
        int bytesRead = fileChannel.read(ByteBuffer.wrap(bytes));
        assert len == bytesRead;
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        String key = readString(fileChannel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints().allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(FileChannel fileChannel) throws IOException {
        // The type of the value.
        // Must be one of the `gguf_metadata_value_type` values.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type value_type;
        // The value.
        return readMetadataValueOfType(value_type, fileChannel); // gguf_metadata_value_t value;
    }

    void readHeader(FileChannel fileChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        this.magic = readInt(fileChannel); //    uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        this.version = readInt(fileChannel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        this.tensorCount = Math.toIntExact(readLong(fileChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(fileChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(FileChannel fileChannel) throws IOException {
        // Any value type is valid, including arrays.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type type;
        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;
        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (value_type) {
            case UINT8, INT8 -> {
                byte[] bytes = new byte[len];
                for (int i = 0; i < len; ++i) {
                    bytes[i] = readByte(fileChannel);
                }
                return bytes;
            }
            case UINT16, INT16 -> {
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(fileChannel);
                }
                return shorts;
            }
            case UINT32, INT32 -> {
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(fileChannel);
                }
                return ints;
            }
            case FLOAT32 -> {
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(fileChannel);
                }
                return floats;
            }
            case BOOL -> {
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(fileChannel);
                }
                return booleans;
            }
            case STRING -> {
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(fileChannel);
                }
                return strings;
            }
            case ARRAY -> {
                Object[] arrays = new Object[len];
                for (int i = 0; i < len; ++i) {
                    arrays[i] = readArray(fileChannel);
                }
                return arrays;
            }
            default -> throw new UnsupportedOperationException("read array of " + value_type);
        }
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 -> readByte(fileChannel);
            case UINT16, INT16 -> readShort(fileChannel);
            case UINT32, INT32 -> readInt(fileChannel);
            case FLOAT32 -> readFloat(fileChannel);
            case UINT64, INT64 -> readLong(fileChannel);
            case FLOAT64 -> readDouble(fileChannel);
            case BOOL -> readBoolean(fileChannel);
            case STRING -> readString(fileChannel);
            case ARRAY -> readArray(fileChannel);
        };
    }

    private byte readByte(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_1);
        assert bytesRead == 1;
        return BB_1.clear().get(0);
    }

    private boolean readBoolean(FileChannel fileChannel) throws IOException {
        return readByte(fileChannel) != 0;
    }

    private short readShort(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_2);
        assert bytesRead == 2;
        return BB_2.clear().getShort(0);
    }

    private int readInt(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_4);
        assert bytesRead == 4;
        return BB_4.clear().getInt(0);
    }

    private long readLong(FileChannel fileChannel) throws IOException {
        int bytesRead = fileChannel.read(BB_8);
        assert bytesRead == 8;
        return BB_8.clear().getLong(0);
    }

    private float readFloat(FileChannel fileChannel) throws IOException {
        return Float.intBitsToFloat(readInt(fileChannel));
    }

    private double readDouble(FileChannel fileChannel) throws IOException {
        return Double.longBitsToDouble(readLong(fileChannel));
    }

    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }

    public int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        return alignment;
    }
}

interface Timer extends AutoCloseable {
	static final Log log = LogFactory.getLog(Timer.class);
    @Override
    void close(); // no Exception

    static Timer log(String label) {
        return log(label, TimeUnit.MILLISECONDS);
    }

    static Timer log(String label, TimeUnit timeUnit) {
        return new Timer() {
            final long startNanos = System.nanoTime();

            @Override
            public void close() {
                long elapsedNanos = System.nanoTime() - startNanos;
                log.info(label + ": "
                        + timeUnit.convert(elapsedNanos, TimeUnit.NANOSECONDS) + " "
                        + timeUnit.toChronoUnit().name().toLowerCase());
            }
        };
    }
}
/**
 * Load model, get GGUF metadata, load vocabulary, create tokenizer, create config, if loadWeights - load tensors, load weights
 * create Llama with config, tokenizer, weights
 */
final class ModelLoader {
    static final String TOKENIZER_GPT2_MODEL = "gpt2"; // ModelRunner uses gpt2!
    static final String TOKENIZER_LLAMA_MODEL = "llama"; // non Llama uses llama!
    public static String model = "gpt2"; // default for Llama models!
    public static String name = null; // Name is based solely on name of model, they all seem to have their own ChatFormat not based on model
    private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
        model = (String) metadata.get("tokenizer.ggml.model");
        name = (String) metadata.get("general.name");
        if(name.toLowerCase().contains("llama")) // Meta Llama etc. etc.
        	name = "llama";
        else
        	if(name.toLowerCase().contains("mistral")) //models--mistralai etc. etc.
        		name="mistral";

        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        if(TOKENIZER_LLAMA_MODEL.equals(model)) {
        	float[] scores = (float[]) metadata.get("tokenizer.ggml.scores");
        	return new Vocabulary(tokens, scores);
        } else {
        	if(TOKENIZER_GPT2_MODEL.equals(model)) {
        		return new Vocabulary(tokens, null);
        	} else {
        		throw new IllegalArgumentException("expected " + TOKENIZER_GPT2_MODEL + " or "+ TOKENIZER_LLAMA_MODEL+ " but found " + model);
        	}
        }
    }

    public static Llama loadModel(Path ggufPath, int contextLength, boolean loadWeights) throws IOException {
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        return loadModel(fileChannel, gguf, contextLength, loadWeights);
    }

    public static Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights) throws IOException {
        try (var ignored = Timer.log("Load model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            if(ModelRunner.DISPLAY_METADATA) {
            	ModelRunner.output.println("Begin GGUF Metadata:");
            	metadata.forEach((k, v) -> {
            		String valueStr;
            		if (v != null && v.getClass().isArray()) {
            			Class<?> componentType = v.getClass().getComponentType();
            			if (componentType == int.class) {
            				valueStr = Arrays.toString((int[]) v);
            			} else if (componentType == byte.class) {
            				valueStr = Arrays.toString((byte[]) v);
            			} else if (componentType == double.class) {
            				valueStr = Arrays.toString((double[]) v);
            			} else if (componentType == boolean.class) {
            				valueStr = Arrays.toString((boolean[]) v);
            			} else if (componentType == char.class) {
            				valueStr = Arrays.toString((char[]) v);
            			} else if (componentType == long.class) {
            				valueStr = Arrays.toString((long[]) v);
            			} else if (componentType == float.class) {
            				valueStr = Arrays.toString((float[]) v);
            			} else if (componentType == short.class) {
            				valueStr = Arrays.toString((short[]) v);
            			} else {
            				valueStr = Arrays.toString((Object[]) v); // for Object arrays
            			}
            		} else {
            			valueStr = String.valueOf(v);
            		}
            		ModelRunner.output.println(k + "=" + valueStr);
            	});
            	ModelRunner.output.println("End GGUF Metadata.\r\n");
            }
            Vocabulary vocabulary = loadVocabulary(metadata);
            TokenizerInterface tokenizer;
            Llama.Configuration config;
            Llama.Weights weights = null;
            String arch = (String) metadata.get("general.architecture");
            if(ModelLoader.name.equals("mistral")) {
           		tokenizer = createLlamaTokenizer(metadata, vocabulary);
        		config = createConfig(arch, metadata, vocabulary, contextLength);
        		if (loadWeights) {
        			// loadTensors corresponds to getTensorEntries in old version
        			Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
        			weights = loadLlamaWeights(tensorEntries, config);
        		}
            } else {
            	if(ModelLoader.name.equals("llama")) {
                   	tokenizer = createGPT2Tokenizer(metadata, vocabulary);
                	config = createConfig(arch, metadata, vocabulary, contextLength);
                    if (loadWeights) {
                    	// loadTensors corresponds to getTensorEntries in old version
                        Map<String, GGMLTensorEntry> tensorEntries = GGUF.loadTensors(fileChannel, gguf.getTensorDataOffset(), gguf.getTensorInfos());
                        weights = loadGPT2Weights(tensorEntries, config);
                    }
            	} else
        			throw new IllegalArgumentException("expected metadata general.name containing mistral oe llama but found "+ModelLoader.name);
            }
            return new Llama(config, tokenizer, weights);
        }
    }
    
    static Llama.Configuration createConfig(String arch, Map<String, Object> metadata, Vocabulary vocabulary, int contextLength) {
        Llama.Configuration config = new Llama.Configuration(
                (int) metadata.get(arch+".embedding_length"),
                (int) metadata.get(arch+".feed_forward_length"),
                (int) metadata.get(arch+".block_count"),
                (int) metadata.get(arch+".attention.head_count"),

                metadata.containsKey(arch+".attention.head_count_kv")
                        ? (int) metadata.get(arch+".attention.head_count_kv")
                        : (int) metadata.get(arch+".attention.head_count"),

                vocabulary.size(),
                (int) metadata.get(arch+".context_length"),
                (float) metadata.getOrDefault(arch+".attention.layer_norm_rms_epsilon", 1e-5f),
                (float) metadata.getOrDefault(arch+".rope.freq_base", 10000f)
        ).withContextLength(contextLength);
        return config;
    }
    

    /**
     * Called from AOT.tryUsePreloaded and ModelLoader.loadModel
     * @param tensorEntries
     * @param config
     * @return
     */
    static Llama.Weights loadGPT2Weights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        float scaleFactor = 8;
        float loFreqFactor = 1;
        float hiFreqFactor = 3;
        int oldContextLength = 8192;
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta,
                ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        Llama.Weights qw = new Llama.Weights(
                loadQuantized(tokenEmbeddings),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                toFloatBuffer(tensorEntries.get("output_norm.weight")),
                FloatBuffer.wrap(ropeFreqsReal),
                FloatBuffer.wrap(ropeFreqsImag),
                // If "output.weight" is not present then the embedding weights are tied/shared with the decoder.
                // This is commonly referred as "tie word embeddings".
                loadQuantized(tensorEntries.getOrDefault("output.weight", tokenEmbeddings))
        );
        return qw;
    }
    
    static Llama.Weights loadLlamaWeights(Map<String, GGMLTensorEntry> tensorEntries, Llama.Configuration config) {
    	   Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta);
           float[] ropeFreqsReal = ropeFreqs.first();
           float[] ropeFreqsImag = ropeFreqs.second();

           Llama.Weights qw = new Llama.Weights(
                   loadQuantized(tensorEntries.get("token_embd.weight")),
                   loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                   loadArrayOfFloatBuffer(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                   loadArrayOfQuantized(config.numberOfLayers, i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                   toFloatBuffer(tensorEntries.get("output_norm.weight")),
                   FloatBuffer.wrap(ropeFreqsReal),
                   FloatBuffer.wrap(ropeFreqsImag),
                   loadQuantized(tensorEntries.get("output.weight"))
           );
           return qw;
    }
    
 
    private static Tokenizer createGPT2Tokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts ->
                        new Pair<>(
                                vocabulary.getIndex(parts[0]).orElseThrow(),
                                vocabulary.getIndex(parts[1]).orElseThrow())
                ).toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        int reservedSpecialTokens = allTokens - baseTokens;
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                i -> specialTokensList.get(i),
                                i -> baseTokens + i)
                        );

        return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
    }
    
    private static MistralTokenizer createLlamaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
        List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
        Map<String, Integer> specialTokens =
                IntStream.range(0, specialTokensList.size())
                        .boxed()
                        .collect(Collectors.toMap(
                                t -> vocabulary.get(t),
                                t -> t)
                        );
        return new MistralTokenizer(vocabulary, null, specialTokens, tokenTypes);
    }
    
    public static FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case F32 -> new F32FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q8_0 -> new Q8_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 -> new Q4_0FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case BF16 -> new BF16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            case F16 -> new F16FloatTensor(FloatTensor.numberOfElements(entry.shape()), entry.memorySegment());
            default -> throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public static FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public static FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 -> tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default -> throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }
}

record Llama(Configuration configuration, TokenizerInterface tokenizer, Weights weights) {
	private static final Log log = LogFactory.getLog(Llama.class);
	private static boolean DEBUG = true;
    public State createNewState(int batchsize, int beginOfText) {
    	if(DEBUG )
    		log.info("Create new state...");
        State state = new State(configuration(), batchsize);
        state.latestToken = beginOfText; // was tokenizer.getSpecialTokens().get("<|begin_of_text|>");, now we get from ChatFormat.beginOfText() which does the same
        return state;
    }

    public static final class Configuration {
        public final int dim; // transformer dimension
        public final int hiddenDim; // for ffn layers
        public final int numberOfLayers; // number of layers
        public final int numberOfHeads; // number of query heads
        public final int numberOfKeyValueHeads; // number of key/value heads (can be < query heads because of multiquery)
        public final int vocabularySize; // vocabulary size, usually 256 (byte-level)
        public final int contextLength; // max sequence length
        public final float rmsNormEps;
        public final float ropeTheta;
        public final int headSize;

        Configuration withContextLength(int newContextLength) {
            if (newContextLength < 0) {
                return this; // no change
            }
            return new Configuration(this.dim, this.hiddenDim, this.numberOfLayers, this.numberOfHeads, this.numberOfKeyValueHeads, this.vocabularySize, newContextLength, this.rmsNormEps, this.ropeTheta);
        }

        public Configuration(int dim, int hiddenDim, int numberOfLayers, int numberOfHeads, int numberOfKeyValueHeads, int vocabularySize, int contextLength, float rmsNormEps, float ropeTheta) {
            this.dim = dim;
            this.hiddenDim = hiddenDim;
            this.numberOfLayers = numberOfLayers;
            this.numberOfHeads = numberOfHeads;
            this.numberOfKeyValueHeads = numberOfKeyValueHeads;
            this.vocabularySize = vocabularySize;
            this.contextLength = contextLength;
            this.rmsNormEps = rmsNormEps;
            this.ropeTheta = ropeTheta;
            this.headSize = dim / numberOfHeads;
        }
    }

    public static final class Weights {
        // token embedding table
        public final FloatTensor token_embedding_table; // (vocab_size, dim)
        // weights for rmsnorms
        public final FloatBuffer[] rms_att_weight; // (layer, dim) rmsnorm weights
        // weights for matmuls
        public final FloatTensor[] wq; // (layer, n_heads * head_size)
        public final FloatTensor[] wk; // (layer, n_kv_heads, head_size)
        public final FloatTensor[] wv; // (layer, n_kv_heads * head_size)
        public final FloatTensor[] wo; // (layer, n_heads * head_size, dim)
        
        public final FloatBuffer[] rms_ffn_weight; // (layer, dim)
        // weights for ffn
        public final FloatTensor[] w1; // (layer, hidden_dim, dim)
        public final FloatTensor[] w2; // (layer, dim, hidden_dim)
        public final FloatTensor[] w3; // (layer, hidden_dim, dim)
        // public final rmsnorm
        public final FloatBuffer rms_final_weight; // (dim,)
        // freq_cis for RoPE relatively positional embeddings
        public final FloatBuffer freq_cis_real; // (seq_len, head_size/2)
        public final FloatBuffer freq_cis_imag; // (seq_len, head_size/2)
        // (optional) classifier weights for the logits, on the last layer
        public final FloatTensor wcls; // (vocab_size, dim)

        public Weights(FloatTensor token_embedding_table, FloatBuffer[] rms_att_weight, FloatTensor[] wq, FloatTensor[] wk, FloatTensor[] wv, FloatTensor[] wo, FloatBuffer[] rms_ffn_weight, FloatTensor[] w1, FloatTensor[] w2, FloatTensor[] w3, FloatBuffer rms_final_weight, FloatBuffer freq_cis_real, FloatBuffer freq_cis_imag, FloatTensor wcls) {
            this.token_embedding_table = token_embedding_table;
            this.rms_att_weight = rms_att_weight;
            this.wq = wq;
            this.wk = wk;
            this.wv = wv;
            this.wo = wo;
            this.rms_ffn_weight = rms_ffn_weight;
            this.w1 = w1;
            this.w2 = w2;
            this.w3 = w3;
            this.rms_final_weight = rms_final_weight;
            this.freq_cis_real = freq_cis_real;
            this.freq_cis_imag = freq_cis_imag;
            this.wcls = wcls;
        }

    }

    public static class State implements Externalizable {
    	private static final long serialVersionUID = -1L;
    	private static final Log log = LogFactory.getLog(State.class);
    	// current wave of activations
    	public int batchsize;
    	public FloatTensor[] x; // activation at current time stamp (dim,)
    	public FloatTensor[] xb; // same, but inside a residual branch (dim,)
    	public FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
    	public FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    	public FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    	public FloatTensor[] q; // query (dim,)
    	public FloatTensor[] k; // key (dim,)
    	public FloatTensor[] v; // value (dim,)
    	public FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
    	public FloatTensor logits; // output logits

    	// kv cache
    	public FloatTensor[] keyCache;   // (n_layer, seq_len, kv_dim)
    	public FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

    	/** last index in previous block */
    	int idxPrevBlock;

    	public int latestToken;

    	public State() {}
    	
    	State(Configuration config, int batchsize) {
    		this.batchsize = batchsize;
    		this.x = allocate(batchsize, config.dim);
    		this.xb = allocate(batchsize, config.dim);
    		this.xb2 = allocate(batchsize, config.dim);
    		this.hb = allocate(batchsize, config.hiddenDim);
    		this.hb2 = allocate(batchsize, config.hiddenDim);
    		this.q = allocate(batchsize, config.dim);
    		this.k = allocate(batchsize, config.dim);
    		this.v = allocate(batchsize, config.dim);
    		this.att = allocate(batchsize, config.numberOfHeads, config.contextLength);
    		idxPrevBlock = -1;

    		this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
    		int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
    		this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
    		this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim)).limit(config.numberOfLayers).toArray(FloatTensor[]::new);
    	}

    	@Override
    	public void writeExternal(ObjectOutput out) throws IOException {
    		out.writeInt(batchsize);
    		out.writeInt(latestToken);
    		out.writeInt(x.length);
    		for(FloatTensor x0: x)
    			out.writeObject(x0); // activation at current time stamp (dim,)
    		out.writeInt(xb.length);
    		for(FloatTensor xb0: xb)
    			out.writeObject(xb0); // same, but inside a residual branch (dim,)
    		out.writeInt(xb2.length);
    		for(FloatTensor xb20: xb2)
    			out.writeObject(xb20); // an additional buffer just for convenience (dim,)
    		out.writeInt(hb.length);
    		for(FloatTensor hb0: hb)
    			out.writeObject(hb0); // buffer for hidden dimension in the ffn (hidden_dim,)
    		out.writeInt(hb2.length);
    		for(FloatTensor hb20: hb2)
    			out.writeObject(hb20); // buffer for hidden dimension in the ffn (hidden_dim,)
    		out.writeInt(q.length);
    		for(FloatTensor q0: q)
    			out.writeObject(q0);// query (dim,)
    		out.writeInt(k.length);
    		for(FloatTensor k0: k)
    			out.writeObject(k0);// key (dim,)
    		out.writeInt(v.length);
    		for(FloatTensor v0: v)
    			out.writeObject(v0); // value (dim,)
    		out.writeInt(att.length);
    		for(FloatTensor att0: att)
    			out.writeObject(att0); // buffer for scores/attention values (n_heads, seq_len)
    		out.writeObject(logits); // output logits

    		// kv cache
    		out.writeInt(keyCache.length);
    		for(FloatTensor keyCache0: keyCache)
    			out.writeObject(keyCache0);   // (n_layer, seq_len, kv_dim)
    		out.writeInt(valueCache.length);
    		for(FloatTensor valueCache0: valueCache)
    			out.writeObject(valueCache0); // (n_layer, seq_len, kv_dim)

    		/** last index in previous block */
    		out.writeInt(idxPrevBlock);
    	}

    	@Override
    	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
    		batchsize = in.readInt();
    		latestToken = in.readInt();
    		int len;
    		// x
    		len = in.readInt();
    		x = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			x[i] = (FloatTensor) in.readObject();
    		}
    		// xb
    		len = in.readInt();
    		xb = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			xb[i] = (FloatTensor) in.readObject();
    		}
    		// xb2
    		len = in.readInt();
    		xb2 = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			xb2[i] = (FloatTensor) in.readObject();
    		}
    		// hb
    		len = in.readInt();
    		hb = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			hb[i] = (FloatTensor) in.readObject();
    		}
    		// hb2
    		len = in.readInt();
    		hb2 = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			hb2[i] = (FloatTensor) in.readObject();
    		}
    		// q
    		len = in.readInt();
    		q = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			q[i] = (FloatTensor) in.readObject();
    		}
    		// k
    		len = in.readInt();
    		k = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			k[i] = (FloatTensor) in.readObject();
    		}
    		// v
    		len = in.readInt();
    		v = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			v[i] = (FloatTensor) in.readObject();
    		}
    		// att
    		len = in.readInt();
    		att = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			att[i] = (FloatTensor) in.readObject();
    		}
    		// logits
    		logits = (FloatTensor) in.readObject();
    		// keyCache
    		len = in.readInt();
    		keyCache = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			keyCache[i] = (FloatTensor) in.readObject();
    		}
    		// valueCache
    		len = in.readInt();
    		valueCache = new FloatTensor[len];
    		for (int i = 0; i < len; i++) {
    			valueCache[i] = (FloatTensor) in.readObject();
    		}
    		// idxPrevBlock
    		idxPrevBlock = in.readInt();
    	}

    }

    static FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    static void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // calculate sum of squares
        float ss = x.reduce(0, size, 0f, (acc, xi) -> acc + xi * xi);
        ss /= size;
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss; // for the lambda
        out.mapWithIndexInPlace(0, size, (value, index) -> weight.get(index) * (finalss * x.getFloat(index)));
    }

    static FloatTensor forward(Llama model, State state, int[] tokens, int position, boolean computeLogits) {
        // a few convenience variables
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads; // integer multiplier of the kv sharing in multiquery
        float sqrtHeadSize = (float) Math.sqrt(headSize);
        final int nTokens = tokens.length;

        // copy the token embedding into x
        Parallel.parallelFor(0, nTokens, t ->
            weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim)
        );

        // forward all the layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            // attention rmsnorm
            // rmsnorm(state.xb, state.x, weights.rms_att_weight[l], dim, config.rmsNormEps);
            final int curLayer = l;
            Parallel.parallelFor(0, nTokens, t ->
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps)
            );

            // qkv matmuls for this position
            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            Parallel.parallelFor(0, nTokens, t -> {
                for (int i = 0; i < dim; i += 2) {
                    int head_dim = i % headSize;
                    float fcr = weights.freq_cis_real.get((position + t) * (headSize / 2) + (head_dim / 2));
                    float fci = weights.freq_cis_imag.get((position + t) * (headSize / 2) + (head_dim / 2));
                    int rotn = i < kvDim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                    for (int vi = 0; vi < rotn; vi++) {
                        FloatTensor vec = vi == 0 ? state.q[t] : state.k[t]; // the vector to rotate (query or key)
                        float v0 = vec.getFloat(i);
                        float v1 = vec.getFloat(i + 1);
                        vec.setFloat(i, v0 * fcr - v1 * fci);
                        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
                    }
                }
            });

            // save key,value at this time step (position) to our kv cache
            //int loff = l * config.seq_len * kvDim; // kv cache layer offset for convenience
            Parallel.parallelFor(0, nTokens, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            // If the logits are not required, the attention and FFN of the last layer can be skipped entirely.
            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                state.idxPrevBlock = nTokens - 1;
                return null;
            }

            // multihead attention. iterate over all heads
            Parallel.parallelForLong(0, (long) nTokens * (long) config.numberOfHeads, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                // get the query vector for this head
                // float* q = s.q + h * headSize;
                int qOffset = h * headSize;

                // attention scores for this head
                // float* att = s.att + h * config.seq_len;
                int attOffset = h * config.contextLength;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= position + token; t++) {
                    // get the key vector for this head and at this timestep
                    // float* k = s.key_cache + loff + t * dim + h * headSize;
                    int keyCacheOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // calculate the attention score as the dot product of q and k
                    float score = state.q[token].dot(qOffset, state.keyCache[curLayer], keyCacheOffset, headSize);
                    score /= sqrtHeadSize;
                    // save the score to the attention buffer
                    state.att[token].setFloat(attOffset + t, score);
                }

                // softmax the scores to get attention weights, from 0..position inclusively
                state.att[token].softmaxInPlace(attOffset, position + token + 1);

                // weighted sum of the values, store back into xb
                // float* xb = s.xb + h * headSize;
                int xbOffset = h * headSize;
                // memset(xb, 0, headSize * sizeof(float));
                state.xb[token].fillInPlace(xbOffset, headSize, 0f);

                for (int t = 0; t <= position + token; t++) {
                    // get the value vector for this head and at this timestep
                    // float* v = s.value_cache + loff + t * dim + h * headSize;
                    int vOffset = /* loff + */ t * kvDim + (h / kvMul) * headSize;
                    // get the attention weight for this timestep
                    float a = state.att[token].getFloat(attOffset + t);
                    // accumulate the weighted value into xb
                    state.xb[token].saxpyInPlace(xbOffset, state.valueCache[curLayer], vOffset, headSize, a);
                }
            });

            // final matmul to get the output of the attention
            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, dim);

            // residual connection back into x
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb2[t]);
            });

            // ffn rmsnorm
            Parallel.parallelFor(0, nTokens, t -> {
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps);
            });

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);

            // SwiGLU non-linearity
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].mapInPlace(value -> value / (float) (1.0 + Math.exp(-value)));
            });

            // elementwise multiply with w3(x)
            Parallel.parallelFor(0, nTokens, t -> {
                state.hb[t].multiplyInPlace(state.hb2[t]);
            });

            // final matmul to get the output of the ffn
            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);

            // residual connection
            Parallel.parallelFor(0, nTokens, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
        }

        // final rmsnorm
        Parallel.parallelFor(0, nTokens, t -> {
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });
        
        if(false) {
        	try (Timer timer = Timer.log("Last vector:"+state.x[nTokens-1].size())) {
        	}
        	try (Timer timer = Timer.log("Signature")) {
        		
        	}
        	try (Timer timer = Timer.log("Store Tensor:")) {
        		//ModelRunner.dbClient.storekv(ModelRunner.tensorAlias, ModelRunner.xid, "index" , state.x[nTokens-1]);
        	}
        }
        
        // classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);
        state.idxPrevBlock = nTokens - 1;

        return state.logits;
    }
    
    /**
     * LLM generation entry point, ingest prompt tokens and generates new tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop token is found.
     * The returned tokens only include generated/inferred tokens.
     *
     * @param model            model to run inference (including weights, configuration, tokenizer ...)
     * @param state            state of the model e.g. key/value caches ... this is mutated by this call
     * @param startPosition    start prompt ingestion + inference at this position in the context e.g. useful if state was kept across calls (chained generation). 0 implies run with no previous context.
     * @param promptTokens     prompt tokens to ingest, all the prompt tokens will be ingested, given there's enough capacity left in the context
     * @param stopTokens       set of tokens that abort generation during inference, stop tokens do not affect prompt ingestion
     * @param maxTokens        maximum number of tokens (can go up to {@link Configuration#contextLength context length}
     *                         if this value is negative or greater than {@link Configuration#contextLength context length}
     * @param sampler          {@link Sampler strategy} used to select tokens
     * @param echo             debugging flag, prints ALL, prompt and inferred tokens, to {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if any e.g. does not include any token from the prompt
     */
    public static List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens, Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
                                               IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position, Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        // log prompt token (different color?)
                        System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    log.info(String.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n", position, promptIndex, promptTokens.size(), Arrays.toString(tokens)));
                }
                // Only compute logits on the very last batch.
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
                forward(model, state, tokens, position, computeLogits);
                position += nTokens - 1; // -1 -> incremented later in the for loop
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
                forward(model, state, new int[]{token}, position, true);
            }
            nextToken = sampler.sampleToken(state.logits);
            if (echo) {
                // log inferred token
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        log.info(String.format("%ncontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                startPosition + promptIndex + generatedTokens.size(), model.configuration().contextLength,
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size()));

        return generatedTokens;
    }
}

interface TokenizerInterface {
	 public Map<String, Integer> getSpecialTokens();
	 public boolean isSpecialToken(int tokenIndex);
	 public String decode(List<Integer> tokens);
	 public List<Integer> encodeAsList(String text);
	 public int getTokenType(int tokenIndex);
	 public Collection<? extends Integer> encode(String text);
}
/**
 * Byte Pair Encoding tokenizer.
 * <p>
 * Based on <a href="https://github.com/karpathy/minbpe">minbpe</a>, algorithmically follows along the
 * <a href="https://github.com/openai/gpt-2/blob/master/src/encoder.py">GPT 2 tokenizer</a>
 */
class Tokenizer implements TokenizerInterface {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<Pair<Integer, Integer>, Integer> merges;
    private final Map<String, Integer> specialTokens;
    private int[] tokenTypes; // qwen2

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return specialTokens.containsValue(tokenIndex);
    }
    @Override
    public int getTokenType(int tokenIndex) {
        return tokenTypes[tokenIndex];
    }
    
    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.merges = new HashMap<>();
        for (Pair<Integer, Integer> pair : merges) {
            int firstIndex = pair.first();
            int secondIndex = pair.second();
            int mergeIndex = vocabulary.getIndex(vocabulary.get(firstIndex) + vocabulary.get(secondIndex)).orElseThrow();
            this.merges.put(pair, mergeIndex);
        }
    }

    public Tokenizer(Vocabulary vocabulary, List<Pair<Integer, Integer>> merges, String regexPattern, Map<String, Integer> specialTokens, int[] tokenTypes) {
    	this(vocabulary, merges, regexPattern, specialTokens);
    	this.tokenTypes = tokenTypes;
    }
    
    private int[] encodeImpl(Collection<? extends Integer> intc) {
    	return intc.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
     * if none_raise, then an error is raised if any special token is encountered in text
     * this is the default tiktoken behavior right now as well
     * any other behavior is either annoying, or a major footgun.
     */
    List<Integer> encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;
        assert getSpecialTokens().keySet().containsAll(special);
        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encodeOrdinary(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        // we can use re.split for this. note that surrounding the pattern with ()
        // makes it into a capturing group, so the special tokens will be included
        String specialPattern = special
                .stream()
                .map(Pattern::quote)
                .collect(Collectors.joining("|", "(", ")"));

        String[] specialChunks = text.split(specialPattern);
        // now all the special characters are separated from the rest of the text
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String part : specialChunks) {
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                ids.add(getSpecialTokens().get(part));
            } else {
                // this is an ordinary sequence, encode it normally
                ids.addAll(encodeOrdinary(part));
            }
        }
        return ids;
    }

    private static List<String> findAll(Pattern pattern, String text) {
        List<String> allMatches = new ArrayList<>();
        Matcher matcher = pattern.matcher(text);
        while (matcher.find()) {
            allMatches.add(matcher.group());
        }
        return allMatches;
    }

    /**
     * Encoding that ignores any special tokens.
     */
    public List<Integer> encodeOrdinary(String text) {
        // split text into chunks of text by categories defined in regex pattern
        List<String> textChunks = findAll(compiledPattern, text);
        // all chunks of text are encoded separately, then results are joined
        List<Integer> ids = new ArrayList<>();
        for (String chunk : textChunks) {
            List<Integer> chunkIds = encodeChunk(chunk);
            ids.addAll(chunkIds);
        }
        return ids;
    }

    private Map<Pair<Integer, Integer>, Integer> getStats(List<Integer> ids) {
        Map<Pair<Integer, Integer>, Integer> map = new HashMap<>();
        for (int i = 0; i + 1 < ids.size(); i++) {
            Pair<Integer, Integer> key = new Pair<>(ids.get(i), ids.get(i + 1));
            map.put(key, map.getOrDefault(key, 0) + 1);
        }
        return map;
    }

    private List<Integer> encodeChunk(String chunk) {
        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        List<Integer> ids = new ArrayList<>();
        for (int b : chunk.toCharArray()) {
            int tokenIndex = this.vocabulary.getIndex(String.valueOf((char) b)).orElseThrow();
            ids.add(tokenIndex);
        }

        while (ids.size() >= 2) {
            // find the pair with the lowest merge index
            Map<Pair<Integer, Integer>, Integer> stats = getStats(ids);
            Pair<Integer, Integer> pair = stats.keySet().stream().min(Comparator.comparingInt(key -> this.merges.getOrDefault(key, Integer.MAX_VALUE))).orElseThrow();
            // subtle: if there are no more merges available, the key will
            // result in an inf for every single pair, and the min will be
            // just the first pair in the list, arbitrarily
            // we can detect this terminating case by a membership check
            if (!this.merges.containsKey(pair)) {
                break; // nothing else can be merged anymore
            }
            // otherwise let's merge the best pair (lowest merge index)
            int idx = this.merges.get(pair);
            ids = merge(ids, pair, idx);
        }
        return ids;
    }

    private static List<Integer> merge(List<Integer> ids, Pair<Integer, Integer> pair, int idx) {
        List<Integer> newids = new ArrayList<>();
        int i = 0;
        while (i < ids.size()) {
            // if not at the very last position AND the pair matches, replace it
            if (ids.get(i).equals(pair.first()) && i < ids.size() - 1 && ids.get(i + 1).equals(pair.second())) {
                newids.add(idx);
                i += 2;
            } else {
                newids.add(ids.get(i));
                i += 1;
            }
        }
        return newids;
    }

    public String decodeImpl(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            sb.append(tokenString);
        }
        return sb.toString();
    }

    /**
     * Returns list of utf-8 byte and a corresponding list of unicode strings.
     * The reversible bpe codes work on unicode strings.
     * This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
     * When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
     * This is a significant percentage of your normal, say, 32K bpe vocab.
     * To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
     * And avoids mapping to whitespace/control characters the bpe code barfs on.
     */
    private static Map<Integer, Integer> bytesToUnicode() {
        List<Integer> bs = new ArrayList<>();
        IntStream.rangeClosed('!', '~').forEach(bs::add);
        IntStream.rangeClosed('¡', '¬').forEach(bs::add);
        IntStream.rangeClosed('®', 'ÿ').forEach(bs::add);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n += 1;
            }
        }

        // return dict(zip(bs, cs))
        return IntStream.range(0, bs.size())
                .boxed()
                .collect(Collectors.toMap(bs::get, cs::get));
    }

    static final Map<Integer, Integer> BYTE_ENCODER = bytesToUnicode();
    static final Map<Integer, Integer> BYTE_DECODER = BYTE_ENCODER.entrySet()
            .stream()
            .collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey));

    public Collection<? extends Integer> encode(String text) {
        StringBuilder sb = new StringBuilder();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        for (byte b : bytes) {
        	sb.appendCodePoint(BYTE_ENCODER.get(Byte.toUnsignedInt(b)));
        }
        return encode(sb.toString(), Set.of());
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }
    @Override
    public List<Integer> encodeAsList(String text) {
        return Arrays.stream(encodeImpl(encode(text))).boxed().toList();
    }
    @Override
    public String decode(List<Integer> tokens) {
        String decoded = decodeImpl(tokens);
        int[] decodedBytesAsInts = decoded.codePoints().map(BYTE_DECODER::get).toArray();
        byte[] rawBytes = new byte[decodedBytesAsInts.length];
        for (int i = 0; i < decoded.length(); i++) {
            rawBytes[i] = (byte) decodedBytesAsInts[i];
        }
        return new String(rawBytes, StandardCharsets.UTF_8);
    }
}

/**
 * Wherein Llama models metadata.get("tokenizer.ggml.model") = gpt2
 * and Mistral uses metadata.get("tokenizer.ggml.model") = llama.
 */
class MistralTokenizer implements TokenizerInterface {
    private final Pattern compiledPattern;
    private final Vocabulary vocabulary;
    private final Map<String, Integer> specialTokens;
    private final int[] tokenType;
    private final int byte0;

    public String regexPattern() {
        if (compiledPattern == null) {
            return null;
        }
        return compiledPattern.pattern();
    }
    @Override
    public Map<String, Integer> getSpecialTokens() {
        return specialTokens;
    }
    @Override
    public boolean isSpecialToken(int tokenIndex) {
        return getTokenType(tokenIndex) != 1;
    }
    @Override
    public int getTokenType(int tokenIndex) {
        return tokenType[tokenIndex];
    }

    public MistralTokenizer(Vocabulary vocabulary, String regexPattern, Map<String, Integer> specialTokens, int[] tokenType) {
        this.vocabulary = vocabulary;
        this.compiledPattern = regexPattern != null ? Pattern.compile(regexPattern) : null;
        this.specialTokens = new HashMap<>(specialTokens);
        this.tokenType = tokenType;
        this.byte0 = vocabulary.getIndex("<0x00>").orElseThrow();
    }

    public List<Integer> encode(String text) {
        return encodeImpl(text.replace(' ', '▁'));
    }

    private List<Integer> encodeImpl(String text) {
        List<Integer> tokens = new ArrayList<>();
        // first encode every individual codepoint in the input string
        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
            cpi = text.codePointAt(i);

            String singleCodepoint = Character.toString(cpi);
            int id = vocabulary.getIndex(singleCodepoint).orElse(-1);

            if (id != -1) {
                // we found this codepoint in vocab, add it as a token
                tokens.add(id);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +byte0 here to skip all the control and special tokens e.g. <unk>, <s>, </s>
                // so the individual bytes only start at token <0x00>
                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
                    tokens.add(Byte.toUnsignedInt(b) + byte0);
                }
            }
        }
        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (int i = 0; i < tokens.size() - 1; ++i) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                String str_buffer = vocabulary.get(tokens.get(i)) + vocabulary.get(tokens.get(i + 1));
                int id = vocabulary.getIndex(str_buffer).orElse(-1);
                if (id != -1 && vocabulary.getScore(id) > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = vocabulary.getScore(id);
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) {
                break; // we couldn't find any more pairs to merge, so we're done
            }

            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens.set(best_idx, best_id);
            tokens.remove(best_idx + 1);
        }

        return tokens;
    }
    @Override
    public String decode(List<Integer> tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            String tokenString = vocabulary.get(token);
            if (isSpecialToken(token)) {
                // some tokens designate raw bytes e.g. '<0x10>'
                String prefix = "<0x";
                String suffix = ">";
                if (tokenString.length() == 6 && tokenString.startsWith(prefix) && tokenString.endsWith(suffix)) {
                    String code = tokenString.substring(prefix.length(), tokenString.length() - suffix.length());
                    int cp = Integer.parseInt(code, 16);
                    tokenString = Character.toString(cp);
                }
            } else {
                tokenString = tokenString.replace('▁', ' ');

            }
            sb.append(tokenString);
        }
        return sb.toString();
    }

    public static String replaceControlCharacters(int[] codePoints) {
        // we don't want to print control characters
        // which distort the output (e.g. \n or much worse)
        // https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
        // http://www.unicode.org/reports/tr44/#GC_Values_Table\
        StringBuilder chars = new StringBuilder();
        for (int cp : codePoints) {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4)); // escape
            } else {
                chars.appendCodePoint(cp); // this character is ok
            }
        }
        return chars.toString();
    }

    public static String replaceControlCharacters(String str) {
        return replaceControlCharacters(str.codePoints().toArray());
    }

    public List<Integer> encodeAsList(String text) {
        return encode(text);
    }
}

final class Parallel {
    public static void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public static void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        if (startInclusive == 0 && endExclusive == 1) {
            action.accept(0);
            return;
        }
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }
}

record Pair<First, Second>(First first, Second second) {
}

record GGMLTensorEntry(MemorySegment mappedFile, String name, GGMLType ggmlType, int[] shape,
                       MemorySegment memorySegment) {
}

enum GGMLType {
    F32(Float.BYTES),
    F16(GGMLType.FLOAT16_BYTES),
    Q4_0(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_1(2 * GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    UNSUPPORTED_Q4_2(Integer.MAX_VALUE), // support has been removed
    UNSUPPORTED_Q4_3(Integer.MAX_VALUE), // support has been removed
    Q5_0(Integer.MAX_VALUE),
    Q5_1(Integer.MAX_VALUE),
    Q8_0(GGMLType.FLOAT16_BYTES + 32 * Byte.BYTES, 32),
    Q8_1(32 * Byte.BYTES + 2 * Float.BYTES, 32),
    // k-quantizations
    Q2_K(Integer.MAX_VALUE),
    Q3_K(Integer.MAX_VALUE),
    Q4_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q5_K(2 * GGMLType.FLOAT16_BYTES + ((GGMLType.QK_K / 16) / 8 * 6) + GGMLType.QK_K / 8 + GGMLType.QK_K / 2, GGMLType.QK_K),
    Q6_K(GGMLType.QK_K / 2 + GGMLType.QK_K / 4 + GGMLType.QK_K / 16 + GGMLType.FLOAT16_BYTES, GGMLType.QK_K),
    Q8_K(Integer.MAX_VALUE),

    IQ2_XXS(Integer.MAX_VALUE),
    IQ2_XS(Integer.MAX_VALUE),
    IQ3_XXS(Integer.MAX_VALUE),
    IQ1_S(Integer.MAX_VALUE),
    IQ4_NL(Integer.MAX_VALUE),
    IQ3_S(Integer.MAX_VALUE),
    IQ2_S(Integer.MAX_VALUE),
    IQ4_XS(Integer.MAX_VALUE),

    I8(Byte.BYTES),
    I16(Short.BYTES),
    I32(Integer.BYTES),
    I64(Long.BYTES),
    F64(Double.BYTES),
    IQ1_M(Integer.MAX_VALUE),
    BF16(GGMLType.BFLOAT16_BYTES),
    Q4_0_4_4(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_4_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    Q4_0_8_8(GGMLType.FLOAT16_BYTES + 16 * Byte.BYTES, 32),
    TQ1_0(Integer.MAX_VALUE),
    TQ2_0(Integer.MAX_VALUE);

    public static final int BFLOAT16_BYTES = 2;
    public static final int FLOAT16_BYTES = 2;

    private static final GGMLType[] VALUES = values();

    private final int typeSize;

    private final int blockSize;

    public int getTypeSize() {
        return typeSize;
    }

    public int getBlockSize() {
        return blockSize;
    }

    public static GGMLType fromId(int id) {
        return VALUES[id];
    }

    GGMLType(int typeSize) {
        this(typeSize, 1);
    }

    public long byteSizeFor(int numberOfElements) {
        long t = numberOfElements * (long) getTypeSize();
        assert t % getBlockSize() == 0;
        return Math.toIntExact(t / getBlockSize());
    }

    public static final int QK_K = 256; // or 64?

    GGMLType(int typeSize, int blockSize) {
        assert blockSize > 0;
        assert typeSize > 0;
        assert isPowerOf2(blockSize);
        this.typeSize = typeSize;
        this.blockSize = blockSize;
    }

    private static boolean isPowerOf2(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
}

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be backed by memory
 * e.g. can represent a sequence of quantized floats.
 */
abstract class FloatTensor implements Externalizable, Comparable {
    static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    static short readShort(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_SHORT, offset);
        //return UNSAFE.getShort(memorySegment.address() + offset);
    }
    
    static int readInt(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_INT, offset);
        //return UNSAFE.getShort(memorySegment.address() + offset);
    }
    
    static float readFloat(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_FLOAT, offset);
        //return UNSAFE.getShort(memorySegment.address() + offset);
    }
    
    static byte readByte(MemorySegment memorySegment, long offset) {
        return memorySegment.get(ValueLayout.JAVA_BYTE, offset);
        //return UNSAFE.getByte(memorySegment.address() + offset);
    }

    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    abstract int size();

    abstract float getFloat(int index);

    abstract void setFloat(int index, float value);

    abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

    abstract GGMLType type();

    public static int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

    static float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }

    void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1)); 
        });
    }

    @FunctionalInterface
    interface AggregateFunction {
        float apply(float acc, float value);
    }

    float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    int argmax() {
        return argmax(0, size());
    }

    @FunctionalInterface
    interface MapFunction {
        float apply(float value);
    }

    @FunctionalInterface
    interface MapWithIndexFunction {
        float apply(float value, int index);
    }

    FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            setFloat(i, mapFunction.apply(getFloat(i)));
        }
        return this;
    }

    FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, size(), mapFunction);
    }

    FloatTensor mapWithIndexInPlace(int thisOffset, int size, FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
        	//System.out.println("setFloat:"+i+" of size:"+size);
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }

    FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, size());
    }

    FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size, (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    FloatTensor multiplyInPlace(FloatTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }

    FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    FloatTensor softmaxInPlace(int thisOffset, int size) {
        // find max value (for numerical stability)
        float maxVal = max(thisOffset, size);
        // exp and sum
        mapInPlace(thisOffset, size, f -> (float) Math.exp(f - maxVal));
        float sum = sum(thisOffset, size);
        // normalize
        return divideInPlace(thisOffset, size, sum);
    }

    FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        // this[thatOffset ... thatOffset + size) = a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
    
    static float cosineSimilarity(FloatTensor a, FloatTensor b) {
    	float dotProduct = a.dot(0, b, 0, a.size());
    	DoubleAdder aNormAdder = new DoubleAdder();
    	DoubleAdder bNormAdder = new DoubleAdder();
    	Parallel.parallelFor(0, a.size(), t -> {
    	    aNormAdder.add(a.getFloat(t) * a.getFloat(t));
    	    bNormAdder.add(b.getFloat(t) * b.getFloat(t));
    	});
    	float aNorm = (float) Math.sqrt(aNormAdder.sum());
    	float bNorm = (float) Math.sqrt(bNormAdder.sum());
    	return (dotProduct / (aNorm * bNorm));
    }
    
    public String toString() {
    	StringBuilder sb = new StringBuilder("[");
    	for(int i = 0; i < size(); i++) {
    		sb.append(getFloat(i));
    		if(i == (size()-1)) 
    			sb.append("]");
    		else
    			sb.append(",");
    	}
    	return sb.toString();
    }
}


/**
 * {@link FloatTensor} quantized in the {@link GGMLType#Q4_0} format.
 * <p>
 * This tensor implementation is not compatible with {@link FloatTensor}, but
 * {@link #dot(int, FloatTensor, int, int)} has a vectorized implementation that is used when
 * the second argument implements {@link FloatTensor}.
 */
final class Q4_0FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

	int size;
    transient MemorySegment memorySegment;

    public Q4_0FloatTensor() {}
    
    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        byte quant;
        int modIndex = index % GGMLType.Q4_0.getBlockSize();
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q4_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q4_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getTypeSize();
        int upperBound = size / GGMLType.Q4_0.getBlockSize() * GGMLType.Q4_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q4_0.getBlockSize(), blockOffset += GGMLType.Q4_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    val = sum0.add(sum2).fma(wScale, val);
                }
                case 256 -> {
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(loBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 0));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(hiBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 0) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length()).mul(tmp.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
		out.flush();
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
	    size = in.readInt();
	    long bs = in.readLong();
	    byte[] bytes = new byte[(int) bs];
	    in.readFully(bytes); // Properly consumes block data
	    memorySegment = MemoryUtils.allocateSegmentFromBytes(bytes, Arena.ofAuto());
	}

	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((Q4_0FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((Q4_0FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

final class Q8_0FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

	int size;
    transient MemorySegment memorySegment;

    public Q8_0FloatTensor() {}
    
    public Q8_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q8_0.getBlockSize();
        int withinBlockIndex = index % GGMLType.Q8_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q8_0.getTypeSize();
        byte quant = readByte(memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + withinBlockIndex);
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return quant * scale;
    }

    public static final ValueLayout.OfShort JAVA_SHORT_LE = ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + startIndex to type().getBlockSize().
        assert Integer.bitCount(GGMLType.Q8_0.getBlockSize()) == 1 : "power of 2";
        int alignmentBound = Math.min(size, -thisOffset & (GGMLType.Q8_0.getBlockSize() - 1));
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getBlockSize() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getTypeSize();
        int upperBound = size / GGMLType.Q8_0.getBlockSize() * GGMLType.Q8_0.getBlockSize();
        for (; j < upperBound; j += GGMLType.Q8_0.getBlockSize(), blockOffset += GGMLType.Q8_0.getTypeSize()) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
                    var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment, blockOffset + GGMLType.FLOAT16_BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 0 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 0));
                        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 1 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1));
                        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2));
                        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + i * 16 + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
		out.flush();
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
	    size = in.readInt();
	    long bs = in.readLong();
	    byte[] bytes = new byte[(int) bs];
	    in.readFully(bytes); // Properly consumes block data
	    memorySegment = MemoryUtils.allocateSegmentFromBytes(bytes, Arena.ofAuto());
	}


	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((Q8_0FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((Q8_0FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

final class BF16FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

    int size;
    transient MemorySegment memorySegment;
    
    public BF16FloatTensor() {}
    
    public BF16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.BF16;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return bfloat16ToFloat(readShort(memorySegment, index * GGMLType.BFLOAT16_BYTES));
    }

    private float bfloat16ToFloat(short bfloat16) {
        return Float.intBitsToFloat(bfloat16 << 16);
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(BF16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bfloat16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * (long) GGMLType.BFLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);
            // BFloat16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───····───[7]─┬─[6]────····────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (7 bits)  │ BFloat16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            FloatVector thizVector = bfloat16
                    .castShape(I_SPECIES, 0) // (int) vi
                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                    .reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
		out.flush();
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
	    size = in.readInt();
	    long bs = in.readLong();
	    byte[] bytes = new byte[(int) bs];
	    in.readFully(bytes); // Properly consumes block data
	    memorySegment = MemoryUtils.allocateSegmentFromBytes(bytes, Arena.ofAuto());
	}


	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((BF16FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((BF16FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

final class F16FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

    int size;
    transient MemorySegment memorySegment;

    public F16FloatTensor() {}
    
    public F16FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.F16;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return Float.float16ToFloat(readShort(memorySegment, index * GGMLType.FLOAT16_BYTES));
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private static float vectorDot(F16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment, (thisOffset + i) * (long) GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);

            var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts(); // (int) bits16
            // Does not support infinities nor NaNs, preserves sign, emulate DAZ (denormals-are-zero).
            // Expects well-formed float16 values only (e.g. model weights).
            // Fast Float16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───···───[10]─┬─[9]────····────[0]─┐
            // │ Sign │ Exponent (5 bits) │ Mantissa (10 bits) │ Float16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            //    │             │                    │
            //    ▼             ▼                    ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            //
            // Shifts and adjustments:
            // - Sign:       float16[15] -> float32[31] (shift 16 bits up)
            // - Exponent:   float16[10-14] -> float32[23-30] (+ bias adjustment)
            // - Mantissa:   float16[0-9] -> float32[13-22] (shift 13 bits up)
            //
            // exp = bits32 & 0x7C00
            // zeroExponentMask = exp == 0 ? 0 : ~0
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31); // = (-exp) >> 31
            bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16) // sign
                    .or(
                            // exponent and mantissa combined
                            bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13)
                                    .and(zeroExponentMask) // -0, +0 and DAZ (denormals-are-zero)

                    );

            FloatVector thizVector = bits32.reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
		out.flush();
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
	    size = in.readInt();
	    long bs = in.readLong();
	    byte[] bytes = new byte[(int) bs];
	    in.readFully(bytes); // Properly consumes block data
	    memorySegment = MemoryUtils.allocateSegmentFromBytes(bytes, Arena.ofAuto());
	}


	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((F16FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((F16FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

final class F32FloatTensor extends FloatTensor implements Externalizable, Comparable {
	private static final long serialVersionUID = -1L;

	int size;
	transient MemorySegment memorySegment;
	
	public F32FloatTensor() {}
	
	public F32FloatTensor(int size, MemorySegment memorySegment) {
		this.size = size;
		this.memorySegment = memorySegment;
	}

	@Override
	int size() {
		return size;
	}

	@Override
	float getFloat(int index) {
		assert 0 <= index && index < size;
		return readFloat(memorySegment, index * 4);
	}

	@Override
	void setFloat(int index, float value) {
		throw new UnsupportedOperationException("setFloat");	
	}

	@Override
	FloatVector getFloatVector(VectorSpecies<Float> species, int offset) {
		 throw new UnsupportedOperationException("getFloatVector");
	}

	@Override
	GGMLType type() {
		return GGMLType.F32;
	}

	@Override
	public void writeExternal(ObjectOutput out) throws IOException {
		out.writeInt(size);
		out.writeLong(memorySegment.byteSize());
		out.write(memorySegment.toArray(ValueLayout.JAVA_BYTE));
		out.flush();
	}

	@Override
	public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
	    size = in.readInt();
	    long bs = in.readLong();
	    byte[] bytes = new byte[(int) bs];
	    in.readFully(bytes); // Properly consumes block data
	    memorySegment = MemoryUtils.allocateSegmentFromBytes(bytes, Arena.ofAuto());
	}


	@Override
	public int compareTo(Object o) {
		for(int i = 0; i < memorySegment.byteSize(); i++) {
			byte b;
			if(i >= ((F32FloatTensor)o).memorySegment.byteSize())
				return 1;
			else
				b = ((F32FloatTensor)o).memorySegment.get(ValueLayout.JAVA_BYTE, i);
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) > b)
				return 1;
			if(memorySegment.get(ValueLayout.JAVA_BYTE,i) < b)
				return -1;
		}
		return 0;
	}
}

final class ArrayFloatTensor extends FloatTensor implements Externalizable, Comparable {

    float[] values;
    
    public ArrayFloatTensor() {}
    
    ArrayFloatTensor(float[] values) {
        this.values = values;
    }

    public static FloatTensor allocate(int... dims) {
        int numberOfElements = FloatTensor.numberOfElements(dims);
        return new ArrayFloatTensor(new float[numberOfElements]);
    }

    @Override
    public int size() {
        return values.length;
    }

    @Override
    public float getFloat(int index) {
        return values[index];
    }

    @Override
    public void setFloat(int index, float value) {
        values[index] = value;
    }

    @Override
    public GGMLType type() {
        return GGMLType.F32;
    }

    @Override
    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        Arrays.fill(values, thisOffset, thisOffset + size, value);
        return this;
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        if (!USE_VECTOR_API) {
            throw new UnsupportedOperationException();
        }
        return FloatVector.fromArray(species, values, index);
    }
    
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        out.writeInt(values.length);
        ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES);
        for (float v : values) buffer.putFloat(v);
        out.write(buffer.array());
        out.flush();
    }

    @Override
    public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
        int vsize = in.readInt();
        byte[] bytes = new byte[vsize * Float.BYTES];
        in.readFully(bytes);
        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        values = new float[vsize];
        for (int i = 0; i < vsize; i++) values[i] = buffer.getFloat();
    }

	@Override
	public int compareTo(Object o) {
		return Arrays.compare(values,((ArrayFloatTensor)o).values);
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

record Vocabulary(String[] tokens, float[] scores, Map<String, Integer> tokenToIndex) {
    public Vocabulary(String[] vocabulary, float[] scores) {
        this(vocabulary, scores,
                IntStream.range(0, vocabulary.length)
                        .boxed()
                        .collect(Collectors.toMap(i -> vocabulary[i], i -> i))
        );
    }

    public String get(int tokenIndex) {
        return tokens[tokenIndex];
    }

    public OptionalInt getIndex(String token) {
        Integer value = tokenToIndex.get(token);
        return value != null ? OptionalInt.of(value) : OptionalInt.empty();
    }

    public int size() {
        return tokens.length;
    }
    /**
     * Added from Mistral Vocabulary - Groff
     * @param tokenIndex
     * @return
     */
    public float getScore(int tokenIndex) {
        return scores[tokenIndex];
    }
    
    public boolean scoresNull() {
    	return scores == null;
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

interface ChatFormatInterface {
	 public TokenizerInterface getTokenizer();
	 public Set<Integer> getStopTokens();
	 public List<Integer> encodeHeader(ChatFormat.Message message);
	 public List<Integer> encodeMessage(ChatFormat.Message message);
	 public List<Integer> encodeDialogPrompt(boolean appendAssistantTurn, List<ChatFormat.Message> dialog);
	 public int getBeginOfText();
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

    public record Message(ChatFormat.Role role, String content) {
    }

    public record Role(String name) {
        public static ChatFormat.Role SYSTEM = new ChatFormat.Role("system");
        public static ChatFormat.Role USER = new ChatFormat.Role("user");
        public static ChatFormat.Role ASSISTANT = new ChatFormat.Role("assistant");

        @Override
        public String toString() {
            return name;
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
                        ModelLoader.loadModel(fileChannel, gguf, ModelRunner.Options.DEFAULT_MAX_TOKENS, false),
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
            Llama.Weights weights = ModelLoader.loadGPT2Weights(tensorEntries, baseModel.configuration());
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
	 * Initialize a new hash table, uses COsing hash family.
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
	 * Query the hash table in parallel for a vector. It calculates the hash for the vector,
	 * and does a lookup in the hash table. If no candidates are found, an empty
	 * list is returned, otherwise, the list of candidates is returned.
	 * 
	 * @param query The query vector.
	 * @return Does a lookup in the table for a query using its hash. If no
	 *         candidates are found, an empty list is returned, otherwise, the
	 *         list of candidates is returned as List<Result> where Result contains timetamp, vector
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public List<Result> queryParallel(List<Integer> query) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException, InterruptedException, ExecutionException {
		List<Result> res = new ArrayList<Result>();
		ArrayList<Object> iq = new ArrayList<Object>();
		for(int i = 0; i < hashTable.size(); i++) {
			Integer combinedHash = hash(hashTable.get(i), normalize(query));
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
	 * Normalizes integer tokens into a zero-centered, unit-length float tensor
	 * for cosine similarity use with Gaussian random projection.
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
	 * Add a vector to the index. Create a UUID and store the vector in a K/V datastore, use the UUID to
	 * reference the vector in the Relatrix relationship.
	 * @param vector the list of tokens
	 * @throws DuplicateKeyException 
	 * @throws IOException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalAccessException 
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 */
	public void add(List<Integer> vector) throws IllegalAccessException, ClassNotFoundException, IOException, InterruptedException, ExecutionException {
		FloatTensor fvec = normalize(vector);
		NoIndex noIndex = NoIndex.create(vector);
		for(int i = 0; i < hashTable.size(); i++) {
			Integer combinedHash = hash(hashTable.get(i), fvec);
			CompletableFuture<Relation> res = dbClient.store(xid, combinedHash, System.currentTimeMillis(), noIndex);
			res.get();
		}
		dbClient.commit(xid);
	}
	/**
	 * Find the nearest candidates using cosine similarity
	 * @param message trget message in tokens
	 * @return the list of tokens representing closest candidates
	 * @throws ExecutionException 
	 * @throws InterruptedException 
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public List<Integer> findNearest(List<Integer> message) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException, InterruptedException, ExecutionException {
		List<Result> nearest = null;
		ArrayList<Integer> results = new ArrayList<Integer>();
		results.addAll(message);
		FloatTensor fmessage = normalize(message);
		nearest = queryParallel(message);
		log.info("Retrieved "+nearest.size()+" entries from LSH index query.");
		if(nearest.isEmpty())
			return results;
		double[] cossim = new double[nearest.size()];
		int cnt = 0;
		TreeMap<Double, Integer> tm = new TreeMap<Double, Integer>();
		for(int i = 0; i  < nearest.size(); i++) {
			Result result = nearest.get(i);
			// timestamp at Result.get(0)
			NoIndex noIndex = (NoIndex) result.get(1);
			List<Integer> restensor = (List<Integer>)noIndex.getInstance();
			FloatTensor cantensor = normalize(restensor);
			double cosDist = FloatTensor.cosineSimilarity(fmessage, cantensor);
			cossim[i] = cosDist;
			tm.put(cosDist, i);
		}
		NavigableMap<Double, Integer> dmap = tm.descendingMap();
		Iterator<Integer> it = dmap.values().iterator();
		while(it.hasNext() && results.size() < (maxTokens - (((float)maxTokens) * .3))) {
			int i = it.next();
			Result result = nearest.get(i);
			// timestamp at Result.get(0)
			NoIndex noIndex = (NoIndex) result.get(1);
			List<Integer> restensor = (List<Integer>)noIndex.getInstance();
			results.addAll(restensor);
		}
		log.info(cnt+" results above threshold. Similarities:"+Arrays.toString(cossim));
		return results;
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


