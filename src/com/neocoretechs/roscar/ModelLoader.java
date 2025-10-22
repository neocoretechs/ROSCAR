package com.neocoretechs.roscar;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.neocoretechs.roscar.GGUF.GGMLTensorEntry;
import com.neocoretechs.roscar.tokenizer.MistralTokenizer;
import com.neocoretechs.roscar.tokenizer.Tokenizer;
import com.neocoretechs.roscar.tokenizer.TokenizerInterface;

/**
 * Load model, get GGUF metadata, load vocabulary, create tokenizer, create config, if loadWeights - load tensors, load weights
 * create Llama with config, tokenizer, weights
 */
public final class ModelLoader {
	static final Log log = LogFactory.getLog(ModelLoader.class);
	static final String TOKENIZER_GPT2_MODEL = "gpt2"; // ModelRunner uses gpt2!
	static final String TOKENIZER_LLAMA_MODEL = "llama"; // non Llama uses llama!
	public static String model = "gpt2"; // default for Llama models!
	public static String name = null; // Name is based solely on name of model, they all seem to have their own ChatFormat not based on model
	private static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

	private static Vocabulary loadVocabulary(Map<String, Object> metadata) {
		model = (String) metadata.get("tokenizer.ggml.model");
		name = (String) metadata.get("general.name");
		if(name.toLowerCase().contains("llama")) // Meta Llama, etc. etc.
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
		if(ModelRunner.DISPLAY_METADATA) {
			ModelRunner.fileWriter = new FileWriter(ggufPath.toString()+".metadata", false);
			ModelRunner.output = new PrintWriter(ModelRunner.fileWriter);
		}
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
			Configuration config;
			Weights weights = null;
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
					throw new IllegalArgumentException("expected metadata general.name containing mistral or llama but found "+ModelLoader.name);
			}
			return new Llama(config, tokenizer, weights);
		}
	}

	static Configuration createConfig(String arch, Map<String, Object> metadata, Vocabulary vocabulary, int contextLength) {
		return new Configuration(
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
	}


	/**
	 * Called from AOT.tryUsePreloaded and ModelLoader.loadModel
	 * @param tensorEntries
	 * @param config
	 * @return
	 */
	static Weights loadGPT2Weights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
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
		Weights qw = new Weights(
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

	static Weights loadLlamaWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
		Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize, config.ropeTheta);
		float[] ropeFreqsReal = ropeFreqs.first();
		float[] ropeFreqsImag = ropeFreqs.second();

		Weights qw = new Weights(
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
		int allTokens = vocabulary.size();
		int baseTokens = 128000; // assume all tokens after the base ones are special.
		int reservedSpecialTokens = allTokens - baseTokens;
		List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

		assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

		Map<String, Integer> specialTokens =
				IntStream.range(0, specialTokensList.size()).boxed().collect(Collectors.toMap(
						i -> specialTokensList.get(i),
						i -> baseTokens + i)
						);
		String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
		List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
				.map(line ->line.split(" "))
				.map(parts->new Pair<>(vocabulary.getIndex(parts[0]).orElseThrow(),vocabulary.getIndex(parts[1]).orElseThrow())).toList();
		return new Tokenizer(vocabulary, merges, LLAMA_3_PATTERN, specialTokens);
	}

	private static MistralTokenizer createLlamaTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
		int[] tokenTypes = (int[]) metadata.get("tokenizer.ggml.token_type");
		List<Integer> specialTokensList = IntStream.range(0, vocabulary.size()).filter(t -> tokenTypes[t] != 1 && tokenTypes[t] != 6).boxed().toList();
		Map<String, Integer> specialTokens = IntStream.range(0, specialTokensList.size()).boxed().collect(Collectors.toMap(
				t -> vocabulary.get(t),
				t -> t));
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


