package com.neocoretechs.roscar;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class State implements Externalizable {
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
	
	static FloatTensor[] allocate(int numTokens, int... dims) {
		return IntStream.range(0, numTokens)
				.mapToObj(i -> ArrayFloatTensor.allocate(dims))
				.toArray(FloatTensor[]::new);
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
