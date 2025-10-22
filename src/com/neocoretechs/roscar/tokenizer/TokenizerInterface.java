package com.neocoretechs.roscar.tokenizer;

import java.util.Collection;
import java.util.List;
import java.util.Map;
import com.neocoretechs.roscar.Pair;
public interface TokenizerInterface {
	 public Map<String, Integer> getSpecialTokens();
	 public boolean isSpecialToken(int tokenIndex);
	 public String decode(List<Integer> tokens);
	 public List<Integer> encodeAsList(String text);
	 public int getTokenType(int tokenIndex);
	 public Collection<? extends Integer> encode(String text);
	 public Map<Pair<Integer, Integer>, Integer> getMerges();
}
