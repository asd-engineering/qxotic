package com.qxotic.jinfer.models.qwen3;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * {@link ModelProvider} service for {@code general.architecture} "qwen3": supports generative
 * models (such as Bonsai-8B and other Qwen3-backbone LMs) as well as the RETRIEVAL family
 * (Qwen3-Embedding and Qwen3-Reranker).
 */
public final class Qwen3Provider implements ModelProvider {
    static final LoadedModel.SamplingDefaults SAMPLING_DEFAULTS =
            new LoadedModel.SamplingDefaults(1.0f, 0.95f, 20, null);

    @Override
    public Set<String> architectures() {
        return Set.of("qwen3");
    }

    @Override
    public LoadedModel<?> loadLanguage(
            FileChannel fileChannel,
            GGUF gguf,
            Path path,
            Arena arena,
            Map<String, Path> companions,
            Tokenizer tokenizer)
            throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena, tokenizer);
        Tokenizer tok = model.tokenizer();
        int eos = gguf.getValueOrDefault(int.class, "tokenizer.ggml.eos_token_id", -1);
        return new LoadedModel<>(
                model,
                tok,
                gguf.getStringOrDefault("tokenizer.chat_template", ""),
                SpecialTokens.stops(tok, eos, "<|im_end|>", "<|endoftext|>"),
                Models.modelSeed(fileChannel),
                Optional.empty(),
                SAMPLING_DEFAULTS);
    }

    @Override
    public LoadedEmbedder<?> loadEmbedder(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena, tokenizer);
        // last-token pooling wants a trailing EOS on every sequence (the llama.cpp convention)
        int eos =
                SpecialTokens.find(model.tokenizer(), "<|endoftext|>")
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "qwen3 vocab has no <|endoftext|>"));
        return new LoadedEmbedder<>(
                model,
                model.tokenizer(),
                IntSequence.empty(),
                IntSequence.of(eos),
                model.configuration().embeddingLength(),
                32, // model card: Matryoshka output supports every width from 32 to native
                path.getFileName().toString(),
                // the card's instructed-query framing, default retrieval task, verbatim
                // (get_detailed_instruct: 'Instruct: {task}\nQuery:{query}' - no space after
                // Query:); documents are embedded bare per the same card
                "Instruct: Given a web search query, retrieve relevant passages that answer the"
                        + " query\nQuery:",
                "");
    }

    @Override
    public LoadedReranker<?> loadReranker(
            FileChannel fileChannel, GGUF gguf, Path path, Arena arena, Tokenizer tokenizer)
            throws IOException {
        Qwen3 model = Qwen3.loadModel(fileChannel, gguf, arena, tokenizer);
        return new LoadedReranker<>(model, new Qwen3Reranker(model), path.getFileName().toString());
    }
}
