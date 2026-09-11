# 🕳️ moe-hot-layers

**Squeezing a 30B-parameter model onto a 4GB GPU by exploiting the fact that it never actually uses all 30B at once.**

This started from one annoying observation: a Mixture-of-Experts model with "30B" in its name doesn't compute anywhere near 30B parameters per token. Qwen3-30B-A3B activates 8 of its 128 experts per layer, per token; the "A3B" literally stands for "Active 3B." Most of the model is sitting there, unused, for any given token. So why does running it still feel like you need enough VRAM for the whole 30B?

Because nothing in the stock tooling asks *which* experts actually get used. It just fits as much of the model onto the GPU as physically fits, first-come-first-served, with zero awareness that some experts get picked constantly and others almost never. This repo is a small, deliberately unglamorous fix for that: **trace which experts actually fire, and put those specific ones on the GPU, not just "the first N tensors that fit."**

Tested on the kind of hardware that has no business running a 30B model at all: a **GTX 1050 Ti (4GB VRAM)** + **16GB RAM**. It works. Numbers below.

## The idea, in one picture

```
Stock llama.cpp -ngl / auto-fit:
  [ GPU: layers 0,1,2,3,4,5... first-N-that-fit ] [ CPU/disk: everything else ]

This repo:
  [ GPU: the 3-8 layers whose experts get reused constantly ] [ CPU/disk: cold layers ]

```

MoE routing is data-dependent (which expert gets picked depends on the token, not a fixed schedule) but it's *not* random. Run a real trace and you'll find some layers have highly repetitive expert selection (the same handful of experts keep getting picked turn after turn) while others are basically uniform noise across the full expert pool. The repetitive ones are the ones worth pinning to GPU. The noisy ones will thrash the GPU cache no matter where you put them, so don't waste VRAM on them.

## How it works, precisely

Two things this repo actually needs to get right, spelled out instead of hand-waved:

**What "hot" means, exactly.** `locality_sorted.tsv` ranks each layer by an `avg_overlap` score: for that layer, look at the set of experts selected on one decode step and the set selected on the *next* decode step, count how many experts are in both sets, and average that count across the whole trace. Qwen3-30B-A3B picks 8 experts per layer per token, so the score ranges from 0 (completely different experts every step, pure noise) to 8 (the exact same 8 experts get reused every single step). A layer scoring 6.17 means, on average, 6 of the next step's 8 experts were *already* used the step before. That's the number that actually predicts whether caching a layer's experts pays off: high overlap means whatever you pin stays relevant step after step, low overlap means you're constantly evicting and refetching regardless of what you pinned.

**Why pinning changes speed at all.** `llama.cpp` loads GGUF files via `mmap`, so weight tensors aren't unconditionally copied into RAM ahead of time. Untouched pages get faulted in from the OS page cache (or disk, on a true cold read) the moment they're actually read for a matmul. Every layer evaluation that touches a "cold" expert means paying that fault cost again, on every single decode step, for the whole session. `llama.cpp`'s `-ot` (override-tensor) flag forces specific tensors to live permanently on a named backend (`CUDA0`, in this repo's case) instead of being subject to that mmap/page-cache lifecycle. Pin a layer whose experts get reused constantly, and you pay the transfer cost roughly once instead of every step. Pin a layer whose experts barely repeat, and you've just spent VRAM on tensors that get swapped out just as often as if you'd left them alone; that's the entire reason placement has to be measured per layer instead of applied uniformly.

## What's actually in here

- **`pick_hot_layers.sh`**: reads `layer_sizes.tsv` (how big each layer's expert tensors are) and `locality_sorted.tsv` (how "hot"/reusable each layer's expert selection is, ranked), greedily fills your free VRAM budget with the hottest layers that fit, and emits a ready-to-use `llama.cpp` `-ot` (override-tensor) string.
- **`get_ot.sh`**: thin wrapper so a `.bat` launcher can call into the bash script with one plain argument instead of nesting quoted logic (this alone cost more debugging time than it deserved).
- **`run_qwen.sh`**: launches `llama-cli` with hot layers auto-placed. One command, no manual `-ot` string editing.
- **`start-chat.sh`**: same idea but starts `llama-server` and opens llama.cpp's own built-in web chat UI. No extra frontend needed.
- **`layer_sizes.tsv` / `locality_sorted.tsv`**: the real trace data from this project's Qwen3-30B-A3B run, included as a working example so you can see the shape of real data and don't have to trace blind on your first try.

The actual "which experts fired, how often, on which layer" tracing was done with a small instrumented harness hooked into `llama.cpp`'s `ggml_backend_sched_eval_callback` (analogous to a PyTorch `forward_hook`, just at the ggml graph level), logging every expert-selection tensor during real generations, then computing a locality score per layer (how much the top-K expert set overlaps turn to turn). That harness isn't published here in its raw form since it's tightly coupled to a specific llama.cpp build; `locality_sorted.tsv` is its output. If there's interest, the tracer itself is a good candidate for a follow-up repo.

## Results (the actual numbers, not vibes)

All on the same box: GTX 1050 Ti, 4GB VRAM (~3.3GB free after desktop overhead), 16GB RAM, Qwen3-30B-A3B, Q4_K_M quant (~17.3GB total, ~15.75GB of that is expert weights).

| Configuration | Generation speed | What it tells you |
|---|---|---|
| CPU-only (no GPU at all) | **1.40 tok/s** | The honest floor with zero acceleration |
| Naive GPU auto-fit (`llama.cpp` default, first-N-that-fit) | **~4.0 tok/s** | GPU is helping, but barely more than you'd expect from *any* 2.7GB of the model being resident, not from *the right* 2.7GB |
| **Hot-layer placement (this repo)** | **5.5 tok/s** (benchmarked, `llama-bench`, r=10) | Same VRAM budget, deliberately chosen layers |
| **Hot-layer placement, live serving** | **7.2 tok/s** | Once genuinely warm (`llama-server`, real chat session) |

The interesting part isn't the raw speedup, it's *why* it happens. Going from CPU-only to naive GPU offload only bought +33% generation speed despite handing real VRAM to real compute. If this were purely compute-bound, that number should have moved a lot more, since only ~3B parameters are active per token in the first place. It barely moved because *which* 2.7GB landed on the GPU was arbitrary, mostly cold layers that get evicted and refetched constantly. Once the *right* layers (measured, not guessed) are pinned, the same VRAM budget does meaningfully more work: **~30-40% faster than naive placement, using no additional hardware.**

## Prior art (because credit matters and reinventing wheels badly is worse than not reinventing them)

This project didn't happen in a vacuum, and two independent projects validate pieces of this same idea, worth knowing about if you're going deeper on this:

- **[tonbistudio/moe-ssd-streaming-windows](https://github.com/tonbistudio/moe-ssd-streaming-windows)**: uses the same core mechanism (`-ot` plus relying on the OS page cache as an LRU) and reports 2.5-4.3 tok/s on this exact model (Qwen3-30B-A3B) on a 12GB-VRAM card. Independent confirmation that the approach works, on different hardware.
- **[emzanautoslide-web/llama.cpp.offload](https://github.com/emzanautoslide-web/llama.cpp.offload)**: a llama.cpp fork explicitly focused on "advanced MoE offloading." Worth a look if you want a more sophisticated, adaptive-cache-style approach than the one-shot static placement this repo uses.

This repo is intentionally the simpler, "measure once, place statically" version of the idea. It's less powerful than a live adaptive cache, but it's about 60 lines of bash and doesn't require patching or rebuilding llama.cpp, so you get most of the win for a fraction of the engineering.

## Usage

Prerequisites: a CUDA (or other GPU-backend) build of [llama.cpp](https://github.com/ggml-org/llama.cpp), and a Qwen3-30B-A3B GGUF (any Q4_K_M-class quant of the base model or a compatible finetune; the *architecture* is what this technique targets, not any specific checkpoint).

```bash
git clone <this-repo>
cd moe-hot-layers

# One-shot generation:
MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf ./run_qwen.sh -p "your prompt here"

# Or a full chat server + browser UI:
MODEL=/path/to/Qwen3-30B-A3B-Q4_K_M.gguf ./start-chat.sh
```

If `llama-cli` / `llama-server` aren't on your `PATH`, point at them explicitly:

```bash
LLAMA_CLI=/path/to/llama-cli MODEL=/path/to/model.gguf ./run_qwen.sh
```

`pick_hot_layers.sh` reads your free VRAM via `nvidia-smi` at launch time, so it adapts automatically if you close other GPU-using programs and free up headroom. No need to re-trace, just re-run.

### Using your own model / hardware

The `layer_sizes.tsv` and `locality_sorted.tsv` shipped here are specific to Qwen3-30B-A3B's 48-layer architecture. They'll work as-is *only* for that model family. For a different MoE model, you'd need to re-trace: log expert selections during a handful of real generations, compute per-layer locality (how much the selected-expert set overlaps between consecutive tokens/turns), and regenerate both TSVs in the same `layer<TAB>value` format. The placement logic in `pick_hot_layers.sh` itself is architecture-agnostic, it just greedily fills a VRAM budget by locality rank.

## Speculation on getting bigger

The core insight, *trace real usage, don't guess placement*, doesn't care about model size. It gets *more* valuable as models get bigger, for a simple reason: the sparsity ratio in modern MoE releases keeps growing. Qwen3-30B-A3B activates ~10% of its total parameters per token. Some newer, much larger MoE models (235B+, 671B+ class) push that ratio even lower, meaning an even smaller fraction of the model is ever "hot" at once, which is exactly the regime where deliberate placement beats naive placement hardest.

A few concrete directions this obviously extends to:

- **Bigger MoE models, same tiny GPU.** The technique doesn't require the model to fit in VRAM, it requires the *hot* subset to fit. A 235B-class MoE with a small enough active fraction could plausibly still get meaningful speedup on a 4GB card, provided RAM/SSD can hold the rest and page it in fast enough. The math changes (bigger cold-tier penalty per miss), but the mechanism is identical.
- **More VRAM equals strictly better, not just "more room."** Every extra GB doesn't just fit more layers, it fits more of the *already-ranked* hot layers, meaning the marginal GB you add is spent on the next-most-valuable thing, not an arbitrary one. That's the whole point of ranking by locality instead of just size.
- **RAM and SSD tiers, not just GPU/CPU.** This repo only builds the GPU/CPU split. The natural extension is a 3-tier hierarchy: hot layers on GPU, warm layers pinned in RAM (bypassing OS page-cache eviction pressure), cold layers explicitly read from SSD on demand. That's closer to what the more sophisticated prior-art project above does with its predictive cache.
- **Live re-ranking instead of static placement.** Right now, placement is computed once per session from a static trace. A model serving many different conversation styles would benefit from periodically re-checking whether the "hot" set has drifted, rather than trusting a snapshot from one earlier trace run.
- **Does NOT trivially transfer to diffusion/image models.** Briefly, honestly: the same MoE sparsity exists in some image-generation MoE architectures, but per-token routing in an autoregressive text model is a very different access pattern from per-denoising-step routing in a diffusion transformer, where attention (not expert dispatch) usually dominates memory pressure instead. That's a different enough problem that it deserves its own investigation rather than a paragraph here. Consider this a "there be dragons" flag, not a dead end.

If you take this further in any of those directions, this repo would love to hear about it.

## Honest caveats

- The benchmarked numbers above are **warm-cache** results (after the relevant file regions have been touched a few times). A cold first-token-of-a-session read will be slower than steady-state; this reflects realistic *sustained* chat performance, not a best-case synthetic number.
- The included `locality_sorted.tsv` was traced against one specific Qwen3-30B-A3B checkpoint. It transfers reasonably to same-architecture finetunes (same routing behavior, same 48 layers), but hasn't been independently re-validated against every possible finetune. If speed feels off on a different checkpoint, re-tracing is cheap insurance.
- This is a static, one-shot placement, not an adaptive cache. It won't self-correct mid-session if usage patterns shift dramatically from what was traced.

## License

MIT, see [LICENSE](LICENSE).
