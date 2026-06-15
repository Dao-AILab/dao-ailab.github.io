---
layout: distill
title: "ReplaySSM: Cache SSM Inputs, Not State"
description:
tags:
giscus_comments: false
date: 2026-06-15
featured: true
bibliography: replayssm.bib
thumbnail: assets/img/2026-06-15-replayssm/headline_approach.png

toc:
  - name: "1. State Space Models (SSMs)"
  - name: "2. Three challenges in SSM decoding"
    subsections:
      - name: "2.1 Memory-bound: all I/O, no compute"
      - name: "2.2 Summarization backfires: a state has no undo"
      - name: "2.3 Loss of parallelism: sequential state dependence"
  - name: "3. The idea: don't store the state, cache the recent inputs"
  - name: "4. One change, three answers"
    subsections:
      - name: "4.1 We cut the memory traffic"
      - name: "4.2 Rollback becomes a buffer operation"
      - name: "4.3 We loosen the requirement and find a new way to decode"
  - name: "5. Algorithm"
    subsections:
      - name: "5.1 Output-only decode"
      - name: "5.2 Mamba-2: standard decoding"
      - name: "5.3 Speculative decoding"
      - name: "5.4 Kernel design"
  - name: "6. Evaluation"
    subsections:
      - name: "6.1 Standard decoding"
      - name: "6.2 Speculative decoding"
  - name: "7. Conclusion and the future"
  - name: "8. Appendix"

authors:
  - name: Ze-Wei Liou
    url:
    affiliations:
      name: Princeton
  - name: Tri Dao
    url:
    affiliations:
      name: Princeton, Together AI

---

<!-- > **TL;DR.** ReplaySSM speeds up SSM decode by caching inputs, not states. It keeps a short window of recent SSM inputs and either reconstructs the state when needed or computes the output directly from the cached inputs. -->

> Current SSM decoding updates and writes recurrent state back to HBM every step. ReplaySSM instead caches recent inputs, reconstructing the state only when needed and otherwise computing the output directly from the cache. 

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/headline_approach.png" caption="Figure 1a. ReplaySSM caches recent SSM inputs instead of storing the recurrent state every step, reconstructing states on the fly and writing back only when the buffer is full."%}

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/headline_results.png" caption="Figure 1b. End-to-end decoding throughput in vLLM (CUDA Graph enabled), normalized to vLLM's standard decoding at serving batch sizes. Left: ReplaySSM speeds up standard decoding by up to 1.43x. Right: vLLM's existing speculative decoding falls below standard-decoding throughput at serving batch sizes, while ReplaySSM speculative decoding delivers 1.87–1.96x. Speculative window = 4 and all models are NVFP4 on B300." %}


<p align="left">
  <a href="https://github.com/Johnny-Liou/ReplaySSM.git"><img src="https://img.shields.io/badge/GitHub-ReplaySSM-blue?logo=github" alt="Code"></a>
</p>


---

## 1. State Space Models (SSMs)

Decoding is the bottleneck for RL post-training and inference serving due to the long chain-of-thought traces. Agent workloads stretch it further, since every task adds rounds of tool calls and extra reasoning. Transformers remain the industry standard, but their KV cache makes them expensive to run with long context. As sequences grow, storing the full token history linearly increases memory traffic and memory capacity requirements, which hurts both tail latency and throughput.

**State space models (SSMs)** were designed to address this bottleneck. Rather than storing the full token history, SSMs such as Mamba-2<d-cite key="mamba2"></d-cite>, Gated DeltaNet (GDN)<d-cite key="yang2025gated"></d-cite>, and Kimi Delta Attention<d-cite key="kimilinear2025"></d-cite> compress past information into a fixed-size recurrent state. As a result, memory traffic and memory capacity requirements remain constant with respect to context length, eliminating the linear growth associated with KV caches.

SSM layers are cheap but suffer from exact recall, while attention layers in Transformers achieve better recall yet are expensive. **Hybrid models** aim to find the sweet spot in between by interleaving a majority of SSM layers with a few attention layers. Many production-level models (e.g., Nemotron-3<d-cite key="blakeman2025nvidia"></d-cite>, Qwen3.5<d-cite key="qwen35blog"></d-cite>, and Kimi Linear<d-cite key="kimilinear2025"></d-cite>) are built this way.

SSM decoding looks like a clean win: memory drops from $$O(N)$$ to $$O(1)$$. But the same mechanism that gives SSMs constant memory is exactly what introduces new challenges in practice: **it summarizes all of history into one fixed-size state, updates it recurrently, and throws the inputs away**.

<!-- We will start by detailing the three main challenges of SSM decoding. Then, we will introduce ReplaySSM, a simple yet effective change that addresses all three at once. -->
ReplaySSM addresses all three by caching the recent inputs instead of writing the state back every step, without changing the output. That alone speeds up standard decoding by up to 1.48x (1.43x on large MoE models). The larger gain is in speculative decoding, where vLLM's existing implementation falls below standard decoding at serving batch sizes, while ReplaySSM unlocks 1.87–1.96x speedup. We cover the three challenges next, then the method.

## 2. Three challenges in SSM decoding

<a id="2-1-memory-bound-all-i-o-no-compute"></a>
### 2.1 Memory-bound: all I/O, no compute

In Mamba-2, the recurrent form for step $$t$$ is:

$$
S_t = a_t\,S_{t-1} + \Delta_t\,(v_t\,k_t^\top), \qquad y_t = S_t\,q_t.
$$

Here, $$S$$ is the state (i.e., summary of the history) and $$y$$ is the output. $$v_t$$ and $$k_t$$ are the new inputs at step $$t$$ that update the summary from $$S_{t-1}$$ to $$S_t$$, and $$q_t$$ is the input that reads out the output from the current state.

<details markdown="1">
<summary><b>Math behind SSMs: the state update and output read out</b> (skip if familiar)</summary>

**State Space Models (SSMs)**

In each decode step, an SSM repeatedly does two things: updates the state (the summary) with new inputs and reads the output from it. In Mamba-2, the recurrent form is:

$$
S_t = a_t\,S_{t-1} + \Delta_t\,(v_t\,k_t^\top) \quad\text{(update)}, \qquad y_t = S_t\,q_t \quad\text{(readout)} .
$$
    
Here, $$S$$ is the state and $$y$$ is the output. $$v_t$$ and $$k_t$$ are the new inputs at time $$t$$ that update the state from $$S_{t-1}$$ to $$S_t$$, and $$q_t$$ is the input that reads out the output from the current state. $$a_t = e^{A\Delta_t}$$ and $$\Delta_t$$ are the per-step scalars. 
(If you are more familiar with Mamba-2's notation, $$x_t, B_t, C_t$$ are our $$v_t, k_t, q_t$$.)
 
<br>

**An SSM variant: Delta-rule family**

The delta-rule family such as Gated DeltaNet (GDN) trades more complex computation for an additional benefit: the ability to erase the state, while Mamba-2 only adds. In GDN, the recurrent form is:
 
$$
S_t = e^{g_t}\,S_{t-1}\,(I - \beta_t\,k_t k_t^\top) + \beta_t\,(v_t\,k_t^\top) \quad\text{(update)}, \qquad y_t = S_t\,q_t \quad\text{(readout)} .
$$

Here, $$I - \beta_t\,k_t k_t^\top$$, called the Householder term, is what erases information from the current state. Similarly, $$g_t$$ and $$\beta_t$$ are per-step scalars.

        
<br>

**Shapes**

For each sequence, $$v_t$$ is a per-head vector with head dimension $$d$$, while $$k_t$$ and $$q_t$$ are shared across a group of heads (ngroups total) with shape dimension $$n$$. The per-head state is a matrix with shape $$(d, n)$$. Typically, $$n$$ and $$d$$ are $$64$$ or $$128$$, with ngroups ranging from $$1$$ to $$16$$.

</details>

The state update is a rank-1 update by the outer product of two vectors, $$v$$ and $$k$$, followed by an accumulation. Output generation is a vector-matrix multiplication between $$q$$ and the fixed-size state. Both operations are lightweight, and neither maps efficiently to modern matrix-multiplication accelerators such as Tensor Cores. Each decoding step is instead dominated by memory I/O<d-footnote>Arithmetic intensity (FLOPs per byte loaded from memory) is only ~1. Far below the arithmetic intensity of matmul in modern hardware, such as ~300 ops per byte on an H100.</d-footnote>, since it must load and store a state of shape $$(nheads, d, n)$$.

**In hybrid models, SSM matters**

One might expect the attention layers in a hybrid model to dominate latency because their cost grows with context length. Figure 2 shows otherwise. The SSM state update kernel remains the primary bottleneck up to 100K tokens, which covers a large fraction of real inference workloads.

Three factors lead to this result. First, the $$O(N)$$ attention cost remains modest at short to middle context lengths. Second, SSM layers typically outnumber attention layers by a factor of three to six.<d-footnote>Qwen3.5 uses a ratio of 3:1; Nemotron-3-Ultra uses a ratio of 4:1; Nemotron-3-Super uses a ratio of 5:1.</d-footnote> Third, optimization efforts have concentrated on attention kernels, while SSM kernels have received far less attention.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/motivation-latencybreakdown.png" caption="Figure 2. Latency breakdown for Nemotron-3-Super-120B-A12B-NVFP4 with batch size 256 on single B300. Despite attention scaling with context length, SSM kernel latency, which is mostly fixed across context length, remains the largest component up to 100K tokens." %}

<!-- | Layer | Per-step load | Per-step store | # layers (Nemotron-3-Ultra)|
| --- | --- | --- | --- |
| SSM | $$O(d_{\text{state}})$$ | $$O(d_{\text{state}})$$ | 48 |
| Attention | $$O(N)$$ | $$O(1)$$ | 12 | -->

<a id="2-2-summarization-backfires-a-state-has-no-undo"></a>
### 2.2 Summarization backfires: a state has no undo

An SSM summarizes token history into a fixed-size state at each step. A Transformer stores the entire history explicitly in the KV cache. This fixed-size summary gives SSMs their efficiency, but the **summary is lossy and irreversible.** Once the state is updated, the model cannot recover the exact tokens that produced it.

This becomes a problem when inference needs to rewind. Speculative decoding, widely used across production-level models such as Nemotron-3, Gemma4<d-cite key="gemma4_2026"></d-cite>, and Qwen3.5, relies on rollback when a draft token is rejected.

Attention handles rollback naturally. It moves the sequence pointer in the KV cache backward, and the rejected keys and values are no longer used. An SSM does not have an explicit token history to point into. The history is **compressed** into the recurrent state, and the raw inputs are gone. Figure 3 contrasts the two.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/rollback.png" caption="Figure 3. Attention rolls back by moving a KV-cache pointer, while an SSM has irreversibly summarized its inputs into the state and cannot undo them." %}

The common workaround, used for example in vLLM, is to store a separate SSM state for every speculative token. On rejection, the system restores the state that corresponds to the last accepted token. This adds $$T$$ times more memory traffic per decoding step to an already memory-bound path, where $$T$$ is the speculative window. This overhead means speculative decoding, a reliable speedup for Transformers, barely helps SSMs at serving batch sizes, creating a major disadvantage.

<a id="2-3-loss-of-parallelism-sequential-state-dependence"></a>
### 2.3 Loss of parallelism: sequential state dependence

The recurrent SSM states are sequentially dependent. Each state depends on the state before it. This dependency makes SSMs harder to parallelize.<d-footnote>State dependence already poses challenges during training. Mamba-2 addresses this with chunk-wise parallelism, which partitions long sequences into chunks, parallelizes computation across chunks, and performs serial state updates within each chunk. Parallel decoding methods such as speculative decoding have no comparable sequence dimension to split.</d-footnote>

Speculative decoding increases parallelism for Transformers because verification over $$T$$ draft tokens can be batched. Many small matrix-vector operations become one larger GEMM over the speculative window.

SSMs do not get the same form of batching because verification over $$T$$ draft tokens requires the state and output at every speculative position. It cannot replace the whole window with one combined state transition, because verification needs the intermediate outputs, not only the final state. The computation is still a length-$$T$$ loop, not a batched GEMM.

<!-- ### Three challenges, one cause

All three challenges come from the same design choice: at each decoding step, an SSM summarizes the new input into a recurrent state and stores that state back to memory.

(1) Because the state is stored back to memory, we pays high state I/O cost.
(2) Because the inputs are summarized into the state, we loss the ability to rollback.
(3) Because the state is updated recurrently, we depend on different states in each step. -->

<a id="3-the-idea-don-t-store-the-state-cache-the-recent-inputs"></a>
## 3. The idea: don't store the state, cache the recent inputs

Step back from all three problems and ask something that sounds too simple to be useful:

> Why do we store the state at all?

An SSM decoding step consists of four stages: **loading** the state, **updating** it with the new inputs, **generating** the output, and **storing** the updated state back to memory. But the only consumer of that written-back state is the **next** step, which loads it just to do the same thing again.

> We store the state only to recurrently update the state. So, do we need it?

We don't, and the reason lies in the definition of the SSM recurrent state update.

$$
S_t = a_t\, S_{t-1} + \Delta_t\,(v_t\,k_t^\top) = \sum_{i \le t} \Big( \textstyle\prod_{i < j \le t} a_j \Big)\, \Delta_i\,(v_i\,k_i^\top) .
$$

These two expressions describe the same state, but they suggest two different ways to compute it:
 
* **Summary route (left):** load the summary $$S_{t-1}$$ and update it with the new inputs $$v_t, k_t$$.
* **History route (right):**<d-footnote>We assume a zero initial state for simplicity. A nonzero initial state adds a decayed-state term.</d-footnote> reconstruct $$S_t$$ from the recent inputs window $$(v_i, k_i)$$.

An SSM recurrence has the flexibility to take either route.

> An SSM can eagerly summarize each step into a state, or keep the recent inputs and reconstruct it. Current decoding always picks eager summarization. We don't have to.

<details markdown="1">
<summary>Does this flexibility also hold for GDN?</summary>

We use Mamba-2 as the example. GDN adds a Householder term $$I - \beta_t k_t k_t^{\top}$$ to erase content from the state, which makes the history route more complex. However, the same concept still holds since it is still a recurrent update. The full algorithm is placed in the Appendix.
</details>

### 3.1 ReplaySSM caches recent inputs $$(v, k)$$

ReplaySSM changes what is stored to memory per step. Instead of storing the recurrent state, ReplaySSM **caches the recent inputs** in a small buffer. For Mamba-2, the buffer stores the per-step $$(v, k)$$ pairs and the decay factors needed to replay them. 

When the model needs the state, ReplaySSM selects the history route to **reconstruct the state from the buffered inputs**. The state is no longer something we update and write back to memory at each step. It is something we recompute when needed.

When the buffer has grown large enough that loading it would cost more than just writing back a state, ReplaySSM **flushes** the buffer. 
It summarizes the buffered history inputs into the state, clears the buffer, and starts caching inputs again. The state write-back happens only at flush steps; most decoding steps cache small SSM inputs to the buffer. Notably, ReplaySSM is mathematically equivalent to original decoding up to floating-point error. It changes how the state is computed, but not the output.

## 4. One change, three answers

<a id="4-1-we-cut-the-memory-traffic"></a>
### 4.1 We cut the memory traffic

On most steps, ReplaySSM does not write the recurrent state back to memory. It still loads the recurrent state, but replaces the full state store with a small buffer load and an append of two vectors, $$(v, k)$$. This roughly halves the dominant state traffic.

<details markdown="1">
<summary>Baseline vs. ReplaySSM memory traffic</summary>

#### Baseline

In baseline decoding, we load the state and the inputs, and we store the state.

Assuming 4-byte states and 2-byte activations, the memory traffic per head is:

$$
8dn + 2(d + 2n + 1)
$$

The dominant term is the state traffic $8dn$.

#### ReplaySSM

ReplaySSM caches recent inputs instead of storing the state. Assume the buffer already caches the most recent $$h$$ inputs.

- We load: the state, the cached $$(v, \Delta, k)$$, and the current inputs
- We store: the current-step $$(v_t, \Delta_t, k_t)$$

The total memory traffic per head is:

$$
4dn + 2h(d + n + 1) + 2(d + 2n + 1) + 2(d + n + 1)
$$

ReplaySSM halves the dominant state traffic from $8dn$ to $4dn$.

</details>

Since SSM decoding is memory-bound, reducing memory traffic directly improves latency. The flush path is more expensive because it summarizes a chunk of recent inputs into the state and writes the full state back once, but this cost is amortized across the whole window.

<a id="4-2-rollback-becomes-a-buffer-operation"></a>
### 4.2 Rollback becomes a buffer operation

ReplaySSM caches the recent SSM inputs (e.g., the draft tokens) explicitly. It doesn't perform the irreversible summarization per step, so rolling back rejected draft tokens only requires removing their buffer entries.<d-footnote>Prior SSM speculative decoding methods such as Mamba-in-the-Llama<d-cite
key="wang2024mamba"></d-cite> and STree<d-cite key="wu2026stree"></d-cite> take a
different approach to rollback and state handling; see Appendix for details.</d-footnote> There is no full state to restore for each rejected token, and no per-token state copy to keep in memory for recovery. Figure 4 illustrates this with a two-step example.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/rollback_solution.png" caption="Figure 4. ReplaySSM caches each draft's raw inputs, so rolling back rejected drafts is just a pointer move with no state write-back." %}

Speculative decoding triggers flushes (state updates when the buffer is full) more frequently because verification appends multiple proposed tokens at once, causing the buffer to fill faster for a fixed buffer size.

However, ReplaySSM still avoids writing the full recurrent state in most steps. Even under speculation, most steps cache the inputs rather than writing the full state back at every speculative position. Rollback becomes cheap, and the amortized state traffic with speculation is even lower than in baseline standard decoding.

<a id="4-3-we-loosen-the-requirement-and-find-a-new-way-to-decode"></a>
### 4.3 We loosen the requirement and find a new way to decode

In the baseline decoder, every token must produce two things: the updated recurrent state and the output. These two are tied together because the next token needs the state. The baseline decoder follows the recurrence directly: it materializes $$S_t$$, reads $$y_t$$ from it, and writes $$S_t$$ back to memory.

**ReplaySSM changes what must be produced**

Between flushes, the checkpoint state does not change. The recent history lives in the buffer. That means most decode steps **only need the output** for the current token. The updated state is needed only when the buffer fills and we summarize the cached inputs into the checkpoint state.

This weaker requirement unlocks the opportunity to use different algorithms in two paths:

1. Build the state from the checkpoint state and the cached inputs, then read the output.
2. Compute the output directly from the checkpoint state and the cached inputs, without materializing the state.

**Bypass the sequential state dependence**

For standard decoding, the second path is an option. For speculative decoding, the second path is needed to bypass the sequential state dependence. It lets ReplaySSM compute multiple draft outputs in parallel and breaks the need to sequentially reconstruct a new state at each draft token.

## 5. Algorithm

<a id="5-1-output-only-decode"></a>
### 5.1 Output-only decode

**Intuition**

To get the intuition, let's first assume a zero initial state. Then, the state is $$S = v\,k^\top$$, and the output (the only value ReplaySSM needs in most steps) is:

$$
y = S q = (v k^\top) q .
$$

This three-way product can be bracketed in two ways:

$$
(v\,k^\top)\,q
\qquad\text{or}\qquad
v\,(k^\top q).
$$

The left route builds the full state with an outer product, $$v k^\top$$, then reads from it. It gives both the state and the output. This route is useful when ReplaySSM needs to flush the buffer and update the checkpoint state.

The right route never materializes the state. It first computes the inner product $$k^\top q$$, then scales $$v$$ with that scalar. It gives the same output, but not the state. Figure 5a contrasts the two routes.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/output_only.png" caption="Figure 5a. Two routes to the same output. One builds the full state $S = v k^\top$ then reads it with $q$, the other forms the scalar $k^\top q$ first and scales $v$, never materializing the state." %}

**ReplaySSM can choose either route.**

Most of the decoding steps only need an output, so ReplaySSM uses the output-only route. If the buffer is full and the checkpoint state must be updated, ReplaySSM selects the state-and-output route.

With a nonzero checkpoint state $$S_0$$ and a buffer of recent inputs, the same idea applies. Suppose the buffer covers positions $$1,\dots,t-1$$ after the last checkpoint. For Mamba-2, 

$$
S_t = \bar a_t S_0
+
\sum_{j=1}^{t}
s_{j,t} (v_j k_j^\top),
$$

with $$\bar a = e^{A\,\mathrm{pre}_t}$$, $$\;s_j = \Delta_j\,e^{A(\mathrm{pre}_t - \mathrm{pre}_j)}$$, and $$\mathrm{pre}_j = \sum_{i\le j}\Delta_i$$

Reading the output from this state gives

$$
y_t = \bar a_t (S_0 q_t)
+
\sum_{j=1}^{t}
s_{j,t} v_j (k_j^\top q_t).
$$

<!-- This is the output-only form. It still reads the checkpoint state and the recent input buffer, but it does not materialize a $$d \times n$$ state per head. Notably, since $$k$$ and $$q$$ are shared across a group of heads, the output-only form also gives us the benefit of precomputing $$k_j^\top q_t$$.

In the following, we use Mamba-2 as the example. The algorithms for Gated DeltaNet are placed in the Appendix. -->

This is the output-only form. It still reads the checkpoint state and the recent input buffer, but it does not materialize a $$d \times n$$ state per head. Notably, since $$k$$ and $$q$$ are shared across a group of heads, the output-only form also gives us the benefit of precomputing $$k_j^\top q_t$$. Figure 5b shows both routes with their matrix shapes.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/output_only_general.png" caption="Figure 5b. The two routes with ReplaySSM. The state-and-output route materializes $S_t$ then reads it with $q_t$, while the output-only route computes $K^\top q_t$ first and never materializes the state." %}

In the following, we use Mamba-2 as the example. The algorithms for Gated DeltaNet are placed in the Appendix.

<details markdown="1">
<summary>What are the FLOPs for each path?</summary>

**What about FLOPs?**

Switching from the outer-product to the inner-product form also changes the FLOPs spent per decoded token. FLOPs don't affect the latency of a memory-bound kernel, but the count is still worth a look to understand what output-only decode actually computes.

<br>

Assume the buffer holds the most recent $$h$$ inputs (with current token, there are $$h+1$$ $$(v, k)$$ pairs). Per head, the state-and-output (outer product) route costs:

1. Build the new-state sum $$\sum_{j=1}^{h+1} s_{j,t}\,(v_j k_j^\top)$$: $$\;2(h+1)dn$$
2. Decay the checkpoint, $$\bar a_t S_0$$, and accumulate: $$\;2dn$$
3. Read out $$S_t\,q_t$$: $$\;2dn$$

<br>

The output-only (inner product) route costs:

1. $$\bar a_t\,(S_0\,q_t)$$: $$\;2dn + d$$
2. Inner products $$s_{j,t}\,(k_j^\top q_t)$$ for $$j = 1,\dots,h+1$$: $$\;2(h+1)n + n$$
3. Weighted sum over values, $$\sum_j (\cdot)\,v_j$$: $$\;2(h+1)d$$

<br>

The outer-product route pays $$2(h+1)dn$$ to materialize the state. The inner-product route replaces that with $$2(h+1)(d+n)$$, about 64x smaller for $$d = n = 128$$. And since $$k$$ and $$q$$ are shared within a head group, the $$k_j^\top q_t$$ products are computed once per group rather than once per head. **Output-only decode needs fewer FLOPs.**
    
The FLOPs count gets messier beyond Mamba-2 standard decode. Speculative decoding adds a quadratic $$T(T+h)$$ term to the inner-product route ($$T$$ draft queries against $$h+T$$ cached keys, where $$T$$ is the speculative window). This is also where the $$k^\top q$$ products become a real GEMM. Mamba-3<d-cite key="lahoti2025mamba3"></d-cite> adds an $$R^2$$ term, where $$R$$ is the rank. GDN's correction terms complicate the FLOPs count further.

<br>

**Tensor Cores**

Notably, even if we enter the compute-bound regime (e.g., ReplaySSM applied to Mamba-3 speculative decoding), the FLOPs count isn't everything. Tensor Cores have far higher throughput than CUDA cores (roughly 989 vs. 67 TFLOP/s on an H100). Take Mamba-2 standard decode as an example. In the outer-product route, the state-construction term is a GEMM with inner dimension $$h+1$$. Given enough cached tokens, it can map onto Tensor Cores and overlap with the matrix–vector terms running on CUDA cores. The inner-product route has no such GEMM in standard decode; every term is a matrix–vector product or a dot product, all on CUDA cores. So whether the FLOP reduction translates into faster compute requires deeper exploration. 

</details>
    

<a id="5-2-mamba-2-standard-decoding"></a>
### 5.2 Mamba-2: standard decoding

**Baseline Mamba-2 decoder**

The baseline Mamba-2 decoder eagerly updates and stores the recurrent state back to memory at each decoding step.
 
> **Algorithm 1: Baseline (recurrent state update)**
>
> State: recurrent state $$S \in \mathbb{R}^{d \times n}$$ in HBM
>
> Input: token inputs $$(v, \Delta, k, q)$$ 
>
> <br>
>
> 1. $a \gets e^{A\Delta}$
> 2. Load $$S$$ from HBM
> 3. $S \gets a\,S + \Delta\,(v\,k^\top)$ 
> 4. $y \gets S\,q$ 
> 5. store $$S$$ to HBM
> 6. Return $$y\;[+\,\text{skip, gate}]$$

<!-- > **Algorithm 2: Outer-product decode (state-and-output)**
>
> State: checkpoint $$S_0$$, buffer $$\{(v_j, \Delta_j, k_j)\}_{j=1}^{L}$$, window $$t = L+1$$
> Input: step $$(v, \Delta, k, q)$$
>
> 1. $$S_t \gets \bar a\,S_0 + \sum_{j} s_j\,(v_j\,k_j^\top)$$  // build the $$d\times n$$ state
> 2. $$y \gets S_t\,q$$  // read out
> 3. If buffer full: $$S_0 \gets S_t$$, clear buffer  // flush: one full-state store
> 4. Else: append $$(v, \Delta, k)$$
> 5. Return $$y\;[+\,\text{skip, gate}]$$
 -->
 
**ReplaySSM**

ReplaySSM keeps a checkpoint state $$S_0$$ and a buffer of recent inputs. For each token, it appends the current inputs to the buffer, computes the output from the checkpoint plus the cached inputs, and materializes the state only when the buffer must be flushed.

> **Algorithm 2: ReplaySSM output-only decode**
>
> State: checkpoint $$S_0$$, buffer $$\mathcal{B} = \{(v_j,\Delta_j,k_j)\}_{j=1}^{h}$$ with capacity $$L$$ in HBM
>
> Input: token inputs $$(v, \Delta, k, q)$$
>
> <br>
>
> 1. Append $$(v, \Delta, k)$$ to the buffer
> 2. Compute the decay weights $$\bar a$$ and $$s_j$$
> 3. $$y \gets \bar a\,(S_0\,q) + \sum_{j=1}^{h+1} s_j\,(k_j^\top q)\,v_j$$  // output only, no state materialized
> 4. If buffer full:
> 5. $$\quad S_0 \gets \bar a\,S_0 + \sum_{j=1}^{h+1} s_j\,(v_j\,k_j^\top)$$, clear buffer  // flush: full-state store
> 6. Return $$y\;[+\,\text{skip, gate}]$$

<a id="5-3-speculative-decoding"></a>
### 5.3 Speculative decoding

**Baseline in vLLM's implementation**

The baseline must store a full state snapshot for every draft position so it can roll back after rejection. It also computes the draft outputs through a length-$$T$$ recurrence loop.

> **Algorithm 3: Standard speculative decode (serial scan)**
>
> State: State for the last accepted token in HBM
> 
> Input: draft inputs $${(v_s, \Delta_s, k_s, q_s)}_{s=1}^{T}$$
>
> <br>
>
> 1. Load $$S$$ from the last accepted snapshot
> 2. For $$s = 1,\dots,T$$:
> 3. $\quad S \gets e^{A\Delta_s}\,S + \Delta_s\,(v_s\,k_s^\top)$
> 4. $\quad y_s \gets S\,q_s$
> 5. $$\quad$$Store $$S$$ as the snapshot for draft token $$s$$
> 6. Return $$\{y_s\}\;[+\,\text{skip, gate}]$$

**ReplaySSM**

ReplaySSM keeps draft inputs in the same buffer used during standard decoding. On rollback, it simply moves the pointer in the buffer to keep accepted entries and discard the rest.

The output-only form also removes the serial state update from verification. Each draft query reads from the **same checkpoint state** and the same buffer window. The only difference across draft positions is the causal mask: draft output $$s$$ can use cached entries up to its own position, but not later drafts.

> **Algorithm 4: ReplaySSM cached speculative decode**
>
> State: checkpoint $$S_0$$, buffer $$\mathcal{B} = \{(v_j,\Delta_j,k_j)\}_{j=1}^{h}$$ with capacity $$L$$ in HBM
>
> Input: draft inputs $$\{(v_s,\Delta_s,k_s,q_s)\}_{s=1}^{T}$$
>
> <br>
>
> 1. Append the draft inputs to the buffer; draft $$s$$ sits at position $$p_s = h + s$$
> 2. For each draft $$s$$, compute the decay weights $$\bar a_s$$ and $$w_{j,s}$$ at position $$p_s$$
> 3. $$H_{:,s} \gets \bar a_s\,(S_0\,q_s)$$  // checkpoint readout for every draft
> 4. $$M_{j,s} \gets k_j^\top q_s$$, masked to $$j \le p_s$$  // GEMM
> 5. $$Y_{:,s} \gets H_{:,s} + \sum_{j \le p_s} w_{j,s}\,M_{j,s}\,v_j$$  // GEMM
> 6. If this step is a flush step:
> 7. $\quad S_0 \gets \bar a_{\mathcal{B}}\,S_0 + \sum_{j=1}^{h} w_j\,(v_j\,k_j^\top)$ 
> 8. Return $$Y_{:,s}\;[+\,\text{skip, gate}]$$

Compared with standard speculative decode, which iterates through draft tokens and materializes each intermediate state, ReplaySSM directly computes the outputs through inner products between $$k$$ and $$q$$. That is a better shape for the hardware. The key-query products become a matrix multiplication over cached keys and draft queries. The weighted sum over values is another matrix multiplication under a causal mask.

This also changes rollback cost. Baseline speculative decode keeps one state snapshot per draft token. ReplaySSM keeps recent inputs. During commit, ReplaySSM advances the pointer by the number of accepted draft tokens and discards the rest. No full-state restore is needed. Notably, in a flush step, only committed cached inputs are summarized into the state. The current step's speculative tokens are not summarized for rollback.

<details markdown="1">
<summary>The flush decision for speculative decoding</summary>

Let $$h$$ be the number of cached tokens currently in the buffer, $$T$$ the speculative window, and $$L$$ the buffer capacity, as in Algorithm 4. ReplaySSM flushes one window early. It summarizes the cached tokens into the checkpoint when
 
$$
h + 2T > L,
$$
 
rather than the natural condition $$h + T > L$$.
 
The natural condition can silently shrink the speculative window. Suppose a step lands at $$h + T = L - 1$$. No flush fires, and all $$T$$ drafts happen to be accepted. The next step then starts at $$h = L - 1$$. The flush fires now, but only one free slot remains for the fresh draft window in that step, so the window is truncated to a single draft and the accepted tokens for that step collapse. Flushing one window early guarantees at least $$T$$ free slots on every step.

</details>

<a id="5-4-kernel-design"></a>
### 5.4 Kernel design

Here, we highlight two key kernel design choices. Feel free to also check out the Appendix for details on how we integrate our approach into vLLM.

**Precomputing shared inner products (Mamba-2)** 

In the output-only form, all heads in a group need the same inner products $$k_j^\top q$$. Computing them inside the main SSM update kernel would repeat the work across the head-dimension grid and add register pressure. ReplaySSM computes them in a small precompute kernel that runs once per group and writes a scratch buffer the main SSM update kernel reads.<d-footnote>We do not use the same precompute kernel for GDN. GDN state update reads the state with two vectors, $k$ and $q$, so ReplaySSM takes the state-and-output route there (Algorithm 6 in Appendix for details), where no shared inner products appear.</d-footnote>

**A ring buffer avoids data-dependent copies** 

In speculative decoding, a flush only summarizes the committed cached inputs into the state. The current step's speculative tokens are not summarized, since rejected ones must still be rolled back. The accepted tokens remain in the buffer as the cached inputs for the next decoding step. To avoid relocating these tokens back to the front of the buffer<d-footnote>Without a ring buffer, the relocation would be needed for each layer in almost every step, due to continuous batching and different accepted tokens per sequence in speculative decoding. Under CUDA Graph, the copy kernel would also launch for every row only to early-exit on the ones not relocating.</d-footnote>, ReplaySSM uses a ring buffer, with indexing in the kernel for correctness, so a rollback becomes purely a pointer move. The ring buffer with indexing is also important for tree-based speculative decoding, where accepted tokens are no longer contiguous in the buffer.


## 6. Evaluation

We evaluate ReplaySSM on two hybrid families with different SSM layers: Nemotron-3 (Mamba-2) and Qwen3.5 (GDN).

| Model | Params | Precision | Hardware |
|---|---|---|---|
| Nemotron-3-Nano-4B | 4B dense | BF16 | 1×H100 |
| Nemotron-3-Super-120B | A12B MoE | NVFP4 | 1×B300 |
| Nemotron-3-Ultra-550B | A55B MoE | NVFP4 | 2×B300, TP2 <d-footnote>Ultra-550B does not fit on a single B300, so we run it with tensor parallelism across two B300 GPUs.</d-footnote>|
| Qwen3.5-4B | 4B dense | BF16 | 1×H100 |
| Qwen3.5-122B | A10B MoE | NVFP4 | 1×B300 |

We implemented ReplaySSM on top of vLLM. All results run in vLLM with CUDA Graph enabled. SSM states are in FP32 and the vectors cached in the buffer are in BF16. For speculative decoding, both families use their MTP heads as the drafter.

Across both families and sizes from 4B to 550B, ReplaySSM speeds up vLLM's standard decoding by up to 1.48x (1.43x on large MoE models) end-to-end and speculative decoding by 1.87–1.96x over vLLM's standard decoding. It also supports 3.0–3.3x more concurrent requests than vLLM's speculative path under a fixed memory budget.

<a id="6-1-standard-decoding"></a>
### 6.1 Standard decoding

<!-- **End-to-end speedup** -->

Figure 6 reports SSM-kernel and end-to-end per-step speedup at batch size 256 over 1K decoding steps, with buffer size 8 for Nemotron-3 and 16 for Qwen3.5 (the best settings from Figure 7).

ReplaySSM makes SSM decoding faster, and the kernel speedup translates into **end-to-end speedup on hybrid models across different SSM families and model sizes (from 4B to 550B).** On Nemotron-3, ReplaySSM reaches 1.43x to 1.84x kernel and 1.20x to 1.48x end-to-end speedup. On Qwen3.5, ReplaySSM reaches 1.43x to 1.64x kernel and 1.20x to 1.27x end-to-end speedup. The end-to-end speedup is smaller because ReplaySSM targets only the SSM kernel, while attention, GEMMs, and the rest are unchanged.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/ar-e2espeedup.png" caption="Figure 6. Kernel-level and end-to-end per-step speedup over vLLM's baseline across the Nemotron-3 and Qwen3.5 families (batch size 256, 1K decoding steps)." %}

**Trade-offs of different buffer capacities**

The buffer size in ReplaySSM introduces a trade-off. A shorter buffer flushes more often, which pays more cost on writing the updated state back to HBM. A longer buffer reduces flush frequency, but each step reads more from the buffer, and eventually turns the kernel compute-bound. Figure 7 shows the resulting bell shape, where a medium buffer (8 for Nemotron-3, 16 for Qwen3.5) balances the two costs.

<!-- The bell-shaped trend in Figure 7 shows this trade-off. **A medium-sized buffer balances the two costs and achieves the largest speedup.** -->

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/ar-buffersize.png" caption="Figure 7. Kernel speedup of ReplaySSM over the baseline for buffer sizes 4, 8, 16, and 32 at batch sizes 64 and 256." %}

<a id="6-2-speculative-decoding"></a>
### 6.2 Speculative decoding

**End-to-end throughput**

We test end-to-end throughput (tokens/s) on vLLM using prompts from the GSM8K dataset<d-cite key="gsm8k"></d-cite> (speculative window = 4, temperature = 0), sweeping batch size up to 512. ReplaySSM preserves the same draft acceptance behavior while consistently outperforming standard decoding and vLLM's speculative decoding baseline. At the largest batches, ReplaySSM reaches 1.87–1.96× over standard decoding and up to 2.14× over the baseline speculative path, and the gap widens with batch size. Since acceptance is identical across all three systems (bottom panels), the throughput gain comes from two sources: **faster verification per step** and **higher concurrency.** We break these down next.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/spec-e2ethroughput.png" caption="Figure 8. End-to-end decoding throughput versus batch size on GSM8K prompts (speculative window 4, temperature 0, MTP drafter). Bottom: accepted tokens per step are identical for baseline and ReplaySSM." %}

**Breakdown 1: faster decode steps**

The baseline's verification cost grows almost linearly with the speculative window, because it stores a full SSM state per draft token on an already memory-bound path. Figure 9 shows this on Qwen3.5-122B. At $$T=6$$ the baseline kernel costs 4.85× the standard decoding kernel. ReplaySSM's state traffic is one checkpoint load plus an occasional flush, so its cost stays near flat, between 1.27× and 1.72× at $$T=6$$ depending on how many drafts are accepted (more acceptance advances the buffer faster and flushes more often).

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/spec-specwindow.png" caption="Figure 9. Speculative decoding kernel latency on Qwen3.5-122B-A10B-NVFP4, normalized to vLLM standard decoding (batch size 128, buffer size 16, 1×B300). The shaded band spans the all-reject to all-accept cases." %}

Figure 10 further shows how kernel speedup propagates to full decode step speedup. The 2.28–3.33× kernel speedup translates to 1.23–1.69× on the verify forward pass, then 1.20–1.58× on the full decode step once draft-model and preprocessing overheads are included. 

<br>

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/spec-speedupbreakdown.png" caption="Figure 10. Speedup over vLLM's speculative decoding baseline at the kernel, verify forward pass, and full decode step levels (speculative window 4)." %}

**Breakdown 2: higher maximum concurrency**
 
The per-draft snapshots also cost capacity. Under a fixed HBM budget (window = 4), the baseline's preallocated states cut the maximum decode batch by roughly 4× relative to standard serving. ReplaySSM caches small input vectors instead of full states, recovering 3.0–3.3× of that concurrency (Figure 11). For a throughput-oriented deployment the **maximum concurrency matters as much as per-step latency**. It determines how many requests the speculative path can serve at all.

{% include figure.liquid loading="eager" path="assets/img/2026-06-15-replayssm/spec-maxconcurrency.png" caption="Figure 11. Maximum decode concurrency under a fixed HBM budget (speculative window 4). ReplaySSM supports 3.0–3.3× more concurrent requests than the baseline speculative path." %}

Together, these two effects explain the trends in Figure 8. Cheaper verification lifts the entire curve, while the smaller memory footprint allows ReplaySSM to continue scaling with batch size where the baseline flattens out.

## 7. Conclusion and the future

ReplaySSM makes a simple change: instead of storing the state, we cache recent inputs. This simple change reduces memory traffic, enables low-cost rollback, and unlocks output-only decoding.

ReplaySSM is not limited to Mamba-2; it also applies to delta-rule models such as GDN. We implemented ReplaySSM in vLLM, where it speeds up standard decoding and removes key obstacles that have long hindered speculative decoding.

Looking ahead, we plan to bring the ideas behind ReplaySSM to more SSM architectures, such as Mamba-3 and GDN2<d-cite key="gdn2"></d-cite>. We are also excited to explore how the flexibility of SSMs, choosing when to summarize inputs and when to cache them explicitly, can go beyond accelerating decoding.

## 8. Appendix

<details markdown="1">
<summary>Algorithms for Gated DeltaNet</summary>

### A.1 Standard decoding

**Baseline GDN decoder**

The GDN baseline mirrors Algorithm 1, with one extra correction term that lets the model erase stale content.
 
> **Algorithm 5: GDN baseline (recurrent state update)**
>
> Input: state $$S \in \mathbb{R}^{d\times n}$$ from HBM, step $$(q, k, v, g, \beta)$$
>
> <br>
>
> 1. $\alpha \gets e^{g}$
> 2. $$S \gets \alpha\,S$$  // gated decay
> 3. $$u \gets \beta\,(v - S\,k)$$  // correction: subtract the state's readout at $$k$$
> 4. $$S \gets S + u\,k^\top$$  // rank-1 write at $$k$$
> 5. $$y \gets S\,q$$  // readout at $$q$$
> 6. store $$S$$ to HBM
> 7. Return $$y$$
 
The key difference between Mamba-2 and GDN is **what we cache**. In Mamba-2, we simply append $$(v, k)$$ to the buffer. This does not work for GDN because of the correction term
 
$$u = \beta\,(v - S\,k).$$
 
Computing $$u_t$$ requires the state $$S_{t-1}$$. If we cached the raw $$v_t$$, replaying the buffer would still need every intermediate state, reintroducing the serial dependency we are trying to eliminate.

The fix is to cache $$u$$ instead of $$v$$. Once $$u_t$$ is known, the GDN update becomes $$S_t = \alpha_t\,S_{t-1} + u_t\,k_t^\top$$. The state then unrolls into a decayed checkpoint plus a weighted sum of $$u_j\,k_j^\top$$, same as the Mamba-2 history route.

<br>

**ReplaySSM**

> **Algorithm 6: ReplaySSM GDN standard decode (state reconstruction)**
>
> State: checkpoint $$S_0$$, buffer $$\{(u_j, k_j, g_j)\}_{j=1}^{h}$$, with $$\alpha_j = e^{g_j}$$
> 
> Input: step $$(q, k, v, g, \beta)$$
>
> <br>
>
> 1. $$S_h \gets \big(\textstyle\prod_j \alpha_j\big)\,S_0 + \sum_j \big(\textstyle\prod_{i>j}\alpha_i\big)\,u_j\,k_j^\top$$  // rebuild from cache
> 2. $\alpha \gets e^{g}$
> 3. $$u \gets \beta\,(v - \alpha\,(S_h\,k))$$  // correction at the current key
> 4. $$y \gets \alpha\,(S_h\,q) + u\,(k\!\cdot\! q)$$  // output for this token
> 5. Append $$(u, k, g)$$ to buffer
> 6. If buffer full: 
> 7. $$\quad$$ $$S_0 \gets \alpha\,S_h + u\,k^\top$$, clear buffer // flush: one state store
> 8. Return $$y$$
 
Unlike Mamba-2, a GDN step needs the state's readout at two vectors, $$k$$ and $$q$$. ReplaySSM therefore takes the state-and-output route. It rebuilds the state from the decayed checkpoint and the outer products $$u_j\,k_j^\top$$, then reads the output from it. 

### A.2 Speculative decoding

**Baseline in vLLM's implementation**

GDN verification suffers from the same serial loop and per-draft state snapshots as Mamba-2. Moreover, each correction $$u_s$$ depends on the state after the previous draft, introducing **sequential dependencies between drafts.**

> **Algorithm 7: GDN standard speculative decode (serial delta-rule scan)**
>
> Input: drafts $$\{(q_s, k_s, v_s, g_s, \beta_s)\}_{s=1}^{T}$$, state snapshots
>
> <br>
>
> 1. $$S \gets$$ snapshot at the last accepted token  // roll back
> 2. For $$s = 1, \dots, T$$:  // serial
> 3. $\quad S \gets e^{g_s}\,S$
> 4. $\quad u_s \gets \beta_s\,(v_s - S\,k_s)$
> 5. $\quad S \gets S + u_s\,k_s^\top$
> 6. $\quad y_s \gets S\,q_s$
> 7. $$\quad$$ store $$S$$ to snapshot $$s$$  // full-state store per draft
> 8. Return $$\{y_s\}$$

ReplaySSM removes the serial loop by applying the **chunk-wise parallelism approach** GDN uses for training. Expanding the recurrence from the reconstructed state $$S_h$$ gives $$u_s = R_s - \sum_{s'<s} A_{s,s'}\,u_{s'}$$, where $$R_s$$ depends only on the drafts and $$S_h$$, and $$A$$ is strictly lower triangular. The $$T$$ corrections can therefore be computed through a single triangular solve rather than $$T$$ sequential state updates. 

<br>

> **Algorithm 8: ReplaySSM GDN speculative decode (chunked delta-rule)**
>
> State: checkpoint $$S_0$$, buffer $$\{(u_j, k_j, g_j)\}$$
> 
> Input: drafts $$\{(q_s, k_s, v_s, g_s, \beta_s)\}_{s=1}^{T}$$, with cumulative gates $$G_s = \sum_{i \le s} g_i$$
>
> <br>
>
> 1. $$S_h \gets$$ rebuild from $$S_0$$ and the buffer (Algorithm 6, step 1)
> 2. $$hq_s \gets S_h\,q_s$$, $$\;hk_s \gets S_h\,k_s$$  // history into each draft (GEMMs)
> 3. $$A_{s,s'} \gets \beta_s\,e^{G_s - G_{s'}}\,(k_s\!\cdot\! k_{s'})$$ for $$s' < s$$  // strictly lower triangular
> 4. $R_s \gets \beta_s\,(v_s - e^{G_s}\,hk_s)$
> 5. $$W \gets (I + A)^{-1}$$, // one $$T \times T$$ inverse, all corrections at once
> 6. $U_s \gets \sum_{s' \le s} W_{s,s'}\,R_{s'}$
> 7. $y_s \gets e^{G_s}\,hq_s + \sum_{s' \le s} e^{G_s - G_{s'}}\,(k_{s'}\!\cdot\! q_s)\,U_{s'}$
> 8. Append $$(U_s, k_s, g_s)$$ to the buffer 
> 9. If this step is a flush step:
> 10. $\quad$ $S_0 \gets S_h$
> 11. Return $$\{y_s\}$$

ReplaySSM removes both per-draft state snapshots and the serial loop. As a result, the entire verification step becomes parallelizable. Besides one $$T \times T$$ triangular solve, all operations are GEMMs.
    
</details>

<details markdown="1">
<summary>Running under CUDA Graph in vLLM</summary>

CUDA Graph is important for inference performance because it removes per-step CPU launch overhead. However, enabling CUDA Graph support for ReplaySSM inside vLLM is not straightforward, due to two issues.
 
The first issue is batch divergence. With continuous batching, different sequences reach their flush steps at different times. Speculative decoding amplifies this effect because each sequence may accept a different number of draft tokens. As a result, at a given decode step, some sequences need to flush while others do not. Since the same captured graph has to handle the whole batch, ReplaySSM treats the flush decision as per-sequence data that the kernel reads and branches at runtime, not a compile-time constant.
 
The second issue is commit and rollback for speculative decoding. The number of accepted tokens is only known after sampling, and it can differ across sequences. Sending those counts back to the host would add a host-device synchronization point on every step, which would stall the pipeline. Instead, ReplaySSM uses a small commit kernel to update buffer pointers directly on the device. For each sequence, the kernel advances the relevant pointers by that sequence’s accepted-token count, allowing one captured graph per batch size to cover the speculative path without host synchronization.

</details>

<details markdown="1">
<summary>Comparison with prior SSM speculative decoding methods</summary>

Prior SSM speculative decoding methods already avoid keeping a separate recurrent
state for each draft token (e.g., Mamba-in-the-Llama<d-cite key="wang2024mamba">
</d-cite> and STree<d-cite key="wu2026stree"></d-cite>), but they still materialize
and write at least one recurrent state back to HBM at every decoding step. Committing
that state each step keeps a sequential state dependency, since the state for the last committed tokens
has to be rebuilt before the current drafts are verified. Both were also evaluated only for Mamba-2 and at batch size 1, where speculative decoding's gain
comes mostly from amortizing weight loads rather than from a faster SSM kernel.
ReplaySSM instead writes the state back to HBM only at flush, generalizes to
delta-rule models such as GDN, and runs in vLLM under CUDA Graph at serving batch
sizes.

</details>