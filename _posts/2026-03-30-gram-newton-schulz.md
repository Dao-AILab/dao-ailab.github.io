---
layout: distill
title: "Gram Newton-Schulz: A Fast, Hardware-Aware Newton-Schulz Algorithm for Muon"
giscus_comments: true
date: 2026-03-30
featured: true
related_posts: false

authors:
  - name: Jack Zhang
    affiliations:
      name: Princeton University
  - name: Noah Amsel
    affiliations:
      name: New York University
  - name: Berlin Chen
    affiliations:
      name: Princeton University
  - name: Tri Dao
    affiliations:
      name: Princeton University
---

<style>
.post img {
  max-width: 100%;
  height: auto;
}

/* Algorithm formatting - tighter spacing between steps */
.post blockquote p {
  margin-top: 0.2em;
  margin-bottom: 0.2em;
  line-height: 1.4;
}

/* More space after algorithm title and input */
.post blockquote p:first-child {
  margin-top: 0;
}

.post blockquote p:nth-child(2),
.post blockquote p:nth-child(3) {
  margin-bottom: 1em;
}

/* Unitalicize algorithm titles */
.post blockquote strong {
  font-style: normal !important;
}

/* Style theorem boxes to look like HackMD */
.post blockquote {
  background-color: transparent;
  border-left: 4px solid var(--global-theme-color, #4c9eff);
  padding: 1rem 1.5rem;
  font-size: inherit;
  color: inherit;
}

/* Center display math equations */
.post blockquote .MJXc-display,
.post blockquote .katex-display {
  text-align: center !important;
  margin: 1em 0 !important;
}

/* Make math equations inherit text color */
.post blockquote .MathJax,
.post blockquote .katex,
.post blockquote .MathJax_Display,
.post blockquote mjx-container,
.post blockquote mjx-math,
.post blockquote mjx-mrow,
.post blockquote .MathJax *,
.post blockquote mjx-container * {
  color: inherit !important;
}

/* Dark mode compatibility */
html[data-theme='dark'] .post blockquote .MathJax,
html[data-theme='dark'] .post blockquote mjx-container,
html[data-theme='dark'] .post blockquote mjx-container * {
  color: var(--global-text-color) !important;
}

/* Make H1 headers match H2 style with underline */
.post h1 {
  font-weight: normal !important;
  font-style: normal !important;
  border-bottom: 1px solid var(--global-divider-color) !important;
  padding-bottom: 0.5rem !important;
}

/* Consistent spacing before headers - larger for higher level headers */
.post h1 {
  margin-top: 3rem !important;
  margin-bottom: 1.5rem !important;
}

.post h2 {
  margin-top: 2.5rem !important;
  margin-bottom: 1.25rem !important;
}

.post h3 {
  margin-top: 2rem !important;
  margin-bottom: 1rem !important;
}

.post h4 {
  margin-top: 1.5rem !important;
  margin-bottom: 0.75rem !important;
}

.post h5, .post h6 {
  margin-top: 1rem !important;
  margin-bottom: 0.5rem !important;
}
</style>

Muon is becoming the optimizer of choice for training state-of-the-art language models like Kimi K2 Thinking and GLM-5.[^kimi][^GLM] Compared to AdamW, Muon needs fewer optimizer steps to reach a given loss, but each step is more expensive. This overhead is due to Muon's Newton-Schulz orthogonalization procedure, a cubic time matrix operation not present in older optimizers.

<!-- Still, on net, Muon is superior in the way that matters most: it trains an equally good model in a shorter amount of time.Or "On balance, Muon is superior in the way that matters most: it produces an equally good model in a shorter amount of time" -->

![icml_optimizer_plot_blackwell (2)](https://hackmd.io/_uploads/ByJX-6BsWg.png)
_Figure 1: AdamW vs. Muon: Wall clock time of optimizer step across LLaMa model sizes, benchmarked on B300._

Muon's superior optimization quality justifies its more expensive optimizer step. However, as model size scales up, the overhead of computing each Muon step grows rapidly. Traditional optimization methods (SGD, AdamW) perform element-wise operations, such as updating the momentum or rescaling it by the second moment. For a weight matrix of size $n \times m$, performing the optimizer step takes $O(mn)$ time given the gradient matrix as input. In contrast, many modern optimizers (Muon, Scion, Dion, SOAP, Shampoo, SPlus, etc.) use orthogonalization or higher-order preconditioning to compute the update to the weights at each training step.[^muon][^dion][^scion][^soap][^shampoo][^splus] These methods require matrix multiplications that cost $O(mn^2)$ time (assuming $n \leq m$). Therefore, the runtime of each call to the optimizer is far greater than for AdamW. Depending on the training setup (global batch size, cluster size, and parallelism settings), Newton-Schulz accounts for between [2% and 17%](#appendix) of end-to-end wall clock time.

While $O(mn^2)$ runtime is an unavoidable cost of these algorithms, there is still significant room for improvement in both FLOPs and wall clock time. As it is typically implemented, the Newton-Schulz routine has several shortcomings:

1. It uses not just one or two, but _ten_ multiplications of $n \times m$ matrices, costing $2mn^2$ FLOPs each. Most weights in popular architectures are rectangular, with $m \gg n$, and those of recent MoE architectures with many fine-grained experts are even _more_ rectangular. Thus, the rectangular matrix multiplications dominate the costs of other operations (like small multiplications of $n \times n$ matrices).
2. Many of the intermediate matrices it computes are symmetric, but no computational advantage is taken of this structure. Half the work used to compute these matrices is redundant.
3. It uses cuBLAS for batched matrix multiplication/addition $\alpha \mathbf A \mathbf B + \beta \mathbf C$, which is not fully optimized for the Hopper GPU architecture.
<!-- 1. Conceptually distinct weights, such as $\mathbf W_Q$, $\mathbf W_K$, and $\mathbf W_V$ in attention layers, are sometimes orthogonalized in concatenated form as a single matrix, like $\begin{bmatrix}\mathbf W_Q \,\vert\, \mathbf W_K \,\vert\, \mathbf W_V\end{bmatrix}$. Given Newton-Schulz's cubic runtime, this form is significantly more expensive. -->

<!-- 1. It misses certain opportunities for fusing matrix addition operations with matrix multiplications. -->

Previous work has sought to improve Newton-Schulz by optimizing its polynomial coefficients or its normalization step. While this can reduce the number of iterations needed for Newton-Schulz to converge, it does not address the shortcomings listed above. Others[^flashmuon] have implemented Newton-Schulz using special-purpose symmetric matrix multiplication routines, but the runtime benefit is limited due to the high number of rectangular and non-symmetric multiplications. While Newton-Schulz and related methods have been studied for decades in the numerical analysis literature, research attention has mostly focused on regimes where high accuracy is required, where algorithms are optimized for CPUs rather than GPUs, or where input matrices are square. In recent years, randomized sketching has been used to design sophisticated algorithms for many computations involving highly rectangular matrices; however (aside from further optimizing the coefficients[^PRISM]) these do not seem to be applicable to Muon.

## Our Contributions

To address these shortcomings, we introduce **Gram Newton-Schulz**, a reworking of the Newton-Schulz routine that **reduces the optimizer time by up to 50%** in trillion-parameter MoE models like Kimi K2. Instead of iterating on the rectangular input matrix $\mathbf{X} \in \mathbb{R}^{n \times m}$, Gram Newton-Schulz iterates on the small square symmetric Gram matrix $\mathbf{XX^\top} \in\mathbb{R}^{n \times n}$, reducing the FLOP cost and enabling a greater use of symmetric GEMM kernels.

Our contributions are as follows. First, we show how to rewrite standard Newton-Schulz in a form that is _mathematically identical_, producing the exact same output up to floating-point error, but that acts mostly on the space of $n \times n$ matrices. Because these matrices are smaller and admit specialized symmetric matrix multiplication routines, each iteration is faster than in standard Newton-Schulz. Only the preprocessing step (forming $\mathbf X \mathbf X^\top$) and the post-processing step (multiplying by $\mathbf X$) require rectangular matrix multiplications. We call this new form [Naive Gram Newton-Schulz](#alg-naive-gram-ns).

Second, we conduct a thorough study of the numerical properties of Naive Gram Newton-Schulz. We identify the potential for numerical instability when using half-precision floating point arithmetic, especially due to spurious negative eigenvalues in the Gram matrix. We remedy this instability using a "restarting" strategy, where we reconstruct the Gram matrix partway through the algorithm. We call this modified algorithm [Stabilized Gram Newton-Schulz](#alg-stable-gram-ns).

<!-- In terms of FLOPs, this algorithm is significantly cheaper than standard Newton-Schulz for rectangular inputs. -->

Third, to take full advantage of the latest generation of GPUs and of the mathematical structure of Newton-Schulz, we implement custom kernels for _symmetric_ matrix multiplication. The kernels, implemented in CuTeDSL for the Hopper and Blackwell architectures, attain state-of-art performance.

<!-- We implement our kernels in CuTeDSL, a language that provides low level, granular control unavailable in PyTorch or Triton. -->

<!-- Third, we design custom GPU kernels with clever tile schedules that coordinate parallel works much more efficiently for *symmetric* matrix multiplication, which characterizes most of Gram Newton-Schulz's workload. The kernels, implemented in CuTeDSL, attain state-of-art performance.  -->

<!-- Second, we show how to rewrite the Newton-Schulz iteration in a form that is *mathematically identical* but computationally advantageous. In this new form—which we call [Naive Gram Newton-Schulz](#alg-naive-gram-ns)—multiplications of large rectangular matrices are replaced with symmetric multiplications of small square matrices. -->

Finally, we replace Muon's Newton-Schulz routine with Gram Newton-Schulz, which we denote **GramMuon**, and observe a 40-50% reduction in the runtime of the orthogonalization step. Experiments confirm that training language models with GramMuon is stable and preserves the optimization quality of the standard version within $0.01$ validation perplexity, making our algorithm a rare instance of "free lunch" performance improvement.

To facilitate the adoption of Gram Newton-Schulz, we release the following open-source implementations:

1. A [drop-in replacement](https://github.com/Dao-AILab/gram-newton-schulz) for Muon's Newton-Schulz routine that is mathematically equivalent, numerically stable, and up to twice as fast.
2. [Fast GPU kernels](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_symmetric.py) for symmetric matrix multiplication ($AB$, $\alpha AB + \beta C$) written in CuTeDSL for Hopper and Blackwell, which may be of independent interest.
   <!-- 1. An up to $2\times$ faster algorithm for Newton-Schulz that iterates on the square, symmetric Gram matrix $\mathbf{XX^\top}$ instead of the expensive rectangular $\mathbf{X}$, minimizing the FLOPs bottleneck created by $\mathbf{X}$'s long dimension and enabling more symmetric GEMM kernels.-->
   <!-- 2. A polynomial restart strategy that stabilizes Gram Newton-Schulz by preventing explosions of half-precision negative eigenvalues formed during the construction of the Gram matrix, ensuring training accuracy matching standard Muon. -->

<!-- We will first [recap Muon](#Muon-Recap) and analyze [standard Newton-Schulz's FLOPs](#Runtime-of-Standard-Newton-Schulz) and bottlenecks. Second, we will [derive Naive Gram Newton-Schulz](#Gram-Newton-Schulz) and understand why iterating on the Gram matrix is faster. Third, we will understand [the numerical properties of iterating on the Gram matrix](#Instability-of-Naive-Gram-Newton-Schulz), identify potential for numerical instability, and [fix this numerical instability](#Stabilizing-Gram-Newton-Schulz-by-Restarting). Fourth, we will provide an overview of the [symmetric GEMM kernel implementation](#Symmetric-GEMM-Kernels-in-CuTeDSL). Finally, we will benchmark Gram Newton-Schulz and show that it preserves training accuracy. --> <!-- Noah: Maybe others disagree I think this is redundant given what we have above. And if people want to jump around, they can use the hyperlinks in the table of contents. -->

We will begin by recapping Muon to see why we need Newton-Schulz in the first place, describing how standard Newton-Schulz works mathematically, and analyzing its performance bottlenecks.

# Muon Recap

The Muon optimizer[^muon] is best described as steepest-direction descent with respect to the spectral norm.[^deriving_muon] At step $k$ of training, let $\mathbf W_k \in \mathbb{R}^{n \times m}$ be a weight matrix and let $\mathbf G_k$ be the gradient of the loss with respect to $\mathbf W_k$. The Muon update rule is

$$
\begin{align*}
\mathbf{M}_k &= \mu \mathbf{M}_{k-1} + \mathbf{G}_k \\
\mathbf{W}_{k+1} &= \mathbf{W}_k - \eta \operatorname{polar}(\mathbf{M}_k)
\end{align*}
$$

where $\mu$ is the momentum coefficient, $\eta$ is the learning rate, and $\mathbf M_k$ is the momentum matrix (with $\mathbf M_0 := 0$).

In most ways, Muon resembles basic stochastic gradient descent (SGD) with momentum.
Its key innovation is using the $\operatorname{polar}$ operation, which is defined as follows:

> **Definition 1: Polar Decomposition**
>
> If $\mathbf X = \mathbf U \mathbf \Sigma \mathbf V^\top$ is the singular value decomposition (SVD) of a matrix, then $\operatorname{polar}(\mathbf X) = \mathbf U \mathbf V^\top$.

Since $\operatorname{polar}(\mathbf X)$ is expensive to compute exactly, Muon uses the Newton-Schulz method to approximate it. Newton-Schulz is an iterative method based on matrix polynomials. Beginning with $\mathbf X_0$, each iteration improves the approximation $\mathbf X_t \approx \operatorname{polar}(\mathbf X)$ according to the update rule

$$
\mathbf X_{t+1} = a_t \mathbf X_t + b_t \mathbf X_t \mathbf X_t^\top \mathbf X_t + c_t \left(\mathbf X_t \mathbf X_t^\top\right)^2 \mathbf X_t.
$$

We can interpret Newton-Schulz by understanding how it affects the singular value decomposition.

Let $\mathbf X_0 = \mathbf U \mathbf \Sigma \mathbf V^\top$ be the SVD. Recall that $\mathbf U$ and $\mathbf V$ have orthonormal columns, such that $\mathbf U^\top \mathbf U = \mathbf V^\top \mathbf V = \mathbf I$, and $\mathbf \Sigma$ is a diagonal matrix whose entries are called the singular values. Then

$$
\mathbf X_0 \mathbf X_0^\top \mathbf X_0 = \left(\mathbf U \mathbf \Sigma \mathbf V^\top\right) \left(\mathbf U \mathbf \Sigma \mathbf V^\top\right)^\top \left(\mathbf U \mathbf \Sigma \mathbf V^\top\right)
= \mathbf U \mathbf \Sigma \mathbf V^\top \mathbf V \mathbf \Sigma \mathbf U^\top \mathbf U \mathbf \Sigma \mathbf V^\top
= \mathbf U \mathbf \Sigma^3 \mathbf V^\top
$$

By the same logic,

$$
\mathbf X_1 = \mathbf U \left(a_1 \mathbf \Sigma + b_1 \mathbf \Sigma^3 + c_1 \mathbf \Sigma^5 \right) \mathbf V^\top = \mathbf U p_1(\mathbf \Sigma) \mathbf V^\top
$$

where we have defined the polynomial $p_1(x) = a_1 x + b_1 x^3 + c_1 x^5$. Since $\mathbf U$ and $\mathbf V$ have orthonormal columns and $p_1(\mathbf \Sigma)$ is diagonal, the right-hand side of this equation must be the SVD of $\mathbf X_1$! This shows that $\mathbf X_1$ shares the same singular vectors $\mathbf U$ and $\mathbf V$ as $\mathbf X_0$, and that its singular values are those of $\mathbf X_0$ transformed according to the polynomial $p_1$. By extension, $\mathbf X_T$ also shares the same singular vectors $\mathbf U$ and $\mathbf V$, and its singular values have been transformed according to the composition of polynomials $(p_T \circ \cdots \circ p_1)(\mathbf \Sigma)$. If $(p_T \circ \cdots \circ p_1)(x) \approx 1$ for all singular values, then $(p_T \circ \cdots \circ p_1)(\mathbf \Sigma) \approx \mathbf I$ and so $\mathbf X_T \approx \mathbf U \mathbf V^\top = \operatorname{polar}(\mathbf X_0)$.

All that remains is to find a sequence of odd polynomials for which $(p_T \circ \cdots \circ p_1)(x) \approx 1$ on the singular values. To make this easier, we first normalize the matrix $\mathbf X_0 = \mathbf X / \|\mathbf X\|_{\mathsf F}$. This ensures that the singular values of $\mathbf X_0$ lie in the interval $[0, 1]$. The developers of Muon identified a sequence of five degree-5 odd polynomials that approximate $1$ for every input on this interval $[0, 1]$, giving a decent approximation to $\operatorname{polar}(\mathbf X)$ for typical inputs $\mathbf X$ in just five iterations.[^muon]

A standard implementation of Newton-Schulz looks like this:

<a id="alg-standard-ns"></a>

> **Algorithm 1: Standard Newton-Schulz**
>
> Input: $\mathbf X \in \mathbb{R}^{n \times m}$, coefficients $\{(a_t, b_t, c_t)\}_{t=1}^5$
>
> 1. $\mathbf X \gets \mathbf X \,/\, (\lVert\mathbf X\rVert_{\mathsf F} + \epsilon)$&emsp;&emsp;&emsp;&emsp;// Normalize sing vals to $[0, 1]$.&emsp;$\epsilon = 10^{-7}$
> 1. $\mathbf X \gets \texttt{bfloat16}(\mathbf X)$&emsp;&emsp;&emsp;&emsp; // Cast to half precision for speed
> 1. If $m < n$:&emsp;&emsp;$\mathbf X \gets \mathbf X^\top$&emsp;&emsp;&emsp;// Trick to make $\mathbf X \mathbf X^\top$ cheaper
> 1. For $t = 1, \ldots, 5$:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; // Apply $p_t(\mathbf X)$
> 1. &emsp;&emsp; $\mathbf A \gets \mathbf X\mathbf X^\top$
> 1. &emsp;&emsp; $\mathbf B \gets b_t \mathbf A + c_t \mathbf A^2$
> 1. &emsp;&emsp; $\mathbf X \gets a_t \mathbf X + \mathbf B \mathbf X$
> 1. If $m < n$:&emsp;&emsp;$\mathbf X \gets \mathbf X^\top$&emsp;&emsp;&emsp;// Undo trick
> 1. Return $\mathbf X$

Successive work has sought to improve Muon in several ways. Most of these proposals modify Muon's update rule so as to reach the same loss in fewer training steps; however, they use the same Newton-Schulz routine described above. Some methods (e.g., Polar Express) do address Newton-Schulz by changing the sequence of polynomials or the normalization step.[^polar-express][^grishina] While they improve its approximation accuracy, they do not change its wall-clock runtime. The Dion optimizer[^dion] reduces the runtime in the distributed setting, when weights and gradients are sharded across different GPUs. It uses a low-rank approximation of Muon to reduce the communication cost and the dimension of $\mathbf X$, but each step still calls the standard Newton-Schulz routine.

In contrast, our work speeds up Newton-Schulz itself. Since Gram Newton-Schulz is mathematically identical to the standard version, it is compatible with nearly all varieties of Muon.

## Runtime of Standard Newton-Schulz

Let's analyze the runtime of Newton-Schulz in FLOPs to help us understand its performance bottlenekcs. We count only the cubic-time matrix multiplication operations, ignoring the lower-order scalar multiplications and matrix additions. For clarity, we let $T$ denote the number of iterations, remembering that within Muon, $T=5$.[^num-iters] We also assume without loss of generality that $n \leq m$ and define the aspect ratio $\alpha = m / n \geq 1$. Intuitively, $\alpha$ measures how rectangular the shape of the matrix is, with $\alpha = 1$ being square and $\alpha \gg 1$ being very rectangular.

Each iteration has three steps. Each step contains a single matrix multiplication costing, respectively,

- $\mathbf X \mathbf X^\top$:&emsp;$2mn^2$
- $\mathbf A^2$:&emsp;$2n^3$
- $\mathbf B \mathbf X$:&emsp;$2mn^2$

for a total cost of $T(4mn^2 + 2n^3) = 2T(2\alpha + 1)n^3$ FLOPs. When $T=5$, the cost is $(20\alpha + 10)n^3$ spread across 15 GEMMs. This analysis highlights two shortcomings of standard Newton-Schulz that inspire our work:

### Symmetric Matrix Multiplication

The matrices $\mathbf A = \mathbf X \mathbf X^\top$ and $\mathbf B = b_t \mathbf A + c_t \mathbf A^2$ computed at each iteration of Newton-Schulz are symmetric by definition. This fact can be exploited to reduce the cost of Newton-Schulz. Instead of calling general matrix multiplication routines as typical implementations of Newton-Schulz do, we can compute the lower triangular part of these matrices in the usual way and then simply copy the results to the upper triangular part. This technique halves the cost of computing $\mathbf X \mathbf X^\top$ and $\mathbf A^2$, giving an overall total of $T(3\alpha + 1)n^3$ FLOPs. We describe our custom CuTeDSL kernels that implement this technique [below](#symmetric-gemm-kernels-in-cutedsl).

<a id="dependence-on-alpha"></a>

### Dependence on $\alpha$

Even using symmetric GEMMs, Newton-Schulz's runtime is dominated by the large rectangular matrix multiplications needed to compute $\mathbf A$ and $\mathbf X$, which together cost $3\alpha n^3$ FLOPs per iteration. A typical implementation with $T=5$ requires 10 of these expensive rectangular multiplications.

This strong dependence on $\alpha$ is unfortunate. Most of the weight matrices in transformer architectures are rectangular, including the MLP weights, MoE weights, and attention projection weights when using GQA or MLA.[^embeddings] Furthermore, we observe that the latest MoE architectures are trending towards finer-grained, sparser experts, meaning that the aspect ratios of their hidden dimensions to intermediate dimensions are increasing as well.[^kimi][^sonicmoe][^qwen][^gpt-oss]

Thus, at large scales, pretraining time would benefit greatly from an algorithm that uses fewer rectangular multiplications and more small symmetric ones.

# Gram Newton-Schulz

<!-- As described [above](#dependence-on-alpha), the cost of standard Newton-Schulz is dominated by its ten rectangular matrix multiplications.  -->

We now show how to rewrite Newton-Schulz to reduce the number of expensive rectangular matrix multiplications by iterating on the small, square, symmetric Gram matrix $\mathbf X \mathbf X^\top$ instead of the rectangular input matrix $\mathbf X$. The output of this algorithm is mathematically identical to that of standard Newton-Schulz, but it is significantly cheaper to compute.

At a high level, our strategy is based on the following formula. If $\mathbf X \in \mathbb{R}^{n \times m}$ with $n \leq m$, then $\mathrm{polar}(\mathbf X) = (\mathbf X \mathbf X^\top)^{-1/2} \mathbf X$. Rather than use an iterative method to approximate $\mathbf X_T \approx \mathrm{polar}(\mathbf X)$ directly, we instead

1. Compute the $n \times n$ Gram matrix $\mathbf X \mathbf X^\top$
2. Use an iterative method to approximate $\mathbf Q_T \approx (\mathbf X \mathbf X^\top)^{-1/2}$
3. Compute $\mathbf Q_T \mathbf X$

Step 2—which comprises almost all of the algorithm's wall clock runtime and FLOP cost—works entirely with small $n \times n$ symmetric matrices. This version uses just two rectangular matrix multiplications: $\mathbf X \mathbf X^\top$ in the beginning, and $\mathbf Q_T \mathbf X$ at the end. It also synergizes well with our symmetric GEMM kernels. Because we now use more symmetric multiplications, our kernels provide an even greater speedup than before. Since this method works on the $n \times n$ Gram matrix of $\mathbf X$, we call it "Gram Newton-Schulz".

How can we turn an iterative polynomial method $(p_T \circ \cdots \circ p_1)(\mathbf X) \approx \operatorname{polar}(\mathbf X)$ like Newton-Schulz into an iterative polynomial method for approximating $\mathbf Y \mapsto \mathbf Y^{-1/2}$? Recall that each $p_t$ is an odd polynomial $p(x) = ax + bx^3 + cx^5$. Any odd polynomial can be rewritten in the form $p(x) = xh(x^2)$, where $h$ is a lower-degree polynomial with the same coefficients, like $h(x) = a + bx + cx^2$. Intuitively, if $p(x) \approx 1$, then $h(y) = p(y^{1/2})y^{-1/2} \approx y^{-1/2}$, so the Newton-Schulz polynomials implicitly provide a way to approximate inverse square roots.

Formally, Gram Newton-Schulz is based on the following theorem. In effect, it shows how to compute $\mathbf X_T$ from $\mathbf X_0$ without ever constructing the intermediate values $\mathbf X_1, \ldots, \mathbf X_{T-1}$:

> **Theorem 1:**
>
> If $p_t(x) = xh_t(x^2)$ for all $t \in \{1, \ldots, T\}$, then $(p_T \circ \cdots \circ p_1)(x) = q_T x$, where $q_T$ is defined by the iteration $r_0 = x^2$, $q_0 = 1$, and
>
> $$z_t = h_t(r_{t-1})$$
>
> $$r_t = r_{t-1}z_t^2$$
>
> $$q_t = q_{t-1}z_t$$
>
> for all $t \in \{1, \ldots, T\}$.

_Proof_. Define $x_0 = x$ and $x_t = p_t(x_{t-1})$ for $t \in \{1, \ldots, T\}$. We will show by induction that $r_t = x_t^2$ and $q_t = x_t / x_0$ for all $t$. The base case $t = 0$ holds by the definition $r_0 = x^2, q_0 = 1$. Now assume the hypothesis holds for $t-1$. By assumption,

$$x_t = p_t(x_{t-1}) = x_{t-1} h_t(x_{t-1}^2)$$

Observe that $h_t(x_{t-1}^2) = h_t(r_{t-1}) = z_t$, so $x_t = x_{t-1} z_t$. Squaring both sides,

$$x_t^2 = x_{t-1}^2 z_t^2 = r_{t-1} z_t^2 = r_t$$

If we instead divide both sides by $x_0$,

$$\frac{x_t}{x_0} = \frac{x_{t-1}}{x_0}z_t = q_{t-1} z_t = q_t$$

Thus, the hypothesis holds for $t$ as well. Finally, observe that $(p_T \circ \cdots \circ p_1)(x) = x_T = q_T x_0$.$\blacksquare$

Note that, as an immediate corollary of the proof, $q_t = x_t / x_0 \to 1/x_0 = \left(x_0^2\right)^{-1/2} = r_0^{-1/2}$. In effect, this shows that $\mathbf Q_T \to (\mathbf X \mathbf X^\top)^{-1/2}$.

To obtain our initial version of Gram Newton-Schulz, we simply lift the iteration from Theorem 1 to matrices. As in standard Newton-Schulz, each matrix operation preserves singular vectors. Therefore, each singular value of $\mathbf R_t$, $\mathbf Q_t$, and $\mathbf Z_t$ evolves independently of the others according to the scalar iteration described above. Note that while this algorithm is mathematically equivalent to standard Newton-Schulz, it is not yet practical due to numerical instability. The only difference between [our proposed method](#alg-stable-gram-ns) and this naive version is the presence of what we call a "restart" at the beginning of iteration 3 of the loop. We will motivate this modification soon.

<a id="alg-naive-gram-ns"></a>

> **Algorithm 2: Naive Gram Newton-Schulz**
>
> Input: $\mathbf X \in \mathbb{R}^{n \times m}$ with $n \leq m$, coefficients $\{(a_t, b_t, c_t)\}_{t=1}^5$
>
> 1. $\mathbf X \gets \mathbf X \,/\, (\lVert\mathbf X\rVert_{\mathsf F} + \epsilon)$&emsp;&emsp;&emsp;&emsp;// Normalize sing vals to $[0, 1]$.&emsp;$\epsilon = 10^{-7}$
> 1. $\mathbf R_0 = \mathbf X \mathbf X^\top$
> 1. $\mathbf Q_0 = \mathbf I$
> 1. For $t = 1, \ldots, 5$:
> 1. &emsp;&emsp; $\mathbf Z_t \gets a_t\mathbf I + b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2$&emsp;&emsp;&emsp; // Apply $h_t(\mathbf R_{t-1})$
> 1. &emsp;&emsp; $\mathbf Q_t \gets \mathbf Q_{t-1} \mathbf Z_t$
> 1. &emsp;&emsp; $\mathbf R_t \gets \mathbf Z_t \mathbf R_{t-1} \mathbf Z_t$
> 1. Return $\mathbf Q_5 \mathbf X$

Gram Newton-Schulz is closely akin to a method proposed in Appendix F of the Polar Express paper.[^polar-express] Both form the Gram matrix and transform standard Newton-Schulz into an iteration on $n \times n$ matrices. Both aim to reduce the FLOP cost of Newton-Schulz. However, our work supersedes the proposal from Appendix F in several ways. First, the precise formulas of Gram Newton-Schulz are different, and we believe more stable. Second, we use symmetric matrix multiplication kernels; the opportunity to use these kernels more is an essential advantage of Gram Newton-Schulz not studied previously, and using symmetric matrix multiplication can have subtly different numerical properties in half-precision that require more careful stability strategies. Third, we undertake a thorough stability analysis and provide practical recommendations that allow Gram Newton-Schulz to be used in practice with minimal ad-hoc hyperparameter tuning.

## Runtime of Naive Gram Newton-Schulz

Let's measure the FLOP count of this new algorithm to see how its runtime improves on standard Newton-Schulz. There are four matrix multiplications per iteration. If we use our symmetric GEMM kernel, these cost:

- $\mathbf R_{t-1}^2$:&emsp;$n^3$
- $\mathbf Q_{t-1} \mathbf X_t$:&emsp;$n^3$
- $\mathbf Z_t \mathbf R_{t-1} \mathbf Z_t$:&emsp;$n^3 + n^3$

The initialization and output steps cost:

- $\mathbf X \mathbf X^\top$:&emsp;$mn^2$
- $\mathbf Q_5 \mathbf X$:&emsp;$2mn^2$&emsp;(not symmetric)

Lastly, computing $\mathbf Q_1 = \mathbf Z_1$ is free since $\mathbf Q_0 = \mathbf I$, and we do not need to compute $\mathbf R_5$:

- Skipping $\mathbf Q_0 \mathbf Z_1,\,\mathbf Z_5 \mathbf R_4 \mathbf Z_5$:&emsp;$-3n^3$

Thus, the total FLOP count is $T\cdot4n^3 + 3mn^2 - 3n^3 = (4T + 3\alpha - 3)n^3$ for general $T$, or $(17 + 3\alpha)n^3$ across 19 GEMMs for $T=5$. Compare this to standard Newton-Schulz's $T(3\alpha + 1)n^3$ FLOPs when using symmetric GEMMs. When $\alpha = 1$, they are equal. When $\alpha > 1$, Gram Newton-Schulz is cheaper. For a typical Muon application ($T=5, \alpha = 4$), **it saves 55% of the FLOPs** used by standard Newton-Schulz with symmetric GEMMs, **or 68%** compared to a typical implementation without symmetric GEMMs.

In practice, when $\alpha=1$, we fall back to [standard Newton-Schulz with our symmetric GEMMs](#kernel-optimizations-for-standard-newton-schulz), since it launches fewer GEMMs and will have a faster wall clock time.

# Instability of Naive Gram Newton-Schulz

Let's try training a transformer LLM with Muon using Naive Gram Newton-Schulz:

![llama_430_no_reset](https://hackmd.io/_uploads/SJiRa6W9Wl.png)
_Figure 2: Naive Gram Newton-Schulz on Llama-430M._

This is no good. Not only do we get loss spikes, but eventually, the output of Gram Newton-Schulz is full of Infs! While Gram Newton-Schulz is mathematically equivalent to standard Newton-Schulz in exact arithmetic, it behaves differently in finite precision, especially in half precision.

We will now pause to explain the source of this instability in detail and motivate our solution. Readers not concerned with these technical details can [skip ahead](#stabilized-gram-newton-schulz) to see the stabilized method. Code for running these stability experiments and generating the figures is available [here](https://github.com/NoahAmsel/PolarExpress/blob/appF-stability/gram_newton_schulz_stability.ipynb).

## Tracking Eigenvalues of Intermediate Matrices

We can understand how matrices evolve and why they diverge by studying their eigenvalues and singular values. Recall that the entries of any matrix are upper bounded by its largest singular value, so if we control the singular values, we will prevent blowups.

If $\mathbf X = \mathbf U \mathbf \Sigma \mathbf V^\top$ is the SVD of the input matrix, then intermediate matrices of Algorithm 2 ($\mathbf R_t$, $\mathbf Q_t$, $\mathbf Z_t$) are square symmetric with eigenvectors $\mathbf V$. In exact arithmetic, $\mathbf U^\top \mathbf R_t \mathbf U$ is a diagonal matrix containing $\mathbf R_t$'s eigenvalues, each of which corresponds to a singular value of $\mathbf X$. We can therefore plot the eigenvalues of $\mathbf R_t$ and $\mathbf Q_t$ against the corresponding singular values of $\mathbf X$ to track how each evolves according to the polynomial update rules—or diverges from them.

To see how things should look, let's start by running Naive Gram Newton-Schulz in full `float64` precision for $10$ steps. We will use a synthetic input—a $128 \times 512$ matrix with an exponentially decaying spectrum. In order to make our plots more readable, with smooth monotonic curves, the experiments in this section use the coefficients $(a_t, b_t, c_t) = (\tfrac{15}8, \tfrac{10}8, \tfrac38)$ at every iteration. The numerical behavior we observe will generalize to other coefficients; those used in practice (like You Jiacheng's or Polar Express) will blow up at an even earlier iteration, matching the behavior we observe in training.[^you][^polar-express] Even though our method does not need to compute the intermediate matrices $\mathbf X_1, \ldots, \mathbf X_{T-1}$, we do so here for demonstration using the formula $\mathbf X_t = \mathbf Q_t \mathbf X_0$, where we label the input $\mathbf X_0$ for clarity.

<!-- <iframe src="../PolarExpress/html_plots/f64_diagnostics.html" width="100%" height=300px></iframe> <!--style="border: none;"-->
<!-- ![f64_diagnostics](https://hackmd.io/_uploads/B16nR9RKbe.gif) -->

![f64_diagnostics](https://hackmd.io/_uploads/Bkm4pJ_iZg.gif)
_Figure 3: Evolution of eigenvalues of $\mathbf R_t$, $\mathbf Q_t$, and $\mathbf X_t$ in Float64 in Naive Gram Newton-Schulz with coefficients $(\tfrac{15}8, \tfrac{10}8, \tfrac38)$._

Initially, we have $r_0 = x_0^2$, and $q_0 = 1$. As the algorithm proceeds, we know that $x_t \to 1$, so we expect $r_t \to 1$ and $q_t = x_t / x_0 \to 1/x_0 = r_0^{-1/2}$ as per Theorem 1. Note that if $x_0$ is close to 1, the method converges quickly, while if $x_0$ is close to zero, it converges slowly. After 10 iterations, the spectrum of $\mathbf X_t$ is visually indistinguishable from $1$, as expected.

Now let's repeat the experiment using `bfloat16` instead of `float64` arithmetic:

<!-- <iframe src="../PolarExpress/html_plots/f16_diagnostics.html" width="100%" height=300px></iframe> <!--style="border: none;"-->
<!-- ![f16_diagnostics](https://hackmd.io/_uploads/ByO605RYWe.gif) -->

![f16_diagnostics](https://hackmd.io/_uploads/B1PA2ydo-e.gif)
_Figure 4: Evolution of eigenvalues of $\mathbf R_t$, $\mathbf Q_t$, and $\mathbf X_t$ in BFloat16 in Naive Gram Newton-Schulz with coefficients $(\tfrac{15}8, \tfrac{10}8, \tfrac38)$._

The first few iterations proceed as before. However, by step 7, we see unexpected behavior in the spectrum of $\mathbf X_t$. The singular values that began near $0$ suddenly jump up above 1, instead of converging to 1 from below. By step 8, the algorithm is returning complete junk. What happened?

We identify two key causes of divergence:

1. Spurious negative eigenvalues of the Gram matrix $\mathbf X \mathbf X^\top$
2. Eigenvector drift

## Spurious Negative Eigenvalues

The main cause of divergence is the presence of negative eigenvalues in the Gram matrix due to half-precision arithmetic. These negative eigenvalues blow up after a few iterations of Gram Newton-Schulz.

If you look closely, you can see that the trouble begins in $\mathbf R_t$. By construction, $r_t = x_t^2 \geq 0$, so in exact arithmetic, $\mathbf R_t$ should be a positive semidefinite matrix.
However, when using `bfloat16`, our plots show that $\mathbf R_t$ has negative eigenvalues!
Because $\mathbf X_0$ is numerically low rank, $\mathbf R_0 = \mathbf X_0 \mathbf X_0^\top$ has eigenvalues that are _numerically_ equal to zero, and in `bfloat16`, a number like $-10^{-5}$ is numerically equal to zero.
Let's transform the y-axis to emphasize values close to zero and replot this:

<!-- <iframe src="../PolarExpress/html_plots/f16_diagnostics_zoomed.html" width="100%" height=300px></iframe> <!--style="border: none;"-->
<!-- ![f16_diagnostics_zoomed](https://hackmd.io/_uploads/BJkJysRFWg.gif) -->

![f16_diagnostics_zoomed](https://hackmd.io/_uploads/ry9ZpJdjWg.gif)
_Figure 5: Evolution of eigenvalues of $\mathbf R_t$, $\mathbf Q_t$, and $\mathbf X_t$ in BFloat16, with y-axis centered around $0$._

Now we see that from the very beginning, $\mathbf R_0$ has tiny negative eigenvalues introduced in the first computation $\mathbf X_0 \mathbf X_0^\top$. Later computations can introduce additional negative eigenvalues to $\mathbf R_t$ too. These eigenvalues represent nothing about the original problem, they are just an artifact of floating point arithmetic. Therefore, we call them "spurious eigenvalues".

These spurious negative eigenvalues start small, but the plot shows that their magnitude grows quickly. Let's understand mathematically why this happens. Recall the update rule:
$$r_t = r_{t-1} z_t^2 = r_{t-1} h_t(r_{t-1})^2$$
If we now substitute $h_t(x) = \tfrac{15}8 - \tfrac{10}8 x + \tfrac38 x^2$ and plot this update rule, we can see the problem:

<!-- <img src="https://hackmd.io/_uploads/SJhVlkr5Wx.svg" width="500"> -->

![r_update_map](https://hackmd.io/_uploads/HkOvAkujWe.svg)

_Figure 6: Negative values of $r_t$ diverge towards negative infinity._

As the plot shows, $r_t < \left(\tfrac{15}{8}\right)^2 r_{t-1}$. Thus, if $r_0 < 0$, the magnitude of the spurious eigenvalues grows exponentially! This sets off a chain reaction. As $r_t \to -\infty$, we get $z_t \to \infty$. This causes $q_t \to \infty$ and therefore also $x_t \to \infty$. This problem cannot be fixed by choosing different polynomials. Conceptually, in the main loop, we are attempting to compute the inverse square root of a negative number. It cannot help but diverge.

To show that the spurious negative eigenvalues of $\mathbf R_0$ are enough to cause this catastrophic failure, let's rerun the method with every operation in `float64` precision, except that we will convert $\mathbf R_0$ from `float64` to `bfloat16` and then back to `float64` to induce a little floating point error. As you can see, even this causes a blowup.

<!-- <iframe src="../PolarExpress/html_plots/posthoc_f16_diagnostics.html" width="100%" height=300px></iframe> <!--style="border: none;"-->
<!-- ![posthoc_f16_diagnostics](https://hackmd.io/_uploads/SkLXkjRK-e.gif) -->

![posthoc_f16_diagnostics](https://hackmd.io/_uploads/HyEH0k_iZx.gif)
_Figure 7: Evolution of eigenvalues of $\mathbf R_t$, $\mathbf Q_t$, and $\mathbf X_t$ when all operations use Float64 except $\mathbf R_0 = \mathbf X \mathbf X^\top$._

Recall that the average magnitude of a matrix's entries (root mean squared) is proportional to its Frobenius norm, which is larger than the largest singular value. Therefore, as $\mathbf Q_t$'s largest singular value blows up, its entries do too.

## Eigenvector Drift

Spurious negative eigenvalues are not the only source of instability. If we take as input a matrix that excludes small singular values (i.e., all $\geq 0.017$), then we do not observe any negative eigenvalues in $\mathbf R_t$, but we still see a moderate blow up in $\mathbf X_t$. The culprit seems to be eigenvector drift.

In exact arithmetic, the eigenvectors of all intermediate matrices match $\mathbf U$, the left singular vectors of $\mathbf X_0$, but in finite precision they do not. This effect can be measured by observing how far $\mathbf U^\top \mathbf R_t \mathbf U$, $\mathbf U^\top \mathbf Q_t \mathbf U$, and $\mathbf U^\top \mathbf X_t \mathbf V$ are from being diagonal matrices. The plot below shows that after several iterations, the eigenvectors of $\mathbf Q_t$ and $\mathbf X_t$ have drifted significantly. At the same time, we see the eigen*values* of $\mathbf Q_t$ (and by extension, those of $\mathbf X_t$) diverge from where they should be in exact arithmetic. The growing eigenvalues of $\mathbf Q_t$ seem to spill into one another. The strength of this effect is less consistent than that of negative eigenvalues, but it is still harmful.

<!-- ![Evolution of eigenvalues of R, Q, and X for a well-conditioned spectrum showing eigenvector drift](../PolarExpress/html_plots/easy_spectrum_diagnostics.svg) -->

![easy_spectrum_diagnostics](https://hackmd.io/_uploads/HyuQYBm5-x.svg)
_Figure 8: As the eigenvectors drift (left) the spectral norms of $\mathbf R_t$, $\mathbf Q_t$, and $\mathbf X_t$ diverge._

## Stabilizing Gram Newton-Schulz by Restarting

If we run Gram Newton-Schulz for more than a few iterations, the spurious negative eigenvalues grow unmanageably large and $\mathbf Q_t$ blows up.
Our solution is simple: run Gram Newton-Schulz for only a few iterations.
Rather than using Gram Newton-Schulz to compute $\mathbf X_T$ directly, we use it to compute, say, $\mathbf X_5$ in a stable manner for coefficients $(\tfrac{15}8, \tfrac{10}8, \tfrac38)$.
While $\mathbf X_5$ is not a good approximation to $\lim_{T \to \infty} \mathbf X_T = \mathrm{polar}(\mathbf X_0)$, we are closer than when we started.
Now we can apply Gram Newton-Schulz a second time on the input $\mathbf X_5$ to compute $\mathbf X_{10}$ stably.
We can repeat this over and over to reach whatever $T$ we like.
This restarting technique sacrifices some of the performance gains of Gram Newton-Schulz, but it still offers a significant speedup over standard Newton-Schulz.

Below we plot the results of this method on the same test matrix used above.
As before, we compute $\mathbf X_t$ for all $t$ for diagnostic purposes, though the algorithm computes only $\mathbf X_5, \mathbf X_{10}, \ldots, \mathbf X_{30}$.
Looking closely, you can see that $\mathbf R_t$ develops some negative eigenvalues, but unlike before, the growth of these eigenvalues is controlled.
Each time we restart, we re-initialize $\mathbf R_t = \mathbf X_t \mathbf X_t^\top$, eliminating any negative eigenvalues of large magnitude.
As you can see, at iteration $5, 10, 20, 25$, and $30$, $\mathbf Q_t$ resets to the identity.
Therefore, the eigenvalues of $\mathbf Q_t$ never grow beyond $\approx 12$, despite the negative eigenvalues in $\mathbf R_t$.
Since the eigenvalues of $\mathbf Q_t$ remain controlled, those of $\mathbf X_t = \mathbf Q_t \mathbf X_{t-5}$ stay strictly smaller than $1$.

<!-- <iframe src="../PolarExpress/html_plots/restart5_diagnostics.html" width="100%" height=300px></iframe> <!--style="border: none;"-->
<!-- ![restart5_diagnostics](https://hackmd.io/_uploads/Hk2NJjCtbg.gif) -->

![restart5_diagnostics](https://hackmd.io/_uploads/BJ5oC1_oZl.gif)
_Figure 9: Restarting prevents the divergence of $\mathbf R_t$._

Restarting also helps control eigenvector drift. We repeat the experiment from above on the same matrix (with all singular values $> 0.017$), but now with a restart after step 5. We observe that the diagonalization error remains $\leq 0.05$ for all matrices, and the maximum eigenvalues now align closely with their values in exact arithmetic. Note that we always measure eigenvector drift relative to the original input $\mathbf X_0$, not the restarted $\mathbf X_5$.

<!-- ![easy_spectrum_restart2_diagnostics](../PolarExpress/html_plots/easy_spectrum_restart2_diagnostics.svg) -->

![easy_spectrum_restart2_diagnostics](https://hackmd.io/_uploads/rkmItSQ5Wx.svg)
_Figure 10: Restarting prevents eigenvector drift._

### When to Restart: Polar Express Coefficients for Muon

At what iteration should we restart?
To avoid numerical trouble, we need to control the magnitude of $\mathbf Q_t$, even when $\mathbf R_0$ has spurious negative eigenvalues. (Because each $q_r \geq 1$, this is equivalent to controlling the condition number of $\mathbf Q_t$.)
So long as each eigenvalue of $\mathbf Q_t$ remains smaller than the inverse of the corresponding eigenvalue from $\mathbf X$, then $\mathbf X_t = \mathbf Q_t \mathbf X$ will have eigenvalues $\leq 1$.

The growth of $\mathbf Q_t$ in turn depends on the size of the spurious negative eigenvalues and the specific sequence of polynomials we use.
Furthermore, since the polynomial $p_t$ changes at each iteration, it may not be ideal to restart at regular intervals.
Instead, we can choose when to restart adaptively, depending on the specific sequence of polynomials we have applied since the previous restart.

For the application to Muon, let's now switch over to using five iterations of the Polar Express polynomials, which are defined as follows:

| $t$ | $a$      | $b$        | $c$       |
| --- | -------- | ---------- | --------- |
| 1   | 8.123737 | -22.232240 | 16.373715 |
| 2   | 4.026529 | -2.776323  | 0.514551  |
| 3   | 3.870284 | -2.739120  | 0.520999  |
| 4   | 3.253351 | -2.343223  | 0.481420  |
| 5   | 2.300652 | -1.668904  | 0.418807  |

In the example above, we observe that the most negative spurious eigenvalue of $\mathbf R_0$ is about $-4 \cdot 10^{-4}$.
Using the scalar analogue Gram Newton-Schulz, let's simulate how the eigenvalues of $\mathbf Q_t$ evolve in full precision when $\mathbf R_0$ has eigenvalues in the range $[-4\cdot 10^{-4}, 1]$.
With no restart, they blow up:

<!-- ![polar_no_restart_growth](https://hackmd.io/_uploads/B1VEyDGc-x.svg) -->

![polar_no_restart_growth (1)](https://hackmd.io/_uploads/BJjkC1dsZl.svg)

_Figure 11: Min/max eigenvalue of $\mathbf R_t$ and $\mathbf Q_t$ without restarts. $\mathbf R_0$ starts with a negative eigenvalue as low as $-4 \times 10^{-4}$._

Now let's repeat the experiment with a restarted version of the algorithm.
To obtain a good balance of stability and speed, let's limit ourselves to a single restart.
When should this restart take place?
We'll try all possibilities.
As above, we begin with eigenvalues in the range $[-4 \cdot 10^{-4}, 1]$.
Every time we restart and form $\mathbf R = \mathbf X\mathbf X^\top$, we subtract $4 \times 10^{-4}\mathbf I$ to simulate a potentially dangerous shift in the eigenvalues due to floating point error. As you can see, restarting after the second iteration ensures that the eigenvalues of $\mathbf R_t$ stay well above $-0.4$ and that the condition number of $\mathbf Q_t$ stay below $\approx 100$ for all iterations, much better than the other options.

<!-- ![Maximum size of Q when restarting at different iterations](../PolarExpress/html_plots/polar1restart_results.svg) -->
<!-- ![polar1restart_results](https://hackmd.io/_uploads/r18ja8fc-g.svg) -->

![polar1restart_results (1)](https://hackmd.io/_uploads/Hk4qTyuiZx.svg)

_Figure 12: Minimum eigenvalue of $\mathbf R_t$ and condition number of $\mathbf Q_t$ if restart is placed after iteration $1$, $2$, $3$, or $4$. $\mathbf R_0$ starts with a negative eigenvalue as low as $-4 \times 10^{-4}$. Restarting after iteration $2$ provides the best bound on $\mathbf Q_t$._

Note that restarting works precisely because we reset the minimum negative eigenvalue of $\mathbf R_t$, which in turn tightens the bound on $\mathbf Q_t$'s eigenvalues.
In [our repo](https://github.com/Dao-AILab/gram-newton-schulz), we provide a utility that performs this analysis. For any given Newton-Schulz coefficients and any number of restarts, it identifies the best iterations at which to restart.

Now let's run the full method with a restart after the second iteration on our test matrix.
Now it converges! All singular values of $\mathbf X_t$ approach 1.

<!-- <iframe src="../PolarExpress/html_plots/final_diagnostics.html" width="100%" height=300px></iframe> <!--style="border: none;"-->
<!-- ![final_diagnostics](https://hackmd.io/_uploads/H1GtkoAY-l.gif) -->

![final_diagnostics](https://hackmd.io/_uploads/HyJU6JOiWl.gif)
_Figure 13: Restarting after $2$ iterations creates a stable polar decomposition of our test matrix with Polar Express coefficients._

## Further Precautions

While restarting greatly improves stability, it is not absolutely foolproof.
The usual numerical snags for Newton-Schulz still apply.

For example, most choices of Newton-Schulz polynomials are designed to converge only when $\lVert\mathbf X_0\rVert \leq 1$; any singular values larger than $1$ may diverge rapidly.

<!-- ![Behavior of Newton-Schulz with sigma_X significantly above 1](figs/X_final_unbounded.png) -->

![X_final_unbounded (4)](https://hackmd.io/_uploads/rkkBqyBcZl.png)
_Figure 14: Theoretical behavior of both standard and Gram Newton-Schulz on $\sigma_{X*0}$ slightly above $1$ using Polar Express coefficients.*

Even with a properly normalized input, perturbed singular values of $\mathbf X_0$ slightly greater than $1$ can develop due to numerical error.
This problem affects standard Newton-Schulz as well, so the Polar Express polynomials are typically adjusted according to the formula
$\tilde p_t(x) = p_t(x / 1.02)$.
This ensures convergence even for singular values as large as $1.02$.
When using Gram Newton-Schulz, roundoff errors like this can worsen due to computations like $\mathbf X\mathbf X^\top$, which do not have built-in safety factors; however, we have never seen this happen when using our recommended setup (`float16` arithmetic with restarting after $2$ iterations). It is generally wise to be extra conservative in the choice of safety factor, for instance, by replacing $1.02$ with $1.05$.

### Float16 vs BFloat16 in Newton-Schulz

In addition, we argue for using `float16` instead of `bfloat16` to implement Newton-Schulz.
Compared to `bfloat16`, `float16` can only represent values from a narrower range, but it has greater precision within that range.
For our purposes, the range of `float16` (roughly $6.1\cdot 10^{-5}$ to $6.5 \cdot 10^4$) suffices because the magnitudes of our matrices are controlled to lie near 1.
And in some cases, we can benefit from using `float16` to reduce numerical errors.

On certain test matrices, we see more accurate $\operatorname{polar}(\mathbf X)$ approximations with `float16`, but in practice, we have not found a case where the pretraining loss is meaningfully different between `float16` and `bfloat16`. Still, we default to `float16`.

### Computing Matrix Quadratics

A key step in Gram Newton-Schulz is computing the matrix quadratic $\mathbf Z_t \gets a_t\mathbf I + b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2$. PyTorch implementations of Newton-Schulz typically do not assemble such polynomials explicitly; to compute $\mathbf X(\mathbf a_t \mathbf I + b_t \mathbf A + c_t \mathbf A^2)$, they partly distribute $\mathbf X$ and use two calls to `torch.baddbmm`, which dispatches to cuBLAS GEMM, as follows

> 1. &emsp;&emsp; $\mathbf B \gets b_t \mathbf A + c_t \mathbf A^2$
> 1. &emsp;&emsp; $\mathbf X \gets a_t \mathbf X + \mathbf B \mathbf X$

Our symmetric GEMM kernel is capable of fusing these matrix quadratics into a single step. In particular, it can fuse the addition of $\gamma \mathbf I$ by adding $\gamma$ to all diagonal entries of the output when they are at the register level. This optimization completely obviates any I/O operations needed for the $\gamma I$ addition, typically outspeeding `gemm_symmetric(A, B, C, alpha, beta) + gamma * I`, which would require loading $\mathbf I$ from general memory to shared memory to registers. Once $\mathbf Z_t$ is assembled, Gram Newton-Schulz can use it in three subsequent multiplications.

<!-- This also allows the remaining $3$ symmetric GEMMs for $Q$, $RZ$, and $R$ to remain in the form of $AB$, without any epilogue addition. -->

However, our tests show that adding $\gamma \mathbf I$ explicitly can be less stable than handling it implicitly in some corner cases. If we stress-test our method by ignoring some of our own advice—restarting after three iterations instead of two and using a Polar Express safety factor of $1.02$ instead of $1.05$, and computing the quadratic with $a_t \mathbf I$ explicitly—then we observe instability. This instability disappears if we use non-symmetric GEMMs (either from `torch` or Quack) instead of our symmetric kernels. We conclude that our fused quadratic kernel can hurt stability in this setting. Since we reproduce this issue by forcing symmetry after calling standard `torch` GEMMs, we know this is not a kernel bug, but a numerical property.

We believe this effect can be explained as follows.
While the fused kernel computes $a_t\mathbf I + b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2$ in `float32` arithmetic under the hood, the result $\mathbf Z_t$ is rounded back down to `float16` at the end of the GEMM. Future computations like $\mathbf Q_t \mathbf Z_t$ suffer from this loss of precision in $a_t$. In contrast, if the $a_t \mathbf I$ term is handled implicitly, all arithmetic involving $a_t$ takes place in `float32`. Therefore, it is more accurate to compute $a_t \mathbf Q_t + \mathbf Q_t\left(b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2\right)$ than $\mathbf Q_t\left(a_t\mathbf I + b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2\right)$.

We reiterate that in all our experiments, this instability can be avoided entirely by restarting correctly or by using a higher safety factor of $1.05$. Out of an abundance of caution, we rearrange the arithmetic of [Naive Gram Newton-Schulz](#alg-naive-gram-ns) to avoid adding $\mathbf I$ explicitly. That is, we change

> 1. &emsp;&emsp; $\mathbf Z_t \gets a_t\mathbf I + b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2$&emsp;&emsp;&emsp; // Apply $h_t(\mathbf R_{t-1})$
> 2. &emsp;&emsp; $\mathbf Q_t \gets \mathbf Q_{t-1} \mathbf Z_t$
> 3. &emsp;&emsp; $$(\mathbf{RZ})_t \gets \mathbf R_{t-1} \mathbf Z_t$$
> 4. &emsp;&emsp; $$\mathbf R_t \gets \mathbf Z_t (\mathbf{RZ})_t$$

to

> 1. &emsp;&emsp; $\mathbf Z_t \gets b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2$
> 2. &emsp;&emsp; $\mathbf Q_t \gets \mathbf Q_{t-1} \mathbf Z_t +  a_t\mathbf Q_{t-1}$
> 3. &emsp;&emsp; $$(\mathbf{RZ})_t \gets \mathbf R_{t-1} \mathbf Z_t + a_t\mathbf R_{t-1}$$
> 4. &emsp;&emsp; $$\mathbf R_t \gets \mathbf Z_t (\mathbf{RZ})_t + a_t(\mathbf{RZ})_t$$

This change fixes all collected examples in which symmetric GEMMs were less stable than non-symmetric GEMMs.

## Takeaways on Stability

While Gram Newton-Schulz is fundamentally more unstable than standard Newton-Schulz, it can be coaxed into behaving equally stably with the proper care.
The understanding gleaned from these experiments gives us the confidence to use Gram Newton-Schulz in practice.
However, users should be willing to monitor the method, and if they find instability, to adjust the hyperparameters (e.g., $1.02 \to 1.05$ above).
For example, a second restart may be required if using a particularly sensitive set of coefficients or if running more than five iterations with Polar Express polynomials.

In the application of Muon for pretraining, we do not need very high polar decomposition accuracy, and our experiments below show that Muon with Gram Newton-Schulz yields effectively identical results to Muon with standard Newton-Schulz in terms of training quality.
However, when high accuracy is desired, the usual warnings about forming the Gram matrix apply.
Since forming $\mathbf X \mathbf X^\top$ immediately squares the condition number, Gram Newton-Schulz may not be appropriate in these cases.

# Stabilized Gram Newton-Schulz

We now present our complete algorithm, which enjoys the speed of naive Gram Newton-Schulz while remaining numerically stable.
We use five iterations of Newton-Schulz with degree-5 polynomials (such as Polar Express).
We use `float16` arithmetic, and we "restart" after the first two iterations by setting $\mathbf X \gets \mathbf Q_2 \mathbf X$ and reinitializing $\mathbf R_2$ and $\mathbf Q_2$.
As in standard Newton-Schulz, we write the logic of our routine assuming that $\mathbf X$ has more columns than rows. If this is not the case, we simply run on $\mathbf X^\top$ and output the transpose of the result.

<a id="alg-stable-gram-ns"></a>

> **Algorithm 3: Stabilized Gram Newton-Schulz**
>
> Input: $\mathbf X \in \mathbb{R}^{n \times m}$ with $n \leq m$, coefficients $\{(a_t, b_t, c_t)\}_{t=1}^5$
>
> 1. $\mathbf X \gets \mathbf X \,/\, (\lVert\mathbf X\rVert_{\mathsf F} + \epsilon)$&emsp;&emsp;&emsp;&emsp;// Normalize sing vals to $[0, 1]$.&emsp;$\epsilon = 10^{-7}$
> 1. $\mathbf X \gets \texttt{float16}(\mathbf X)$&emsp;&emsp;&emsp;&emsp; // Cast to half precision for speed
> 1. If $m < n$:&emsp;&emsp;$\mathbf X \gets \mathbf X^\top$&emsp;&emsp;&emsp;// Trick to make $\mathbf X \mathbf X^\top$ cheaper
> 1. $\mathbf R_{0} \gets \mathbf X \mathbf X^\top$
> 1. $\mathbf Q_{0} \gets \mathbf I$
> 1. For $t = 1, \ldots, 5$:
> 1. &emsp;&emsp; If $t = 3$:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;// Restart to stabilize
> 1. &emsp;&emsp;&emsp;&emsp; $\mathbf X \gets \mathbf Q_2 \mathbf X$
> 1. &emsp;&emsp;&emsp;&emsp; $\mathbf R_2 \gets \mathbf X \mathbf X^\top$
> 1. &emsp;&emsp;&emsp;&emsp; $\mathbf Q_2 \gets \mathbf I$
> 1. &emsp;&emsp; $\mathbf Z_t \gets b_t \mathbf R_{t-1} + c_t \mathbf R_{t-1}^2$
> 1. &emsp;&emsp; $\mathbf Q_t \gets \mathbf Q_{t-1} \mathbf Z_t +  a_t\mathbf Q_{t-1}$
> 1. &emsp;&emsp; $$(\mathbf{RZ})_t \gets \mathbf R_{t-1} \mathbf Z_t + a_t\mathbf R_{t-1}$$
> 1. &emsp;&emsp; $$\mathbf R_t \gets \mathbf Z_t (\mathbf{RZ})_t + a_t(\mathbf{RZ})_t$$
> 1. $\mathbf X \gets \mathbf Q_4 \mathbf X$
> 1. If $m < n$:&emsp;&emsp;$\mathbf X \gets \mathbf X^\top$&emsp;&emsp;// Undo trick
> 1. Return $\mathbf X$

## Runtime of Stabilized Gram Newton-Schulz

Above, we showed that Naive Gram Newton-Schulz uses $(4T + 3\alpha - 3)n^3$ FLOPs.
How does restarting change this?
It requires two additional matrix multiplications:

- $\mathbf X \gets \mathbf Q_2 \mathbf X$:&emsp;$2mn^2$
- $\mathbf R_2 \gets \mathbf X \mathbf X^\top$:&emsp;$mn^2$

Since the initial value of $\mathbf R_2$ is discarded and $\mathbf Q_2 = \mathbf I$, it also allows us to skip three matrix multiplications:

- $\mathbf R_2 \gets \mathbf Z_2 \mathbf R_1\mathbf Z_2$:&emsp;$-n^3 - n^3$
- $\mathbf Q_3 \gets \mathbf Q_2 \mathbf Z_3$:&emsp;$-n^3$

Therefore, Stabilized Gram Newton-Schulz with one restart uses $(4T + 6\alpha - 6)n^3$ FLOPs.
As before, this matches standard Newton-Schulz for $\alpha = 1$ and improves on it for $\alpha > 1$.
For $T=5, \alpha = 4$, our algorithm reduces the number of FLOPs by 42% compared to standard Newton-Schulz with symmetric GEMMs, or by 58% compared to typical implementations lacking symmetric GEMMs.

Observe that if we hypothetically used more restarts, each would increase the FLOPs by $3mn^2 - 3n^3$. With $T-1$ restarts, Gram Newton-Schulz would be exactly the same algorithm as standard Newton-Schulz.

In this sense, adding restarts can be viewed as trading wall clock time for greater guaranteed stability, with the extrema being Naive Gram Newton-Schulz and standard Newton-Schulz.

# Symmetric GEMM Kernels in CuTeDSL

To take advantage of the greater share of symmetric matrix multiplications enabled by Gram Newton-Schulz, we implement kernels for the operations $\mathbf A \mathbf B$ and $\alpha \mathbf A \mathbf B + \beta \mathbf C$ that assume $\mathbf A \mathbf B$ and $\mathbf C$ are symmetric. Symmetric kernels also accelerate standard Newton-Schulz; this idea has been around for a while for the construction of the Gram $\mathbf{XX^\top}$, but to our knowledge, hasn't been explored for fused symmetric matrix multiplication with addition.[^flashmuon][^laker] We target the Hopper and Blackwell GPU architectures and [open source](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_symmetric.py) our implementation in the [Quack](https://github.com/Dao-AILab/quack) library of CuTeDSL kernels developed by our lab.

![gemm_benchmarks (1)](https://hackmd.io/_uploads/BJq8sVPibl.png)
_Figure 15: SOTA Symmetric GEMM Kernels benchmarked on Hopper and Blackwell against cuBLAS._

## Layout Engineering and Work Scheduling

GEMM implementations of $\mathbf A \mathbf B$ and $\alpha \mathbf A \mathbf B + \beta \mathbf C$ can be broken down into the following components:

1. How do we schedule GEMM output tiles as work among groups of workers?
2. Once assigned a tile, how does a group of workers compute the tile?

In most GEMM and fused GEMM kernels, tiles are computed in the same way, with the following components:

1. The prologue, in which rows of $A$ and columns of $B$ needed for the current tile are loaded in from general memory (high-bandwidth memory) to shared memory (SRAM)
2. Matrix-Multiply Accumulate (MMA), in which those rows and columns are multiplied and written to the register file in Hopper or tensor memory in Blackwell
3. The epilogue, in which additional tensors needed for the fusion are loaded in, the fused arithmetic occurs, and the final values are written to the output tensor(s), from the register file to shared to general memory. An example is loading in $C$, $\alpha$, $\beta$, and then scaling $\mathbf A \mathbf B$ with $\alpha$ and adding $\beta \mathbf C$.

Our symmetric GEMM kernel and the standard GEMM kernel only differ in how they schedule and partition output tiles as work and how they implement their epilogues.

### Triangular Scheduler

In the standard GEMM, the entire output matrix is divided into work tiles that are load balanced and evenly partitioned amongst clusters of thread blocks, where thread blocks in the same cluster can access the same shared memory and are therefore scheduled to run together. Then, each cluster computes its assigned work tiles in succession.

Our tile scheduler in the symmetric GEMM is almost identical. The only difference is that only the work tiles in the lower triangle of the matrix are partitioned amongst the clusters, and work tiles in the upper triangle are unassigned, since their values are identical to the transposed values of the lower triangle.

Instead of using the standard tile scheduler which evenly divides the tiles of both triangles among the clusters, we use a _triangular scheduler_ to evenly divide only the tiles of only the lower triangle among the clusters. This ensures that every cluster gets assigned the same number of tiles that actually need to be worked on, ensuring load balancing.

### Epilogue: Writing to the Transposed Tile

In the GEMM epilogue, when the computed values of the lower triangle are written to their assigned tile in general memory (HBM), they are also written to their transposed tile location in the upper triangle.

![symm_gemm_diagram (1)](https://hackmd.io/_uploads/SkkoEAVq-e.png)
_Figure 16: Symmetric GEMM only computes $256 \times 256$ work tiles on the diagonal and in the lower triangle, copying and transposing each lower tile to its transposed location in the upper triangle._

We implement all of our symmetric GEMM kernels with square cluster work tiles. Hopper uses cluster size $(2, 1)$ and thread block tile size $(128, 256)$, and Blackwell uses cluster size $(2, 1)$ and 2-CTA collaboration, in which the 2 thread blocks in the cluster collaborate on the same big $(256, 256)$ tile.

Notably, highly optimized custom GEMM kernels on Hopper typically use Ping Pong Scheduling, in which the MMA of tile $i$ and the epilogue of tile $i-1$ are overlapped in two consumer warp groups[^ping-pong]. However, Ping Pong Scheduling uses more registers at once, and $(128, 256)$ is too large of a tile size for Ping Pong Scheduling, leading to register spillage. This is much slower than standard single producer warp, single consumer warp scheduling. Thus, our Hopper symmetric kernels do not use Ping Pong Scheduling. Blackwell GEMM kernels have no explicit conception of Ping Pong Scheduling, since by default in both cuBLAS and Quack, two accumulators are kept in the new tensor memory hierarchy, and MMA is computed on one accumulator while the epilogue is computed on the other.

As a small implementation detail, note that the main diagonal of $256 \times 256$ cluster work tiles is part of the work assigned by the triangular scheduler. Since their transposed locations are identical to their current locations, we only write those values to general memory once - writing twice can cause inaccurate values or NaN's.

## Implementation Strategy in Code

There are only two differences between the symmetric GEMM kernel and the standard GEMM kernel: the triangular scheduler and the transposed tile write in the epilogue. Quack is designed around abstracting the standard GEMM kernel to enable lightweight but maximally performant GEMM epilogue fusions. Using these abstractions, we are able to implement the symmetric GEMM kernel for both Hopper and Blackwell in just 160 lines, while achieving SOTA performance.

We override the standard tile scheduler with our triangular scheduler and wrap the symmetric GEMM class around the [GEMM with activation](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_act.py) class. GEMM with activation itself is a wrapper around the [Blackwell](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm100.py) and [Hopper](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py) default GEMMs. It supports writing two output tensors - the standard GEMM output (the preactivation) and the standard GEMM output with an activation function such as SwiGLU or ReLU applied (the postactivation). We define the activation function to be the identity and the postactivation tensor to be the inplace transpose of the preactivation tensor. Then, when the GEMM with activation class writes to the postactivation, it is really writing to the upper triangle with a tranposed layout - this is exactly the intent of the symmetric GEMM kernel. We override the epilogue of GEMM with activation just to ensure we don't write twice to the diagonal tiles, for the correctness reasons mentioned previously.

We're super excited about this simplicity! Without this abstraction, the initial implementation was close to 1500 lines of CuTeDSL. It shows the convenience of principled abstractions in kernel engineering, specifically that of the GEMM main loop + Epilogue paradigm.[^JCZ_anecdote]

## Kernel Optimizations for Standard Newton-Schulz

Using just our Quack CuTeDSL kernels, we can accelerate standard Newton-Schulz with two changes.

1. **Symmetric Matrix Multiplication**
   As discussed above, the matrices $\mathbf A = \mathbf X \mathbf X^\top$ and $\mathbf B = b_t \mathbf A + c_t \mathbf A^2$ computed at each iteration of Newton-Schulz are symmetric by definition. Therefore, we use our symmetric GEMM kernels for these operations, reducing their FLOP cost by half.
2. **Fused GEMM + Add**
   The typical way to implement the non-symmetric multiplication $\mathbf X \gets a_t \mathbf X + \mathbf B \mathbf X$ is to use `torch.baddbmm`, which calls cuBLAS under the hood. However, Quack offers a much faster implementation of this "Fused GEMM + Add" operation for Hopper. Unlike cuBLAS, Quack supports [Ping Pong Scheduling for Hopper](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_sm90.py), which better hides the epilogue addition of $a_t \mathbf X$.

<!-- 3. **Adding $\gamma \mathbf I$ for Free**
We rearrange the arithmetic of Newton-Schulz as follows. Instead of the [usual implementation](#alg-standard-ns)
   > 1. &emsp;&emsp; $\mathbf A \gets \mathbf X\mathbf X^\top$
   > 1. &emsp;&emsp; $\mathbf B \gets b_t \mathbf A + c_t \mathbf A^2$
   > 1. &emsp;&emsp; $\mathbf X \gets a_t \mathbf X + \mathbf B \mathbf X$

   we instead do
   > 1. &emsp;&emsp; $\mathbf A \gets \mathbf X\mathbf X^\top$
   > 1. &emsp;&emsp; $\mathbf B \gets a_t \mathbf I + b_t \mathbf A + c_t \mathbf A^2$
   > 1. &emsp;&emsp; $\mathbf X \gets \mathbf B \mathbf X$

   We perform step 2 using our symmetric kernel for the operation $\alpha \mathbf A \mathbf B + \beta \mathbf C + \gamma \mathbf I$. This kernel fuse the addition of $\gamma \mathbf I$ by adding $\gamma$ to all diagonal entries of the output when they are at the register level. This is practically free, so $\alpha \mathbf A \mathbf B + \beta \mathbf C + \gamma \mathbf I$ has effectively the same runtime as $\alpha \mathbf A \mathbf B + \beta \mathbf C$. This optimization completely obviates any I/O operations needed for the $\gamma I$ addition, outspeeding `gemm_symmetric(A, B, C, alpha, beta) + gamma * I`, which would require loading $\mathbf I$ from general memory to shared memory to registers. This reformulation of standard Newton-Schulz eliminates a rectangular matrix addition in step 3 without increasing the runtime of step 2, providing a small speedup.
 -->

This table shows that the total runtime of applying standard Newton-Schulz to all weight matrices of various LLMs decreases by about 25\% when combining these kernel optimizations on the Hopper architecture.

| Model      | `torch.compile` (Pure cuBLAS) | 1. CuTeDSL Symmetric GEMMs | 1. + 2. Fused GEMM Add | Final Speedup over `torch.compile` |
| ---------- | ----------------------------- | -------------------------- | ---------------------- | ---------------------------------- |
| Llama-430M | 18.909 ms                     | 16.114 ms                  | 13.71 ms               | **27% faster**                     |
| Qwen-600M  | 24.751 ms                     | 21.939 ms                  | 17.606 ms              | **29% faster**                     |
| Gemma-1B   | 75.055 ms                     | 66.063 ms                  | 55.444 ms              | **26% faster**                     |

_Table 1: On Hopper, using symmetric kernels and Ping Pong Scheduling in GEMM + Add accelerates standard Newton-Schulz by around $25\%$ already._

# Training Experiments and Benchmarks

We validate Gram Newton-Schulz's training quality and performance gain on Llama-430M, Qwen-600M, Gemma-1B, and a custom MoE-1B architecture with ~20% active parameters across 1 billion total parameters.[^llama][^qwen][^gemma]

We train on FineWeb-Edu. The number of training tokens for each dense model is given by the Chinchilla scaling law and for MoE-1B by twice the Chinchilla scaling law with respect to its active parameters. We use a cosine learning rate scheduler with the following base learning rates:

| Model      | Learning Rate |
| ---------- | ------------- |
| Llama-430M | 3e-3          |
| Qwen-600M  | 1.5e-3        |
| Gemma-1B   | 3e-4          |
| MoE-1B     | 2.5e-3        |

For both profiling and full training runs, our Muon setup is as follows:

1. Weights orthogonalized by Muon include $\mathbf W_q, \mathbf W_k, \mathbf W_v$ (the projection matrices for attention), $\mathbf W_o$ (the out-projection matrix following attention), $\mathbf W_{MLP_{UP}}$, $\mathbf W_{MLP_{GATE}}$, and $\mathbf W_{MLP_{DOWN}}$ (the SwiGLU MLP weights), and $\mathbf W_{router}$ (the token router matrix for MoE).
2. Each instance of $\mathbf W_q, \mathbf W_k, \mathbf W_v, \mathbf W_o, \mathbf W_{MLP_{UP}}, \mathbf W_{MLP_{GATE}}$, $\mathbf W_{MLP_{DOWN}},$ and $\mathbf W_{router}$ is batched across all transformer layers; that is, we execute a Newton-Schulz call for all the $\mathbf W_q$'s at once, for all the $\mathbf W_k$'s at once, etc. Maximizing the batch size of Newton-Schulz improves efficiency by making the batched GEMM operations as compute-bound as possible.

Muon is generally combined with a learning rate adjustment that scales the effective step size for each weight matrix based on its dimensions. We find that using Moonshot AI's strategy of scaling the update by $0.2 \sqrt{\max(\mathrm{fan\_out}, \mathrm{fan\_in})}$—roughly matching the RMS of Muon's update with that of AdamW—yields the best loss curves.[^moonshot-muon-is-scalable]

### Splitting the Weights

We draw special attention to the fact that we split $\mathbf W_{MLP_{UP}}$ from $\mathbf W_{MLP_{GATE}}$ and orthogonalize them separately. Ordinarily, MLPs are implemented as Linear + SwiGlu + Linear, where the weight matrix of the first linear layer is a concatenation of $\mathbf W_{MLP_{UP}}$ and $\mathbf W_{MLP_{GATE}}$. However, the gradients flowing back into the $\mathbf W_{MLP_{UP}}$ and $\mathbf W_{MLP_{GATE}}$ halves are calculated quite differently since their contributions to the activation are fundamentally different. We observe that orthogonalizing them separately improves the final loss; for example, in Llama-430M, we observe an improvement of $\approx 0.2$ in perplexity. In addition, splitting $\mathbf W_{MLP_{UP/GATE}}$ halves its small dimension in MoE architectures, where the intermediate size is smaller than the hidden size, leading to greater speedup from Gram Newton-Schulz.

<!-- In addition, orthogonalizing two small matrices is faster than orthogonalizing one big matrix. (JCZ: I don't think this is true for MLP with intermediate size > h dim, if you use NS Flop count)-->

Likewise, while earlier implementations of Muon orthogonalized the combined matrix $\begin{bmatrix} \mathbf W_q \,\vert\, \mathbf W_k \,\vert\, \mathbf W_v\end{bmatrix}$, we orthogonalize each piece separately.

We are also aware that in some settings, including pretraining GLM-5, Muon benefits from splitting Multi-Latent Attention weights ($\mathbf W^{UQ}$, $\mathbf W^{UK}$, and $\mathbf W^{UV}$) by attention head before orthogonalizing.[^GLM] This choice is principled, since the actual matrix multiplications happening in attention are between attention heads rather than the full query, key, value, and out projections. On our test models, we experimented with splitting $\mathbf W_q$, $\mathbf W_k$, $\mathbf W_v$, and $\mathbf W_o$ by attention heads to form $H$ matrices each of size $\tfrac{d}{H} \times d$, where $d$ is the embedding dimension and $H$ is the number of heads. However, we observed higher losses throughout training when using this design.

Still, we believe that there are other settings like GLM-5 where this strategy works well. Such cases would benefit _immensely_ from Gram Newton-Schulz, since the aspect ratio of these weight matrices would be the number of heads $H$. For a standard attention weight like $\mathbf W_q$ with $H=16$ and $T=5$, Gram Newton-Schulz on the little matrices would use **$80\times$** fewer FLOPs than orthogonalizing the big matrix!

## Model Quality is Preserved

We see loss preserved as follows, when both using the Polar Express coefficients and the coefficients derived by You Jiacheng:[^you]
![validation_perplexity_hopper](https://hackmd.io/_uploads/ryyhZAroWe.png)
_Figure 17: Validation perplexity is always preserved within 0.01. We train with Muon using the Chinchilla scaling law on Hopper._
![moe_1b_blackwell_ppl](https://hackmd.io/_uploads/ryrgMRrsZx.png)
_Figure 18: Validation perplexity is preserved within 0.01. We train with Muon using the Chinchilla scaling law on Blackwell._

## Our Method Speeds up the Optimizer Step

**Newton-Schulz Performance**
We observe that our method speeds up the runtime of the Newton-Schulz step in each iteration by up to $2\times$, especially as weights become more rectangular. The tables below report these speed-ups for each model, benchmarked on both H100 and B300. In these experiments, we use standard Newton-Schulz as the fallback when $m = n$:

![icml_ns_breakdown (6)](https://hackmd.io/_uploads/BJy11AZoWe.png)
_Figure 19: Hopper architecture Newton-Schulz time per model weight. Very rectangular weights like Up/Gate and Down in Gemma-1B will especially benefit from Gram Newton-Schulz, while square weights like Llama-430m's attention weights will just benefit from the kernels. The speedup on MoE-1B for Up/Gate and Down doesn't even take advantage of the symmetric kernel, since the small intermediate size of $256$ is exactly the tile size. The speedup is fully algorithmic._

![icml_ns_breakdown_b300](https://hackmd.io/_uploads/Bk1YJCZj-x.png)
_Figure 20: Blackwell architecture Newton-Schulz time per model weight. The speedup on MoE-1B for Up/Gate and Down is fully algorithmic, like in Figure 17._

**End-to-End Optimizer Performance**
The following figure shows the end-to-end wall clock time of the optimizer step for each method. For Muon, these timings include the AdamW updates for weights not assigned to Muon (such as the embedding layer and the vector-valued weights), PyTorch operations for splitting and reconcatenating weights, and learning rate scaling.

![icml_optimizer_plot2 (4)](https://hackmd.io/_uploads/r1s2fCZjWg.png)
_Figure 21: Hopper architecture end-to-end optimizer step during training, including matrix splitting and recombination for QKV and MLP, LR scaling, master weight updates, and the scalar optimizer (AdamW) step for non-2D weights._

These results allow us to measure the impact of our optimized kernels separately from that of our Gram Newton-Schulz algorithm. We see that both pieces contribute significantly to the speedup. We observe that Llama-430M's and Qwen-600M's smaller, square weights benefit from the kernels - again, we stress that square architectures are the rare case. Meanwhile, Gemma benefits from both the algorithm and kernels, seeing the biggest speedup due to its MLP weights' higher aspect ratio of $8$ instead of $4$.

We run our experiments on a single GPU. The speedup of using our method in different parallelism configurations should be the same as on one GPU in most cases.

### Gram Newton-Schulz time in Kimi K2

Kimi K2 is a trillion parameter sparse, fine-grained MoE model with $384$ experts per layer, a hidden size of $7168$, and a small expert intermediate dimension of $2048$. Since models are trending towards finer-grained MoE architectures and Kimi K2 was trained with Muon, this is a perfect setting to benchmark Gram Newton-Schulz.

In the [Appendix](#appendix), we approximate the exposed Newton-Schulz wall clock time of a global training step of Kimi K2 to be that of:

- 216 expert up/gate/down weights of shape $2048 \times 7168$
- 1 dense up/gate/down weight of shape $7168\times18432$

![kimi (2)](https://hackmd.io/_uploads/B1w5VCZjbe.png)
_Figure 22: On Hopper, Gram Newton-Schulz is $2\times$ faster than standard Newton-Schulz in Kimi K2's pipeline parallelism configuration._

![kimi_b300](https://hackmd.io/_uploads/HkQAVRZoWe.png)
_Figure 23: On Blackwell, Gram Newton-Schulz is $2\times$ faster than standard Newton-Schulz in Kimi K2's pipeline parallelism configuration._

Observe that the speedup of Gram Newton-Schulz over standard Newton-Schulz in `torch` is twice the speedup of standard Newton-Schulz in CuTeDSL over standard Newton-Schulz in `torch`, showing the contribution of the new algorithm.

# Impact on End-to-End Training Time

In the previous section, we showed that Gram Newton-Schulz significantly speeds up the optimizer step time. This improvement is most impactful when the optimizer time is a large share of the global training step time. Many factors affect the relative runtimes of the optimizer step and the forward and backward passes. In this section, we describe several common settings where the optimizer step is a meaningful performance bottleneck.

### Low precision training

In low precision training, the forward and backward passes are computed in 4 bit or 8 bit precision, greatly speeding up their wall clock time. However, Newton-Schulz must be computed in 16 bit precision for stability and accuracy. Therefore, the optimizer time will occupy a greater share of training time.

### Small global batch size

When global batch size decreases, fewer microbatches are needed, so fewer forward and backward passes will occur per global training step. The optimizer time will remain the same, since it is agnostic to batch size. Therefore, the optimizer step will occupy a greater share of training time. For example, when SFT and RL use Muon, as in Kimi K2's post-training pipeline, batch sizes are significantly smaller than in pretraining.[^kimi][^SFT]

### Optimizer step frequency is bottlenecked by optimizer duration

Fixing the total number of tokens used in training, smaller global batch sizes are typically preferable to larger global batch sizes for model quality, since they allow for more frequent weight updates.[^allen] However, when using pipeline parallelism at scale, smaller batch sizes can come with a performance tradeoff. The backward pass of pipeline stage $i-1$ needs to hide the optimizer step of pipeline stage $i$ as much as possible, and increasing the batch size to better hide the optimizer step with a longer backward pass can increase throughput.

Gram Newton-Schulz decreases the optimizer step time, allowing the backward pass to hide the optimizer at smaller batch sizes. Thus, Gram Newton-Schulz can improve model quality by allowing for smaller batch sizes and more frequent updates without a throughput tradeoff.

### Large cluster size

A larger cluster size allows for more data parallel groups, decreasing the forward and backward pass time of a global training step. The optimizer step time will usually be the same. Distributing the Newton-Schulz work of a GPU's model parameters across its corresponding rank in the other data parallel groups is possible, but it invokes significant internode communication overhead and occupies bandwidth that is usually not worth the cost.

<!-- We use a single H100 to approximate the savings of Gram Newton-Schulz given this training setup, with the following assumptions:
1. For the routed expert FFNs, we take one expert per layer in order to fit them in one Newton-Schulz call without exceeding the GPU's memory, and then multiply the recorded time by the number of experts per layer for an approximation.
2. For all other weights, we run Newton-Schulz normally.

![kimi](https://hackmd.io/_uploads/HJH5lBzqbe.png)
*Figure : Newton-Schulz time per weight for Kimi K2 Thinking.*


![kimi_total](https://hackmd.io/_uploads/r1M5rHzqbg.png)
*Figure : Total Estimated Newton-Schulz time on a single H100 for Kimi K2 Thinking.*

**Gram Newton-Schulz is 1.88× faster than standard Newton-Schulz implemented with `torch.compile` and 1.51× faster than standard Newton-Schulz implemented with our CuteDSL kernels. This speedup is fundamentaly algorithmic (and huge)!**


The script we use for Kimi K2 Benchmarking can be found [here](https://github.com/JackCharlesZhang/square-muon/blob/main/benchmarks/benchmark_kimi_k2_mock.py). -->

# Conclusion

We hope our analysis and experiments will encourage researchers to try Gram Newton-Schulz. Our results show that Gram Newton-Schulz preserves training quality and speeds up the optimizer step by up to $2\times$ on popular model architectures, providing a rare case of free lunch performance.

We release an [implementation of Gram Newton-Schulz](https://github.com/Dao-AILab/gram-newton-schulz/blob/main/gram_newton_schulz/gram_newton_schulz.py) that serves as a drop-in replacement for the standard five-step Newton-Schulz used in Muon along with the [symmetric GEMM kernels](https://github.com/Dao-AILab/quack/blob/main/quack/gemm_symmetric.py) that accelerate it.
We believe that the stability analysis provided in this blog post lays the foundation for easily adapting Gram Newton-Schulz to other use cases. The only hyperparameter that needs to be retuned at all is the set of iterations at which to restart. To this end, we provide an [autotuning script](https://github.com/Dao-AILab/gram-newton-schulz/blob/main/gram_newton_schulz/autotune_restarts.py) that takes a series of coefficients (for instance, 10 steps of Polar Express) and suggests the optimal set of restarts according to [our analysis above](#stabilized-gram-newton-schulz).

<!-- For future work, we deeply care about the following: TODO -->

## Citing this blog post

```bibtex
@misc{GramNewtonSchulz,
  title   = {Gram Newton-Schulz},
  author  = {Jack Zhang and Noah Amsel and Berlin Chen and Tri Dao},
  year    = {2026},
  url     = {https://dao-ailab.github.io/blog/2026/gram-newton-schulz/}
}
```

[^muon]: https://kellerjordan.github.io/posts/muon/
[^deriving_muon]: https://jeremybernste.in/writing/deriving-muon
[^ping-pong]: https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/
[^num-iters]: Some variants set $T=6$ or $T=4$, but never anything else.
[^polar-express]: https://openreview.net/forum?id=yRtgZ1K8hO
[^PRISM]: https://arxiv.org/abs/2601.22137
[^grishina]: https://arxiv.org/abs/2506.10935
[^dion]: https://arxiv.org/abs/2504.05295
[^embeddings]: In the case of standard attention, the $\mathbf W_{QKV}$ matrix is rectangular with aspect ratio 3, but for unrelated reasons we divide it into three square matrices and apply Newton-Schulz to each as we discuss [here](#training-experiments-and-benchmarks). Other authors subdivide these matrices into separate weights for each head, making them highly rectangular. The embedding and unembedding matrices are also rectangular, but these are not typically optimized using Muon.
[^moonshot-muon-is-scalable]: See section 2.2 of https://arxiv.org/abs/2502.16982.
[^sonicmoe]: https://arxiv.org/abs/2512.14080
[^llama]: https://arxiv.org/abs/2407.21783
[^kimi]: https://arxiv.org/abs/2507.20534
[^GLM]: https://arxiv.org/abs/2602.15763
[^qwen]: https://arxiv.org/abs/2505.09388
[^gpt-oss]: https://arxiv.org/abs/2508.10925
[^deepseek]: https://arxiv.org/abs/2412.19437
[^SFT]: https://arxiv.org/abs/2404.18922
[^scion]: https://arxiv.org/abs/2502.07529
[^soap]: https://arxiv.org/abs/2409.11321
[^splus]: https://arxiv.org/abs/2506.07254
[^shampoo]: https://arxiv.org/abs/1802.09568
[^you]: https://x.com/YouJiacheng/status/1905861218138804534
[^gemma]: https://arxiv.org/abs/2503.19786
[^laker]: https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf
[^flashmuon]: https://github.com/nil0x9/flash-muon
[^JCZ_anecdote]: We had previously mentioned a fused symmetric quadratic kernel for $\mathbf a_t \mathbf I + b_t \mathbf A + c_t \mathbf A^2$ that we ended up passing on for stability reasons. Quack's abstraction was so convenient that Claude and I were able to write the register-level $\mathbf a_t \mathbf I$ fusion in 5 minutes on a car ride.
[^allen]: https://allenai.org/blog/critical-batch-size

## References

1. Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. "Muon: An optimizer for hidden layers in neural networks." Blog post, 2024. Available at: https://kellerjordan.github.io/posts/muon/
2. Jeremy Bernstein. "Deriving Muon." Blog post, 2025. Available at: https://jeremybernste.in/writing/deriving-muon
3. Less Wright and Adnan Hoque. "CUTLASS Ping-Pong GEMM Kernel." PyTorch Blog, November 1, 2024. Available at: https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/
4. Noah Amsel, David Persson, Christopher Musco, and Robert M. Gower. "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm." International Conference on Learning Representations (ICLR), 2026.
5. Ekaterina Grishina, Matvey Smirnov, and Maxim Rakhuba. "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials." arXiv preprint arXiv:2506.10935 (2025).
6. Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, and John Langford. "Dion: Distributed Orthonormalized Updates." arXiv preprint arXiv:2504.05295 (2025).
7. Jingyuan Liu et al. "Muon is Scalable for LLM Training." arXiv preprint arXiv:2502.16982 (2025).
8. Kimi Team. "Kimi K2: Open Agentic Intelligence." arXiv preprint arXiv:2507.20534 (2026).
9. Aaron Grattafiori et al. "The Llama 3 Herd of Models." arXiv preprint arXiv:2407.21783 (2024).
10. GLM-5 Team et al. "GLM-5: From Vibe Coding to Agentic Engineering." arXiv preprint arXiv:2602.15763 (2026).
11. An Yang et al. "Qwen3 Technical Report." arXiv preprint arXiv:2505.09388 (2025).
12. OpenAI et al. "gpt-oss-120b & gpt-oss-20b Model Card." arXiv preprint arXiv:2508.10925 (2025).
13. DeepSeek-AI et al. "DeepSeek-V3 Technical Report." arXiv preprint arXiv:2412.19437 (2025).
14. Han Zhong, Zikang Shan, Guhao Feng, Wei Xiong, Xinle Cheng, Li Zhao, Di He, Jiang Bian, and Liwei Wang. "DPO Meets PPO: Reinforced Token Optimization for RLHF." arXiv preprint arXiv:2404.18922 (2025).
15. Thomas Pethick, Wanyun Xie, Kimon Antonakopoulos, Zhenyu Zhu, Antonio Silveti-Falls, and Volkan Cevher. "Training Deep Learning Models with Norm-Constrained LMOs." arXiv preprint arXiv:2502.07529 (2025).
16. Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. "SOAP: Improving and Stabilizing Shampoo using Adam." arXiv preprint arXiv:2409.11321 (2025).
17. Kevin Frans, Sergey Levine, and Pieter Abbeel. "A Stable Whitening Optimizer for Efficient Neural Network Training." arXiv preprint arXiv:2506.07254 (2025).
18. Vineet Gupta, Tomer Koren, and Yoram Singer. "Shampoo: Preconditioned Stochastic Tensor Optimization." International Conference on Machine Learning. PMLR, 2018.
19. Gemma Team et al. "Gemma 3 Technical Report." arXiv preprint arXiv:2503.19786 (2025).
20. Laker Newhouse, Dakota Goldberg, and Ricardo Ruiz. "Faster Symmetric Matrix Multiplication with ThunderKittens."Available at: https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf
21. Tianyang Lin. "Flash-Muon: An Efficient Implementation of Muon Optimizer." GitHub repository, 2025. Available at: https://github.com/nil0x9/flash-muon
22. Will Merrill. "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training." arXiv preprint arXiv:2505.23971 (2025). Available at: https://allenai.org/blog/critical-batch-size

# Appendix

The share of end-to-end training time taken up by Newton-Schulz can vary widely depending on the training setup. To explain this variability, we analyze two idealized scenarios. In one, standard Newton-Schulz takes 2% of training time; in the another it takes 17%.

### Case Study 1: Standard Newton-Schulz accounts for 2% of Kimi K2 training time

The following analysis gives a very optimistic estimate of the optimizer's wall clock time. We assume an efficient training infrastructure with highly optimized pipeline parallelism. Moreoveer, we assume that the optimizer step of each pipeline stage is completely hidden behind the backward pass of the next pipeline stage.

Kimi K2 Thinking is a $1.1$ trillion parameter model with $32$ billion active parameters. It has $1$ dense layer followed by $60$ MoE layers.[^kimi] It is pretrained with $256$-GPU model parallel groups, $16$-way pipeline parallelism, $16$-way expert parallelism within each pipeline stage, and a huge batch size of $67$ million tokens.

We use a single H100 to approximate the share of each training step's runtime occupied by Newton-Schulz in this setting under the following assumptions:

1. The training cluster is $2048$ H100s across $256$ nodes (8 GPUs per node), connected with NDR 400 Gb/s InfiniBand inter-node (8 NICs per node, 1:1 NIC-to-GPU ratio) and NVLink 4.0 intra-node. This is the size of the cluster used to train DeepSeekV3, with upgraded hardware.[^deepseek] This means there are $\frac{2048}{256} = 8$ groups in data parallel.
2. Training in `bfloat16` hits $35\%$ to $45\%$ MFU, which is a typical range for MoEs at this scale on H100 clusters.
3. The only non-overlapped optimizer wall clock time is of the last pipeline stage that completes its backward (i.e. pipeline stage $1$ of $16$). The optimizer steps of pipeline stages $2$ to $16$ are fully hidden behind the backwards of stages $1$ to $15$.
4. Pipeline stage $1$ has the dense layer and $3$ MoE layers.

Under these assumptions, the optimal way to partition the Newton-Schulz work of pipeline stage 1 is as follows:

1. Each of the $16$ GPUs in pipeline stage $1$'s expert parallel group gets $$\frac{384 \text{ experts/layer} \times 3 \text{ MoE layers}}{16 \text{ GPUs}} = 72 \text{ experts/GPU} = 216 \text{ expert up-gate-down/GPU}$$ Each of the 16 GPUs has its own unique expert weights, so no communication is needed.
2. The four shared experts' weights and the dense MLP's weights are divided evenly based on orthogonalization wall clock time amongst the 16 expert parallel GPUs, which run Newton-Schulz in parallel. The dense MLP's three $7168\times18432$ weights (up/gate/down) dominate the wall clock time, so they are sent to 3 different GPUs, with the rest of the weights split amongst the remaining 13. Thus, the total Newton-Schulz time for all these weights when the 16 GPUs run in parallel is the same as the time to run Newton-Schulz on one of the dense up/gate/down weights. An `all_gather` is required between the two nodes to collect the distributed orthogonalized gradients, but we assume it is substantially faster than redundant Newton-Schulz work.

Then, the total Newton-Schulz time of Pipeline Stage 1 is that of 216 expert up/gate/down weights and 1 dense up/gate/down weight.

Per our assumptions, Pipeline Stage 1's Newton-Schulz time is the only non-overlapped Newton-Schulz time. As benchmarked [here](#gram-newton-schulz-time-in-kimi-k2), standard Newton-Schulz in `torch` will take 315 ms.

Let's estimate the end-to-end wall clock time of an entire Kimi K2 global training step.

**Given:**

- Active parameters: $N = 32 \times 10^9$
- H100 peak: $P = 989 \times 10^{12}$ FLOP/s
- Cluster size: $G = 2048$ GPUs
- Global batch size: $B = 67 \times 10^6$ tokens

$$\text{sec/batch} = \frac{B \times 6N}{P \times \text{MFU} \times G} = \frac{67 \times 10^6 \times 6 \times 32 \times 10^9}{989 \times 10^{12} \times \text{MFU} \times 2048} = \frac{6.351}{\text{MFU}}$$

For realistic estimates of MFU, we have

| MFU | sec/batch |
| --- | --------- |
| 35% | 18.14 s   |
| 45% | 14.11 s   |

Thus, Newton-Schulz takes approximately $\frac{315\text{ ms}}{18140\text{ ms} + 315\text{ ms}} = 1.7\%$ to $\frac{315\text{ ms}}{7060\text{ ms}+315\text{ ms}} = 2.2\%$ of total pretraining wall clock time in this setting.

### Case Study 2: Standard Newton-Schulz occupies 17% of Llama3-70B SFT time

Llama3-70B is a 80-layer dense model with hidden size 8192, intermediate size 28672, and grouped query attention with $1024 \times 8192$ $\mathbf W_k, \mathbf W_v$ weights and $8192 \times 8192$ $\mathbf W_q, \mathbf W_o$ weights.[^llama] Supervised finetuning (SFT) typically uses small batch sizes, ranging from $32$ to $256$ sequences.[^SFT][^deepseek]

We construct the following SFT case:

1. Training uses $32$ H100s across $4$ nodes (8 GPUs per node).
2. Training in `bfloat16` hits $40\%$ MFU.
3. Weights are sharded evenly across GPUs using FSDP, and the exposed Newton-Schulz time is that of $\frac{80 \text{ layers}}{32 \text{ GPUs}} \approx 3 \text{ layers}$. Each layer has 3 up-gate-down weights, 2 $\mathbf W_q, \mathbf W_o$ weights, and 2 $\mathbf W_k, \mathbf W_v$ weights.

According to our benchmarking, standard Newton-Schulz of

- Nine $8192 \times 28672$ weights takes 738.731 ms
- Six $8192 \times 8192$ weights takes 156.368 ms
- Six $1024 \times 8192$ weights takes 2.318 ms

totalling 250 ms.

**Given:**

- Parameters: $N = 70 \times 10^9$
- H100 peak: $P = 989 \times 10^{12}$ FLOP/s
- Cluster size: $G = 32$ GPUs
- Global batch size: $B = 64 \text{ sequences} \times 2048 \text{ tokens/sequence} = 131{,}072$ tokens

$$\text{sec/batch} = \frac{B \times 6N}{P \times \text{MFU} \times G} = \frac{131{,}072 \times 6 \times 70 \times 10^9}{989 \times 10^{12} \times 0.40 \times 32} = 4.35\text{ s}$$

Newton-Schulz takes approximately $\frac{897.417\text{ ms}}{4350\text{ ms} + 897.417\text{ ms}} = 17\%$ of total SFT wall clock time in this parallelism setting.
