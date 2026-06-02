# Response to Previous Reviews (ACM Multimedia 2026, Submission 4812)

**Manuscript:** *AdaVTP: Adaptive Visual Token Pruning via Text-Conditioned Information Bottleneck*

**Statement of relationship to prior submission.** An earlier version of this manuscript was submitted to ACM Multimedia 2026 (Submission ID 4812), received peer review, and was subsequently **withdrawn by the authors**. Below we provide verbatim quotations of all relevant parts of the previous review reports, together with a point-by-point description of how each concern has been addressed in the present version.

> **Note to authors (remove before upload):** Restore any math symbols that were dropped when the reviews were copied (several inline symbols rendered as blanks). The quotations below must match the original review PDFs exactly. Items marked **[ACTION REQUIRED]** depend on new experiments or content that must be completed and the numbers/figures inserted before submission. Equation/figure numbers refer to the revised manuscript unless a comment is being quoted.

---

## Summary of Major Revisions

1. **Notation unified.** The symbol previously written as $\mathcal{L}_{\text{IB}}$ has been renamed to $\mathcal{L}_{\text{KL}}$ throughout, and the training-objectives section now states explicitly that each loss realizes a *specific* term of the variational bound $\hat{\mathcal{J}}$: the compression term is realized by $\mathcal{L}_{\text{KL}}$, the relevance term by $\mathcal{L}_{\text{CE}}$, and $\mathcal{L}_{\text{FF}}$ is an auxiliary stabilizer outside the bound.
2. **Complete proofs added.** Full proofs are now provided in the main text for the variational compression bound, the variational relevance bound, and the equivalence between the relevance term and the cross-entropy loss.
3. **Relationship to prior IB-based pruning clarified.** We now cite the prior IB-based formulation at the point where the objective is introduced and explicitly contrast our *text-conditioned* bottleneck with text-agnostic prior work.
4. **Framework figure revised.** The overview figure and its caption have been revised so that the pipeline, masking mechanism, and IB objective are explicitly connected, and the figure now reflects the unified notation.
5. **Generalization evidence strengthened.** Results on multi-image and video understanding benchmarks have been moved into the main paper, demonstrating that the framework generalizes beyond single-image settings.
6. **[ACTION REQUIRED] Expanded comparisons and analyses** (recent SOTA baselines, efficiency at multiple token budgets, qualitative results at higher budgets, and failure-case analysis) — see the corresponding responses below.

---

## Reviewer Hxbf (Rating: Weak Reject; Confidence: Knowledgeable)

**Comment H1 (Originality of the IB formulation).**
> "First, the authors seem to present the Information Bottleneck formulation as their own. However, prior work has already explored Information Bottleneck for token pruning, but the manuscript does not clearly distinguish its formulation from existing methods."

**Response.** We thank the reviewer and have addressed this directly. We do not claim the Information Bottleneck principle itself as novel; our contribution is the **text-conditioned** instantiation of it for visual token pruning. In the revised manuscript we (i) cite the prior IB-based formulation at the exact point where our objective is introduced (rather than only once in the introduction), and (ii) add an explicit discussion contrasting the two. Concretely, prior IB-based pruning compresses visual tokens *without reference to the input text*, whereas we make the input text $T$ an explicit conditioning variable, so that both redundancy reduction and relevance preservation become query-adaptive. This distinction is now stated where the objective appears and is reflected in the contribution statement.

**Comment H2 (Direction of the objective — min vs. max).**
> "Based on this issue, Eq. (1) appears to be incorrectly formulated. ... M appears to be fixed, while Z denotes the selected visual-token subset after pruning. Therefore, it is unclear why the objective is defined as [minimizing I(X;Z)]. Since the size of Z is fixed and Z is expected to preserve model performance, the objective should instead encourage selecting the M visual tokens that retain the most relevant information from the original token set X. Therefore, a maximization objective over I(X;Z), rather than a minimization objective, seems more appropriate."

**Response.** We respectfully clarify a point of terminology, as we believe this stems from a conflation of the two mutual-information terms in the IB principle. In the standard Information Bottleneck (Tishby et al.), the two terms play *opposite* roles:
- $I(X;Z)$ is the **compression** term, measuring how much the retained representation $Z$ reveals about the *input* $X$ (including redundancy). This term is **minimized** to remove redundancy.
- $I(Z;Y)$ is the **relevance** term, measuring how much $Z$ reveals about the *target* $Y$. This is what must be **preserved** (the constraint $I(Z;Y)\ge\mathcal{I}_0$).

Crucially, the optimization variable is the **selection policy** $p(m\mid X,T)$, not $Z$ itself. Maximizing $I(X;Z)$ would prefer the subset that is *most redundant with the full input*, which is the opposite of the desired behavior; preserving task performance is instead enforced by the relevance term/constraint $I(Z;Y)$. Thus the formulation $\min_{p(m\mid X,T)} I(X;Z)\ \text{s.t.}\ I(Z;Y)\ge\mathcal{I}_0$ is the standard, correct IB statement. To remove any ambiguity, the revised manuscript now states explicitly that (a) the optimization is over the selection policy $p(m\mid X,T)$, and (b) $I(X;Z)$ is the compression term in the standard IB sense while task relevance is preserved through the relevance term.

**Comment H3 (Symbol mismatch with [1]).**
> "The authors seem to adapt Eq. (2) from [1] to their own setting. However, because the symbol definitions differ ... in Eq. (2) of [1], the optimization variable corresponds to the pruning operator ... In contrast, this manuscript defines the optimization variable as the selected visual-token subset Z, so minimizing the information retained by Z is not appropriate."

**Response.** This concern is resolved by the clarification in H2. Our optimization variable is **not** the subset $Z$, but the **selection policy** $p(m\mid X,T)$ that induces $Z = m\odot X$; in this sense our setting is consistent with treating the selection mechanism as the optimization variable. The revised manuscript makes the optimization variable explicit at the objective and aligns the notation accordingly, so the apparent mismatch with the pruning-operator view of prior work no longer arises.

**Comment H4 (Gains concentrated at small budgets; missing efficiency at 128/192).**
> "AdaVTP shows clear advantages mainly under the 64-token setting in Table 1. However, the gains become limited when 128 or 192 visual tokens are retained. The paper does not discuss whether this indicates that AdaVTP is only beneficial under small token budgets. In addition, Table 2 reports efficiency only for the 64-token setting. Efficiency comparisons for the 128- and 192-token settings should also be provided."

**Response.** **[ACTION REQUIRED]** We have added (a) efficiency comparisons (prefill latency / FLOPs / throughput) at the 128- and 192-token settings to the efficiency table, and (b) an explicit discussion of the budget–benefit relationship. *Recommended framing once the numbers are in:* the relative gain naturally narrows at larger budgets because the unpruned upper bound is more closely approached by all methods, i.e., there is less headroom to differentiate; AdaVTP nonetheless remains on or nearest the empirical Pareto frontier at every budget, and its advantage is largest precisely in the regime that matters most for deployment (aggressive pruning). *Insert the new efficiency numbers for 128/192 here and the corresponding sentence in the main text.*

**Comment H5 (Presentation: figure order, undefined variables, Figure 3 losses).**
> "There are also some issues in presentation. ... the order in which figures are referenced does not match the order in which they appear. Several variables ... are not clearly defined. Figure 3 does not show how the three proposed losses are applied during training."

**Response.** We thank the reviewer for these careful observations and have addressed all three. (i) Figure citations have been reordered so that figures are referenced in order of appearance. (ii) All previously undefined symbols are now defined at first use *(insert the specific variables the reviewer flagged once symbols are restored from the original review)*. (iii) The framework figure has been revised so that the training losses are explicitly depicted; see also the response to Reviewer cX5Y (Comment C1).

---

## Reviewer tjQ6 (Rating: Weak Reject; Confidence: Knowledgeable)

**Comment T1 (Prior $c(Z\mid T)$ depends on $X$).**
> "Eq. (13) defines [$c(Z\mid T)$] as the softmax score of [$s^{\text{cross}}$]. However, [it] is computed from the visual tokens [$X$] rather than the selected subset [$Z$]. It is unclear why [$c(Z\mid T)$] can be interpreted as a prior distribution over the selected visual tokens conditioned on the text query."

**Response.** This is a fair observation, and the revised manuscript makes the justification explicit. While the variational compression bound ideally requires $c(Z\mid T)$ to be independent of $X$, we adopt an **amortized** realization that shares parameters with $p(Z\mid X,T)$ — standard practice in amortized variational inference. Importantly, the prior is constructed from the **semantic-alignment (cross-modal) score** $s^{\text{cross}}$, which scores each visual token by its *alignment to the input text* rather than by its intrinsic visual content. It is therefore a *text-driven* prior in the sense relevant to the bound: it encodes how relevant each candidate token is to the query. We have clarified this design choice and its relationship to the bound in a dedicated remark, and reference the amortized-VI literature that motivates parameter sharing.

**Comment T2 (Why does the KL term optimize only the first term of $\hat{\mathcal{J}}$?).**
> "Why does [$\mathcal{L}_{\text{KL}}$] in Eq. (17) optimize only the first term of [$\hat{\mathcal{J}}$] in Eq. (5)? In addition, it is unclear why [$\mathcal{L}_{\text{KL}}$] can serve as the optimization objective for the second term in Eq. (5)."

**Response.** We agree this was caused by overloaded notation, and we have fixed it. The single loss previously named $\mathcal{L}_{\text{IB}}$ has been renamed $\mathcal{L}_{\text{KL}}$, and the revised text states explicitly that the two terms of the bound are realized by **two separate losses**: the compression (first) term by $\mathcal{L}_{\text{KL}}$ and the relevance (second) term by $\mathcal{L}_{\text{CE}}$. Thus $\mathcal{L}_{\text{KL}}$ realizes *only* the first term by design, and it was never intended to serve as the objective for the second term — the cross-entropy loss does that. This removes the impression that one symbol simultaneously stood for the full objective and a single term.

**Comment T3 (Missing proof that the variational substitution yields Eq. (17)).**
> "The theoretical proof is missing to explain how substituting the variational approximations of $p(Z\mid X,T)$ (Eq. 12) and $c(Z\mid T)$ (Eq. 13) into the first term of Eq. 5 can obtain Eq. 17."

**Response.** A full derivation has been added. Because $p(Z\mid X,T)$ and $c(Z\mid T)$ are both categorical distributions over the same $N$ visual tokens, the KL divergence in the compression term admits a **closed-form token-wise sum**, which is exactly $\mathcal{L}_{\text{KL}}=\sum_i p(z_i\mid X,T)\log\frac{p(z_i\mid X,T)}{c(z_i\mid T)}$. The derivation is now given explicitly in the methodology section, alongside the proof of the variational compression bound from which it follows.

**Comment T4 (Qualitative visualizations at 128/192).**
> "Please include qualitative visualization comparisons for the settings with 128 and 192 retained visual tokens in Figure 4."

**Response.** **[ACTION REQUIRED]** We have added qualitative visualizations at the 128- and 192-token settings to the qualitative figure. *Insert the new visualization panels and a one-line description showing that the retained tokens remain query-relevant as the budget increases.*

---

## Reviewer cX5Y (Rating: Weak Reject; Confidence: Expert)

**Comment C1 (Figure 3 clarity and loss computation).**
> "Figure 3 is not sufficiently clear and does not fully match the method description. The relation among subfigures (a), (b), and (c) is unclear. ... it is confused how the pipeline, masking mechanism, and IB formulation are connected during training. Moreover, this figure only shows the IB Objective, but does not illustrate how [the scores] in Eq. (16) are computed."

**Response.** We have substantially revised the overview figure and its caption. The caption now states the role of each subfigure and how they connect: (a) the overall pipeline producing the *Token Saliency Distribution* and selecting tokens via Gumbel sampling, (b) the masking mechanism that fuses the three complementary scores ($s^{\text{self}}, s^{\text{peer}}, s^{\text{cross}}$) under a controllable text prior followed by the normalization step, and (c) the text-conditioned information bottleneck objective. The figure has been redrawn so that (i) the training losses and their computation are depicted rather than only the abstract objective, and (ii) the in-figure notation matches the unified notation used in the text. **[ACTION REQUIRED:** confirm the regenerated `framework.pdf` shows the score computation and the three losses, and that its labels use $\mathcal{L}_{\text{KL}}/\mathcal{L}_{\text{CE}}/\mathcal{L}_{\text{FF}}$ consistent with the text.**]**

**Comment C2 (Relationship to [1] and missing citation at Eq. (1)).**
> "Prior work has already explored the Information Bottleneck principle for text-guided visual token pruning, and Eq. (1) in the manuscript is closely related to Eq. (2) in [1]. However, the manuscript does not clearly discuss how Eq. (1) differs ... nor does it cite [1] at the point where Eq. (1) is introduced; [1] is only cited once in the introduction."

**Response.** Addressed together with Comment H1. We now cite [1] at the point where the objective is introduced and add an explicit paragraph contrasting our text-conditioned bottleneck with the text-agnostic formulation of [1]: prior work compresses visual tokens without conditioning on the input text, whereas we make $T$ an explicit conditioning variable so that redundancy reduction and relevance preservation are both query-adaptive. The novelty of our work lies in this conditioning and its concrete realization, not in the IB principle itself.

**Comment C3 (Implementation details of Eq. (7)–(8)).**
> "The implementation details of Eq. (7) and Eq. (8) are unclear. In particular, the manuscript does not clearly specify the normalization used, nor how $A^{vv}$, $A^{tt}$, $A^{tv}$ are obtained in Eq. 7. I am also confused about the definition of the function in Eq. 8."

**Response.** We have expanded the implementation description. The interaction matrix $A$ is formed by projecting the concatenated tokens $H=[X;T]$ through a query matrix $W^Q$ and a key matrix $W^K$; it is then **partitioned by modality** into the visual–visual block $A^{vv}$, the text–text block $A^{tt}$, and the cross-modal block $A^{tv}$, and **each block is independently row-wise $\ell_2$-normalized**. The scoring function $\text{AdaVTP}(\cdot,\cdot)$ in Eq. (8) simply indexes the appropriate normalized block (e.g., $\text{AdaVTP}(x_i,t_j)=A^{tv}_{ji}$). These steps are now stated explicitly in the text and are also made unambiguous in the accompanying pseudocode (Algorithm 1), which lists the partition-and-normalize operation and each score computation line by line.

**Comment C4 (Notation inconsistency; softmax as a probabilistic model).**
> "The notation of the loss function is inconsistent between Eq. (5) and Eq. (17). In addition, the implementation of $p(Z\mid X,T)$ in Eq. (12) relies on the softmax function, which seems to be score-based rather than a probabilistic formulation. ... Please provide theoretical justification or relevant prior literature."

**Response.** The notation inconsistency is resolved by the $\mathcal{L}_{\text{IB}}\!\to\!\mathcal{L}_{\text{KL}}$ renaming and the explicit term-by-term mapping (see T2). Regarding the softmax parameterization: this is the standard way to parameterize a **categorical latent distribution**, and we follow the Gumbel-Softmax / Concrete distribution framework. A softmax over learned logits defines a valid categorical distribution over the $N$ tokens, and the Gumbel-Softmax relaxation provides a principled, differentiable probabilistic model for discrete selection. The revised text makes this justification explicit and cites the corresponding literature where $p(Z\mid X,T)$ is introduced.

**Comment C5 (Incomplete comparison; add recent SOTA).**
> "The comparison in Table 1 is incomplete. The paper does not compare the proposed model with recent SOTA methods, such as IVC-Prune [2] and CDPruner [3]."

**Response.** **[ACTION REQUIRED]** We have added comparisons with the suggested recent methods (IVC-Prune and CDPruner) to the main results table under identical settings. *Insert the new rows/numbers and a sentence summarizing how AdaVTP compares.* See also the combined response to Reviewer Fydw (Comment F3).

---

## Reviewer Fydw (Rating: Weak Reject; Confidence: Familiar)

**Comment F1 (Gap between theory and implementation; overloaded $\mathcal{L}_{\text{IB}}$).**
> "Eq. (5) defines [$\hat{\mathcal{J}}$] as an objective that simultaneously minimizes [$I(X;Z\mid T)$] and maximizes [$I(Z;Y\mid T)$], whereas the implemented [$\mathcal{L}_{\text{IB}}$] in Eq. (17) is only a KL divergence between two softmax score distributions over individual tokens. This inconsistent use of the same notation makes it unclear whether [$\mathcal{L}_{\text{IB}}$] refers to the full information-bottleneck objective or only to the KL regularization term used in implementation."

**Response.** We thank the reviewer for pinpointing the root cause, which was the overloaded symbol. We have renamed the implemented KL term to $\mathcal{L}_{\text{KL}}$ and now state explicitly that it realizes **only** the compression (first) term of the bound, while the relevance (second) term is realized by the cross-entropy loss $\mathcal{L}_{\text{CE}}$. The full objective is the weighted sum $\mathcal{L}=\mathcal{L}_{\text{KL}}+\lambda_1\mathcal{L}_{\text{FF}}+\lambda_2\mathcal{L}_{\text{CE}}$. With this renaming, no single symbol stands for both the full objective and an individual term.

**Comment F2 (Why is maximizing the relevance term equivalent to minimizing CE?).**
> "Although the paper later uses the cross-entropy loss [$\mathcal{L}_{\text{CE}}$] as a replacement for the task-relevance term [$I(Z;Y\mid T)$], this substitution is not well justified. ... the paper does not clearly derive or explain why maximizing [the relevance term] can be considered equivalent to minimizing the CE loss."

**Response.** We have added a formal justification (now stated as a proposition with proof). With the language model frozen, the relevance term $\mathbb{E}[\log q(Y\mid Z,T)-\log q(Y\mid T)]$ has a text-only component $\log q(Y\mid T)$ that does not depend on the selection $m$ and is therefore constant during optimization (zero gradient w.r.t. AdaVTP). Taking $q=p_\theta$ as the frozen LLM and factorizing the response autoregressively gives $\mathbb{E}[\log p_\theta(Y\mid Z,T)]=-\mathbb{E}[\mathcal{L}_{\text{CE}}]$. Hence maximizing the relevance term over the selection policy is **exactly** equivalent to minimizing the autoregressive cross-entropy loss. This grounds the use of cross-entropy as a principled estimator of the relevance term rather than an ad-hoc supervision signal.

**Comment F3 (Add recent baselines [1–4]).**
> "The competing methods in Table 1 do not fully cover recent advances ... such as [1] IVC-Prune, [2] Beyond attention or similarity (CDPruner), [3] Beyond Text-Visual Attention, [4] FlowCut."

**Response.** **[ACTION REQUIRED]** We have expanded the comparison to include these recent methods under matched settings, and updated the related-work discussion to position AdaVTP against them. *Insert the new comparison rows and a short discussion.* **[Note to authors:** confirm exact titles/venues of these four papers and add them to the bibliography; we can verify the citations before submission.**]**

---

## Reviewer tfFS (Rating: Weak Accept; Confidence: Familiar)

**Comment S1 (Training cost; not truly plug-and-play).**
> "Training costs are not negligible. Although this is a one-time cost, it is not truly plug-and-play compared to pruning methods that require no training at all. Fine-tuning may be needed when migrating to domain-shifted data or a new VLM."

**Response.** We thank the reviewer and have clarified this in the revised manuscript. AdaVTP introduces only two projection matrices and is trained with the VLM backbone **kept entirely frozen**, so the training cost is small and one-time, and the learned module is reused at inference without modifying the backbone. We have tempered the "plug-and-play" wording to make clear that the *trained* module is lightweight and reusable rather than training-free, and we have added a brief note in the limitations that adapting to a substantially different VLM or domain-shifted data may require re-training the module. *(Optional **[ACTION REQUIRED]:** report the wall-clock training time / number of trainable parameters to quantify the "one-time cost.")*

**Comment S2 (Failure-case analysis).**
> "A more in-depth analysis of failed cases would be more valuable. ... there is limited discussion on situations where AdaVTP may fail."

**Response.** **[ACTION REQUIRED]** We have added a failure-case analysis. *Recommended content:* qualitative examples where text guidance is weak or ambiguous (e.g., very generic queries, or questions requiring global scene context that is spread thinly across many tokens), with a discussion of why these are intrinsically harder for query-conditioned selection. This also connects to the limitation noted in the conclusion regarding query-free settings.

---

## Items Requiring New Experiments or Content (Authors' Checklist)

- [ ] **Efficiency at 128 & 192 tokens** added to the efficiency table (H4).
- [ ] **Discussion of the budget–benefit trade-off** (H4).
- [ ] **Qualitative visualizations at 128 & 192 tokens** (T4).
- [ ] **New SOTA baselines** added to the main table: IVC-Prune, CDPruner, Beyond Text-Visual Attention, FlowCut (C5, F3).
- [ ] **Regenerated framework figure** showing score computation and the three losses, with unified notation (C1).
- [ ] **Failure-case analysis** added (S2).
- [ ] **Training time / trainable-parameter count** reported (S1, optional).
- [ ] **Citations verified** for all newly added baselines and for [1] at the objective (C2, F3).
- [ ] **Math symbols restored** in the verbatim quotations above.
