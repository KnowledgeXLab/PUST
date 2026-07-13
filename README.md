# PUST: Proxy-guided Update Signal Transfer

> **Decoupling Exploration and Alignment: Proxy-Guided Update Signal Transfer for LLM Post-Training**

PUST decouples LLM post-training into three stages: **proxy exploration** → **update-signal extraction** → **signal transfer**. A lightweight proxy model performs low-cost trial-and-error, while the primary model aligns to relative improvement signals—enabling asynchronous, reusable, and weak-to-strong cross-model post-training.

<p align="center">
  <a href="figs/pipeline_v2.pdf"><b>Figure 1</b> — Post-training pipeline comparison</a>
</p>

## Method

<p align="center">
  <a href="figs/method.pdf"><b>Figure 3</b> — Overview of PUST</a>
</p>

The core update signal is the relative improvement of the proxy policy:

$$\Delta_\phi(a \mid s_t) = \log \frac{\pi_\phi^+(a \mid s_t)}{\pi_\phi(a \mid s_t)}$$

During transfer, a calibration coefficient $\lambda$ penalizes over-updating on the primary model. See Section 3 of the paper for details.

## Results

Evaluated on Qwen3 models (Math: DeepMath-103K, Code: Eurus-RL-Code):

- Signals from weaker proxies (1.7B / 4B) effectively improve the 8B primary model
- The same signal can be reused across primary models of different scales in only 50 steps
- Multi-hop transfer is supported (e.g., 4B → 1.7B → 8B)

<p align="center">
  <img src="figs/table1_math.png" width="95%"/>
  <br/><em>Table 1. Math benchmarks (Mean@16)</em>
</p>

<p align="center">
  <img src="figs/table2_code.png" width="80%"/>
  <br/><em>Table 2. Code benchmarks (Mean@16)</em>
</p>

<p align="center">
  <img src="figs/table3_math.png" width="95%"/>
  <br/><em>Table 3. Reusability of proxy update signals</em>
</p>

<p align="center">
  <img src="figs/table4_transitivity.png" width="90%"/>
  <br/><em>Table 4. Transitivity of proxy update signals</em>
</p>

<p align="center">
  <a href="figs/sensitivity_analysis.pdf"><b>Figure 4</b> — Sensitivity to proxy model and calibration coefficient λ</a>
</p>

<!-- ## Citation

```bibtex
@article{pust2026,
  title   = {Decoupling Exploration and Alignment: Proxy-Guided Update Signal Transfer for LLM Post-Training},
  author  = {author 11 and author 2 and author 3},
  year    = {2026}
}
``` -->
