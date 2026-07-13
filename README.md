# PUST: Proxy-guided Update Signal Transfer

> **Decoupling Exploration and Alignment: Proxy-Guided Update Signal Transfer for LLM Post-Training**

PUST decouples LLM post-training into three stages: **proxy exploration** → **update-signal extraction** → **signal transfer**. A lightweight proxy model performs low-cost trial-and-error, while the primary model aligns to relative improvement signals—enabling asynchronous, reusable, and weak-to-strong cross-model post-training.

**Figure 1** — Post-training pipeline comparison

## Method

**Figure 3** — Overview of PUST

The core update signal is the relative improvement of the proxy policy:

$$\Delta_\phi(a \mid s_t) = \log \frac{\pi_\phi^+(a \mid s_t)}{\pi_\phi(a \mid s_t)}$$

During transfer, a calibration coefficient $\lambda$ penalizes over-updating on the primary model. See Section 3 of the paper for details.

## Results

Evaluated on Qwen3 models (Math: DeepMath-103K, Code: Eurus-RL-Code):

- Signals from weaker proxies (1.7B / 4B) effectively improve the 8B primary model
- The same signal can be reused across primary models of different scales in only 50 steps
- Multi-hop transfer is supported (e.g., 4B → 1.7B → 8B)

  
*Table 1. Math benchmarks (Mean@16)*

  
*Table 2. Code benchmarks (Mean@16)*

  
*Table 3. Reusability of proxy update signals*

  
*Table 4. Transitivity of proxy update signals*

**Figure 4** — Sensitivity to proxy model and calibration coefficient λ

