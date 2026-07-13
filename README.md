# Proxy Exploration and Reusable Guidance: A Modular Post-Training Paradigm via Proxy-Guided Update Signals

PUST decouples LLM post-training into three stages: **proxy exploration** → **update-signal extraction** → **signal transfer**. A lightweight proxy model performs low-cost trial-and-error, while the primary model aligns to relative improvement signals—enabling asynchronous, reusable, and weak-to-strong cross-model post-training.

<p align="center">
  <img src="figs/pipeline_v2.png" width="90%"/>
  <br/><em>Figure 1. Post-training pipeline comparison</em>
</p>

## Method

<p align="center">
  <img src="figs/method.png" width="85%"/>
  <br/><em>Figure 3. Overview of PUST</em>
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


Figure 4 sweeps the calibration coefficient $\lambda$ when transferring signals from Qwen3-1.7B and Qwen3-4B to Qwen3-8B. Performance peaks at $\lambda^* \approx 1.51$ for the 1.7B proxy and $\lambda^* \approx 1.08$ for the 4B proxy—both above 1.0, suggesting that proxy signals should be down-scaled during transfer to avoid over-updating. The 4B proxy also reaches a higher peak than the 1.7B proxy, indicating that stronger proxies yield higher-quality transferable signals.

<p align="center">
  <img src="figs/sensitivity_analysis.png" width="70%"/>
  <br/><em>Figure 4. Sensitivity to proxy model and calibration coefficient λ</em>
</p>
