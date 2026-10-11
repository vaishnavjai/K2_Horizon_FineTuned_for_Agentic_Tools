# 🔬 K2 Horizon: Tool-Use Fine-Tuning Experiment

A personal learning experiment exploring parameter-efficient fine-tuning (PEFT/LoRA) on the **K2 Horizon** open foundation model family, focusing on agentic tool calling and schema adherence.

---

## 🎯 Goal & Context

This repository documents an educational deep-dive into the end-to-end LLM fine-tuning lifecycle. The objective was to test whether a lightweight LoRA fine-tuning run on a specialized tool-calling dataset could measurably improve function call precision and syntax validity on the K2 Horizon base architecture.

> **Key Takeaway:** As shown in the evaluation curves below, **the net performance improvement was marginal**. This experiment serves as a clear case study on base model prior capabilities, data mixture requirements, and the challenges of domain adaptation for structured generation.

---

## 📊 Results & Observations

| Metric / Stage | Base Model | LoRA Fine-Tuned | Delta |
| :--- | :--- | :--- | :--- |
| **Tool Call Validity** | High baseline | Marginal gain | ~Minimal |
| **Schema Strictness** | Strong out-of-the-box | Slight edge-case fixes | +Marginal |
| **Loss Trajectory** | — | Rapid plateau | Early saturation |

### What the Loss & Eval Curves Show:
1. **High Base Capability:** Modern K2 Horizon models already carry significant instruction/chat priors for JSON structures; small SFT datasets do not dramatically shift the baseline without substantial distributional diversity.
2. **Early Saturation:** Validation loss flattened out quickly, indicating the adapter was mostly fitting syntax style rather than unlocking new multi-step reasoning capabilities.
3. **Data Quality > Volume:** Minor performance gains highlight the need for harder negative examples, failed-tool recovery paths, and longer multi-turn tool trajectories rather than generic single-turn function pairs.

---

## 🛠️ Pipeline Overview

- **Base Architecture:** K2 Horizon
- **Technique:** LoRA / QLoRA using `peft` and Hugging Face `trl` (`SFTTrainer`)
- **Dataset Focus:** Function definitions, structured JSON arguments, and mock API call-and-response traces
- **Target Layers:** Attention projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`)

---

## 📁 Repository Structure

```text
├── notebooks/
│   └── fine_tune_k2_horizon_agentic.ipynb   # Full interactive training, loss logging, and eval run
├── requirements.txt                         # Python dependencies
└── README.md
```
## 🚀 Running the Notebook
Install dependencies:

```Bash
pip install -r requirements.txt
Launch Jupyter:
```
```Bash
jupyter notebook notebooks/fine_tune_k2_horizon_agentic.ipynb
```
Follow the notebook steps: environment setup, dataset tokenization, LoRA adapter configuration, training loop, and evaluation charting.

## 💡 Lessons Learned
Benchmark the base thoroughly first: Quantitative probing of the unmodified model reveals whether LoRA is even addressing a real bottleneck.

Agentic data needs error loops: Simply fine-tuning on "happy path" API calls produces trivial gains; true agent resilience requires training on syntax repair and error feedback.

Learning curve over benchmark chasing: Setting up clean training loops, tracking loss plateaus, and understanding why an experiment didn't yield huge improvements is far more informative than an uninspected win.
