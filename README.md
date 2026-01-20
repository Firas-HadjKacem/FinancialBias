# 📘 **Financial Bias: On Detecting Bias in Financial Language Models**

[![Paper](link_to_paper)](#)  
[![Demo](https://huggingface.co/spaces/FirasHadjKacem/FBDF)](#)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

This is a reproducible research framework for detecting, quantifying, and attributing *demographic bias* in financial language models.  
It combines bias detection using **metamorphic testing**, bias quantification using **probability-shift metrics** (Jensen–Shannon Divergence & Cosine Similarity), and bias explanation using **word-level attribution** (an adaptation of TokenSHAP) to uncover **atomic and hidden intersectional bias** in classifier-based and generative financial models.

This repository accompanies the research paper:

📄 **“On detecting bias across Financial Small and Large
Language Model”**  
🔗 *(Link Here)*  

🎥 **Live Demo**  
🔗 *(https://huggingface.co/spaces/FirasHadjKacem/FBDF)*


> **Note:**  
> The public demo currently runs **only the classifier-based models**  
> (FinBERT, DeBERTa-v3, DistilRoBERTa) due to computation and memory constraints.  
> 
> The **full codebase** supports all models listed in the
> [Supported Models & Tokenizers](#-supported-models--tokenizers) section,
> including FinMA and FinGPT, when run locally on suitable hardware.

---

## 🔗 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Financial Language Models Used](#financial-language-models-used)
- [Running the Pipeline](#running-the-pipeline)
- [Finding Label IDs (LLaMA Models)](#finding-label-ids-llama-models)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Results Overview](#results-overview)
- [Citing](#citing)
- [License](#license)

---

# 📌 Overview

Financial language models are widely deployed in real-world decision-making. However, they often exhibit **bias toward demographic attributes** (race, gender, body descriptors).

This framework evaluates bias using:

1. **Mutations (atomic & intersectional)** applied to financial statements  
2. **Prediction comparison** between original and mutated inputs  
3. **Probability-shift metrics** (JSD, Cosine)  
4. **Word-level attribution** using a **JSD-based TokenSHAP** variant  
5. **Bias lexicon activation** and word-importance ranking  
6. **Simulation of input prioritization strategies**  
7. **Support for multiple financial language models**

The evaluation follows the same methodology discussed in the paper.

---

# ⭐ Key Features

### 🔹 Metamorphic Input Generation (via HInter)

This framework **does not generate mutations by itself**.

To reproduce the metamorphic testing setup described in the paper, users must first generate **atomic** and **intersectional** mutant sentences using the **HInter** framework:

👉 https://github.com/BadrSouani/HInter

HInter provides:
- SBIC-derived demographic lexicons  
- Mutation operators  
- Generation of Word 1 / Word 2 / Replacement 1 / Replacement 2  
- Atomic + intersectional mutant generation  

Once users generate the mutation `.pkl` files with HInter, this framework consumes them directly for:

- bias detection  
- probability-shift measurement  
- word-level attribution (TokenSHAP-JSD adaptation)  
- multi-model evaluation  

This ensures full compatibility with the mutation pipeline used in the paper.

### 🔹 Unified Model API  
Works across:
- FinMA  
- FinGPT  
- FinBERT  
- DeBERTa-v3  
- DistilRoBERTa  
- And any custom LLaMA-based or classifier financial language model  

### 🔹 JSD-Based TokenSHAP  
We replace the default cosine/logit value function from TokenSHAP with **Jensen–Shannon Divergence over probability distributions**, which better captures subtle distributional bias shifts.

### 🔹 Multi-GPU Acceleration  
Automatically distributes mutant files across available GPUs.

### 🔹 Transparent, Reproducible Output  
All results (predictions, probabilities, bias flags, SHAP values) are saved as **Excel spreadsheets** for analysis.

---

# ⚙ Installation

### 1. Clone the repo
```bash
git clone https://github.com/Firas-HadjKacem/FinancialBias
cd FinancialBias
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Login to HuggingFace
```bash
huggingface-cli login
```

---

# 📦 Dataset Format

Each mutant file is a `.pkl` containing:

```
[ metadata, mutants ]
```

Where:

### `metadata["header"]` (list of column names)  
Example:
```
[
  "Word 1", "Replacement 1",
  "Word 2", "Replacement 2",
  "Mutant_Word_1", "Mutant_Word_2",
  "Mutant_Intersectional",
  "Index", "Original", "Similarity"
]
```

### `mutants` (list of rows aligned with header)

Index resolution is **dynamic**, not hardcoded.

---

# 🔗 Financial Language Models Used

The following financial language models were evaluated in the study.  
Use these exact HuggingFace model IDs when running the pipeline.

| Model Name         | HuggingFace ID | Tokenizer ID |
|--------------------|----------------|--------------|
| **FinMA (7B)**     | `ChanceFocus/finma-7b-full` | `ChanceFocus/finma-7b-full` |
| **FinGPT (LLaMA-2 base)** | `oliverwang15/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT` | `meta-llama/Llama-2-7b-chat-hf` |
| **FinBERT**        | `ProsusAI/finbert` | N/A |
| **DeBERTa-v3 (financial fine-tuned)** | `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` | N/A |
| **DistilRoBERTa (financial fine-tuned)** | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` | N/A |

### Notes
- **LLaMA-based models require a base tokenizer** separate from the fine-tuned model weights.  
  This is why FinGPT uses `meta-llama/Llama-2-7b-chat-hf` as its tokenizer.
- All BERT-family models (FinBERT, DeBERTa, DistilRoBERTa) use the **same ID** for both model and tokenizer, and the tokenizer does not have to be inserted.
- FinMA uses its **own tokenizer**, identical to the model ID.

---

# 🚀 Running the Pipeline

Run the pipeline:

## 1. Generate Mutants
```bash
python scripts/process_mutants.py \
  --model_type <bert|llama> \
  --model_name <hf-model-id> \
  --mutants_dir <path-to-mutant-pkl-files> \
  --output_dir <path-to-output-dir> \
  --sampling_ratio <float> \
  --max_combinations <int> \
  --batch_size <int> \
  --num_gpus <int> \
  [--shap] \
  [--base_tokenizer_id <llama-tokenizer-id>] \
  [--positive_ids <comma-separated-ids>] \
  [--negative_ids <comma-separated-ids>] \
  [--neutral_ids <comma-separated-ids>]
```

Examples:

- For FinBERT, a BERT classifier:
```bash
python scripts/process_mutants.py \
  --model_type bert \
  --model_name ProsusAI/finbert \
  --mutants_dir data/mutants \
  --output_dir outputs/finbert \
  --sampling_ratio 0.3 \
  --max_combinations 1000 \
  --batch_size 128 \
  --num_gpus 1
```

- For FinGPT, a LLaMA-based generative model:
```bash
python scripts/process_mutants.py \
  --model_type llama \
  --model_name oliverwang15/FinGPT_v32_Llama2_Sentiment_Instruction_LoRA_FT \
  --base_tokenizer_id meta-llama/Llama-2-7b-chat-hf \
  --positive_ids 6374 \
  --negative_ids 8178,22198 \
  --neutral_ids 21104 \
  --mutants_dir data/mutants \
  --output_dir outputs/fingpt \
  --sampling_ratio 0.3 \
  --max_combinations 1000 \
  --batch_size 16 \
  --num_gpus 1 \
  --shap
```
---

# 🔍 Finding Label IDs (LLaMA Models)

For generative models, sentiment label words (positive/negative/neutral) may map to **multiple token IDs**.

Extract all valid IDs using:

```bash
python utils/find_label_ids.py <base_tokenizer_id>
```

---

# 📂 Project Structure

```
BiasEvaluation/
│
├── analysis/
│   ├── __init__.py           # Package init for analysis components
│   ├── bias_analyzer.py      # High-level bias analysis logic
│   └── tokenShap.py          # JSD-based TokenSHAP variant
│
├── core/
│   ├── __init__.py           # Package init for core abstractions
│   ├── base.py               # Base SHAP engine / model base classes
│   ├── models.py             # Wrappers for BERT / LLaMA financial models
│   └── splitters.py          # String / tokenizer-based splitters
│
├── utils/
│   ├── __init__.py           # Package init for utilities
│   ├── data_loader.py        # Unified data loading + multi-GPU processing
│   └── helpers.py            # JSD metric, prompt builder, I/O helpers
│
├── data/                     # Contains the SBIC-derived bias dictionary and example mutant files
│   └── bias                  #The bias dictionary
│       └── body
│       └── gender
│       └── race
│   └── examples              #The example mutant files
├── scripts/
│   ├── find_label_ids.py     # Script to discover sentiment label token IDs
│   └── process_mutants.py    # Main entry point to run the pipeline
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

# 🔁 Reproducibility

We ensure **research-grade reproducibility**:

### ✔ Deterministic sampling  
SHAP always includes *omit-1* combinations.

### ✔ Fixed seeds  
```python
random.seed(42)
np.random.seed(42)
set_seed(42)
```

### ✔ Consistent prompting across models  
A unified prompt template ensures comparability.

### ✔ Version-locked dependencies  
All requirements are tested for compatibility.

### ✔ Full Excel output  
All metrics are human-readable.

---

# 📚 Citing

```
@inproceedings{FinancialBias2026,
  title     = {On detecting bias across Financial Small and Large
Language Models},
  author    = {Firas Hadj Kacem, Ahmed Khanfir, Mike Papadakis},
  booktitle = {...},
  year      = {2026}
}
```

---

# 📜 License

Distributed under the MIT License.  
See `LICENSE` for details.

---

# 🎉 Acknowledgments

- HInter authors (mutation inspiration)  
- SBIC lexicon contributors  
- TokenSHAP authors (attribution backbone)  
- FinMA, FinGPT, FinBERT, DeBERTa, DistilRoBERTa creators  
- University of Luxembourg; SnT: Interdisciplinary Centre for Security, Reliability and Trust; SerVal Group
- Reviewers & collaborators  

---

# 📬 Contact

📧 firashadjkacem@ieee.org  
🐙 GitHub: @Firas-HadjKacem  