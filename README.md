# R2 Update

1. The RQ2 and RQ3 training scripts have been updated.

**Files updated:**
- `training_scripts/rq2/run_rq2_ae.py`
- `training_scripts/rq2/run_rq2_codebert.py`
- `training_scripts/rq2/run_vllm_rq2_qwen.py`
- `training_scripts/rq3/run_rq3_rnn.py`
- `training_scripts/rq3/run_rq3_codebert.py`
- `training_scripts/rq3/run_rq3_ae.py`
- `training_scripts/rq3/run_vllm_rq3_qwen.py`

2. The original obfuscation-based semantic attack (renaming variables to random identifiers like `v_123_name`) has been replaced with a **synonym-substitution** approach using the NLTK WordNet lexical database.

**New behavior:**
- Splits camelCase/PascalCase identifiers into constituent words (e.g., `ProcessData` -> `["process", "data"]`).
- Queries WordNet for semantically similar synonyms.
- Reassembles substituted words back into camelCase/PascalCase matching the original naming convention (e.g., `ProcessData` -> `HandleInformation`, `reader` -> `accessor`, `data` -> `information`, `path` -> `location`).
- Maintains a file-level consistent mapping so every occurrence of the same identifier within a file is replaced identically.
- Skips language keywords, strings, and comments.

**Files updated:**
- `data_curation_scripts/attacks/java_semantic_attack.py`
- `data_curation_scripts/attacks/csharp_semantic_attack.py`



# Replication Package

This replication package contains a clean, sanitized, and complete pipeline to reproduce the core results of our study on the robustness of deep learning-based code smell detection models.

## Project Structure

*   `config/`: Configuration files enforcing data integrity.
    *   `dataset_splits.zip`: The compressed core configuration file that defines the exact train/test split IDs to ensure deterministic replication and 1:1 data distributions without data leakage. Before running the scripts, please extract it to `config/dataset_splits.json`.

*   `data_curation_scripts/`: Scripts used for dataset extraction, tokenization, and our complete suite of adversarial attack generation.
    *   `data_curation_main.py`: The master runner script that extracts the clean source code using the defined splits.
    *   `csharp/cs_code_split_runner.py`: C# utility for splitting repositories into methods and classes.
    *   `csharp/cs_designite_runner.py`: C# utility for running the Designite smell detection tool.
    *   `csharp/cs_learning_data_generator.py`: C# utility for extracting labeled learning data from raw files.
    *   `java/java_codeSplit_runner.py`: Java utility for splitting source code.
    *   `java/java_designite_runner.py`: Java utility for running DesigniteJava.
    *   `java/java_learning_data_generator.py`: Java utility for generating labeled datasets.
    *   `attacks/csharp_semantic_attack.py`: Generates semantic and structural attacks for C# source code.
    *   `attacks/java_semantic_attack.py`: Generates variable renaming semantic attacks for Java source code.
    *   `attacks/java_structural_attack.py`: Generates dead-code injection structural attacks for Java.
    *   `attacks/generate_nlp_attack.py`: Uses TextAttack to generate typo injection (NLP) attacks for both languages.
    *   `utils/fast_parallel_tokenizer_adv.py`: Accelerated tokenizer for processing adversarial source code.
    *   `utils/tokenizer_runner.py`: Standard tokenizer used during the clean data curation phase.

*   `results/`: Contains the raw CSV performance outputs for all tested models (RNN, CodeBERT, AE, Qwen) across all configurations. These files hold the performance metrics that are parsed to generate the exact manuscript tables.
    *   `rq1_results.csv`: Clean baseline performances (F1, AUC).
    *   `rq2_results.csv`: Performance under adversarial attacks (Semantic, Structural, Hybrid, NLP).
    *   `rq3_clean_results.csv`: Clean-test performance after adversarial training at 10%, 30%, 50% ratios.
    *   `rq3_adv_results.csv`: Adversarial-test performance after adversarial training at 10%, 30%, 50% ratios.

*   `table_generation/`: Contains python scripts to parse the results and output the exact performance values presented in the paper's tables directly to the console.
    *   `rq1.py`: Reads `rq1_results.csv` and generates the RQ1 Baseline Model Performance table.
    *   `rq2.py`: Reads `rq1_results.csv` and `rq2_results.csv` and generates the RQ2 Model Degradation tables (F1 and AUC).
    *   `rq3.py`: Reads all result CSVs and generates the RQ3 Adversarial Training Performance tables for both Clean and Adversarial test datasets.

*   `training_scripts/`: The sanitized, pure Python training, attack, and defense scripts, categorized by Research Question (RQ1, RQ2, RQ3).
    *   `rq1/run_optimized_rq1.py`: Trains and evaluates the baseline RNN-LSTM model on clean data.
    *   `rq1/run_codebert.py`: Fine-tunes and evaluates the baseline CodeBERT model.
    *   `rq1/run_optimized_ae.py`: Trains and evaluates the baseline AutoEncoder (AE) model.
    *   `rq1/run_vllm_qwen_baseline.py`: Evaluates the zero-shot baseline capability of the Qwen3.5-9B LLM.
    *   `rq2/run_rq2_rnn.py`: Evaluates the trained RNN model against the four adversarial attack modes.
    *   `rq2/run_rq2_codebert.py`: Evaluates the fine-tuned CodeBERT against adversarial attacks.
    *   `rq2/run_rq2_ae.py`: Evaluates the AutoEncoder against adversarial attacks.
    *   `rq2/run_vllm_rq2_qwen.py`: Evaluates Qwen3.5-9B against adversarial attacks.
    *   `rq3/run_rq3_rnn.py`: Performs adversarial training for the RNN at 10%, 30%, and 50% ratios.
    *   `rq3/run_rq3_codebert.py`: Performs adversarial training for CodeBERT.
    *   `rq3/run_rq3_ae.py`: Performs adversarial training for the AutoEncoder.
    *   `rq3/run_vllm_rq3_qwen.py`: Implements Adversarial In-Context Learning (Adv-ICL) for Qwen3.5-9B.
    *   `utils/inputs.py`: Data loading and mapping module for model pipelines.
    *   `utils/rnn_lstm.py`: Model architecture definition for the core sequence models.
    *   `utils/rq1_rnn_emb_lstm.py`: Training routines specific to the RNN model.
    *   `utils/metrics_util.py`: Shared functions for computing F1, AUC, and MCC metrics.
    *   `utils/path_config.py`: Path mapping utility for finding the correct datasets.

*   `test_reproduction.py`: An automated test script ensuring the CSV files parse successfully and aggregate without errors.

---

## 1. Data Setup & Adversarial Generation

Because the raw source code dataset (containing hundreds of thousands of `.java` and `.cs` files) is extremely large, it is not bundled directly inside this repository.

### Step 1.1: Download Original Dataset
Please clone the original raw source code dataset repository from Sharma et al.:
```bash
git clone https://github.com/tushartushar/DeepLearningSmells
```
Extract or arrange the dataset such that the root path corresponds to:
`../original/DeepLearningSmells` (or adjust the paths in the scripts accordingly to point to your clone).

### Step 1.2: Run Data Curation (Extraction & Filtering)
Navigate to the `data_curation_scripts/` directory. This directory contains a streamlined pipeline to curate the raw data.
```bash
cd data_curation_scripts
python3 data_curation_main.py
```
*   **What this does:** The `data_curation_main.py` acts as a master runner that utilizes the scripts in `csharp/` and `java/` (such as `cs_code_split_runner.py` and `java_learning_data_generator.py`). It reads the unzipped **`config/dataset_splits.json`** to extract the exact positive and negative samples required for our balanced, deduplicated dataset, completely preventing data leakage between the training and testing sets. **This JSON file is absolutely essential as it enforces the exact, reproducible data splits used in our study.**

### Step 1.3: Generate Adversarial Attacks
Once the clean data is extracted, you can generate the adversarial test sets and augmented training sets using the scripts in the `data_curation_scripts/attacks/` folder.
```bash
# Generate Semantic / Structural / Hybrid attacks for C#
python3 attacks/csharp_semantic_attack.py

# Generate Semantic attacks for Java
python3 attacks/java_semantic_attack.py

# Generate Structural (Dead-code injection) attacks for Java
python3 attacks/java_structural_attack.py

# Generate NLP (TextAttack typo injection) attacks for both languages
python3 attacks/generate_nlp_attack.py
```
Ensure all processed output data is placed inside a `data/` folder at the root of this replication package before proceeding to model training.

---

## 2. How to Execute the Experiments

### Step 2.1: Run the Training and Evaluation Pipeline
To reproduce the experimental results from scratch, navigate to the `training_scripts/` directory and execute the scripts sequentially.

> **Note:** The scripts output their metric results directly into the `../results/` directory as CSV files.

```bash
cd training_scripts

# Example: Run an RQ1 Baseline
python3 rq1/run_codebert.py Java ComplexMethod

# Example: Run an RQ2 Attack Evaluation
python3 rq2/run_rq2_codebert.py Java ComplexMethod semantic

# Example: Run an RQ3 Defense Evaluation (at 30% ratio)
python3 rq3/run_rq3_codebert.py Java ComplexMethod semantic 0.3
```

## 3. Table Generation & Verification

Once the models have finished executing and populated the `results/` folder (which comes pre-populated in this replication package with the data matching the manuscript), you can generate the exact tables used in the manuscript directly in your console.

### Step 3.1: Run the Table Generators
```bash
cd table_generation

# Generate Tables for RQ1
python3 rq1.py

# Generate Tables for RQ2
python3 rq2.py

# Generate Tables for RQ3
python3 rq3.py
```
The scripts will automatically group the raw data, calculate the exact statistical averages, and print formatted, human-readable console tables.

### Step 3.2: Automated Test Validation
To ensure that all data is correctly formatted, parsed, and contains no missing values (NaNs), run the reproduction test script from the root directory:
```bash
python3 test_reproduction.py
```
This guarantees that all output tables correctly aggregate and match the integrity of the submitted manuscript.
