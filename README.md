# ClinDiag

This is the official repo for the paper [**ClinDiag: Grounding Large Language Model in Clinical Diagnostics**](https://github.com/geteff1/ClinDiag).


## Table of Contents

1. [Installation](#installation)
    * [Set up a virtual environment](#venv)
    * [Install dependencies](#dep)
2. [Usage](#usage)
    * [Human+LLM](#human-llm)
    * [Human Alone](#human)
    * [Ablation Study](#ablation)
        * [Multi-doctor agents](#multi-doctor)
        * [Critic agent](#critic)
        * [Expert prompt](#expert)
3. [Datasets](#datasets)
    * [ClinDiag-Benchmark](#benchmark)
    * [Standardized Patients](#patients)
    * [Fine-Tuning Data](#fine-tune)


## Installation <a name="installation"></a>

### Set up a virtual environment <a name="venv"></a>

When using pip it is generally recommended to install packages in a virtual environment to avoid modifying system state. We use [`conda`](https://www.anaconda.com/download/) as an example here:

Create and activate:
```bash
$ conda create -n clindiag python==3.11.1
$ conda activate clindiag
```

To deactivate later, run:
```bash
(clindiag) conda deactivate
```

### Install dependencies <a name="dep"></a>

```bash
(clindiag) pip install -r requirements.txt
```

## Usage <a name="usage"></a>

Before running a script, go to [`configs/OAI_Config_List.json`](https://github.com/geteff1/ClinDiag/blob/main/configs/OAI_Config_List.json) to fill in your model and API key. 
```json
{
    "model": "gpt-4o-mini",
    "api_key": "[YOUR_API_KEY]",
    "base_url": "[YOUR_BASE_URL]",
    "tags": [
        "x_gpt4omini"
    ]
}
```
The tags will be used to filter selected model(s) for each stage, see `parse_args()` for details.

### Human+LLM <a name="human-llm"></a>

This script implements a human-LLM collaboration framework where LLMs serve as an assistant to answer physician's questions.

```bash
(clindiag) python code/test_human_llm.py --data_dir benchmark_dataset
```

### Human Alone <a name="human"></a>

This is to simulate the human-alone scenario where a physician performs the clinical diagnostic procedure all by itself in the ClinDiag framework.

```bash
(clindiag) python code/test_human_alone.py --data_dir benchmark_dataset
```

### Ablation Study <a name="ablation"></a>

The following scripts were used for ablation study. We examined the effects of (1) multi-doctor collaboration, (2) introducing a critic agent, and (3) prompt engineering on diagnostic performance. 

#### 1. Multi-doctor agents <a name="multi-doctor"></a>

We tested the effect of having 2–3 doctor agents collaborate in the clinical decision making process. 

```bash
(clindiag) python code/trial_stepwise_multiagent_converse.py --data_dir benchmark_dataset --num_specialists 2
```

`--num_specialists`: number of doctor agents, defaults to 3

#### 2. Critic agent <a name="critic"></a>

This framework incorporates a critic agent to suggest further revisions on doctor agent's questions.

```bash
(clindiag) python code/trial_stepwise_nochain_critic.py --data_dir benchmark_dataset --model_name_critic x_gpt4omini
```

`--model_name_critic`: model used for the critic agent, defaults to gpt-4o-mini

#### 3. Expert prompt <a name="expert"></a>

This script adopts expert-generated prompts.

```bash
(clindiag) python code/trial_stepwise_nochain_expert_prompt.py --data_dir benchmark_dataset
```

## Datasets <a name="datasets"></a>

### ClinDiag-Benchmark (n=4,421) <a name="benchmark"></a>

`./benchmark_dataset.zip`

(To uncompress, run `unzip benchmark_dataset.zip` in the root directory)

A comprehensive clinical dataset comprising 4,421 real-world cases, encompassing both rare and common diseases across 32 specialties.

### Standardized Patients (n=35) <a name="patients"></a>

`./human_examiner_scripts/`

A set of 35 patient scripts sourced from the hospital’s Objective Structured Clinical Examination (OSCE) test dataset for standardized patient training.

### Fine-Tuning Data (n=7,616) <a name="fine-tune"></a>

`./finetune_data.zip`

(To uncompress, run `unzip finetune_data.zip` in the root directory)

The multi-turn chat dataset used for fine-tuning a chat model. Each conversation example was constructed from a quality-checked real-world case and structured to adhere to standard clinical diagnostic practice. The data is available in both `jsonl` and `json` formats. 

`finetune_data_messages.jsonl`:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`finetune_data_conversations.json`:
```json
{
    "conversations": [
        [
            {"from": "system", "value": "..."},
            {"from": "user", "value": "..."},
            {"from": "assistant", "value": "..."},
        ],
        [
            {"from": "system", "value": "..."},
            {"from": "user", "value": "..."},
            {"from": "assistant", "value": "..."},
        ]
    ]
}
```