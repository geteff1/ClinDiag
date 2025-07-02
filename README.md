# ClinDiag

This is the official repo for the paper [**ClinDiag: Grounding Large Language Model in Clinical Diagnostics**](https://github.com/geteff1/ClinDiag).

🔗 **Demo website: https://clindiag.streamlit.app/**


## Table of Contents

1. [Installation](#installation)
    * [Set up a virtual environment](#venv)
    * [Install dependencies](#dep)
    * [Add API configs](#configs)
2. [Demo](#demo)
    * [ClinDiag-GPT](#clindiag-gpt)
    * [ClinDiag-Framework](#clindiag-framework)
3. [Datasets](#datasets)
    * [ClinDiag-Benchmark](#benchmark)
    * [Standardized Patients](#patients)
    * [Fine-Tuning Data](#fine-tune)
4. [Usage](#usage)
    * [Human+LLM](#human-llm)
        * [Human as *Doctor*](#human-doctor)
        * [Human as *Provider*](#human-provider)
    * [Human Alone](#human)
    * [Ablation Study](#ablation)
        * [Multi-doctor agents](#multi-doctor)
        * [Critic agent](#critic)
        * [Expert prompt](#expert)


## Installation <a name="installation"></a>

This installation requires *no* non-standard hardware.

The installation time depends on internet connection bandwidth and typically takes less than 30 mins.

### Set up a virtual environment <a name="venv"></a>

When using pip it is generally recommended to install packages in a virtual environment to avoid modifying system state. We use [`conda`](https://www.anaconda.com/download/) as an example here:

Create and activate:
```bash
$ conda create -n clindiag python==3.12
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

### Add API configs <a name="configs"></a>

Before running a script, go to [`configs/OAI_Config_List.json`](https://github.com/geteff1/ClinDiag/blob/main/configs/OAI_Config_List.json) to fill in your model name and API key. 
```json
{
    "model": "gpt-4o-mini",
    "api_key": "[YOUR_API_KEY]",
    "base_url": "[YOUR_BASE_URL](optional)",
    "tags": [
        "gpt-4o-mini"
    ]
}
```
The tags will be used to filter selected model(s) for each stage, see `parse_args()` in code scripts for details.


## Demo <a name="demo"></a>

### 💬 ClinDiag-GPT <a name="clindiag-gpt"></a>

- 🔗 **Demo website: https://clindiag.streamlit.app/**

Trained on our [fine-tuning dataset](#fine-tune), *ClinDiag-GPT* showed superior performance in clinical diagnostic procedures. Although we can't provide direct API access to our fine-tuned model for security and cost considerations, feel free to chat with ClinDiag-GPT on our [demo website](https://clindiag.streamlit.app/).

Running the demo on our website doesn't require any additional installation or configuration. It normally takes 10-20 mins to go through a full demo case.

### 🏗 ClinDiag-Framework <a name="clindiag-framework"></a>

To test out the 2-agent *ClinDiag-Framework*, run:

```bash
(clindiag) python code/trial_doctor_provider.py --data_dir sample_data
```

- `--data_dir`: root directory of input case folders. Here we use `sample_data` for a quick demo
- `--output_dir`: directory to save output files, defaults to `output`
- `--model_name_{history/pe/test/diagnosis}`: models used for the doctor agent in each stage, defaults to `gpt-4o-mini`
- `--model_name_provider`: model used for the provider agent, defaults to `gpt-4o-mini`


## Datasets <a name="datasets"></a>

### ClinDiag-Benchmark <a name="benchmark"></a>

`./benchmark_dataset/`

A comprehensive clinical dataset of 2,021 real-world cases, encompassing both rare and common diseases across 32 specialties.

> **Note:** The full ClinDiag-Benchmark (n=4,421) used in our study comprises three subsets: 
>   1. Challenging Case Subset (n=1,719)
>   2. Rare Disease Subset (n=302)
>   3. Emergency Case Subset (n=2,400)
>
> The provided `benchmark_dataset/` only contains the former two subsets.
>
> The Emergency Case Subset is derived from *MIMIC-IV-Ext Clinical Decision Making Dataset*, which is officially available at https://physionet.org/. Users should follow the guidelines provided there to gain access to the MIMIC dataset and adhere to their data use policy. 

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


## Usage <a name="usage"></a>

Below are instructions to run experiments on the full [benchmark dataset](#benchmark).

### Human+LLM <a name="human-llm"></a>

#### Human as *Doctor* <a name="human-doctor"></a>

This script implements a version of our human-LLM collaboration framework where LLMs serve as an assistant to answer physician's questions.

```bash
(clindiag) python code/human_as_doctor.py --data_dir benchmark_dataset --output_dir output
```

By default, output files will be saved to `./output/human_as_doctor/...`. You can set your desired output directory by specifying `--output_dir` (same for all scripts below).

#### Human as *Provider* <a name="human-provider"></a>

Our framework also allows human to act as the information provider, while LLMs are the doctors who drive the diagnostic process.

```bash
(clindiag) python code/human_as_provider.py --data_dir benchmark_dataset
```

### Human Alone <a name="human"></a>

This is to simulate the human-alone scenario where a physician performs the clinical diagnostic procedure all by itself within the ClinDiag-Framework.

```bash
(clindiag) python code/human_alone.py --data_dir benchmark_dataset
```

### Ablation Study <a name="ablation"></a>

The following scripts were used for ablation study. We examined the effects of (1) multi-doctor collaboration, (2) introducing a critic agent, and (3) prompt engineering on diagnostic performance. 

#### 1. Multi-doctor agents <a name="multi-doctor"></a>

We tested the effect of having 2–3 doctor agents collaborate in the clinical decision-making process. 

```bash
(clindiag) python code/trial_multidoctor.py --data_dir benchmark_dataset --num_specialists 2
```

- `--num_specialists`: number of doctor agents, defaults to `3`

#### 2. Critic agent <a name="critic"></a>

This framework incorporates a critic agent to suggest further revisions on doctor agent's questions.

```bash
(clindiag) python code/trial_critic.py --data_dir benchmark_dataset --model_name_critic gpt-4o-mini
```

- `--model_name_critic`: model used for the critic agent, defaults to `gpt-4o-mini`

#### 3. Expert prompt <a name="expert"></a>

This script adopts expert-generated prompts.

```bash
(clindiag) python code/trial_expert_prompt.py --data_dir benchmark_dataset
```
