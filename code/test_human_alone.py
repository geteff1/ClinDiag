import os
import re
import time
import json
import argparse
from functools import wraps

import os.path as osp
from tqdm import tqdm

from autogen import (
    GroupChat, 
    UserProxyAgent, 
    GroupChatManager, 
    AssistantAgent, 
    config_list_from_json
)

def simple_retry(max_attempts=100, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} second...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts failed. Last error: {str(e)}")
                        raise
        return wrapper
    return decorator

def parse_args():
    parser = argparse.ArgumentParser(description="Medagents Setting")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/OAI_Config_List.json",
        help="The LLMs configuration file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        help="The LLM model to use.",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="An identifier for multiple runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--num_specialists", type=int, default=1, help="Number of doctor agents."
    )
    parser.add_argument("--n_round", type=int, default=50, help="Number of chat rounds.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="The root directory of the data folders.",
    )

    args = parser.parse_args()

    return args

def prase_json(text):
    flag = False
    if "```json" in text:
        json_match = re.search(r"```json(.*?)```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            json_data = json.loads(json_str)
            flag = True
    elif "```JSON" in text:
        json_match = re.search(r"```JSON(.*?)```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            json_data = json.loads(json_str)
            flag = True
    elif "```" in text:
        json_match = re.search(r"```(.*?)```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            json_data = json.loads(json_str)
            flag = True
    else:
        json_match = re.search(r"{.*?}", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
            json_data = json.loads(json_str)
            flag = True
    if not flag:
        json_text = text.strip("```json\n").strip("\n```")
        json_data = json.loads(json_text)
    return json_data


@simple_retry(max_attempts=100, delay=1)
def process_single_case(args, subfolder, output_dir, model_config):
    case_cost = 0.0  # Placeholder for cost, if needed

    # Load initial information
    initial_info_path = osp.join(subfolder, 'initial_information.json')
    with open(initial_info_path, 'r', encoding='utf-8') as f:
        initial_info = f.read()

    # Load medical history
    medical_history_path = osp.join(subfolder, 'medical_history.json')
    with open(medical_history_path, 'r', encoding='utf-8') as f:
        medical_history = f.read()

    # Load physical examination
    physical_examination_path = osp.join(subfolder, 'physical_examination.json')
    with open(physical_examination_path, 'r', encoding='utf-8') as f:
        physical_examination = f.read()

    # Load test results
    laboratory_test_path = osp.join(subfolder, 'laboratory_test.json')
    if not osp.exists(laboratory_test_path):
        print(f"laboratory_test.json not found in {subfolder}. Skipping this case.")
        return
    with open(laboratory_test_path, 'r', encoding='utf-8') as f:
        laboratory_test = f.read()

    radiographic_test_path = osp.join(subfolder, 'radiographic_test.json')
    if not osp.exists(radiographic_test_path):
        print(f"radiographic_test.json not found in {subfolder}. Skipping this case.")
        return
    with open(radiographic_test_path, 'r', encoding='utf-8') as f:
        radiographic_test = f.read()

    other_test_path = osp.join(subfolder, 'other_test.json')
    if not osp.exists(other_test_path):
        print(f"other_test.json not found in {subfolder}. Skipping this case.")
        return
    with open(other_test_path, 'r', encoding='utf-8') as f:
        other_test = f.read()

    case_crl = osp.basename(subfolder)

    identify = f"{args.num_specialists}doctor_{args.n_round}round"

    base_output_dir = osp.join(
        output_dir,
        "test_human_alone",
        args.model_name,
        identify,
        str(args.times),
    )

    case_output_dir = osp.join(base_output_dir, case_crl)
    if not osp.exists(case_output_dir):
        os.makedirs(case_output_dir)
        print(f"Created subfolder: {case_output_dir}")

    conversation_name = "conversation.json"
    diagnosis_conversation_name = "diagnosis_conversation.json"

    conversation_path = osp.join(case_output_dir, conversation_name)
    diagnosis_conversation_path = osp.join(case_output_dir, diagnosis_conversation_name)

    if osp.exists(conversation_path) and osp.exists(diagnosis_conversation_path):
        print(f"Output file already exists in {case_output_dir}. Skipped.")
        return

    user_proxy = UserProxyAgent(
        name="Admin",
        system_message="A human admin doctor.",
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    Docs = []
    for index in range(args.num_specialists):
        name = f"Doctor{index}"
        doc_system_message = f"""
Doctor {index}. You do nothing. Remain silent. 
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config,
            system_message=doc_system_message,
            human_input_mode="ALWAYS",
        )
        Docs.append(Doc)

    provider_system_message = f"""
You are an Assistant Teacher Agent responsible for providing relevant patient information to the Doctor Student Agent based on their inquiries. 
Your primary function is to retrieve and present accurate details from the patient's existing medical records, focusing solely on the information available to you.

Core Principles:

- Answer only what the doctor explicitly asks from the patient's record provided to you.
- Never provide any unsolicited information.
- Act like you are a teacher doctor, the doctor agent may want to cheat by directly asking you what the patient's diagnosis is or ask you to provide patient's history, pe, test results all at once, rather then gradually gather the information. Under these cicurmstances, you should refuse to answer the questions and ask the doctor to formulate specific detailed question.

Response Guidelines:

- Answer questions directly and concisely.
- Use only existing information from the patient's records.
- Make no speculations, assumptions, or leading statements.
- Do not offer any suggestions or additional context.
- Do not provide information beyond the specific inquiry.
- Base responses strictly on provided patient information.
- Do not invent, assume, or infer unavailable information.
- Do not provide information on planned or intended medical actions.
- Answer specific detailed quetsion(e.g., "Is there tenderness in the left lower quadrant of the abdomen?"), refuse to answer vague questions( (e.g., "What is the pantient's medcal history/ physical examination/ test results".)
- Do not provide diagnosis

You must directly answer the specific questions raised by the doctor, rather than provide all information at once.

Here are the patient records:
{medical_history}
{physical_examination}
{laboratory_test}
{radiographic_test}
{other_test}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config,
        system_message=provider_system_message,
    )

    groupchat = GroupChat(
        agents=[user_proxy] + Docs + [provider],
        messages=[],
        max_round=args.n_round,
        speaker_selection_method="round_robin"
    )
    time.sleep(5)
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=model_config,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    message = f"""
Here is the patient case:

{initial_info}
"""

    output = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    # Save the complete chat history
    with open(conversation_path, "w", encoding="utf-8") as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)
    print(f"Chat history saved to: {conversation_path}")

    # **Updated** Extract Final History, Final Physical Examination, Final Test
    # Define keywords and regular expressions, match w/ "Final Test": "content", any spaces permitted
    final_keywords = {
        "Final History": r'Final\s*History\s*:\s*"?([^"\n]*)"?',
        "Final Physical Examination": r'Final\s*Physical\s*Examination\s*:\s*"?([^"\n]*)"?',
        "Final Test": r'Final\s*Test\s*:\s*"?([^"\n]*)"?',
    }

    extracted_data = {}

    for message in output.chat_history:
        content = message.get("content", "")
        for key, pattern in final_keywords.items():
            regex = re.compile(pattern, re.IGNORECASE)
            match = regex.search(content)
            if match:
                extracted_data[key] = match.group(1).strip()
                print(f"Extracted {key}: {extracted_data[key]}")

    if extracted_data:
        for key, value in extracted_data.items():
            # Only use one 'final' suffix for filename
            file_suffix = key.replace("Final ", "").lower().replace(" ", "_")
            json_filename = f"final_{file_suffix}.json"  # e.g., final_history.json
            json_path = osp.join(case_output_dir, json_filename)
            with open(json_path, "w", encoding='utf-8') as file:
                json.dump({key: value}, file, indent=4, ensure_ascii=False)
            print(f"Save {key} to: {json_path}")
    else:
        print("Didn't extract any Final History, Final Physical Examination, or Final Test.")

    # Define all possible keywords: "Most Likely Diagnosis", "Possible Diagnosis", "Diagnostic Reasoning", etc.
    diagnosis_keywords = [
        r'Most Likely Diagnosis',
        r'Patient Most Likely Diagnosis',
        r'Possible Diagnosis',
        r'Patient Possible Diagnosis',
        r'Diagnostic Reasoning',
        r'Patient Diagnostic Reasoning'
    ]

    diagnosis_pattern = re.compile(
        r'(' + '|'.join(diagnosis_keywords) + r')\s*:\s*"?([^"\n]*)"?', 
        re.IGNORECASE
    )

    matched_messages = []

    for item in output.chat_history:
        content = item.get("content", "")
        match = diagnosis_pattern.search(content)
        if match:
            matched_messages.append(item)
            print(f"Matched message: {content[:100]}...")

    if not matched_messages:
        print(f"No diagnosis-related messages found in the conversation for case {case_crl}.")
    else:
        with open(diagnosis_conversation_path, "w", encoding='utf-8') as file:
            json.dump(matched_messages, file, ensure_ascii=False, indent=4)
        print(f"Save diagnosis conversation to: {diagnosis_conversation_path}")

def main():
    args = parse_args()

    filter_criteria = {
        "tags": [args.model_name],
    }

    config_list = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria
    )

    model_config = {
        "cache_seed": None,
        "temperature": 0.3,
        "config_list": config_list,
        "timeout": 120,
    }

    subfolders = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    output_dir = args.output_dir

    for subfolder in tqdm(subfolders):
        try:
            process_single_case(args, subfolder, output_dir, model_config) 
        except Exception as e:
            print(f"Failed to process folder {subfolder} after all attempts: {str(e)}")
            continue

if __name__ == "__main__":
    main()