import os
import os.path as osp
import re
import json
import argparse
import time
from functools import wraps
from tqdm import tqdm

from autogen import (
    GroupChat,
    UserProxyAgent,
    GroupChatManager,
    AssistantAgent,
    config_list_from_json
)

import dashscope
from types import SimpleNamespace


class CustomModelClient:
    """
    Custom client to interact with the Qwen model API for text generation.
    """
    def __init__(self, config, **kwargs):
        print(f"CustomModelClient config: {config}")
        self.model = config.get("model", "")
        self.api_key = config.get("api_key", os.getenv("DASHSCOPE_API_KEY"))

        # params are set by the user and consumed by the user since they are providing a custom model
        # so anything can be done here
        gen_config_params = config.get("params", {})
        self.max_length = gen_config_params.get("max_length", 256)

        print(f"Loaded model {self.model}")


    def create(self, params):
        messages = params["messages"]
        num_of_responses = params.get("n", 1)
        
        # can create your own data response class
        # here using SimpleNamespace for simplicity
        # as long as it adheres to the ModelClientResponseProtocol

        response = SimpleNamespace()
        response.choices = []
        response.model = self.model  # should match the OAI_CONFIG_LIST registration

        for _ in range(num_of_responses):
            res = dashscope.Generation.call(
                api_key=self.api_key,
                model=self.model,
                messages=messages,
                result_format='message',
                temperature=0.3,
            )
            if res.status_code != 200:
                print(f"Request ID: {response.request_id}")
                print(f"HTTP return code: {response.status_code}")
                print(f"Error code: {response.code}")
                print(f"Error message: {response.message}")
                # Ref: https://help.aliyun.com/zh/model-studio/developer-reference/error-code
                raise RuntimeError(f"DashScope request failed: {response.code} - {response.message}")
            
            text = res.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")
            choice = SimpleNamespace()
            choice.message = SimpleNamespace()
            choice.message.content = text
            choice.message.function_call = None
            response.choices.append(choice)
        
        return response

    def message_retrieval(self, response):
        """Retrieve the messages from the response."""
        choices = response.choices
        return [choice.message.content for choice in choices]

    def cost(self, response) -> float:
        """Calculate the cost of the response."""
        response.cost = 0
        return 0

    @staticmethod
    def get_usage(response):
        # returns a dict of prompt_tokens, completion_tokens, total_tokens, cost, model
        # if usage needs to be tracked, else None
        return {}

def simple_retry(max_attempts=10, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        print(f"Attempt {attempt + 1} failed: {str(e)}. Retry in {delay} sec...")
                        time.sleep(delay)
                    else:
                        print(f"All {max_attempts} attempts Failed. Final error: {str(e)}")
                        raise
        return wrapper
    return decorator

def extract_final_section(chat_history, case_output_dir, section):
    """
    Extract specific part from chat history (e.g., Final History, Final Physical Examination, 
    Final Test, etc.) and save as JSON file.
    """
    extracted_data = None

    section_prefix = f"{section}:"

    for idx, item in enumerate(chat_history):
        content = item.get("content", "").strip()
        
        # 1st attempt: try JSON decoding, extract doctor_action
        try:
            content_json = json.loads(content)
            doctor_action = content_json.get("doctor_action", "").strip()

            if doctor_action.startswith(section_prefix):
                # extract content after ':'
                extracted_data = doctor_action[len(section_prefix):].strip()
                print(f"Extracted {section} (from message index {idx}, JSON format): {extracted_data}")
                break  # once found, exit loop

        except json.JSONDecodeError:
            pass
        
        # 2nd attempt: JSON decoding failed or doctor_action doesn't match, search in pure text
        if extracted_data is None:
            # RegExp：match w/ "Final Physical Examination: ..." all the way to the end of the string
            pattern = re.compile(rf"{re.escape(section_prefix)}(.*)", re.IGNORECASE)
            match = pattern.search(content)
            if match:
                extracted_data = match.group(1).strip()
                print(f"Extracted {section} (from message index {idx}, pure text): {extracted_data}")
                break

    if extracted_data:
        # Generate filename
        file_suffix = section.lower().replace(" ", "_")
        json_filename = f"final_{file_suffix}.json"  # e.g., final_history.json
        json_path = osp.join(case_output_dir, json_filename)
        
        # Save as JSON file
        with open(json_path, "w", encoding='utf-8') as file:
            json.dump({section: extracted_data}, file, indent=4, ensure_ascii=False)
        
        print(f"{section} saved to: {json_path}")
        return extracted_data
    else:
        print(f"Didn't extract any {section} content.")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Integrated Medagents Setting")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/OAI_Config_List.json",
        help="The LLM models configuration file.",
    )
    parser.add_argument(
        "--model_name_history",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        help="The LLM model for medical history.",
    )
    parser.add_argument(
        "--model_name_pe",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        help="The LLM model for physical examination.",
    )
    parser.add_argument(
        "--model_name_test",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        help="The LLM model for diagostic tests.",
    )
    parser.add_argument(
        "--model_name_diagnosis",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        help="The LLM model for diagnosis.",
    )
    parser.add_argument(
        "--model_name_provider",
        type=str,
        default="gpt-4o-mini",
        choices=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "claude-3-haiku", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        help="The LLM model for the Provider Agent.",
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

@simple_retry(max_attempts=10, delay=1)
def history_gathering(args, subfolder, case_output_dir, model_config_history, model_config_provider):
    """
    Stage 1: History Gathering
    """
    initial_info_path = osp.join(subfolder, 'initial_information.json')
    with open(initial_info_path, 'r', encoding='utf-8') as f:
        initial_info = f.read()

    conversation_path = osp.join(case_output_dir, "history_conversation.json")
    final_history_path = osp.join(case_output_dir, "final_history.json")

    if osp.exists(conversation_path) and osp.exists(final_history_path):
        print(f"History files already exist in {case_output_dir}. Skip history gathering.")
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
You are a Doctor Agent specialized in acquiring and analyzing a patient's medical history.
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_history,
            system_message=doc_system_message,
            human_input_mode="ALWAYS",
        )
        if next((item for item in model_config_history["config_list"] if item.get("model_client_cls") == "CustomModelClient"), None):
            Doc.register_model_client(model_client_cls=CustomModelClient)
        Docs.append(Doc)

    medical_history_path = osp.join(subfolder, 'medical_history.json')
    with open(medical_history_path, 'r', encoding='utf-8') as f:
        medical_history = f.read()

    provider_system_message = f"""
You are an Assistant Agent responsible for providing relevant patient information to the Doctor Agent based on their inquiries. 
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

You must directly answer the specific questions raised by the doctor, rather than provide all information at once.
Remember you are not doctor, your should strictly adhere to the guidelines provided to you.
It is the doctor's job to summarize the finding, not the assistant's.

Here is the patient record:
{medical_history}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_provider,
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
        llm_config=model_config_history,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    message = f"""
Here is a patient case for analysis, ask about patient history and provide the final patient history.

{initial_info}
"""

    output = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    with open(conversation_path, "w", encoding='utf-8') as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)
    print(f"History conversation saved to: {conversation_path}")

    extracted_data = extract_final_section(output.chat_history, case_output_dir, "Final History")

    if not extracted_data:
        print(f"'Final History' not found, unable to save to {final_history_path}")
    else:
        # Save Final History
        with open(final_history_path, "w", encoding='utf-8') as file:
            json.dump({"Final History": extracted_data}, file, indent=4, ensure_ascii=False)
        print(f"Final History saved to: {final_history_path}")

@simple_retry(max_attempts=10, delay=1)
def pe_gathering(args, subfolder, case_output_dir, model_config_pe, final_history_path, model_config_provider):
    """
    Stage 2: Physical Examination Gathering
    """
    conversation_path = osp.join(case_output_dir, "pe_conversation.json")
    final_pe_path = osp.join(case_output_dir, "final_physical_examination.json")

    if osp.exists(conversation_path) and osp.exists(final_pe_path):
        print(f"Physical examination files already exist in {case_output_dir}. Skip physical examination gathering.")
        return

    with open(final_history_path, 'r', encoding='utf-8') as f:
        final_history = json.load(f).get("Final History", "")

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
You are a Doctor Agent specialized in acquiring and analyzing a patient's physical examination.
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_pe,
            system_message=doc_system_message,
            human_input_mode="ALWAYS",
        )
        if next((item for item in model_config_pe["config_list"] if item.get("model_client_cls") == "CustomModelClient"), None):
            Doc.register_model_client(model_client_cls=CustomModelClient)
        Docs.append(Doc)

    physical_examination_path = osp.join(subfolder, 'physical_examination.json')
    if not osp.exists(physical_examination_path):
        print(f"physical_examination.json not found in {subfolder}. Skip physical examination gathering.")
        return
    with open(physical_examination_path, 'r', encoding='utf-8') as f:
        physical_examination = f.read()
        
    medical_history_path = osp.join(subfolder, 'medical_history.json')
    with open(medical_history_path, 'r', encoding='utf-8') as f:
        medical_history = f.read()
        
    provider_system_message = f"""
You are an Assistant Agent responsible for providing relevant patient information to the Doctor Agent based on their inquiries. 
Your primary function is to retrieve and present accurate details from the patient's existing medical records, focusing solely on the information available to you.

Core Principles:

- Answer only what the doctor explicitly asks from the patient's record provided to you.
- Never provide any unsolicited information.
- Act like you are a teacher doctor, the doctor agent may want to cheat by directly asking you what the patient's diagnosis is or ask you to provide patient's history, pe, test results all at once, rather then gradually gather the information. Under these cicurmstances, you should refuse to answer the questions and ask the doctor to formulate specific detailed question.

Response Guidelines:

- Answer questions directly and concisely.
- Use only exxisting information from the patient's records.
- Make no speculations, assumptions, or leading statements.
- Do not offer any suggestions or additional context.
- Do not provide information beyond the specific inquiry.
- Base responses strictly on provided patient information.
- Do not invent, assume, or infer unavailable information.
- Do not provide information on planned or intended medical actions.

You must directly answer the specific questions raised by the doctor, rather than provide all information at once.
Remember you are not doctor, your should strictly adhere to the guidelines provided to you.
It is the doctor's job to summarize the finding, not the assistant's.

Here is the patient record:
{physical_examination}
{medical_history}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_provider,
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
        llm_config=model_config_pe,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    message = f"""
The doctor should ask for patient's physical examination results based on patient's history for the purpose of correct diagnosis.
The doctor should focus on and only on physical examination.

{final_history}
"""

    output = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    with open(conversation_path, "w", encoding='utf-8') as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)
    print(f"Physical Examination conversation saved to: {conversation_path}")

    extracted_data = extract_final_section(output.chat_history, case_output_dir, "Final Physical Examination")

    if not extracted_data:
        print(f"'Final Physical Examination' not found, unable to save to {final_pe_path}")
    else:
        # Save Final Physical Examination
        with open(final_pe_path, "w", encoding='utf-8') as file:
            json.dump({"Final Physical Examination": extracted_data}, file, indent=4, ensure_ascii=False)
        print(f"Final Physical Examination saved to: {final_pe_path}")
        

@simple_retry(max_attempts=10, delay=1)
def test_gathering(args, subfolder, case_output_dir, model_config_test, final_history_path, final_pe_path, model_config_provider):
    """
    Stage 3: Test Gathering
    """
    conversation_path = osp.join(case_output_dir, "test_conversation.json")
    final_test_path = osp.join(case_output_dir, "final_test.json")
    
    if osp.exists(conversation_path) and osp.exists(final_test_path):
        print(f"Test files already exist in {case_output_dir}. Skip test gathering.")
        return

    with open(final_history_path, 'r', encoding='utf-8') as f:
        final_history = json.load(f).get("Final History", "")
    with open(final_pe_path, 'r', encoding='utf-8') as f:
        final_pe = json.load(f).get("Final Physical Examination", "")

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
You are a Doctor Agent specialized in acquiring and analyzing a patient's test results, including lab tests, radiographic tests, and other diagnostic tests.
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_test,
            system_message=doc_system_message,
            human_input_mode="ALWAYS",
        )
        if next((item for item in model_config_test["config_list"] if item.get("model_client_cls") == "CustomModelClient"), None):
            Doc.register_model_client(model_client_cls=CustomModelClient)
        Docs.append(Doc)

    laboratory_test_path = osp.join(subfolder, 'laboratory_test.json')
    if not osp.exists(laboratory_test_path):
        print(f"laboratory_test.json not found in {subfolder}. Skip this case.")
        return
    with open(laboratory_test_path, 'r', encoding='utf-8') as f:
        laboratory_test = f.read()

    radiographic_test_path = osp.join(subfolder, 'radiographic_test.json')
    if not osp.exists(radiographic_test_path):
        print(f"radiographic_test.json not found in {subfolder}. Skip this case.")
        return
    with open(radiographic_test_path, 'r', encoding='utf-8') as f:
        radiographic_test = f.read()

    other_test_path = osp.join(subfolder, 'other_test.json')
    if not osp.exists(other_test_path):
        print(f"other_test.json not found in {subfolder}. Skip this case.")
        return
    with open(other_test_path, 'r', encoding='utf-8') as f:
        other_test = f.read()
        
    physical_examination_path = osp.join(subfolder, 'physical_examination.json')
    with open(physical_examination_path, 'r', encoding='utf-8') as f:
        physical_examination = f.read()
        
    medical_history_path = osp.join(subfolder, 'medical_history.json')
    with open(medical_history_path, 'r', encoding='utf-8') as f:
        medical_history = f.read()
        
    provider_system_message = f"""
You are an Assistant Agent responsible for providing relevant patient information to the Doctor Agent based on their inquiries. 

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

You must directly answer the specific questions raised by the doctor, rather than provide all information at once.
Remember you are not doctor, your should strictly adhere to the guidelines provided to you.
It is the doctor's job to summarize the finding, not the assistant's.

Here is the patient record:
{laboratory_test}
{radiographic_test}
{other_test}
{physical_examination}
{medical_history}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_provider,
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
        llm_config=model_config_test,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )

    message = f"""
The doctor should ask for patient's test results based on patient's history and physical examination results for the purpose of correct diagnosis.
The doctor should focus on and only on diagnostic test.

Here is the patient's history and physical examination.
{final_history}
{final_pe}
"""

    output = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    with open(conversation_path, "w", encoding='utf-8') as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)
    print(f"Test conversation saved to: {conversation_path}")

    extracted_data = extract_final_section(output.chat_history, case_output_dir, "Final Test")

    if not extracted_data:
        print(f"'Final Test' not found, unable to save to: {final_test_path}")
    else:
        with open(final_test_path, "w", encoding='utf-8') as file:
            json.dump({"Final Test": extracted_data}, file, indent=4, ensure_ascii=False)
        print(f"Final Test saved to: {final_test_path}")


@simple_retry(max_attempts=10, delay=1)
def diagnosis_stage(args, subfolder, case_output_dir, model_config_diagnosis, final_history_path, final_pe_path, final_test_path, model_config_provider):
    """
    Stage 4: Diagnosis
    """
    conversation_path = osp.join(case_output_dir, "diagnosis_conversation.json")
    final_diagnosis_path = osp.join(case_output_dir, "final_diagnosis.json")

    if osp.exists(conversation_path):
        print(f"Diagnosis files already exist in {case_output_dir}. Skip diagnosis.")
        return

    # Loead Final History, Final PE, and Final Test
    with open(final_history_path, 'r', encoding='utf-8') as f:
        final_history = json.load(f).get("Final History", "")
    with open(final_pe_path, 'r', encoding='utf-8') as f:
        final_pe = json.load(f).get("Final Physical Examination", "")
    with open(final_test_path, 'r', encoding='utf-8') as f:
        final_test = json.load(f).get("Final Test", "")

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
You are a Doctor Agent specialized in making final diagnoses based on the patient's history, physical examination, and test results.
"""

        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_diagnosis,
            system_message=doc_system_message,
            human_input_mode="ALWAYS",
        )
        if next((item for item in model_config_diagnosis["config_list"] if item.get("model_client_cls") == "CustomModelClient"), None):
            Doc.register_model_client(model_client_cls=CustomModelClient)
        Docs.append(Doc)

    provider_system_message = f"""
you do nothing and remain silent, output nothing.
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_provider,
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
        llm_config=model_config_diagnosis,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    )
    
    message = f"""
Provide the final diagnosis, differential diagnosis and diagnostic reasoning based on the patient information.
Focuse on and only on diagnosis.
Do not ask question, analyze with the information you have and provide answer.

Here is the patient information:
Final History: {final_history}
Final Physical Examination: {final_pe}
Final Test: {final_test}
"""

    output = user_proxy.initiate_chat(
        manager,
        message=message,
    )

    with open(conversation_path, "w", encoding='utf-8') as file:
        json.dump(output.chat_history, file, indent=4, ensure_ascii=False)
    print(f"Diagnosis conversation saved to: {conversation_path}")

    extracted_diagnosis = extract_final_section(output.chat_history, case_output_dir, "Final Diagnosis")
    extracted_differential = extract_final_section(output.chat_history, case_output_dir, "Differential Diagnosis")
    extracted_reasoning = extract_final_section(output.chat_history, case_output_dir, "Diagnostic Reasoning")

    if not extracted_diagnosis or not extracted_differential or not extracted_reasoning:
        print(f"Complete diagnosis info not found, unable to save to {final_diagnosis_path}")
    else:
        # Save Final Diagnosis
        final_diagnosis = {
            "Final Diagnosis": extracted_diagnosis,
            "Differential Diagnosis": extracted_differential,
            "Diagnostic Reasoning": extracted_reasoning
        }
        with open(final_diagnosis_path, "w", encoding='utf-8') as file:
            json.dump(final_diagnosis, file, indent=4, ensure_ascii=False)
        print(f"Final Diagnosis saved to {final_diagnosis_path}")



def main():
    args = parse_args()

    # Load model configs for each stage
    filter_criteria_history = {
        "tags": [args.model_name_history],
    }
    config_list_history = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria_history
    )
    model_config_history = {
        "cache_seed": None,
        "temperature": 0.3,
        "config_list": config_list_history,
        "timeout": 120,
    }

    filter_criteria_pe = {
        "tags": [args.model_name_pe],
    }
    config_list_pe = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria_pe
    )
    model_config_pe = {
        "cache_seed": None,
        "temperature": 0.3,
        "config_list": config_list_pe,
        "timeout": 120,
    }

    filter_criteria_test = {
        "tags": [args.model_name_test],
    }
    config_list_test = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria_test
    )
    model_config_test = {
        "cache_seed": None,
        "temperature": 0.3,
        "config_list": config_list_test,
        "timeout": 120,
    }

    filter_criteria_diagnosis = {
        "tags": [args.model_name_diagnosis],
    }
    config_list_diagnosis = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria_diagnosis
    )
    model_config_diagnosis = {
        "cache_seed": None,
        "temperature": 0.3,
        "config_list": config_list_diagnosis,
        "timeout": 120,
    }
    
    filter_criteria_provider = {
        "tags": [args.model_name_provider],
    }
    config_list_provider = config_list_from_json(
        env_or_file=args.config, filter_dict=filter_criteria_provider
    )
    model_config_provider = {
        "cache_seed": None,
        "temperature": 0,
        "config_list": config_list_provider,
        "timeout": 120,
    }


    subfolders = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    output_dir = args.output_dir

    for subfolder in tqdm(subfolders, desc="Processing Cases"):
        try:
            case_crl = os.path.basename(subfolder)
            identify = f"{args.num_specialists}doctor_{args.n_round}round"

            base_output_dir = osp.join(
                output_dir,
                "test_human_llm",
                args.model_name_diagnosis,  # Adjust as needed
                identify,
                str(args.times),
            )
            case_output_dir = osp.join(base_output_dir, case_crl)
            if not osp.exists(case_output_dir):
                os.makedirs(case_output_dir)
                print(f"Create subfolders: {case_output_dir}")

            history_gathering(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_history,
                model_config_provider
            )
            pe_gathering(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_pe, 
                osp.join(case_output_dir, "final_history.json"),
                model_config_provider
            )
            test_gathering(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_test, 
                osp.join(case_output_dir, "final_history.json"), 
                osp.join(case_output_dir, "final_physical_examination.json"),
                model_config_provider
            )
            diagnosis_stage(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_diagnosis, 
                osp.join(case_output_dir, "final_history.json"), 
                osp.join(case_output_dir, "final_physical_examination.json"), 
                osp.join(case_output_dir, "final_test.json"),
                model_config_provider
            )

        except Exception as e:
            print(f"Processing folder {subfolder} failed, error: {str(e)}")
            continue

if __name__ == "__main__":
    main()