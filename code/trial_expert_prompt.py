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
    # Define keywords and regular expressions:
    # Match w/ "Final <Section>": "content" or "Final <Section>": { ... }, any spaces permitted
    patterns = {
        "Final History": r'"Final\s*History"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Final Physical Examination": r'"Final\s*Physical\s*Examination"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Final Test": r'"Final\s*Test"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Final Diagnosis": r'"Final\s*Diagnosis"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Differential Diagnosis": r'"Differential\s*Diagnosis"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Diagnostic Reasoning": r'"Diagnostic\s*Reasoning"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Final Updates During Physical Examination": r'"Final\s*Updates\s*During\s*Physical\s*Examination"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
        "Final Updates During Test": r'"Final\s*Updates\s*During\s*Test"\s*:\s*(?:"([^"]*)"|(\{[\s\S]*?\}))',
    }

    if section not in patterns:
        print(f"Unknown extraction: {section}")
        return None

    pattern = patterns[section]
    regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)

    extracted_data = None

    all_content = "\n".join([item.get("content", "") for item in chat_history])
    print(f"Extracting {section} content...")
    print(all_content)
    
    match = regex.search(all_content)
    if match:
        string_value = match.group(1)
        json_object = match.group(2)
        if string_value:
            extracted_data = string_value.strip()
        elif json_object:
            try:
                extracted_data = json.loads(json_object.strip())
            except json.JSONDecodeError as e:
                print(f"JSON error in decoding {section}: {str(e)}")
                extracted_data = json_object.strip()
        print(f"Extracted {section}: {extracted_data}")
    else:
        print(f"Didn't find content matching {section}.")

    if extracted_data:
        file_suffix = section.replace("Final ", "").replace("Differential ", "").lower().replace(" ", "_")
        json_filename = f"final_{file_suffix}.json"  # e.g., final_history.json
        json_path = osp.join(case_output_dir, json_filename)
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
        "--num_specialists", type=int, default=3, help="Number of doctor agents."
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
def history_gathering(args, subfolder, case_output_dir, model_config_history):
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
Medical Doctor {index}. You are a Doctor Agent specialized in acquiring and analyzing a patient's medical history. 
Your sole responsibility is to gather comprehensive details about the patient's history to understand their health background better. 
You should ask specific, targeted questions and reason about what to ask next based on the feedback you receive.

Primary Objectives:

- Determine necessary inquiries and formulate specific, relevant questions to acquire patient's medical history information.
- Engage in iterative information gathering through dialogue with the Assistant.
- Assess when sufficient information has been obtained to cease further inquiries.

Constraints:
- Focus exclusively on medical history. Do not perform examinations, order tests, make diagnoses, or suggest treatments.
- May inquire about past examinations, test results, diagnoses, and treatments.
- Ensure each question is directly related to obtaining the patient's medical history.
- Ask specific questions (e.g., "Do you have a history of hypertension?") rather than broad ones (e.g., "Do you have any other diseases?").

Interaction Process:

- Initiate inquiry with specific questions about the patient's medical history.
- Receive and analyze feedback from the Assistant.
- Determine the next set of specific questions based on this analysis.
- Continue the cycle until sufficient information is gathered.

Guidelines:
- Form your diagonsis hypothesis, you might form serveral hypothesis, ask questions around these hypothesis
- Formulate clear, specific, and relevant questions.
- Adapt questioning based on responses and feedback.

Clinical GUideline:
1. Gather History Guidelines

The purpose of this phase is to systematically collect comprehensive information from the patient to inform diagnosist.
A. Chief Complaint (CC):
Record the patient’s primary concern in their own words.
Clarify vague statements with follow-up questions (e.g., "What do you mean by feeling unwell?").
B. History of Present Illness (HPI):
Use a structured approach like OLDCART or OPQRST to gather details about the main issue:
Onset: When did it start? Was it sudden or gradual?
Location: Where is the symptom located? Does it radiate?
Duration: How long does it last? Has it been constant or intermittent?
Character/Quality: How does the patient describe it (e.g., sharp, dull, throbbing)?
Aggravating/Alleviating Factors: What makes it better or worse?
Radiation: Does the symptom spread to other areas?
Timing: Is there a pattern (e.g., morning vs. evening)?
Identify associated symptoms and ask targeted questions based on potential differential diagnoses (e.g., for chest pain, ask about shortness of breath, palpitations, or diaphoresis).
C. Past Medical History (PMH):
Chronic Illnesses: Note the details of any prior diagnoses (e.g., hypertension, diabetes).
Surgical History: Include procedures, dates, and outcomes.
Hospitalizations: Document reasons, treatments, and follow-up care.
Past Episodes: Explore whether similar symptoms occurred in the past.
D. Diagnostic and Treatment History:
Ask about previous tests, imaging, and lab work related to the current complaint.
Include prior treatment attempts, response to treatments, and side effects.
E. Medication History:
Record current medications (prescription, OTC, supplements) with dosages and frequency.
Include history of noncompliance or changes in medication.
F. Allergies:
Document any allergic reactions (e.g., to medications, foods, or environmental factors) and the type/severity of reaction.
G. Family History:
Note genetic or hereditary conditions, especially those relevant to the presenting complaint (e.g., heart disease, cancer).
H. Social History:
Explore factors affecting health, such as:
Smoking, alcohol, or drug use.
Occupational risks or exposures.
Exercise, diet, and stress levels.
Living situation, support systems, and barriers to care.
I. Review of Systems (ROS):
Conduct a comprehensive review of symptoms across all body systems, emphasizing areas related to the chief complaint.
Capture positive findings (symptoms present) and negative findings (symptoms absent) to support or rule out potential diagnoses.

2. Summarize History Guidelines
In this phase, the aim is to accurately condense the collected information into a structured summary. Focus on clarity, completeness, and logical organization. Highlight findings that support or refute differential diagnoses.
A. Present the information in a clear and logical sequence:
B. Include Key Positive and Negative Findings:
Highlight positive findings that support a diagnosis (e.g., "Patient reports sharp chest pain radiating to the left arm with exertion").
Document negative findings that help rule out other conditions (e.g., "No fever, no shortness of breath, no hemoptysis").
C. Avoid Omissions:
Ensure that all systems or components are covered, even if findings are normal or not immediately relevant.
Example: “No prior history of similar symptoms, no personal or family history of cardiovascular disease.”
D. Use Specific, Measurable Details:
Where applicable, include precise descriptions:
"Pain severity rated as 8/10."
"Blood pressure 140/90 mmHg at home measurements."
E. Ensure Clarity:
Avoid medical jargon when summarizing for patients or non-clinical audiences, but maintain precise terminology for clinical communication.

For each inquiry, use the following format:

{{
"doctor_reasoning": "Doctor's reasoning based on current information",
"doctor_action": "Doctor's instructions to the assistant, phrased in third person (maximum 5 specific questions)"
}}

When all necessary information has been gathered, present the complete history in this format:

{{
    "Final History": "[patient's medical history]"
}}

Important note: sometimes you will continuously not receive feedback for the information you ask for, you should end the conversation and continue to summarize. If you do not have any informative information for either category, you can conclude "no valid information has been gathered".
After you provide final history, it is your duty to reply with "TERMINATE" to end the conversation.
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_history,
            system_message=doc_system_message,
        )
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

Here is the patient record:
{medical_history}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_history,
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
Here is a patient case, ask about patient history and provide the final patient history.

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
def pe_gathering(args, subfolder, case_output_dir, model_config_pe, final_history_path):
    """
    Stage 2: Physical Examination Gathering
    """
    conversation_path = osp.join(case_output_dir, "pe_conversation.json")
    final_pe_path = osp.join(case_output_dir, "final_physical_examination.json")
    final_updates_pe_path = osp.join(case_output_dir, "final_updates_during_physical_examination.json")  

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
Medical Doctor {index}. You are a Doctor Agent specialized in acquiring and analyzing a patient's physical examination. 
Your sole responsibility is to gather comprehensive details about the patient's physical examination that would be helpful in reaching the final diagnosis, based on the patient history provided to you.
You should ask specific, targeted questions and reason about what to ask next based on the feedback you receive.

Primary Objectives:

- Determine necessary inquiries and formulate specific, relevant questions to acquire patient's physical examination information.
- Engage in iterative information gathering through dialogue with the Assistant.
- Assess when sufficient information has been obtained to cease further inquiries.

Constraints:
- Focus exclusively on physical examinations. Do not perform lab tests, radiographic tests, or other tests, do not make diagnoses, or suggest treatments.
- Ensure each question is directly related to obtaining the patient's physical examination.
- Ask specific questions (e.g., "Is there tenderness in the left lower quadrant of the abdomen?") rather than broad ones (e.g., "What is the result for abdominal physical examination?").
- Do not ask questions that are related to the patient's lab tests, radiographic tests and other diagnostic tests that are not categorized as physcial examination.
- Do not provide treatment recommendations.

Interaction Process:

- Initiate inquiry with specific questions about the patient's physical examination.
- Receive and analyze feedback from the Assistant.
- Determine the next set of specific questions based on this analysis.
- Continue the cycle until sufficient information is gathered.
- After you get results from physical examination, you can go back and ask for new information about patient history that has not been mentioned in the final history provided to you, if it is helpful in reaching diagnosis.

Guidelines:
- Form your diagonsis hypothesis, you might form serveral hypothesis, ask questions around these hypothesis 
- Formulate clear, specific, and relevant questions.
- Adapt questioning based on responses and feedback.

Clinical Guideline
1. Gathering Physical Examination Findings
The purpose of this phase is to perform a targeted physical examination based on the patient’s presenting complaints and known medical history, followed by a comprehensive system-based examination to ensure no significant findings are missed.
A. General Principles:
Prioritize Based on Patient Information:
Start with focused assessments related to the patient’s chief complaint, history of present illness (HPI), and known medical conditions.
For example:
A patient with chest pain should have a detailed cardiovascular and respiratory examination.
A patient with abdominal pain requires a thorough abdominal assessment.
Expand to a Full Systematic Examination:
After addressing the targeted area, proceed to a head-to-toe examination to identify any additional findings or conditions not directly related to the presenting problem.
Sequential and Methodical:
Conduct the examination in a logical, system-based order to ensure consistency and completeness.

B. Step-by-Step Approach:
General Appearance and Vital Signs:
Observation: Note the patient’s overall state, including level of consciousness, distress, or discomfort.
Vital Signs: Include temperature, blood pressure, heart rate, respiratory rate, and oxygen saturation.
Targeted Examination Based on Patient’s Condition:
Comprehensive Systematic Examination:
After the targeted assessment, expand to a complete physical examination in this sequence:
a. Skin and Peripheral Findings:
Inspect for color changes (e.g., cyanosis, pallor, jaundice), rashes, scars, or lesions.
Palpate for temperature, moisture, and peripheral edema.
b. Head and Neck:
Inspect facial symmetry, eyes (sclera, pupils), ears, nose, and throat.
Palpate lymph nodes, thyroid gland, and carotid pulses.
c. Respiratory System:
Inspect chest movements and deformities.
Palpate for fremitus and tenderness.
Percuss for dullness or hyperresonance.
Auscultate for breath sounds, wheezes, or crackles.
d. Cardiovascular System:
Inspect for jugular venous distension.
Palpate for thrills, heaves, and peripheral pulses.
Auscultate heart sounds (S1, S2, murmurs, rubs).
e. Abdomen:
Inspect for distension or visible masses.
Auscultate for bowel sounds and vascular bruits.
Percuss for organ size and tympany.
Palpate for tenderness, rigidity, or organomegaly.
f. Musculoskeletal System:
Inspect for deformities, swelling, or erythema.
Test active and passive range of motion.
Palpate for joint tenderness or crepitus.
g. Neurological System:
Assess mental status, cranial nerves, motor strength, reflexes, and sensory function.
h. Genitourinary System (if applicable):
Conduct specific examinations only when clinically indicated, respecting patient comfort.
2. Summarizing Physical Examination Findings
The purpose of this phase is to organize and document the findings in a logical and clinically relevant manner. Emphasize positive findings, key negative findings. Be comprehensive and avoid omission.
A. Include Positive and Negative Findings:
B. Maintain Logical 
C. Avoid Omissions
D. Be Quantitative and Specific:
E. Highlight Red Flags and Diagnostic Clues:

For each inquiry, use the following format:

{{
"doctor_reasoning": "Doctor's reasoning based on current information",
"doctor_action": "Doctor's instructions to the assistant, phrased in third person (maximum 5 specific questions)"
}}

When all necessary information has been gathered, present the complete physical examination in this format:

{{
    "Final Physical Examination": "[patient's physical examination results]"
}}

If you get new information from patient's history that is not provided in the patient record, please output the new information, use this format:
{{
    "Final Updates During Physical Examination": "[new information found regarding patient history, that is not included in the original patient record]"
}}
Important note: sometimes you will continuously not receive feedback for the information you ask for, you should end the conversation and continue to summarize. If you do not have any informative information for either category, you can conclude "no valid information has been gathered".
After you provide final physical examination, it is your duty to reply with "TERMINATE" to end the conversation.

Here is the patient's record, based on these information you should determine what physical examination should be performed in the following conversation:
{final_history}
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_pe,
            system_message=doc_system_message,
        )
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

Here is the patient record:
{physical_examination}
{medical_history}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_pe,
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
The doctor should ask for patient's physical examination results.
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

    extracted_updates_pe = extract_final_section(output.chat_history, case_output_dir, "Final Updates During Physical Examination")

    if not extracted_updates_pe:
        print(f"'Final Updates During Physical Examination' not found, unable to save to {final_updates_pe_path}")
    else:
        # Save Final Updates During Physical Examination
        with open(final_updates_pe_path, "w", encoding='utf-8') as file:
            json.dump({"Final Updates During Physical Examination": extracted_updates_pe}, file, indent=4, ensure_ascii=False)
        print(f"Final Updates During Physical Examination saved to: {final_updates_pe_path}")
        

@simple_retry(max_attempts=10, delay=1)
def test_gathering(args, subfolder, case_output_dir, model_config_test, final_history_path, final_pe_path):
    """
    Stage 3: Test Gathering
    """
    # Paths
    conversation_path = osp.join(case_output_dir, "test_conversation.json")
    final_test_path = osp.join(case_output_dir, "final_test.json")
    final_updates_test_path = osp.join(case_output_dir, "final_updates_during_test.json")  
    
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
Medical Doctor {index}. You are a Doctor Agent specialized in acquiring and analyzing a patient's test results, including lab tests, radiographic tests, and other diagnostic tests. 

Your sole responsibility is to gather comprehensive details about the patient's test results that would be helpful in reaching the final diagnosis, based on the patient's history and physical examination provided to you.

You should ask specific, targeted questions and reason about what to ask next based on the feedback you receive.

Primary Objectives:

- Determine necessary inquiries and formulate specific, relevant questions to acquire patient's test result information.
- Engage in iterative information gathering through dialogue with the Assistant.
- Assess when sufficient information has been obtained to cease further inquiries.

Constraints:
- Focus exclusively on lab tests, radiographic tests, and other diagnostic tests. Do not perform tests, make diagnoses, or suggest treatments.
- Ensure each question is directly related to obtaining the patient's test results.
- Ask specific questions (e.g., "Do you have information on the patient's head CT scan?") rather than broad ones (e.g., "What are the radiographic test results of the patient?").
- Do not ask questions that are not related to the patient's lab tests, radiographic tests, and other diagnostic tests.
- Do not provide treatment recommendations.

Interaction Process:

- Initiate inquiry with specific questions about the patient's test results.
- Receive and analyze feedback from the Assistant.
- Determine the next set of specific questions based on this analysis.
- Continue the cycle until sufficient information is gathered.
- After you get results from tests, you can go back and ask for new information about patient history and physical examination that is not included in the patient record provided to you, if it is helpful in reaching diagnosis.

Guidelines:
- Form your diagonsis hypothesis, you might form serveral hypothesis, ask questions around these hypothesis
- Formulate clear, specific, and relevant questions.
- Adapt questioning based on responses and feedback.
- Aim to obtain complete patient's test information.

Clinical Guideline:
1. Gathering Test Results
The objective of gathering test results is to ensure that all relevant diagnostic tests are identified, with a focus on obtaining specific findings that lead to a definitive diagnosis rather than vague or incomplete information. Follow these principles:
A. General Approach:
Understand the Clinical Context:
Refer to the patient’s history and physical examination findings to determine which tests are most relevant.
Focus on tests that directly address differential diagnoses or clarify ambiguous findings.
Ask About Diagnostic-Specific Tests:
When a test result is broad or non-specific, inquire about follow-up or confirmatory tests.
Examples:
"A chest CT shows a pulmonary nodule—has a biopsy or PET scan been done to assess malignancy?"
"Urinalysis suggests a urinary tract infection—were additional tests (e.g., urine culture, imaging) performed to localize the infection and identify the causative pathogen?"
Inquire About Timing and Trends:
Ask if serial testing has been performed to monitor progression or resolution of a condition.
Examples:
"Were follow-up tumor markers obtained after the initial abnormal result?"
"Were repeat imaging studies done to track the size or appearance of the lesion?"
Consider Comprehensive Diagnostic Workups:
Ask about related tests or examinations that provide a complete diagnostic picture.
Examples:
"For anemia identified on CBC, was iron panel, vitamin B12, folate levels, or a bone marrow biopsy performed?"
"For suspected renal dysfunction on basic labs, were urine studies, renal ultrasound, or biopsy conducted?"
Clarify Ambiguous or Uncertain Results:
Ask for additional testing performed to resolve inconclusive or borderline findings.
Example: "Was a borderline troponin elevation followed by serial testing or a stress test?"
B. Focused Areas of Inquiry for Tests:
Laboratory Tests:
Routine Tests: Ask about detailed values (e.g., CBC, BMP) and trends over time.
Confirmatory Tests: Inquire about additional tests that refine a diagnosis.
Example: "For suspected infection, was a blood culture or pathogen-specific PCR performed?"
Imaging Studies:
Initial Findings: Clarify specific details (e.g., size, location, enhancement patterns).
Follow-Up Imaging: Ask if further imaging (e.g., MRI, PET, contrast studies) was done for clarification.
Example: "For liver lesions on ultrasound, was an MRI or biopsy done to confirm benign vs. malignant nature?"
Functional or Specialized Tests:
Examples: Stress tests, pulmonary function tests, nerve conduction studies.
Ask about specific results and diagnostic thresholds: "Did the stress test indicate ischemia or arrhythmias?"
Pathology and Biopsy Results:
Ask for detailed findings, including histology, molecular markers, or genetic mutations.
Example: "Was the lung biopsy indicative of malignancy, and if so, what type (e.g., adenocarcinoma, squamous cell)?"
Microbiological Studies:
Ensure causative organisms are identified along with sensitivities.
Example: "Was the urinary pathogen isolated, and were antibiotic sensitivities determined?"
Procedural Findings:
For endoscopies, catheterizations, or other interventions, ask for specific observations or diagnoses made during the procedure.
Example: "Were biopsies taken during the colonoscopy, and what were the histopathology results?"
2. Summarizing Test Results
The goal of summarizing test results is to present findings in a clear, comprehensive and clinically relevant manner, emphasizing logical reasoning.
A. Highlight Positive and Negative Results:
Positive Results: Clearly describe abnormalities and their implications.
Example: "Lung CT shows a 2.5 cm spiculated nodule in the right upper lobe with FDG uptake on PET scan, consistent with malignancy."
Negative Results: Highlight findings that rule out significant conditions.
Example: "MRI showed no evidence of metastasis, ruling out systemic spread."
B. Prioritize Diagnostic-Specific Findings:
Focus on test results that directly contribute to the diagnosis or management.
Example: "Urine culture identified E. coli sensitive to ciprofloxacin, confirming uncomplicated UTI."
C. Maintain Logical Flow:
Present results in a sequence that supports the diagnostic process.
Example:
Initial finding: "CBC revealed normocytic anemia."
Follow-up: "Iron studies showed low ferritin, indicating iron deficiency."
Additional testing: "Endoscopy revealed gastric ulcers as the source of chronic blood loss."
D. Be Quantitative and Specific:
Include precise values, sizes, and descriptive details.
Example: "Pulmonary nodule measured 2.3 cm with irregular margins and central cavitation."
E. Incorporate Trends:
Highlight dynamic changes over time where applicable.
Example: "Serial troponin levels showed a rise from 0.03 ng/mL to 1.2 ng/mL over six hours, consistent with myocardial infarction.

For each inquiry, use the following format:

{{
"doctor_reasoning": "Doctor's reasoning based on current information",
"doctor_action": "Doctor's instructions to the assistant, phrased in third person (maximum 5 specific questions)"
}}

When all necessary information has been gathered, organize the final test into lab tests, radiographic tests, and other tests, and present the complete test information in this format:
{{
    "Final Test": "[patient's lab tests results, radiographic tests results, other diagnostic tests results]"
}}

If you get new information from patient's history that is not provided in the final history, please output the new information, do not output origingal content from final history, use this format:
{{
    "Final Updates During Test": "[new information found regarding patient history and physical exmination, that is not included in the original patient record provided to you]"
}}

Important note: sometimes you will continuously not receive feedback for the information you ask for, you should end the conversation and continue to summarize. If you do not have any informative information for either category, you can conclude "no valid information has been gathered".
After you provide the final test, it is your duty to reply with "TERMINATE" to end the conversation.

Here is the patient's record, based on these information you should determine what tests should be performed in the following conversation:
{final_history}
{final_pe}
"""
        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_test,
            system_message=doc_system_message,
        )
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

Here is the patient record:
{laboratory_test}
{radiographic_test}
{other_test}
{physical_examination}
{medical_history}
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_test,
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
The doctor should ask for patient's test results.
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

    extracted_updates_test = extract_final_section(output.chat_history, case_output_dir, "Final Updates During Test")

    if not extracted_updates_test:
        print(f"'Final Updates During Test' not found, unable to save to {final_updates_test_path}")
    else:
        # Save Final Updates During Test
        with open(final_updates_test_path, "w", encoding='utf-8') as file:
            json.dump({"Final Updates During Test": extracted_updates_test}, file, indent=4, ensure_ascii=False)
        print(f"Final Updates During Test saved to: {final_updates_test_path}")

@simple_retry(max_attempts=10, delay=1)
def diagnosis_stage(args, subfolder, case_output_dir, model_config_diagnosis, final_history_path, final_pe_path, final_test_path):
    """
    Stage 4: Diagnosis
    """
    conversation_path = osp.join(case_output_dir, "diagnosis_conversation.json")
    final_diagnosis_path = osp.join(case_output_dir, "final_diagnosis.json")

    if osp.exists(conversation_path) and osp.exists(final_diagnosis_path):
        print(f"Diagnosis files already exist in {case_output_dir}. Skip diagnosis.")
        return

    # Load Final History, Final PE, and Final Test
    with open(final_history_path, 'r', encoding='utf-8') as f:
        final_history = json.load(f).get("Final History", "")
    with open(final_pe_path, 'r', encoding='utf-8') as f:
        final_pe = json.load(f).get("Final Physical Examination", "")
    with open(final_test_path, 'r', encoding='utf-8') as f:
        final_test = json.load(f).get("Final Test", "")

    # Load Final Updates (if exists)
    final_updates_pe_path = osp.join(case_output_dir, "final_updates_during_physical_examination.json")
    final_updates_test_path = osp.join(case_output_dir, "final_updates_during_test.json")

    final_updates_pe = ""
    final_updates_test = ""

    if osp.exists(final_updates_pe_path):
        with open(final_updates_pe_path, 'r', encoding='utf-8') as f:
            final_updates_pe = json.load(f).get("Final Updates During Physical Examination", "")
        print(f"'Final Updates During Physical Examination' Loaded.")
    else:
        print(f"'Final Updates During Physical Examination' file not found. Skip loading.")

    if osp.exists(final_updates_test_path):
        with open(final_updates_test_path, 'r', encoding='utf-8') as f:
            final_updates_test = json.load(f).get("Final Updates During Test", "")
        print(f"'Final Updates During Test' Loaded.")
    else:
        print(f"'Final Updates During Test' file not found. Skip loading.")

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
Medical Doctor {index}. You are a Doctor Agent specialized in making final diagnoses based on the patient's history, physical examination, and test results.

Your sole responsibility is to analyze the comprehensive information gathered in the previous stages to arrive at the most accurate diagnosis.

You should consider all provided information and reason thoroughly to make informed diagnostic decisions.

Primary Objectives:

- Integrate the patient's history, physical examination, and test results.
- Formulate the most likely diagnosis based on the integrated data.
- Provide differential diagnoses if applicable.
- Provide your diagnostic reasoning
- List key informaiton that help you to reach diagnosis. Give weights to the information you analyze, the weights should reflect how important it is for you to reach the final diagnosis and all weights sum up to 1, example: CT should dialated pancrease(weights 0.55)
- Assign possibility for each diagnosis you finally provded, the possibility should add up to 100%, example: appendicitis 80%, pancreatis 20%.

Output in the following format:
{{
    "key_information":"[key information with weights]"
    "diagnosis_possibility":"[diagnosis with possibility]
}}

after analysis, output the final diagnosis, use the following format:

{{
"Final Diagnosis": "[Your Diagnosis]",
"Differential Diagnosis": "[List of Differential Diagnoses]",
"Diagnostic Reasoning": "[Your Diagnostic Reasoning]"
}}
Reply "TERMINATE" After you made the diagnosis

Here is the patient record:
Final History: {final_history}
Final Physical Examination: {final_pe}
Final Test: {final_test}
"""

        if final_updates_pe:
            doc_system_message += f"Final Updates During Physical Examination: {final_updates_pe}\n"

        if final_updates_test:
            doc_system_message += f"Final Updates During Test: {final_updates_test}\n"

        Doc = AssistantAgent(
            name=name,
            llm_config=model_config_diagnosis,
            system_message=doc_system_message,
        )
        Docs.append(Doc)

    provider_system_message = f"""
You do nothing, remain silent, output nothing.
"""

    provider = AssistantAgent(
        name="Assistant",
        llm_config=model_config_diagnosis,
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
Provide the final diagnosis.
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
        print(f"Final Diagnosis saved to:; {final_diagnosis_path}")


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

    subfolders = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]

    output_dir = args.output_dir

    for subfolder in tqdm(subfolders, desc="Processing Cases"):
        try:
            case_crl = os.path.basename(subfolder)
            identify = f"{args.num_specialists}doctor_{args.n_round}round"

            base_output_dir = osp.join(
                output_dir,
                "trial_expert_prompt",
                args.model_name_diagnosis,  # Adjust as needed
                identify,
                str(args.times),
            )
            case_output_dir = osp.join(base_output_dir, case_crl)
            if not osp.exists(case_output_dir):
                os.makedirs(case_output_dir)
                print(f"Creating subfolder: {case_output_dir}")

            history_gathering(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_history
            )
            pe_gathering(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_pe, 
                osp.join(case_output_dir, "final_history.json")
            )
            test_gathering(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_test, 
                osp.join(case_output_dir, "final_history.json"), 
                osp.join(case_output_dir, "final_physical_examination.json")
            )
            diagnosis_stage(
                args, 
                subfolder, 
                case_output_dir, 
                model_config_diagnosis, 
                osp.join(case_output_dir, "final_history.json"), 
                osp.join(case_output_dir, "final_physical_examination.json"), 
                osp.join(case_output_dir, "final_test.json")
            )

        except Exception as e:
            print(f"Processing folder {subfolder} failed, error: {str(e)}")
            continue

if __name__ == "__main__":
    main()
