import os
from collections import Counter
from typing import TypeAlias, List, Optional, Union
import argparse
import random
import time
import base64
from openai import OpenAI
from typing import Dict, Optional, Sequence, List, TypeAlias
import json
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed

import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

def update_dict(new_dict, line):
    for key, value in line.items():
        if key == "options":
            flattened_options = [item[0] for item in value]
            new_dict[key] = flattened_options
        else:
            new_dict[key] = value[0]
    return new_dict

class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]

        formatted_options = "\n".join([f"{chr(65 + i)}) {option}" for i, option in enumerate(line["options"])])

        qs = ("Question:\n"
              "{}\n\n"
              "Choices:\n"
              "{}\n\n"
              "Only one option is correct.\n"
              "Present the answer in the form X).\n\n").format(line["question"], formatted_options)

        folder_name = os.path.join(line["image_folder"], "frame_aligned_videos") #frame_aligned_videos
        image_ego = os.path.join(self.image_folder, folder_name, line["cam_ego"], line["frame"])
        image_exo = os.path.join(self.image_folder, folder_name, line["cam_exo"], line["frame"])

        line["options"] = [option for option in line["options"]]
        index = line["options"].index(line["answer"])
        ans = f"{chr(65 + index)}) {line['answer']}"

        return qs, image_ego, image_exo, ans, line

    def __len__(self):
        return len(self.questions)


def create_data_loader_default(questions, image_folder, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=None)
    return data_loader

system = {
    "both": "You are a helpful assistant.\n"
            "You are provided with two visual inputs in sequence, each captured from a different perspective:\n"
            "1. The view from the camera worn by the user ('I').\n"
            "2. The view captured by an external camera observing the user ('I').\n\n"
            "The first image shows what the user ('I') sees from their perspective.\n"
            "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n\n"
            "The second image shows both the user and the environment from a third-person perspective with a broad view.\n"
            "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
            "These two images capture the same event at the same time.\n"
            "Your task is to analyze both images along with the question and provide the most accurate response based on the visual information from both perspectives.\n",
}

space_reasoning_forward = (
    "Background Knowledge:\n"
    "This image shows what the user ('I') sees from their perspective.\n"
    "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n\n"
    "Task:\n"
    "For the provided image and its associated question, generate a scene graph in JSON format that includes the following:\n"
    "1. Objects that are relevant to answering the question.\n"
    "2. Object attributes that are relevant to answering the question.\n"
    "3. Object relationships that are relevant to answering the question.\n\n"
    "Just generate the scene graph in JSON format. Do not say extra words.\n\n"
)
space_reasoning_forward_2 = (
    "Background Knowledge:\n"
    "This image shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
    "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
   "Task:\n"
    "For the provided image from a different view and the scene graph generated from the previous view, refine the scene graph in JSON format as follows:\n"
    "1. Review and Update Existing Objects and Relationships:\n"
    "Examine the objects and relationships in the initial scene graph. Update their attributes or positions based on observations from both views. Remove only elements that are clearly erroneous (e.g., annotation errors or duplicates)\n\n"
    "2. Incorporate New Information:\n"
    "Identify and add any new objects or relationships that appear in the new view.\n\n"
    "3. Align and Reconcile Across Views:\n"
"For overlapping objects and relationships, align them using spatial proximity and semantic similarity. If attribute discrepancies arise, select values that best reflect the combined observations.\n\n"
    "Ensure that the updated scene graph is logically and physically consistent, avoiding contradictions or impossible configurations.\nJust generate the refined scene graph in JSON format. Do not say extra words.\n\n"
)

space_reasoning_forward_3 = "Use the images and the refined scene graph as context and answer the following question.\n"

space_reasoning_reverse = (
    "Background Knowledge:\n"
    "This image shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
    "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
    "Task:\n"
    "For the provided image and its associated question, generate a scene graph in JSON format that includes the following:\n"
    "1. Objects that are relevant to answering the question.\n"
    "2. Object attributes that are relevant to answering the question.\n"
    "3. Object relationships that are relevant to answering the question.\n\n"
    "Just generate the scene graph in JSON format. Do not say extra words.\n\n"
)
space_reasoning_reverse_2 = (
    "Background Knowledge:\n"
    "This image shows what the user ('I') sees from their perspective.\n"
    "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n\n"
   "Task:\n"
    "For the provided image from a different view and the scene graph generated from the previous view, refine the scene graph in JSON format as follows:\n"
    "1. Review and Update Existing Objects and Relationships:\n"
    "Examine the objects and relationships in the initial scene graph. Update their attributes or positions based on observations from both views. Remove only elements that are clearly erroneous (e.g., annotation errors or duplicates)\n\n"
    "2. Incorporate New Information:\n"
    "Identify and add any new objects or relationships that appear in the new view.\n\n"
    "3. Align and Reconcile Across Views:\n"
"For overlapping objects and relationships, align them using spatial proximity and semantic similarity. If attribute discrepancies arise, select values that best reflect the combined observations.\n\n"
    "Ensure that the updated scene graph is logically and physically consistent, avoiding contradictions or impossible configurations.\nJust generate the refined scene graph in JSON format. Do not say extra words.\n\n"
)

space_reasoning_reverse_3 = "Use the images and the refined scene graph as context and answer the following question.\n"

space_reasoning_unified = (
    "Background Knowledge:\n"
    "Image 1 shows what the user ('I') sees from their perspective.\n"
    "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n"
    "Image 2 shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
    "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n"
    "Task:\n"
    "Using the provided two images and their associated question, generate a unified scene graph in JSON format that includes the following:\n"
    "1. Objects that are relevant to answering the question.\n"
    "2. Object attributes that are relevant to answering the question.\n"
    "3. Object relationships that are relevant to answering the question.\n"
    "4. Ensure that objects and relationships from both perspectives are appropriately aligned, integrated, and refined to provide a complete scene representation.\n\n"
    "Just generate the unified scene graph in JSON format. Do not say extra words.\n\n"
)

space_reasoning_unified_reverse = (
    "Background Knowledge:\n"
    "Image 1 shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
    "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n"
    "Image 2 shows what the user ('I') sees from their perspective.\n"
    "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n"
    "Task:\n"
    "Using the provided two images and their associated question, generate a unified scene graph in JSON format that includes the following:\n"
    "1. Objects that are relevant to answering the question.\n"
    "2. Object attributes that are relevant to answering the question.\n"
    "3. Object relationships that are relevant to answering the question.\n"
    "4. Ensure that objects and relationships from both perspectives are appropriately aligned, integrated, and refined to provide a complete scene representation.\n\n"
    "Just generate the unified scene graph in JSON format. Do not say extra words.\n\n"
)

space_reasoning_unified_2 = "Use the images and the unified scene graph as context and answer the following question.\n"

async def eval_model(args):
    # seed setting
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    model_name = args.model
    image_dir = args.image_dir
    question_dir = args.question_dir
    max_new_token = args.max_new_token
    output_dir = os.path.join(args.output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)
    question_files = [f for f in os.listdir(question_dir) if f.endswith(".jsonl")]

    def client_generation(key=None):
        if key is None:
            key = "API_KEY"
        return OpenAI(
            api_key=key,
        )

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    async def gpt_response_async(
            text_content: str,
            image_content: Optional[Union[str, List[str]]] = None,
            system_prompt: Optional[str] = None,
            history: Optional[List[Dict]] = None,
            model_client=None,
            model_name: str = "gpt-4o-2024-11-20",
            max_new_token: int = 2048,
    ) -> List[Dict]:
        if history is None:
            history = []

        if isinstance(image_content, str):
            image_content = [image_content]

        content = []
        if isinstance(image_content, list):
            for img_base64 in image_content:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": "high"
                    }
                })

        content.append({"type": "text", "text": text_content})
        user_message = {"role": "user", "content": content}

        if system_prompt:
            system_message = {"role": "system", "content": system_prompt}
            if not any(msg["role"] == "system" for msg in history):
                history.insert(0, system_message)

        history.append(user_message)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,  # default ThreadPoolExecutor
            lambda: model_client.chat.completions.create(
                model=model_name,
                messages=history,
                stream=False,
                temperature=0.1,
                max_tokens=max_new_token,
            )
        )

        assistant_output = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_output})

        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return assistant_output, history, usage

    async def get_VLM_response_async(
            text_content: str,
            image_content: Optional[Union[str, List[str]]] = None,
            system_prompt: Optional[str] = None,
            history: Optional[List[Dict]] = None,
            model_client=None,
            model_name: str = "gpt-4o-2024-11-20",
            max_new_token: int = 2048,
    ) -> List[Dict]:
        max_retries = 10
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                response, history_out, usage = await gpt_response_async(
                    text_content=text_content,
                    image_content=image_content,
                    system_prompt=system_prompt,
                    history=history,
                    model_client=model_client,
                    model_name=model_name,
                    max_new_token=max_new_token
                )
                return response, history_out, usage
            except Exception as e:
                print(f"Error - retries {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Exceeded max retries.")
                    return "E) Error", history or []

    all_files_cum_start_time = time.time()
    for i, question_file in enumerate(question_files):
        if args.category != "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
            continue
        question_file_path = os.path.join(question_dir, question_file)
        answers_file = os.path.join(output_dir, question_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        questions = [json.loads(q) for q in open(question_file_path, "r")]

        data_loader = create_data_loader_default(questions, image_dir)

        print(f"***Processing file: {question_file}***Model name: {args.model}***")
        cum_start_time = time.time()
        with open(answers_file, "w") as ans_file:
            for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                start_time = time.time()
                both1 = system["both"]
                both2 = system["both"]
                both3 = system["both"]
                history = {"both1": '', "both2": '', "both3": ''}

                client1 = client_generation()
                client2 = client_generation()
                client3 = client_generation()

                client_usage = {
                    "both1": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "both2": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "both3": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

                pixel_value_ego = encode_image(image_ego[0])
                pixel_value_exo = encode_image(image_exo[0])
                pixel_value_both = [pixel_value_ego, pixel_value_exo]

                max_rounds = 1
                for round_number in range(max_rounds):
                    question = qs[0]


                    ######### initial SG generation #########
                    # Prepare the three calls for initial scene graphs in parallel
                    tasks = await asyncio.gather(
                        get_VLM_response_async(
                            text_content=question + space_reasoning_forward,
                            image_content=pixel_value_ego,
                            system_prompt=both1,
                            model_client=client1
                        ),
                        get_VLM_response_async(
                            text_content=question + space_reasoning_reverse,
                            image_content=pixel_value_exo,
                            system_prompt=both2,
                            model_client=client2
                        ),
                        get_VLM_response_async(
                            text_content=question + space_reasoning_unified,
                            image_content=pixel_value_both,
                            system_prompt=both3,
                            model_client=client3
                        )
                    )

                    # Unpack responses and update history
                    (response_1_both1, history["both1"], usage1), (response_1_both2, history["both2"], usage2), (
                    response_1_both3, history["both3"], usage3) = tasks
                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        client_usage["both1"][key] += usage1[key]
                        client_usage["total"][key] += usage1[key]

                        client_usage["both2"][key] += usage2[key]
                        client_usage["total"][key] += usage2[key]

                        client_usage["both3"][key] += usage3[key]
                        client_usage["total"][key] += usage3[key]

                    ######### initial SG generation #########

                    ######### initial SG refinement #########
                    tasks_refine = await asyncio.gather(
                        get_VLM_response_async(
                            text_content=question + space_reasoning_forward_2,
                            image_content=pixel_value_exo,
                            history=history["both1"],
                            model_client=client1
                        ),
                        get_VLM_response_async(
                            text_content=question + space_reasoning_reverse_2,
                            image_content=pixel_value_ego,
                            history=history["both2"],
                            model_client=client2
                        )
                    )

                    (response_2_both1, history["both1"], usage1), (response_2_both2, history["both2"], usage2) = tasks_refine
                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        client_usage["both1"][key] += usage1[key]
                        client_usage["total"][key] += usage1[key]

                        client_usage["both2"][key] += usage2[key]
                        client_usage["total"][key] += usage2[key]


                    ######### initial SG refinement #########



                    ######### initial Answer #########
                    answer_prompt = "Use the images and the unified scene graph as context and answer the following question:\n"
                    text= answer_prompt + question
                    tasks_initial_answers = await asyncio.gather(
                        get_VLM_response_async(
                            text_content=text,
                            image_content=pixel_value_both,
                            history=history["both1"],
                            model_client=client1
                        ),
                        get_VLM_response_async(
                            text_content=text,
                            image_content=pixel_value_both,
                            history=history["both2"],
                            model_client=client2
                        ),
                        get_VLM_response_async(
                            text_content=text,
                            image_content=pixel_value_both,
                            history=history["both3"],
                            model_client=client3
                        )
                    )

                    (response_a_both1, history["both1"], usage1), (response_a_both2, history["both2"], usage2), (
                    response_a_both3, history["both3"], usage3) = tasks_initial_answers

                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        client_usage["both1"][key] += usage1[key]
                        client_usage["total"][key] += usage1[key]

                        client_usage["both2"][key] += usage2[key]
                        client_usage["total"][key] += usage2[key]

                        client_usage["both3"][key] += usage3[key]
                        client_usage["total"][key] += usage3[key]

                    ######### initial Answer #########

                    answers = []
                    for choice in ['A)', 'B)', 'C)', 'D)']:
                        if choice in response_a_both1:
                            answers.append(choice)
                        if choice in response_a_both2:
                            answers.append(choice)
                        if choice in response_a_both3:
                            answers.append(choice)

                    answer_counter = Counter(answers)

                    if len(answer_counter) == 1:
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Time taken: {elapsed_time:.2f} seconds")
                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{response_a_both1}',
                            "label": ans[0],
                            "model_name": model_name,
                            # "debate_prompt": f'initial step:{space_reasoning_reverse_2}',
                            "Time": f"{elapsed_time:.2f} seconds",
                            "Phase": "Finished at first phase",
                            "response_1_both1": response_1_both1,
                            "response_2_both1": response_2_both1,
                            "response_1_both2": response_1_both2,
                            "response_2_both2": response_2_both2,
                            "response_1_both3": response_1_both3,
                            "response_a_both1": response_a_both1,
                            "response_a_both2": response_a_both2,
                            "response_a_both3": response_a_both3,
                            "usage_both1": client_usage["both1"],
                            "usage_both2": client_usage["both2"],
                            "usage_both3": client_usage["both3"],
                            "usage_total": client_usage["total"],
                        }
                        total_tokens = client_usage["total"]
                        print(
                            f"** {j}-th Final answer: 0Turn{new_dict['model_answer']} Label: {ans}**\n")


                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")
                        break
                    other_agent_message_both1 = f"One scene graph: {response_2_both2}\n\nOne scene graph: {response_1_both3}\n"
                    other_agent_message_both2 = f"One scene graph: {response_2_both1}\n\nOne scene graph: {response_1_both3}\n"
                    other_agent_message_both3 = f"One scene graph: {response_2_both2}\n\nOne scene graph: {response_2_both1}\n"

                    user_message_both1 = [
                        "Background Knowledge:\n"
                        "Image 1 shows what the user ('I') sees from their perspective.\n"
                        "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n"
                        "Image 2 shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
                        "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
                        "Task:\n"
                        "Below are different scene graphs generated using different reasoning methods:\n" +
                        f"{other_agent_message_both1}" +
                        "Using the scene graphs generated from different methods as additional context, generate a refined scene graph in JSON format for the provided images and their associated question as follows:\n"
                        "1. Review the objects and relationships from the scene graphs and make any necessary adjustments to better align with both views.\n"
                        "2. Ensure that overlapping objects or relationships between the two views are appropriately aligned and refined, enhancing the accuracy of the scene graph.\n\n"
                        "Just generate the refined scene graph in JSON format. Do not say extra words.\n\n"
                    ]
                    user_message_both1 = question + user_message_both1[0]

                    user_message_both2 = [
                        "Background Knowledge:\n"
                        "Image 1 shows what the user ('I') sees from their perspective.\n"
                        "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n"
                        "Image 2 shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
                        "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
                        "Task:\n"
                        "Below are different scene graphs generated using different reasoning methods:\n" +
                        f"{other_agent_message_both2}" +
                        "Using the scene graphs generated from different methods as additional context, generate a refined scene graph in JSON format for the provided images and their associated question as follows:\n"
                        "1. Review the objects and relationships from the scene graphs and make any necessary adjustments to better align with both views.\n"
                        "2. Ensure that overlapping objects or relationships between the two views are appropriately aligned and refined, enhancing the accuracy of the scene graph.\n\n"
                        "Just generate the refined scene graph in JSON format. Do not say extra words.\n\n"
                    ]
                    user_message_both2 = question + user_message_both2[0]

                    user_message_both3 = [
                        "Background Knowledge:\n"
                        "Image 1 shows what the user ('I') sees from their perspective.\n"
                        "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n"
                        "Image 2 shows both the user ('I') and the environment from a third-person perspective with a broad view.\n"
                        "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
                        "Task:\n"
                        "Below are different scene graphs generated using different reasoning methods:\n" +
                        f"{other_agent_message_both3}" +
                        "Using the scene graphs generated from different methods as additional context, generate a refined scene graph in JSON format for the provided images and their associated question as follows:\n"
                        "1. Review the objects and relationships from the scene graphs and make any necessary adjustments to better align with both views.\n"
                        "2. Ensure that overlapping objects or relationships between the two views are appropriately aligned and refined, enhancing the accuracy of the scene graph.\n\n"
                        "Just generate the refined scene graph in JSON format. Do not say extra words.\n\n"
                    ]
                    user_message_both3 = question + user_message_both3[0]

                    ######### Unified SG Refinement #########
                    tasks_refined_sg = await asyncio.gather(
                        get_VLM_response_async(
                            text_content=user_message_both1,
                            image_content=pixel_value_both,
                            history=history["both1"],
                            model_client=client1
                        ),
                        get_VLM_response_async(
                            text_content=user_message_both2,
                            image_content=pixel_value_both,
                            history=history["both2"],
                            model_client=client2
                        ),
                        get_VLM_response_async(
                            text_content=user_message_both3,
                            image_content=pixel_value_both,
                            history=history["both3"],
                            model_client=client3
                        )
                    )

                    (response_fin_both1, history["both1"], usage1), (response_fin_both2, history["both2"], usage2), (
                    response_fin_both3, history["both3"], usage3) = tasks_refined_sg

                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        client_usage["both1"][key] += usage1[key]
                        client_usage["total"][key] += usage1[key]

                        client_usage["both2"][key] += usage2[key]
                        client_usage["total"][key] += usage2[key]

                        client_usage["both3"][key] += usage3[key]
                        client_usage["total"][key] += usage3[key]

                    ######### Unified SG Refinement #########

                    additional_prompt_both1 = (
                        f"{space_reasoning_unified_2}"
                    )

                    additional_prompt_both2 = (
                        f"{space_reasoning_unified_2}"
                    )

                    additional_prompt_both3 = (
                        f"{space_reasoning_unified_2}"
                    )

                    ######### Final answer #########
                    final_tasks = await asyncio.gather(
                        get_VLM_response_async(
                            text_content=additional_prompt_both1 + question,
                            image_content=pixel_value_both,
                            history=history["both1"],
                            model_client=client1
                        ),
                        get_VLM_response_async(
                            text_content=additional_prompt_both2 + question,
                            image_content=pixel_value_both,
                            history=history["both2"],
                            model_client=client2
                        ),
                        get_VLM_response_async(
                            text_content=additional_prompt_both3 + question,
                            image_content=pixel_value_both,
                            history=history["both3"],
                            model_client=client3
                        )
                    )

                    (response_3_both1, history["both1"], usage1), (response_3_both2, history["both2"], usage2), (
                    response_3_both3, history["both3"], usage3) = final_tasks
                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        client_usage["both1"][key] += usage1[key]
                        client_usage["total"][key] += usage1[key]

                        client_usage["both2"][key] += usage2[key]
                        client_usage["total"][key] += usage2[key]

                        client_usage["both3"][key] += usage3[key]
                        client_usage["total"][key] += usage3[key]

                    ######### Final answer #########
                    torch.cuda.empty_cache()
                    answers = []
                    for choice in ['A)', 'B)', 'C)', 'D)']:
                        if choice in response_3_both1:
                            answers.append(choice)
                        if choice in response_3_both2:
                            answers.append(choice)
                        if choice in response_3_both3:
                            answers.append(choice)

                    answer_counter = Counter(answers)

                    if len(answer_counter) == 1:
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{response_3_both1}',
                            "label": ans[0],
                            "model_name": model_name,
                            "Time": f"{elapsed_time:.2f} seconds",
                            # "debate_prompt": f'{other_agent_message_both1}',
                            "Phase": "Converged Answer after SG refinement",
                            "response_1_both1": response_1_both1,
                            "response_2_both1": response_2_both1,
                            "response_1_both2": response_1_both2,
                            "response_2_both2": response_2_both2,
                            "response_1_both3": response_1_both3,
                            "response_a_both1": response_a_both1,
                            "response_a_both2": response_a_both2,
                            "response_a_both3": response_a_both3,
                            "response_fin_both1": response_fin_both1,
                            "response_fin_both2": response_fin_both2,
                            "response_fin_both3": response_fin_both3,
                            "response_3_both1": response_3_both1,
                            "response_3_both2": response_3_both2,
                            "response_3_both3": response_3_both3,
                            "usage_both1": client_usage["both1"],
                            "usage_both2": client_usage["both2"],
                            "usage_both3": client_usage["both3"],
                            "usage_total": client_usage["total"],
                        }
                        total_tokens = client_usage["total"]
                        print(
                            f"** {j}-th Final answer: 1Turn1answer{new_dict['model_answer']} Label: {ans}**\n"
                        f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")
                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")
                        break

                    elif len(answer_counter) == 2 and any(
                            count >= 2 for count in answer_counter.values()) and round_number == max_rounds - 1:

                        for answer, count in answer_counter.items():
                            if count >= 2:
                                model_answer = str(answer + ')')
                                break
                        end_time = time.time()
                        elapsed_time = end_time - start_time

                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{model_answer}',
                            "label": ans[0],
                            "model_name": model_name,
                            "Time": f"{elapsed_time:.2f} seconds",
                            # "debate_prompt": f'{other_agent_message_both1}',
                            "Phase": "Majority vote done",
                            "response_1_both1": response_1_both1,
                            "response_2_both1": response_2_both1,
                            "response_1_both2": response_1_both2,
                            "response_2_both2": response_2_both2,
                            "response_1_both3": response_1_both3,
                            "response_a_both1": response_a_both1,
                            "response_a_both2": response_a_both2,
                            "response_a_both3": response_a_both3,
                            "response_fin_both1": response_fin_both1,
                            "response_fin_both2": response_fin_both2,
                            "response_fin_both3": response_fin_both3,
                            "response_3_both1": response_3_both1,
                            "response_3_both2": response_3_both2,
                            "response_3_both3": response_3_both3,
                            "usage_both1": client_usage["both1"],
                            "usage_both2": client_usage["both2"],
                            "usage_both3": client_usage["both3"],
                            "usage_total": client_usage["total"],
                        }
                        total_tokens = client_usage["total"]

                        print(
                            f"****************** {j}-th Final answer: {new_dict['model_answer']} ********************\n"
                            f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")
                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")
                    elif round_number == max_rounds-1:
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{response_3_both1}',
                            "label": ans[0],
                            "model_name": model_name,
                            "Time": f"{elapsed_time:.2f} seconds",
                            # "debate_prompt": f'{other_agent_message_both1}',
                            "tt": True,
                            "Phase": "All 3 different answers",
                            "response_1_both1": response_1_both1,
                            "response_2_both1": response_2_both1,
                            "response_1_both2": response_1_both2,
                            "response_2_both2": response_2_both2,
                            "response_1_both3": response_1_both3,
                            "response_a_both1": response_a_both1,
                            "response_a_both2": response_a_both2,
                            "response_a_both3": response_a_both3,
                            "response_fin_both1": response_fin_both1,
                            "response_fin_both2": response_fin_both2,
                            "response_fin_both3": response_fin_both3,
                            "response_3_both1": response_3_both1,
                            "response_3_both2": response_3_both2,
                            "response_3_both3": response_3_both3,
                            "usage_both1": client_usage["both1"],
                            "usage_both2": client_usage["both2"],
                            "usage_both3": client_usage["both3"],
                            "usage_total": client_usage["total"],
                        }
                        total_tokens = client_usage["total"]


                        print(
                            f"****************** {j}-th Final answer: {new_dict['model_answer']} ********************\n"
                            f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")
                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")
        cum_end_time = time.time()
        elapsed_time = cum_end_time - cum_start_time
        print(f"elapsed time for {question_file}: {elapsed_time:.2f} seconds")
    all_files_cum_end_time = time.time()
    elapsed_time = all_files_cum_end_time - all_files_cum_start_time
    print(f"elapsed time: {elapsed_time:.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= "Model")
    parser.add_argument("--image-dir", type=str, default="/path/to/E3VQA/paired_images/")
    parser.add_argument("--output-dir", type=str, default="/path/to/output_dir/")
    parser.add_argument("--question-dir", type=str, default="/path/to/E3VQA/E3VQA_benchmark/")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--max-new-token", type=int, default="2048")
    parser.add_argument("--category", type=str, default='total')
    parser.add_argument("--perspective", type=str, default='')
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    asyncio.run(eval_model(args))
