import os
from typing import TypeAlias, List, Optional, Union, Tuple
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

        folder_name = os.path.join(line["image_folder"], "frame_aligned_videos")
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
            "The user's full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n\n"
            "The second image shows both the user and the environment from a third-person perspective with a broad view.\n"
            "The user's full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
            "These two images capture the same event at the same time.\n"
            "Your task is to analyze both images along with the question and provide the most accurate response based on the visual information from both perspectives.\n",
}

cocot_prompt = "Please tell me the similarities and differences of these two images, and answer to the question.\n"

ddcot_prompt = (
    "Given the images and question, please think step-by-step about the preliminary knowledge to answer the question, "
    "deconstruct the problem as completely as possible down to necessary sub-questions.\n"
    "Then with the aim of helping humans answer the original question, try to answer the sub-questions.\n"
    "The expected answering form is as follows:\n\n"
    "Sub-questions:\n"
    "1. <sub-question 1>\n"
    "2. <sub-question 2>\n"
    "...\n\n"
    "Sub-answers:\n"
    "1. <sub-answer 1>\n"
    "2. <sub-answer 2>\n"
    "...\n\n"
)
ddcot_prompt_2 = "Give your answer of the question according to the sub-questions and sub-answers.\n"

ccot_prompt = (
    "For the provided images and their associated question, generate a scene graph in JSON format that includes the following:\n"
    "1. Objects that are relevant to answering the question.\n"
    "2. Object attributes that are relevant to answering the question.\n"
    "3. Obect relationships that are relevant to answering the question.\n\n"
    "Just generate the scene graph in JSON format. Do not say extra words.\n\n"
)
ccot_prompt_2 = "Use the images and scene graph as context and answer the following question.\n"

def eval_model(args):
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

    def gpt_response(text_content: str,
                     image_content: Optional[Union[str, List[str]]] = None,
                     system_prompt: Optional[str] = None,
                     history: Optional[List[Dict]] = None,
                     model_client=None,
                     temperature: float = 0.1,
                     model_name: str = "gpt-4o-2024-11-20",
                     n: int = 1
                     ) -> Union[
        Tuple[List[str], List[Dict], Dict[str, int]],
        Tuple[List[List[str]], List[List[Dict]], List[Dict[str, int]]]
    ]:

        if history is None:
            history = []

        if isinstance(image_content, str):
            image_content = [image_content]

        content = []
        if isinstance(image_content, list):
            for img in image_content:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                        "detail": "high"
                    }
                })

        content.append({"type": "text", "text": text_content})
        user_message = {"role": "user", "content": content}

        if system_prompt:
            if not any(msg["role"] == "system" for msg in history):
                history.insert(0, {"role": "system", "content": system_prompt})

        base_history = history + [user_message]

        if model_client is not None:
            response = model_client.chat.completions.create(
                model=model_name,
                messages=base_history,
                stream=False,
                temperature=temperature,
                max_tokens=max_new_token,
                n=n
            )

            outputs = [choice.message.content for choice in response.choices]

            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            else:
                usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }

            return outputs, base_history + [{"role": "assistant", "content": outputs[0]}], usage

        return [], base_history, {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def get_VLM_response(text_content: str,
                         image_content: Optional[Union[str, List[str]]] = None,
                         system_prompt: Optional[str] = None,
                         history: Optional[Dict[str, List[Dict]]] = None,
                         model_client=None,
                         temperature: float = 0.1,
                         model_name: str = "gpt-4o-2024-11-20",
                         n: int = 1):
        max_retries = 10000000
        retry_delay = 10

        for attempt in range(max_retries):
            try:
                response, history, usage = gpt_response(text_content, image_content, system_prompt, history, model_client, temperature, model_name, n)
                return response, history, usage
            except:
                print(f"Error - retried {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    print(f"delayed for {retry_delay} seconds")
                else:
                    print("Exceeded max retries.")
                    return "E) Error", history, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, question_file in enumerate(question_files):
        if args.category != "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
            continue
        question_file_path = os.path.join(question_dir, question_file)
        answers_file = os.path.join(output_dir, question_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        questions = [json.loads(q) for q in open(question_file_path, "r")]
        data_loader = create_data_loader_default(questions, image_dir)

        print(f"***Processing file: {question_file}***Model name: {args.model}***")
        cum_start_time=time.time()
        with open(answers_file, "w") as ans_file:

            for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                start_time = time.time()
                both1 = system["both"]

                client1 = client_generation()

                pixel_value_ego = encode_image(image_ego[0])
                pixel_value_exo = encode_image(image_exo[0])
                pixel_value_both = [pixel_value_ego, pixel_value_exo]

                question = qs[0]
                usage_dict = {
                    "both1": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }

                # -------------------------------
                # cot_type 분기 추가
                # -------------------------------
                if args.cot_type == "ccot":
                    question_prompt = question + ccot_prompt
                    response_1, history, usage_1 = get_VLM_response(
                        text_content=question_prompt,
                        image_content=pixel_value_both,
                        system_prompt=both1,
                        temperature=0.1,
                        model_client=client1)
                    for key in usage_dict["both1"]:
                        usage_dict["both1"][key] += usage_1[key]
                        usage_dict["total"][key] += usage_1[key]

                    question_prompt_2 = f"Scene Graph:\n{response_1}\n" + ccot_prompt_2 + "\n" + question
                    response_2, history, usage_2 = get_VLM_response(
                        text_content=question_prompt_2,
                        image_content=pixel_value_both,
                        system_prompt=both1,
                        history=history,
                        temperature=0.1,
                        model_client=client1)
                    for key in usage_dict["both1"]:
                        usage_dict["both1"][key] += usage_2[key]
                        usage_dict["total"][key] += usage_2[key]

                elif args.cot_type == "cocot":
                    question_prompt_2 = question + cocot_prompt
                    response_2, history, usage_2 = get_VLM_response(
                        text_content=question_prompt_2,
                        image_content=pixel_value_both,
                        system_prompt=both1,
                        temperature=0.1,
                        model_client=client1)
                    for key in usage_dict["both1"]:
                        usage_dict["both1"][key] += usage_2[key]
                        usage_dict["total"][key] += usage_2[key]

                elif args.cot_type == "ddcot":
                    question_prompt = question + ddcot_prompt
                    response_1, history, usage_1 = get_VLM_response(
                        text_content=question_prompt,
                        image_content=pixel_value_both,
                        system_prompt=both1,
                        temperature=0.1,
                        model_client=client1)
                    for key in usage_dict["both1"]:
                        usage_dict["both1"][key] += usage_1[key]
                        usage_dict["total"][key] += usage_1[key]

                    question_prompt_2 = f"Context:\n{response_1}\n" + ddcot_prompt_2 + "\n" + question
                    response_2, history, usage_2 = get_VLM_response(
                        text_content=question_prompt_2,
                        image_content=pixel_value_both,
                        system_prompt=both1,
                        history=history,
                        temperature=0.1,
                        model_client=client1)
                    for key in usage_dict["both1"]:
                        usage_dict["both1"][key] += usage_2[key]
                        usage_dict["total"][key] += usage_2[key]

                else:  # baseline
                    question_prompt_2 = question
                    response_2, history, usage_2 = get_VLM_response(
                        text_content=question_prompt_2,
                        image_content=pixel_value_both,
                        system_prompt=both1,
                        temperature=0.1,
                        model_client=client1)
                    for key in usage_dict["both1"]:
                        usage_dict["both1"][key] += usage_2[key]
                        usage_dict["total"][key] += usage_2[key]
                # -------------------------------

                end_time = time.time()
                elapsed_time = end_time - start_time
                print("elapsed time: {:.2f}s".format(elapsed_time))
                new_dict = {
                    "question_prompt": question,
                    "model_answer": response_2,
                    "label": ans[0],
                    "model_name": model_name,
                    "usage": usage_dict,
                    "elapsed_time": elapsed_time,
                    "cot_type": args.cot_type
                }
                new_dict = update_dict(new_dict, line)
                ans_file.write(json.dumps(new_dict) + "\n")

        cum_end_time = time.time()
        elapsed_time = cum_end_time - cum_start_time
        print(f"elapsed time for {question_file}: {elapsed_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Model")
    parser.add_argument("--image-dir", type=str, default="/path/to/E3VQA/paired_images/")
    parser.add_argument("--output-dir", type=str, default="/path/to/output_dir/")
    parser.add_argument("--question-dir", type=str, default="/path/to/E3VQA/E3VQA_benchmark/")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--max-new-token", type=int, default=2048)
    parser.add_argument("--category", type=str, default='total')
    parser.add_argument("--perspective", type=str, default='')
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--cot_type", type=str, default="baseline", choices=["ccot", "cocot", "ddcot", "baseline"])
    args = parser.parse_args()
    eval_model(args)
