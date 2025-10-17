import argparse
import torch
import os
from typing import Dict, Optional, Sequence, List, TypeAlias
import json
from tqdm import tqdm
from llava.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers import AutoProcessor, AutoModelForVision2Seq, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers import AutoModel
from qwen_vl_utils import process_vision_info
from transformers.image_utils import load_image
import PIL.Image
from google import genai
from google.genai import types
from openai import OpenAI
import base64
from PIL import Image
import anthropic
import time
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import random

def update_dict(new_dict, line):
    for key, value in line.items():
        if key == "options":
            flattened_options = [item[0] for item in value]
            new_dict[key] = flattened_options
        else:
            new_dict[key] = value[0]
    return new_dict

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.standard_b64encode(image_file.read()).decode('utf-8')

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

        line["options"] = [option.lower() for option in line["options"]]

        try:
            index = line["options"].index(line["answer"].lower())
            ans = f"{chr(65 + index)}) {line['answer']}"
        except:
            ans = "E) Error"

        return qs, image_ego, image_exo, ans, line

    def __len__(self):
        return len(self.questions)


def create_data_loader_default(questions, image_folder, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=None)
    return data_loader

def eval_model(args):
    model_name = args.model
    image_dir = args.image_dir
    question_dir = args.question_dir
    output_dir = os.path.join(args.output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)
    question_files = [f for f in os.listdir(question_dir) if f.endswith(".jsonl")]

    system = (
        "You are a helpful assistant.\n"
            "You are provided with two visual inputs in sequence, each captured from a different perspective:\n"
            "1. The view from the camera worn by the user ('I').\n"
            "2. The view captured by an external camera observing the user ('I').\n\n"
            "The first image shows what the user ('I') sees from their perspective.\n"
            "The user's ('My') full body cannot be visible; you may only see parts of their body, like their hand, foot, or arm, or in some cases, none of the user's body at all.\n\n"
            "The second image shows both the user and the environment from a third-person perspective with a broad view.\n"
            "The user's ('My') full body is visible, but due to the fixed viewpoint, some parts may not be visible.\n\n"
            "These two images capture the same event at the same time.\n"
            "Your task is to analyze both images along with the question and provide the most accurate response based on the visual information from both perspectives.\n"
    )

    torch.cuda.empty_cache()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if "gemini" in model_name.lower():
        detail_model_name = "gemini-2.0-flash"
        client = genai.Client(api_key="API_KEY")
        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]
            data_loader = create_data_loader_default(questions, image_dir)
            seed=int(args.seed)
            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    both = client.chats.create(
                        model=detail_model_name,
                        config=types.GenerateContentConfig(
                            max_output_tokens=2048,
                            temperature=0.1,
                            system_instruction=system,
                            seed= seed,
                        )
                    )
                    image1 = PIL.Image.open(image_ego[0])
                    image2 = PIL.Image.open(image_exo[0])
                    question_prompt = qs[0]

                    user_message = [image1, image2, question_prompt]
                    response_1 = both.send_message(user_message)
                    response_1 = response_1.text

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response_1,
                        "label": ans[0],
                        "model_name": model_name,
                        "debate_prompt": question_prompt,
                    }
                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")

    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= "Model_name")
    parser.add_argument("--image-dir", type=str, default="/path/to/E3VQA/paired_images/")
    parser.add_argument("--output-dir", type=str, default="/path/to/output_dir/")
    parser.add_argument("--question-dir", type=str, default="/path/to/E3VQA/E3VQA_benchmark/")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--max-new-token", type=int, default="2048")
    parser.add_argument("--category", type=str, default='')
    parser.add_argument("--perspective", type=str, default='')
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    eval_model(args)