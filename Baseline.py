import argparse
import torch
import os
import sys
from typing import Dict, Optional, Sequence, List, TypeAlias
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import PIL.Image
from PIL import Image
from google import genai
from google.genai import types
from openai import OpenAI
import base64
import anthropic
import time
import random
import warnings
import copy
import re

# Transformers / LLaVA / DeepSeek Imports
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoProcessor,
    AutoModelForVision2Seq, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration,
    AutoModel, set_seed
)
from transformers.image_utils import load_image
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from qwen_vl_utils import process_vision_info

# Torchvision Imports
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


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

# LLaVA-NeXT specific preprocessing function
def preprocess_qwen(sources, tokenizer: AutoTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens
    input_ids, targets = [], []
    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]
    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target
    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


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

    if "qwen-vl-chat" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto",
                                                          trust_remote_code=True).eval()
        model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat",
                                                                        trust_remote_code=True)
        model.generation_config.top_p = None
        model.generation_config.do_sample = False
        model.generation_config.top_k = None

        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    query = tokenizer.from_list_format([
                        {'image': image_ego[0]},
                        {'image': image_exo[0]},
                        {'text': qs[0]},
                    ])


                    response, history = model.chat(tokenizer, query=query, history=None, system=system)

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }

                    new_dict = update_dict(new_dict, line)

                    ans_file.write(json.dumps(new_dict) + "\n")




    elif "internvl2" in model_name.lower():
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def intern_load_image(image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values


        tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-8B", trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2-8B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto").eval()
        generation_config = dict(max_new_tokens=1024, do_sample=False)

        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    pixel_values1 = intern_load_image((image_ego[0]), max_num=12).to(
                        torch.bfloat16).cuda()
                    pixel_values2 = intern_load_image(image_exo[0], max_num=12).to(
                        torch.bfloat16).cuda()
                    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

                    question = "Image-1: <image>\nImage-2: <image>\n{}".format(qs[0])

                    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                                   num_patches_list=num_patches_list,
                                                   history=None, return_history=True)

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }

                    new_dict = update_dict(new_dict, line)

                    ans_file.write(json.dumps(new_dict) + "\n")
                    ans_file.flush()

    elif "internvl3" in model_name.lower():
        # Intern functions
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size):
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
            return transform

        def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

        def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
            orig_width, orig_height = image.size
            aspect_ratio = orig_width / orig_height

            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
                i * j <= max_num and i * j >= min_num)
            target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

            # find the closest aspect ratio to the target
            target_aspect_ratio = find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

            # calculate the target width and height
            target_width = image_size * target_aspect_ratio[0]
            target_height = image_size * target_aspect_ratio[1]
            blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

            # resize the image
            resized_img = image.resize((target_width, target_height))
            processed_images = []
            for i in range(blocks):
                box = (
                    (i % (target_width // image_size)) * image_size,
                    (i // (target_width // image_size)) * image_size,
                    ((i % (target_width // image_size)) + 1) * image_size,
                    ((i // (target_width // image_size)) + 1) * image_size
                )
                # split the image
                split_img = resized_img.crop(box)
                processed_images.append(split_img)
            assert len(processed_images) == blocks
            if use_thumbnail and len(processed_images) != 1:
                thumbnail_img = image.resize((image_size, image_size))
                processed_images.append(thumbnail_img)
            return processed_images

        def intern_load_image(image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        path = 'OpenGVLab/InternVL3-8B' #TODO change to 'OpenGVLab/InternVL3-14B' for 14B model

        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue

            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]
            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()

                    pixel_values1 = intern_load_image(image_ego[0], max_num=12).to(torch.bfloat16).cuda()
                    pixel_values2 = intern_load_image(image_exo[0], max_num=12).to(torch.bfloat16).cuda()
                    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                    generation_config = dict(max_new_tokens=1024, do_sample=True)

                    question = qs[0]

                    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                                   history=None, return_history=True)

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }
                    new_dict = update_dict(new_dict, line)

                    ans_file.write(json.dumps(new_dict) + "\n")



    elif "mantis" in model_name.lower():
        processor = AutoProcessor.from_pretrained(
            "TIGER-Lab/Mantis-8B-Idefics2")
        model = AutoModelForVision2Seq.from_pretrained(
            "TIGER-Lab/Mantis-8B-Idefics2",
            device_map="cuda",
            torch_dtype=torch.bfloat16
        )
        generation_kwargs = {
            "max_new_tokens": 1024,
            "do_sample" : False,
        }


        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()

                    image1 = load_image(image_ego[0])
                    image2 = load_image(image_exo[0])
                    images = [image1, image2]

                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": system},
                            ]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "image"},
                                {"type": "text", "text": qs[0]},
                            ]
                        }
                    ]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    inputs = processor(text=prompt, images=images, return_tensors="pt")
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}


                    generated_ids = model.generate(**inputs, **generation_kwargs)
                    response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }
                    new_dict = update_dict(new_dict, line)

                    ans_file.write(json.dumps(new_dict) + "\n")


    elif "gpt" in model_name.lower() :
        detail_model_name = "gpt-4o"
        api_key = "API_Key"
        client = OpenAI(
                api_key=api_key,
                )

        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    image1 = encode_image(image_ego[0])
                    image2 = encode_image(image_exo[0])

                    response = client.chat.completions.create(
                        model= detail_model_name,
                        messages=[
                        {"role": "system", "content": system},
                          {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image1}",
                                        "detail": "high"
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image2}",
                                        "detail": "high"
                                    },
                                },
                                {"type": "text", "text": qs[0]},
                            ],
                          }
                        ],
                        stream=False,
                        temperature=0.1,
                        max_tokens=2048,
                        seed=2000
                    )

                    response = response.choices[0].message.content

                    new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": response,
                            "label": ans[0],
                            "model_name": model_name,
                        }

                    new_dict = update_dict(new_dict,line)

                    ans_file.write(json.dumps(new_dict) + "\n")


    elif "claude" in model_name.lower():
        detail_model_name = "claude-3-5-sonnet-20241022"
        api_key = "API Key"
        client = anthropic.Anthropic(
            api_key=api_key,
        )

        def response_exception(prompt, image1, image2):
            retries = 0
            max_retries = 1000000

            while retries < max_retries:
                try:
                    message = client.messages.create(
                        model=detail_model_name,
                        max_tokens=1000,
                        temperature=0,
                        system=system,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Image 1:"
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": image1,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": "Image 2:"
                                    },
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": image2,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ],
                            }
                        ],
                    )
                    if message and message.content and isinstance(message.content, list):
                        return str(message.content[0].text)
                    else:
                        print("Unexpected response format:", message)
                        return None
                except anthropic.RateLimitError:
                    wait_time = 10
                    print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                except anthropic.APIError as e:
                    if "Overloaded" in str(e) or "overloaded_error" in str(e):
                        wait_time = min(60, 2 ** retries)
                        print(f"Server overload detected. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        print(f"Unexpected API error: {e}")
                        break


            print("Exceeded maximum retries.")
            return None

        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]
            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "a") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):

                    image1 = encode_image(image_ego[0])
                    image2 = encode_image(image_exo[0])

                    response = response_exception(qs[0], image1, image2)

                    new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": response,
                            "label": ans[0],
                            "model_name": model_name,
                        }
                    # print(f"***answer: {response}***lable: {ans}***")
                    new_dict = update_dict(new_dict,line)

                    ans_file.write(json.dumps(new_dict) + "\n")

    elif "gemini" in model_name.lower():
        detail_model_name = "gemini-2.0-flash"
        client = genai.Client(api_key="API Key")
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
                    question = qs[0]

                    question_prompt = question

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

    elif "qwen2-vl" in  model_name.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).eval()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": system},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_ego[0]},
                                {"type": "image", "image": image_exo[0]},
                                {"type": "text", "text": qs[0]},
                            ],
                        }
                    ]

                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")

                    generated_ids = model.generate(**inputs, max_new_tokens=1024,
                                                   do_sample=False)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }

                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")


    elif "deepseek-vl-7b-chat" in model_name.lower():

        vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/deepseek-vl-7b-chat")
        tokenizer = vl_chat_processor.tokenizer

        vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-vl-7b-chat", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda")
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        for i, question_file in enumerate(question_files):
            if args.category!= "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    conversation = [
                        {
                            "role": "User",
                            "content": "<image_placeholder>, <image_placeholder> \n{}".format(qs[0]),
                            "images": [image_ego[0], image_exo[0]]
                        },
                        {
                            "role": "Assistant",
                            "content": ""
                        }
                    ]

                    pil_images = load_pil_images(conversation)
                    prompt = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=conversation,
                                                                                         system_prompt=system)
                    prepare_inputs = vl_chat_processor(
                        prompt=prompt,
                        images=pil_images,
                        force_batchify=True
                    ).to(vl_gpt.device)

                    # run image encoder to get the image embeddings
                    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                    # run the model to get the response
                    outputs = vl_gpt.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=prepare_inputs.attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        max_new_tokens=1024,
                        use_cache=True,
                        do_sample=False,
                    )
                    response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }

                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")

    elif "qwen2.5-vl" in model_name.lower():
        base64str: TypeAlias = str
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            device_map="auto").eval()
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        for i, question_file in enumerate(question_files):
            if args.category != "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]

            data_loader = create_data_loader_default(questions, image_dir)

            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    messages = [
                        {"role": "system",
                         "content": system},
                        {"role": "user",
                         "content": [{"type": "image", "image": image_ego[0]},
                                     {"type": "image", "image": image_exo[0]},
                                     {"type": "text", "text": qs[0]}]
                         }
                    ]

                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to(model.device)
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)


                    new_dict = {
                        "question_prompt": qs[0],
                        "model_answer": response,
                        "label": ans[0],
                        "model_name": model_name,
                    }

                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")

    elif "llava-next-interleave" in model_name.lower():
        warnings.filterwarnings("ignore")
        model_path = "lmms-lab/llava-next-interleave-qwen-7b"
        llava_model_name = "llava_qwen"
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, llava_model_name,
                                                                               device_map="auto")
        for i, question_file in enumerate(question_files):
            if args.category != "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]
            data_loader = create_data_loader_default(questions, image_dir)
            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    image_tensors = []
                    image_ego_pil = Image.open(image_ego[0])
                    image_tensor_ego = image_processor.preprocess(image_ego_pil, return_tensors='pt')['pixel_values']
                    image_tensors.append(image_tensor_ego.half().cuda())
                    image_exo_pil = Image.open(image_exo[0])
                    image_tensor_exo = image_processor.preprocess(image_exo_pil, return_tensors='pt')['pixel_values']
                    image_tensors.append(image_tensor_exo.half().cuda())
                    question = DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_IMAGE_TOKEN + '\n' + qs[0]
                    prompt = [{"from": "human", "value": question}, {'from': 'gpt', 'value': None}]
                    input_ids = preprocess_qwen(prompt, tokenizer, has_image=True, system_message=system).cuda()
                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids, images=image_tensors, temperature=0.1, top_k=50,
                            top_p=0.95, do_sample=True, num_beams=1, max_new_tokens=1024, use_cache=True
                        )
                    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    new_dict = {"question_prompt": qs[0], "model_answer": response, "label": ans[0],
                                "model_name": model_name}
                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")

    elif "llava-onevision" in model_name.lower():
        warnings.filterwarnings("ignore")
        model_path = "lmms-lab/llava-onevision-qwen2-7b-ov"
        llava_model_name = "llava_qwen"
        tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, None, llava_model_name,
                                                                              device_map="auto")
        model.eval()
        for i, question_file in enumerate(question_files):
            if args.category != "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
                continue
            question_file_path = os.path.join(question_dir, question_file)
            answers_file = os.path.join(output_dir, question_file)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            questions = [json.loads(q) for q in open(question_file_path, "r")]
            data_loader = create_data_loader_default(questions, image_dir)
            print(f"***Processing file: {question_file}***Model name: {args.model}***")
            with open(answers_file, "w") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()
                    image_ego_pil = Image.open(image_ego[0])
                    image_exo_pil = Image.open(image_exo[0])
                    image_tensor = process_images([image_ego_pil, image_exo_pil], image_processor, model.config)
                    image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
                    conv_template = "qwen_1_5"
                    question = DEFAULT_IMAGE_TOKEN + "\n" + DEFAULT_IMAGE_TOKEN + "\n" + qs[0]
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX,
                                                      return_tensors="pt").unsqueeze(0).to("cuda")
                    image_sizes = [image_ego_pil.size, image_exo_pil.size]
                    cont = model.generate(
                        input_ids, images=image_tensor, image_sizes=image_sizes, max_new_tokens=1024,
                        temperature=0.1, top_k=50, top_p=0.95, do_sample=True,
                    )
                    response = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                    new_dict = {"question_prompt": qs[0], "model_answer": response, "label": ans[0],
                                "model_name": model_name}
                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")

    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= "Model_name")
    parser.add_argument("--image-dir", type=str, default="/path/to/E3VQA/paired_images")
    parser.add_argument("--output-dir", type=str, default="/path/to/E3VQA/E3VQA_result")
    parser.add_argument("--question-dir", type=str, default="/path/to/E3VQA/annotations")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--max-new-token", type=int, default="2048")
    parser.add_argument("--category", type=str, default='total')
    parser.add_argument("--perspective", type=str, default='ego')
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    eval_model(args)