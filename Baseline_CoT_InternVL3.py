import argparse
import torch
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import set_seed
import base64
from PIL import Image
import random
import time

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
    return base64.b64encode(image_file.read()).decode('utf-8')


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]

        random.shuffle(line["options"])

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
        index = line["options"].index(line["answer"].lower())
        ans = f"{chr(65 + index)}) {line['answer']}"

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

    if "intern" in model_name.lower():
        model_name = args.model
        image_dir = args.image_dir
        question_dir = args.question_dir
        output_dir = os.path.join(args.output_dir, model_name)

        os.makedirs(output_dir, exist_ok=True)
        question_files = [f for f in os.listdir(question_dir) if f.endswith(".jsonl")]

        torch.cuda.empty_cache()
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        set_seed(seed)
        print(f"**************Using seed: {seed}*****************")

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

        def load_image(image_file, input_size=448, max_num=12):
            image = Image.open(image_file).convert('RGB')
            transform = build_transform(input_size=input_size)
            images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            return pixel_values

        path = 'OpenGVLab/InternVL3-14B'


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
            with open(answers_file, "a") as ans_file:
                for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                    torch.cuda.empty_cache()

                    pixel_values1 = load_image(image_ego[0], max_num=12).to(torch.bfloat16).cuda()
                    pixel_values2 = load_image(image_exo[0], max_num=12).to(torch.bfloat16).cuda()
                    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                    generation_config = dict(max_new_tokens=1024, do_sample=True)

                    question = qs[0]

                    if args.cot_type == 'ccot':
                        question_prompt = question + ccot_prompt
                        response_1, history = model.chat(tokenizer, pixel_values, question_prompt, generation_config,
                                                         history=None, return_history=True)
                        question_prompt_2 = f"Scene Graph:\n{response_1}\n" + ccot_prompt_2 + "\n" + question
                        response_2, history = model.chat(tokenizer, pixel_values, question_prompt_2, generation_config,
                                                         history=None, return_history=True)
                    elif args.cot_type == 'cocot':
                        question_prompt_2 = question + cocot_prompt
                        response_2, history = model.chat(tokenizer, pixel_values, question_prompt_2, generation_config,
                                                         history=None, return_history=True)
                    elif args.cot_type == 'ddcot':
                        question_prompt = question + ddcot_prompt
                        response_1, history = model.chat(tokenizer, pixel_values, question_prompt, generation_config,
                                                         history=None, return_history=True)
                        question_prompt_2 = f"Context:\n{response_1}\n" + ddcot_prompt_2 + "\n" + question
                        response_2, history = model.chat(tokenizer, pixel_values, question_prompt_2, generation_config,
                                                         history=None, return_history=True)
                    else:
                        raise ValueError(f"Invalid cot_type: {args.cot_type}. Must be 'ccot', 'cocot', or 'ddcot'.")


                    new_dict = {
                        "question_prompt": question_prompt_2,
                        "model_answer": response_2,
                        "label": ans[0],
                        "model_name": model_name,
                    }
                    new_dict = update_dict(new_dict, line)
                    ans_file.write(json.dumps(new_dict) + "\n")
                    ans_file.flush()

    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= "Model")
    parser.add_argument("--image-dir", type=str, default="/path/to/E3VQA/paired_images/")
    parser.add_argument("--output-dir", type=str, default="/path/to/output_dir/")
    parser.add_argument("--question-dir", type=str, default="/path/to/E3VQA/E3VQA_benchmark/")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--max-new-token", type=int, default="2048")
    parser.add_argument("--category", type=str, default='total')
    parser.add_argument("--perspective", type=str, default='ego')
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--cot_type", type=str, default='ccot', choices=['ccot', 'cocot', 'ddcot'],
                        help="Type of Chain-of-Thought prompting to use (ccot, cocot, ddcot).")
    args = parser.parse_args()
    eval_model(args)