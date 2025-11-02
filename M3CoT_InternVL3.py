from collections import Counter
import argparse
import random
import math
import os
from typing import Dict, Optional, Sequence, List, TypeAlias
import json
from tqdm import tqdm
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
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




def eval_model(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    print(f"**************Using seed: {seed}*****************")
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

    model_name = args.model
    image_dir = args.image_dir
    question_dir = args.question_dir
    max_new_token = args.max_new_token
    output_dir = os.path.join(args.output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)
    question_files = [f for f in os.listdir(question_dir) if f.endswith(".jsonl")]

    path = 'OpenGVLab/InternVL3-14B'
    model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    def get_VLM_response(text_content: str, image_content: Optional[List[str]] = None,
                         history: Optional[List[str]] = None):  # image need to be list or none
        generation_config = dict(max_new_tokens=max_new_token, do_sample=False)
        with torch.no_grad():
            response, history = model.chat(tokenizer, image_content, text_content, generation_config,
                                           history=history, return_history=True)
        return [response, history]

    for i, question_file in enumerate(question_files):
        if args.category != "total" and question_file != f"{args.category}_{args.perspective}.jsonl":
            continue
        question_file_path = os.path.join(question_dir, question_file)
        answers_file = os.path.join(output_dir, question_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        questions = [json.loads(q) for q in open(question_file_path, "r")]
        data_loader = create_data_loader_default(questions, image_dir)

        print(f"***Processing file: {question_file}***Model name: {args.model}***")
        with open(answers_file, "a") as ans_file: #TODO revise it! it may cause more trouble
            for j, (qs, image_ego, image_exo, ans, line) in tqdm(enumerate(data_loader), total=len(data_loader)):
                both1 = system["both"]
                both2 = system["both"]
                both3 = system["both"]
                history = {"both1": '', "both2": '', "both3": ''}

                pixel_value_ego = load_image(image_ego[0], max_num=12).to(torch.bfloat16).cuda()
                pixel_value_exo = load_image(image_exo[0], max_num=12).to(torch.bfloat16).cuda()
                pixel_value_both = torch.cat((pixel_value_ego, pixel_value_exo), dim=0)

                max_rounds = 1
                for round_number in range(max_rounds):
                    question = qs[0]

                    ######### both1 #########
                    text = question + space_reasoning_forward
                    response_1_both1, history["both1"] = get_VLM_response(text_content=text,
                                                                          image_content=pixel_value_ego)
                    text = question+ space_reasoning_forward_2
                    response_2_both1, history["both1"] = get_VLM_response(text_content=text,
                                                                          image_content=pixel_value_exo, history=history["both1"])
                    ######### both1 #########


                    ######### both2 #########
                    text = question + space_reasoning_reverse
                    response_1_both2, history["both2"] = get_VLM_response(text_content=text,
                                                                          image_content=pixel_value_exo)

                    text = question + space_reasoning_reverse_2
                    response_2_both2, history["both2"] = get_VLM_response(text_content=text,
                                                                          image_content=pixel_value_ego, history=history["both2"])
                    ######### both2 #########


                    ######### both3 #########
                    text = question + space_reasoning_unified
                    response_1_both3, history["both3"] = get_VLM_response(text_content=text,
                                                                          image_content=pixel_value_both)
                    torch.cuda.empty_cache()
                    ######### both3 #########

                    #Initial answer
                    answer_prompt = "Use the images and the unified scene graph as context and answer the following question:\n"
                    text= answer_prompt + question
                    response_a_both1, history["both1"] = get_VLM_response(text_content=text,
                                                        image_content=pixel_value_both, history=history["both1"])
                    response_a_both2, history["both2"] = get_VLM_response(text_content=text,
                                                        image_content=pixel_value_both, history=history["both2"])
                    response_a_both3, history["both3"] = get_VLM_response(text_content=text,
                                                        image_content=pixel_value_both, history=history["both3"])

                    torch.cuda.empty_cache()
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
                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{response_a_both1}',
                            "label": ans[0],
                            "model_name": model_name,
                            "debate_prompt": f'initial step',
                        }
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

                    response_fin_both1, history["both1"] = get_VLM_response(text_content=user_message_both1,
                                                        image_content=pixel_value_both, history=history["both1"])
                    response_fin_both2, history["both2"] = get_VLM_response(text_content=user_message_both2,
                                                        image_content=pixel_value_both, history=history["both2"])
                    response_fin_both3, history["both3"] = get_VLM_response(text_content=user_message_both3,
                                                        image_content=pixel_value_both, history=history["both3"])
                    torch.cuda.empty_cache()
                    additional_prompt_both1 = (
                        f"{space_reasoning_unified_2}"
                    )

                    additional_prompt_both2 = (
                        f"{space_reasoning_unified_2}"
                    )

                    additional_prompt_both3 = (
                        f"{space_reasoning_unified_2}"
                    )

                    text= additional_prompt_both1 + question
                    response_3_both1, history["both1"] = get_VLM_response(text_content=text,
                                                        image_content=pixel_value_both, history=history["both1"])

                    text = additional_prompt_both2 + question
                    response_3_both2, history["both2"] = get_VLM_response(text_content=text,
                                                        image_content=pixel_value_both, history=history["both2"])

                    text = additional_prompt_both3 + question
                    response_3_both3, history["both3"] = get_VLM_response(text_content=text,
                                                        image_content=pixel_value_both, history=history["both3"])
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
                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{response_3_both1}',
                            "label": ans[0],
                            "model_name": model_name,
                            "debate_prompt": f'{other_agent_message_both1}',
                        }
                        print(
                            f"** {j}-th Final answer: 1Turn1answer{new_dict['model_answer']} Label: {ans}**\n")
                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")
                        break

                    elif len(answer_counter) == 2 and any(
                            count >= 2 for count in answer_counter.values()) and round_number == max_rounds - 1:

                        for answer, count in answer_counter.items():
                            if count >= 2:
                                model_answer = str(answer + ')')
                                break

                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{model_answer}',
                            "label": ans[0],
                            "model_name": model_name,
                            "debate_prompt": f'{other_agent_message_both1}',
                        }
                        print(
                            f"** {j}-th Final answer: 1Turn2answer{new_dict['model_answer']} Label: {ans}**\n")
                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")
                    elif round_number == max_rounds-1:
                        new_dict = {
                            "question_prompt": qs[0],
                            "model_answer": f'{response_3_both1}',
                            "label": ans[0],
                            "model_name": model_name,
                            "debate_prompt": f'{other_agent_message_both1}',
                            "tt" : True,
                        }
                        print(f"** {j}-th Final answer: 1Turn3answer{new_dict['model_answer']} Label: {ans}**\n")
                        new_dict = update_dict(new_dict, line)
                        ans_file.write(json.dumps(new_dict) + "\n")



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
    args = parser.parse_args()
    eval_model(args)