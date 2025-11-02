import argparse
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import base64
from collections import Counter
from google import genai
from google.genai import types
import PIL.Image
import time
import io
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=3)




def preprocess_user_message(user_message):
    processed = []
    for item in user_message:
        if isinstance(item, str) and item.lower().endswith(('.jpg', '.jpeg', '.png')):
            with open(item, 'rb') as f:
                image_data = f.read()
                image = PIL.Image.open(io.BytesIO(image_data)).convert("RGB")
                processed.append(image)
        else:
            processed.append(item)
    return processed


def blocking_response(agent, user_message):
    max_retries = 20
    retry_delay = 10
    processed_message = preprocess_user_message(user_message)

    for attempt in range(max_retries):
        try:
            response = agent.send_message(processed_message)
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count,
            }
            return response.text, usage

        except Exception as e:
            print(f"Error - retries {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                retry_delay *= 2
                time.sleep(retry_delay)
                print(f"Delayed for {retry_delay} seconds")
            else:
                return "E) Error", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}



def update_usage(usage_dict, usage, client_key):
    usage_dict[client_key]["prompt_tokens"] += usage["prompt_tokens"]
    usage_dict[client_key]["completion_tokens"] += usage["completion_tokens"]
    usage_dict[client_key]["total_tokens"] += usage["total_tokens"]

    usage_dict["total"]["prompt_tokens"] += usage["prompt_tokens"]
    usage_dict["total"]["completion_tokens"] += usage["completion_tokens"]
    usage_dict["total"]["total_tokens"] += usage["total_tokens"]


async def async_response(agent, user_message):
    return await asyncio.get_event_loop().run_in_executor(
        executor, blocking_response, agent, user_message
    )

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
    "Examine the objects and relationships in the initial scene graph. Update their attributes or positions based on observations from both views. "
    "Remove only elements that are clearly erroneous (e.g., annotation errors or duplicates)\n\n"
    "2. Incorporate New Information:\n"
    "Identify and add any new objects or relationships that appear in the new view.\n\n"
    "3. Align and Reconcile Across Views:\n"
    "For overlapping objects and relationships, align them using spatial proximity and semantic similarity. "
    "If attribute discrepancies arise, select values that best reflect the combined observations.\n\n"
    "Ensure that the updated scene graph is logically and physically consistent, avoiding contradictions or impossible configurations.\n"
    "Just generate the refined scene graph in JSON format. Do not say extra words.\n\n"
)

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
    "Examine the objects and relationships in the initial scene graph. Update their attributes or positions based on observations from both views. "
    "Remove only elements that are clearly erroneous (e.g., annotation errors or duplicates)\n\n"
    "2. Incorporate New Information:\n"
    "Identify and add any new objects or relationships that appear in the new view.\n\n"
    "3. Align and Reconcile Across Views:\n"
    "For overlapping objects and relationships, align them using spatial proximity and semantic similarity. "
    "If attribute discrepancies arise, select values that best reflect the combined observations.\n\n"
    "Ensure that the updated scene graph is logically and physically consistent, avoiding contradictions or impossible configurations.\n"
    "Just generate the refined scene graph in JSON format. Do not say extra words.\n\n"
)

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


async def eval_model(args):

    model_name = args.model
    image_dir = args.image_dir
    question_dir = args.question_dir
    output_dir = os.path.join(args.output_dir, model_name)

    os.makedirs(output_dir, exist_ok=True)

    if "gemini-cot-debate" in model_name.lower():
        cum_start_time = time.time()
        detail_model_name = "gemini-2.0-flash"
        api_key = "API_KEY"
        client = genai.Client(api_key=api_key)
        question_files = [f for f in os.listdir(question_dir) if f.endswith(".jsonl")]
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
                    start_time = time.time()

                    # image1 = PIL.Image.open(image_ego[0])
                    # image2 = PIL.Image.open(image_exo[0])

                    max_rounds = 1
                    for round_number in range(max_rounds):

                        both1 = client.chats.create(
                            model=detail_model_name,
                            config=types.GenerateContentConfig(
                                max_output_tokens=2048,
                                temperature=0.1,
                                system_instruction=system["both"],
                                seed=2000,
                            )
                        )
                        both2 = client.chats.create(
                            model=detail_model_name,
                            config=types.GenerateContentConfig(
                                max_output_tokens=2048,
                                temperature=0.1,
                                system_instruction=system["both"],
                                seed=2000,

                            )
                        )
                        both3 = client.chats.create(
                            model=detail_model_name,
                            config=types.GenerateContentConfig(
                                max_output_tokens=2048,
                                temperature=0.1,
                                system_instruction=system["both"],
                                seed=2000,
                            )
                        )
                        usage_dict = {
                            "both1": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "both2": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "both3": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                            "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        }

                        if round_number == 0:
                            question = qs[0]
                            # images = [image1, image2]

                            ######### initial SG #########
                            # Async batch call
                            responses = await asyncio.gather(
                                async_response(both1, [image_ego[0], question + space_reasoning_forward]),
                                async_response(both2, [image_exo[0], question + space_reasoning_reverse]),
                                async_response(both3, [image_ego[0], image_exo[0], question + space_reasoning_unified])
                            )

                            (response_1_both1, usage1), (response_1_both2, usage2), (response_1_both3, usage3) = responses

                            update_usage(usage_dict, usage1, "both1")
                            update_usage(usage_dict, usage2, "both2")
                            update_usage(usage_dict, usage3, "both3")
                            ######### initial SG #########

                            ######### initial SG refinement for both1 and both2 #########
                            responses = await asyncio.gather(
                                async_response(both1, [image_exo[0], question + space_reasoning_forward_2]),
                                async_response(both2, [image_ego[0], question + space_reasoning_reverse_2]),
                            )

                            (response_2_both1, usage1), (response_2_both2, usage2) = responses

                            update_usage(usage_dict, usage1, "both1")
                            update_usage(usage_dict, usage2, "both2")

                            ######### initial SG refinement for both1 and both2 #########

                            ######### Initial answer response #########
                            answer_prompt = "Use the images and the unified scene graph as context and answer the following question:\n"
                            user_message_a = [image_ego[0], image_exo[0], answer_prompt, question]
                            responses = await asyncio.gather(
                                async_response(both1, user_message_a),
                                async_response(both2, user_message_a),
                                async_response(both3, user_message_a)
                            )

                            (response_a_both1, usage1), (response_a_both2, usage2), (response_a_both3, usage3) = responses

                            update_usage(usage_dict, usage1, "both1")
                            update_usage(usage_dict, usage2, "both2")
                            update_usage(usage_dict, usage3, "both3")

                            ######### Initial answer response #########

                            answers = []
                            for choice in ['A)', 'B)', 'C)', 'D)']:
                                if choice in response_a_both1:
                                    answers.append(choice[0])
                                if choice in response_a_both2:
                                    answers.append(choice[0])
                                if choice in response_a_both3:
                                    answers.append(choice[0])

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
                                    "usage_both1": usage_dict["both1"],
                                    "usage_both2": usage_dict["both2"],
                                    "usage_both3": usage_dict["both3"],
                                    "usage_total": usage_dict["total"],
                                }
                                total_tokens = usage_dict["total"]
                                print(
                                    f"****************** {j}-th Final answer: {new_dict['model_answer']} ********************\n"
                                f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")

                                new_dict = update_dict(new_dict, line)
                                ans_file.write(json.dumps(new_dict) + "\n")
                                break



                        if round_number == 0:
                            other_agent_message_both1 = f"One scene graph: {response_2_both2}\n\nOne scene graph: {response_1_both3}\n"
                            other_agent_message_both2 = f"One scene graph: {response_2_both1}\n\nOne scene graph: {response_1_both3}\n"
                            other_agent_message_both3 = f"One scene graph: {response_2_both2}\n\nOne scene graph: {response_2_both1}\n"

                        else:
                            other_agent_message_both1 = f"One scene graph: {response_fin_both2}\n\nOne scene graph: {response_fin_both3}\n"
                            other_agent_message_both2 = f"One scene graph: {response_fin_both1}\n\nOne scene graph: {response_fin_both3}\n"
                            other_agent_message_both3 = f"One scene graph: {response_fin_both2}\n\nOne scene graph: {response_fin_both1}\n"

                        user_message_both1 = [
                            image_ego[0], image_exo[0],  question,

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

                        user_message_both2 = [
                            image_ego[0], image_exo[0], question,
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

                        user_message_both3 = [
                            image_ego[0], image_exo[0], question,
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

                        ######## Unified scenegraph ########
                        responses = await asyncio.gather(
                            async_response(both1, user_message_both1),
                            async_response(both2, user_message_both2),
                            async_response(both3, user_message_both3)
                        )

                        (response_fin_both1, usage1), (response_fin_both2, usage2), (response_fin_both3, usage3) = responses

                        update_usage(usage_dict, usage1, "both1")
                        update_usage(usage_dict, usage2, "both2")
                        update_usage(usage_dict, usage3, "both3")

                        ######## Unified scenegraph ########

                        additional_prompt_both1 = (
                            f"Use the images and the unified scene graph as context and answer the following question.\n"
                        )

                        additional_prompt_both2 = (
                            f"Use the images and the unified scene graph as context and answer the following question.\n"
                        )

                        additional_prompt_both3 = (
                            f"Use the images and the unified scene graph as context and answer the following question.\n"
                        )



                        user_message_final_1 = [image_ego[0], image_exo[0], additional_prompt_both1 , question]
                        user_message_final_2 = [image_ego[0], image_exo[0], additional_prompt_both2 ,question]
                        user_message_final_3 = [image_ego[0], image_exo[0], additional_prompt_both3 ,question]

                        ######## final answer ########
                        responses = await asyncio.gather(
                            async_response(both1, user_message_final_1),
                            async_response(both2, user_message_final_2),
                            async_response(both3, user_message_final_3)
                        )

                        (response_3_both1, usage1), (response_3_both2, usage2), (response_3_both3, usage3) = responses

                        update_usage(usage_dict, usage1, "both1")
                        update_usage(usage_dict, usage2, "both2")
                        update_usage(usage_dict, usage3, "both3")

                        ######## final answer ########


                        answers = []
                        for choice in ['A)', 'B)', 'C)', 'D)']:
                            if choice in response_3_both1:
                                answers.append(choice[0])
                            if choice in response_3_both2:
                                answers.append(choice[0])
                            if choice in response_3_both3:
                                answers.append(choice[0])


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
                                "usage_both1": usage_dict["both1"],
                                "usage_both2": usage_dict["both2"],
                                "usage_both3": usage_dict["both3"],
                                "usage_total": usage_dict["total"],
                            }
                            total_tokens = usage_dict["total"]

                            print(f"****************** {j}-th Final answer: {new_dict['model_answer']} ********************\n"
                                  f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")
                            new_dict = update_dict(new_dict, line)
                            ans_file.write(json.dumps(new_dict) + "\n")
                            break

                        elif len(answer_counter) == 2 and any(count >= 2 for count in answer_counter.values()):
                            for answer, count in answer_counter.items():
                                if count >= 2:
                                    model_answer = str(answer+')')
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
                                "usage_both1": usage_dict["both1"],
                                "usage_both2": usage_dict["both2"],
                                "usage_both3": usage_dict["both3"],
                                "usage_total": usage_dict["total"],
                            }
                            total_tokens = usage_dict["total"]

                            print(f"****************** {j}-th Final answer: {new_dict['model_answer']} ********************\n"
                                  f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")
                            new_dict = update_dict(new_dict, line)
                            ans_file.write(json.dumps(new_dict) + "\n")
                            break


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
                                "tt" : True,
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
                                "usage_both1": usage_dict["both1"],
                                "usage_both2": usage_dict["both2"],
                                "usage_both3": usage_dict["both3"],
                                "usage_total": usage_dict["total"],
                            }
                            total_tokens = usage_dict["total"]



                            print(f"****************** {j}-th Final answer: {new_dict['model_answer']} ********************\n"
                                  f"****************** Time taken: {elapsed_time:.2f} seconds****************** ")
                            new_dict = update_dict(new_dict, line)
                            ans_file.write(json.dumps(new_dict) + "\n")
                cum_end_time = time.time()
                elapsed_time = cum_end_time - cum_start_time
                print(f"elapsed time for {question_file}: {elapsed_time:.2f} seconds")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= "Model")
    parser.add_argument("--image-dir", type=str, default="/path/to/E3VQA/paired_images/")
    parser.add_argument("--output-dir", type=str, default="/path/to/output_dir/")
    parser.add_argument("--question-dir", type=str, default="/path/to/E3VQA/E3VQA_benchmark/")
    parser.add_argument("--question-file", type=str, default="")
    parser.add_argument("--category", type=str, default='total')
    parser.add_argument("--perspective", type=str, default='')
    parser.add_argument("--question-file-list", type=str, nargs='+', default=[])
    args = parser.parse_args()
    asyncio.run(eval_model(args))
