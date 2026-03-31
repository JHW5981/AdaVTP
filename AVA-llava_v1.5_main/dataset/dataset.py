import os
import json
from torch.utils.data import Dataset
from PIL import Image
import copy
import torch
import transformers
from dataclasses import dataclass
import dataset.conversation as conversation_lib
from dataset.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, data_args):
    process_image_type = getattr(data_args, "process_image_type", 'square')
    new_images = []
    new_image_sizes = []
    new_patch_sizes = []
    if process_image_type == 'square':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image.unsqueeze(0))
            new_image_sizes.append(image.shape[-2:])
            new_patch_sizes.append(1)
    else:
        raise ValueError(f"process_image_type {process_image_type} is not supported")
    return new_images, new_image_sizes, new_patch_sizes


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                # print(
                #     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                #     f" (ignored)"
                # )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_llama_2(
    sources,
    tokenizer,
    has_image,
    has_video
):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image and not has_video:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    elif has_video and not has_image:
        raise NotImplementedError("视频数据暂不支持")
    elif has_video and has_image:
        raise NotImplementedError("视频和图片数据同时存在暂不支持")
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image and not has_video:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            elif has_video and not has_image:
                raise NotImplementedError("视频数据暂不支持")
            elif has_video and has_image:
                raise NotImplementedError("视频和图片数据同时存在暂不支持")
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings,
                 tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human": # 间隔的special token ### 不学
            target[cur_idx:cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == "gpt": # ### ASSISTANT:不学
            target[cur_idx: cur_idx+7] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess(
    sources,
    tokenizer,
    has_image,
    has_video
):
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    else:
        # add end signal and concatenate together
        conversations = []
        for source in sources:
            header = f"{conversation_lib.default_conversation.system}\n\n"
            conversation = _add_speaker_and_signal(header, source)
            conversations.append(conversation)
        # tokenize conversations
        def get_tokenize_len(prompts):
            return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

        if has_image:
            input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
        else:
            conversations_tokenized = _tokenize_fn(conversations, tokenizer)
            input_ids = conversations_tokenized["input_ids"]

        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            if has_image:
                tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
            else:
                tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            _mask_targets(target, tokenized_lens, speakers)

        return dict(input_ids=input_ids, labels=targets)

def build_dataset(is_train, args, tokenizer, image_processor):
    if is_train:
        dataset = MyDataset(load_dataset(args.trdata_path), tokenizer, image_processor, args)
        dataset_names = []
    else:
        dataset = []
        dataset_dicts = load_dataset(args.valdata_path)

        dataset_names = list(dataset_dicts.keys())
        for dataset_name in dataset_names:
            dataset.append(MyDataset(dataset_dicts[dataset_name], tokenizer, image_processor, args))
    return dataset, dataset_names

class MyDataset(Dataset):
    def __init__(self, data, tokenizer, image_processor, data_args):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.data_args = data_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 推理数据
        if "question_id" in self.data[index]:
            sources = self.data[index]
            if 'image' in sources:
                idx = sources['question_id']
                image_files = [os.path.join(self.data_args.image_folder, sources['image'])]
                image_list = [Image.open(image_file).convert('RGB') for image_file in image_files]
                images, image_sizes, patch_sizes = process_images(image_list, self.image_processor, self.data_args)
                prompt = sources["text"]
                if "<image>" not in prompt:
                    sources = [[
                    {
                        "from": "human",
                        "value": f"<image>\n {prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                ]]
                else:
                    sources = [[{
                        "from": "human",
                        "value": f"{prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": ""
                    }
                    ]]
            else:
                idx = sources['question_id']
                prompt = sources["text"]
                sources = [[{
                    "from": "human",
                    "value": f"{prompt}"
                },
                {
                    "from": "gpt",
                    "value": ""
                }
                ]]
                images = None
                image_sizes = None
                patch_sizes = None
            
            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=('image' in self.data[index]),
                has_video=False)

            # image exist in the data
            if 'image' in self.data[index]:
                data_dict['images'] = torch.cat(images, dim=0)
                data_dict['image_sizes'] = image_sizes
                data_dict['patch_sizes'] = patch_sizes

            data_dict['question_id'] = idx
            data_dict['prompt'] = prompt
            return data_dict  
        else:
            # 训练数据
            sources = self.data[index]
            # 单纯只有图片的情况
            if 'images' in sources and 'video' not in sources:
                image_files = sources['images']
                assert isinstance(image_files, list), "image files必须是以列表方式存储"
                image_list = [Image.open(image_file).convert('RGB') for image_file in image_files]
                images, image_sizes, patch_sizes = process_images(image_list, self.image_processor, self.data_args)
                sources = copy.deepcopy([sources["conversations"]])
            # 单纯只有视频的情况
            elif 'video' in sources and 'image' not in sources:
                raise NotImplementedError("视频数据暂不支持")
            elif 'video' in sources and 'image' in sources:
                raise NotImplementedError("视频和图片数据同时存在暂不支持")
            else:
                sources = copy.deepcopy([sources["conversations"]])
                images = None
                image_sizes = None
                patch_sizes = None
                
            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=('images' in self.data[index]),
                has_video=('video' in self.data[index]))

            # image exist in the data
            if 'images' in self.data[index]:
                data_dict['images'] = torch.cat(images, dim=0)
                data_dict['image_sizes'] = image_sizes
                data_dict['patch_sizes'] = patch_sizes

            # 如果是Eval数据，会有answer
            if 'answer' in self.data[index]:
                data_dict['answer'] = self.data[index]['answer']
            return data_dict 

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        if 'question_id' in instances[0]:
            input_ids, labels = tuple([instance[key][0] for instance in instances]
                                    for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
            if 'images' in instances[0]:
                images = [instance['images'] for instance in instances]
                image_sizes = [instance['image_sizes'] for instance in instances]
                patch_sizes = [instance['patch_sizes'] for instance in instances]
                batch['images'] = images
                batch['image_sizes'] = image_sizes
                batch['patch_sizes'] = patch_sizes
            question_id = [instance['question_id'] for instance in instances]
            batch['question_id'] = question_id
            prompt = [instance['prompt'] for instance in instances]
            batch['prompt'] = prompt
            return batch
        else:
            input_ids, labels = tuple([instance[key][0] for instance in instances]
                                    for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            input_ids = input_ids[:, :self.tokenizer.model_max_length]
            labels = labels[:, :self.tokenizer.model_max_length]
            batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )

            if 'images' in instances[0]:
                images = [instance['images'] for instance in instances]
                image_sizes = [instance['image_sizes'] for instance in instances]
                patch_sizes = [instance['patch_sizes'] for instance in instances]
                batch['images'] = images
                batch['image_sizes'] = image_sizes
                batch['patch_sizes'] = patch_sizes
            
            if 'answer' in instances[0]:
                answers = [instance['answer'] for instance in instances]
                batch['answers'] = answers

            return batch


if __name__ == '__main__':
    from main import get_args_parser
    args = get_args_parser().parse_args()
    args.trdata_path = '/home/jihuawei/storage/jihuawei2/projects/llava-nan/dataset/dataset.json'
    dataset = build_dataset(is_train=True, args=args)
    print(dataset[0])