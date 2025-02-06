import argparse
import os
from functools import partial

from transformers import AutoTokenizer
from datasets import load_dataset
from litdata import optimize

# モデル識別子（Hugging Face Hub のモデル名）
MODELS = {
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "stablelm": "stabilityai/stablelm-zephyr-3b",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",  # 例。実際の名称に合わせてください。
}

# 最大トークン長
MAX_LENGTH = 4096
MAX_OUTPUT_LENGTH = 512
DATESET = "tokyotech-llm/swallow-magpie-ultra-v0.1"

def tokenize(example, tokenizer):
    """
    データ例からチャット形式のテキストを作成し、トークン化する関数。
    もともと "messages" あるいは "chosen" フィールドを想定していましたが、
    新しいデータセットの場合は "instruction" と "response" にも対応します。
    """
    global generation_prompt  # メインで定義する生成プロンプトを参照

    if "messages" in example:
        # すでにチャット形式の場合
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
    elif "instruction" in example and "response" in example:
        # instruction と response からチャットテキストを作成
        # ※　"User:" などの文字列はお好みで変更してください
        text = f"User: {example['instruction']}\n{generation_prompt}{example['response']}"
    elif "chosen" in example:
        # 旧形式のフォールバック
        text = tokenizer.apply_chat_template(
            example["chosen"],
            tokenize=False,
            add_generation_prompt=False
        )
    elif "input" in example and "output" in example:
        messages = example["input"].copy()
        messages.append(example["output"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        raise ValueError("Example does not contain expected keys (messages/instruction/response/chosen).")

    # 生成プロンプトを区切り文字として利用して入力部と出力部に分割
    messages = text.split(generation_prompt)
    input_text = generation_prompt.join(messages[:-1]) + generation_prompt
    output_text = messages[-1]

    input_ids = tokenizer(text, return_tensors="pt").input_ids
    res = {"model_inputs": {"input_ids": input_ids, "labels": input_ids.clone()}}

    gen_input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    res["model_inputs_gen"] = {"input_ids": gen_input_ids}
    res["response"] = output_text
    return res


def filter_length(example, max_input_len, max_output_len):
    """
    トークン数が上限内に収まっているかフィルタする関数。
    """
    max_length = max_input_len + max_output_len
    if example["model_inputs"]["input_ids"].size(1) > max_length:
        return False
    if example["model_inputs_gen"]["input_ids"].size(1) > max_input_len:
        return False
    # 出力部分のトークン数もチェック
    output_tokens = tokenizer(example["response"], return_tensors="pt").input_ids
    if output_tokens.size(1) > max_output_len:
        return False
    return True


def fn(index, data):
    yield data[index]


def prepare_train(args, tokenizer):
    # 新しいデータセット "tokyotech-llm/lmsys-chat-1m-synth" の train split を読み込む
    dataset = load_dataset(DATESET, split="train")
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        desc="Applying chat template",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(
        filter_length,
        fn_kwargs={
            "max_input_len": MAX_LENGTH - MAX_OUTPUT_LENGTH,
            "max_output_len": MAX_OUTPUT_LENGTH,
        },
        num_proc=args.num_proc,
    )
    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "train"),
        num_workers=16,
        chunk_bytes="500MB",
    )


def prepare_test(args, tokenizer):
    # 評価用データは、もし test split が無い場合は train split から分割する例
    dataset = load_dataset(DATESET, split="train")
    column_names = list(dataset.features)
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=args.num_proc,
        desc="Applying chat template",
        remove_columns=column_names,
    )
    dataset = dataset.with_format("torch")
    dataset = dataset.filter(
        filter_length,
        fn_kwargs={
            "max_input_len": MAX_LENGTH - MAX_OUTPUT_LENGTH,
            "max_output_len": MAX_OUTPUT_LENGTH,
        },
        num_proc=args.num_proc,
    )
    # 例として、テスト用に 2000 件を抽出
    ds = dataset.train_test_split(test_size=2000, seed=42, shuffle=True)
    dataset = ds["test"]

    os.makedirs(args.output_dir, exist_ok=True)

    optimize(
        fn=partial(fn, data=dataset),
        inputs=list(range(len(dataset))),
        output_dir=os.path.join(args.output_dir, args.model_type, "test"),
        num_workers=2,
        chunk_bytes="500MB",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=list(MODELS.keys()),
        default="phi-3",
        help="Teacher type",
    )
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--num_proc", type=int, default=64, help="number of workers for processing"
    )
    args = parser.parse_args()

    # 指定したモデルのトークナイザーを読み込む
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model_type])

    # 各モデル固有の設定例（必要に応じて調整してください）
    if args.model_type == "phi-3":
        # Phi‑3 では pad_token を unk_token に設定（無限生成防止のため）
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "right"
    elif args.model_type == "qwen2.5-32b":
        # Qwen‑2.5‑32B も、pad_token が未設定の場合の処理例
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = "right"

    # モデルごとに利用する生成プロンプトを設定（※実際の仕様に合わせて変更してください）
    if args.model_type in ["phi-3", "stablelm"]:
        generation_prompt = "<|assistant|>\n"
    elif args.model_type in ["llama-2"]:
        generation_prompt = " [/INST] "
    elif args.model_type in ["qwen2.5-32b"]:
        generation_prompt = "<|im_start|>assistant\\n"
    else:
        raise NotImplementedError(args.model_type)

    prepare_train(args, tokenizer)
    prepare_test(args, tokenizer)
