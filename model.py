"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(
    model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit',
    max_seq_length=256
):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""

    return sum(p.numel() for p in model.parameters())

# Step 3 - is_model_4bit_quantized
import bitsandbytes as bnb

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""

    return any(isinstance(module, bnb.nn.Linear4bit) for module in model.modules())

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""

    return ['q_proj', 'k_proj', 'v_proj', 'o_proj']

# Step 6 - attach_lora_adapters
from unsloth import FastLanguageModel


def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""

    if target_modules is None:
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
    )

    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # Return the fraction of parameters that are trainable.
    return float(trainable_count) / float(total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""

    return [
        {
            "instruction": "What is Python?",
            "response": "Python is a high-level programming language known for its simple and readable syntax."
        },
        {
            "instruction": "Explain what a variable is.",
            "response": "A variable is a name that refers to a value stored in a program."
        },
        {
            "instruction": "What does a loop do?",
            "response": "A loop repeatedly executes a block of code while a specified condition or sequence allows it."
        },
        {
            "instruction": "Give me a simple tip for debugging code.",
            "response": "Read the error message carefully and isolate the smallest piece of code that reproduces the problem."
        },
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""

    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['response']}"
    )

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""

    return [format_instruction_example(example) for example in examples]

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""

    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""

    return tokenizer(text, padding=False, truncation=False)["input_ids"]

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""

    return len(input_ids)

# Step 15 - build_training_arguments
import torch
from transformers import TrainingArguments

def build_training_arguments(
    output_dir='./sft_out',
    max_steps=5,
    learning_rate=2e-4
):
    """Return featherweight TrainingArguments for the SFT run."""

    use_bf16 = torch.cuda.is_bf16_supported()

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=1,
        optim="adamw_8bit",
    )

# Step 16 - build_sft_trainer (not yet solved)
# TODO: implement

# Step 17 - run_sft_training (not yet solved)
# TODO: implement

# Step 18 - switch_to_inference_mode (not yet solved)
# TODO: implement

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

