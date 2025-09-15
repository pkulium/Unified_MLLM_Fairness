import logging
import os
import torch
import transformers

from torch.utils.data import Dataset
from transformers import HfArgumentParser, AutoConfig
from transformers import set_seed
from typing import Dict, Tuple, cast

from vila_u import conversation as conversation_lib
from vila_u.data import make_supervised_data_module
from vila_u.model import VILAULlamaModel, VILAULlamaConfig
from vila_u.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from vila_u.train.vila_u_trainer import VILAUTrainer, VILAUBPOLOSS
from vila_u.train.args import TrainingArguments, ModelArguments, DataArguments
from vila_u.train.callbacks.autoresume_callback import AutoResumeCallback
from vila_u.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    mprint,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

local_rank = None

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "VILA-U"

import re
import torch.nn as nn

def get_lora_target_modules(model, pattern: str = ".*llm.*"):
    """
    Automatically identify target modules for LoRA in the LLM component.

    Args:
        model: The pre-trained model.
        pattern: Regex pattern to match module names. Defaults to matching LLM layers.

    Returns:
        A list of module names suitable for LoRA in the LLM component.
    """
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and re.match(pattern, name, re.IGNORECASE) and 'vision_tower' not in name:
            if any(submodule in name for submodule in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                target_modules.append(name)
    return target_modules

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = cast(Tuple[ModelArguments, DataArguments, TrainingArguments], parser.parse_args_into_dataclasses())
    training_args.run_name = training_args.output_dir.split("/")[-1]
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    set_seed(training_args.seed)

    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
    else:
        resume_from_checkpoint = False
        model_cls = VILAULlamaModel
        config = VILAULlamaConfig.from_pretrained(
            model_args.model_name_or_path,
            resume=resume_from_checkpoint
        )
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path
    
    prepare_config_for_training(config, model_args, training_args, data_args)
    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    mprint(model)

    model.llm.config.use_cache = False
    model.get_llm().requires_grad_(training_args.tune_language_model)
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")

    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        if isinstance(model.get_vision_tower(), RQVAESIGLIPTransformerVisionTower):
            model.get_vision_tower().vision_tower.rqvaesiglip.eval()
            model.get_vision_tower().vision_tower.rqtransformer.requires_grad_(True)
        else:
            raise NotImplementedError()
        print(f"vision tower {training_args.tune_vision_tower}")
        print(f"mm projector {training_args.tune_mm_projector}")

    if not any([training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]):
        logging.warning(
            "You are not tuning any part of the model. Please check if this is intended."
        )

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = model.tokenizer
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.num_video_frames = data_args.num_video_frames
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_use_vi_start_end = data_args.mm_use_vi_start_end = (
            model_args.mm_use_vi_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_vi_start_end = model_args.mm_use_vi_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    callbacks = [AutoResumeCallback()]
    if training_args.use_peft:

        if training_args.adapter_checkpoint_path:
            model = PeftModel.from_pretrained(
                model,
                training_args.adapter_checkpoint_path,
                is_trainable=True
            )
            
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # checkpoint = torch.load('/scratch/bcey/hchen10/pkulium/code/VLM_Bias/vila-u/checkpoints/vila-u-lora-generation-race-v6/checkpoint-6000/global_step6000/mp_rank_00_model_states.pt', map_location=device)

            # import pdb 
            # model.load_state_dict(checkpoint['module'], strict=True)
            # del checkpoint
            # torch.cuda.empty_cache()

        else:
            # Automatically identify target modules
            target_modules = get_lora_target_modules(model, pattern=".*attn.*|.*mlp.*")
            print(f"Identified target modules for LoRA: {target_modules}")

            # Define LoRA configuration
            lora_config = LoraConfig(
                # task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=32,  # Rank
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=target_modules,  # Set automatically identified modules
            )
            
            # Apply LoRA to the model
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Optional: To verify trainable parameters
       

    # if "bpo" in training_args.loss_type:
    #     trainer = VILAUBPOTrainer(
    #         model=model,
    #         tokenizer=tokenizer,
    #         args=training_args,
    #         callbacks=callbacks,
    #         **data_module,
    #     )
    # elif "sharpness" in training_args.loss_type:
    #     trainer = VILAUSharpnessTrainer(
    #         model=model,
    #         tokenizer=tokenizer,
    #         args=training_args,
    #         callbacks=callbacks,
    #         **data_module,
    #     )
    # else:

    base_model = model.base_model.model

    base_model.beta = 0
    base_model.alpha = 1
    if training_args.beta:
        base_model.beta = training_args.beta
        base_model.alpha = training_args.alpha
        base_model.loss_type = training_args.loss_type
        base_model.threshold = training_args.threshold
        base_model.bpo_loss_func = VILAUBPOLOSS(training_args.loss_type, training_args.beta, training_args.threshold)
        
    trainer = VILAUTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
        flush=True,
    )
    print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        flush=True,
    )

    print('=' * 32 + "old weight" + '=' * 32)
    print(model.model.llm.model.layers[0].mlp.down_proj.lora_A.default.weight)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir

    print('=' * 32 + "new weight" + '=' * 32)
    print(model.model.llm.model.layers[0].mlp.down_proj.lora_A.default.weight)
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir
    )

if __name__ == "__main__":
    train()