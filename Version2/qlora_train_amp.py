# coding=utf-8

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import time
import torch, gc
import peft
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from transformers import BitsAndBytesConfig
from lora_preprocess import get_dataloader
from lora_config import ProjectConfig
from common_utils import second2time, print_trainable_parameters

def evaluate_model(config, model, data_loader):
    model.eval()
    running_loss = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            with torch.autocast(device_type=config.device, dtype=torch.float16):
                if config.batch_size == 1:
                    loss = model(input_ids=batch['input_ids'].to(dtype=torch.long, device=config.device),
                                 labels=batch['labels'].to(dtype=torch.long, device=config.device)).loss
                else:
                    loss = model(input_ids=batch['input_ids'].to(dtype=torch.long, device=config.device),
                                 attention_mask=batch['attention_mask'].to(dtype=torch.long, device=config.device),
                                 labels=batch['labels'].to(dtype=torch.long, device=config.device)).loss
            if not math.isnan(loss):
                running_loss.append(float(loss.cpu().detach()))
    model.train()
    if len(running_loss) == 0:
        avg_loss = float('inf')
    else:
        avg_loss = sum(running_loss) / len(running_loss)
    return avg_loss

def train_model(config, template, known_template, query_template, label_template, target_modules):
    tokenizer = AutoTokenizer.from_pretrained(config.pre_model, padding_side='right', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    """
    https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/llama#transformers.LlamaForCausalLM
    """
    model = AutoModelForCausalLM.from_pretrained(config.pre_model, quantization_config=nf4_config, device_map="auto", trust_remote_code=True)
    print(f'Base model:\n{model}')
    model = peft.prepare_model_for_kbit_training(model)

    if config.checkpointing_flag:
        model.gradient_checkpointing_enable()  
        model.enable_input_require_grads() 
       
        model.config.use_cache = False

    train_dataloader, val_dataloader = get_dataloader(config.trainfile_path, config.valfile_path, template,
                                                      known_template, query_template, label_template, tokenizer,
                                                      config.train_joint_symbol, config.inference_joint_symbol,
                                                      config.inference_data_num, config.train_ratio, config.entry_size,
                                                      config.batch_size)
    
    num_update_steps_per_epoch = len(train_dataloader)
    
    max_train_steps = config.epochs * num_update_steps_per_epoch
    
    warm_steps = int(config.warmup_ratio * max_train_steps)

    
    if config.lora_type == "lora" or config.lora_type == "dora":
        use_dora = False
        if config.lora_type == "dora":
            use_dora = True
        lora_config = peft.LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            init_lora_weights=config.init_lora_weights,
            use_dora=use_dora,
            task_type=config.lora_task_type,
            target_modules=target_modules,
            inference_mode=False,
        )
       
        model = peft.get_peft_model(model, lora_config, adapter_name=config.lora_type)
    else:
        lora_config = peft.AdaLoraConfig(
            tinit=int(config.adalora_init_warmup_ratio * max_train_steps),
            tfinal=int(max_train_steps * config.adalora_final_warmup_ratio),
            total_step=max_train_steps,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            init_lora_weights=config.init_lora_weights,
            task_type=config.lora_task_type,
            target_modules=target_modules,
            inference_mode=False,
        )
        
        model.add_adapter(lora_config, adapter_name=config.lora_type)
    print(f'LoRA model:\n{model}')

   
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 1e-8,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    scaler = torch.cuda.amp.GradScaler()
    # model.print_trainable_parameters()
    print_trainable_parameters(model)

    cur_time = time.time()
    best_eval_loss = float('inf')
    print('Begin to train：')
    for epoch in range(config.epochs):
        running_loss = []
        best_save_step = 0
        for cur_step, batch in enumerate(train_dataloader):
            with torch.autocast(device_type=config.device, dtype=torch.float16):
                if config.batch_size == 1:
                    loss = model(input_ids=batch['input_ids'].to(dtype=torch.long, device=config.device),
                                 labels=batch['labels'].to(dtype=torch.long, device=config.device)).loss
                else:
                    loss = model(input_ids=batch['input_ids'].to(dtype=torch.long, device=config.device),
                                 attention_mask=batch['attention_mask'].to(dtype=torch.long, device=config.device),
                                 labels=batch['labels'].to(dtype=torch.long, device=config.device)).loss
            scaler.scale(loss).backward()
           
            if config.gradient_accumulation_flag:
                if cur_step % config.gradient_accumulation_steps == 0 or cur_step == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            running_loss.append(float(loss.cpu().detach()))

            if (cur_step + 1) % config.logging_steps == 0:
                time_diff = time.time() - cur_time
                loss_avg = sum(running_loss) / len(running_loss)
                print("epoch: %d, cur step %d ( %02.2f%% ) , loss: %.5f, speed: %.2f step/s, ETA: %s" % (
                epoch + 1, cur_step + 1, (epoch * num_update_steps_per_epoch + (cur_step + 1)) / max_train_steps * 100,
                loss_avg, config.logging_steps / time_diff, second2time(
                    int((max_train_steps - epoch * num_update_steps_per_epoch - (cur_step + 1)) / (
                                config.logging_steps / time_diff)))))
                cur_time = time.time()
            if (cur_step + 1) % config.valid_steps == 0:
                eval_loss = evaluate_model(config, model, val_dataloader)
                print("Evaluation Loss: %.5f" % (eval_loss))
                if eval_loss < best_eval_loss:
                    print(f"Min eval loss has been updated: {best_eval_loss:.5f} --> {eval_loss:.5f}")
                    best_eval_loss = eval_loss
                    cur_save_dir = os.path.join(config.save_dir, "lora_model_best")
                    model.save_pretrained(cur_save_dir)
                    tokenizer.save_pretrained(cur_save_dir)
                    print(f'Best model has saved at {cur_save_dir}.')
                    best_save_step = cur_step
            if (cur_step - best_save_step) >= config.non_save_count * config.valid_steps:
                break
            if (epoch + 1) == 1 and cur_step >= 2:
                gc.collect()
                torch.cuda.empty_cache()
    print('End')


if __name__ == '__main__':
    config = ProjectConfig()
    template = """
    You are a top-tier mathematician who is exceptionally skilled at predicting missing values of a symmetric sparse matrix.
    Note that the values of a symmetric sparse matrix are all decimal numbers ranging between 0 and 1.
    The current task is to predict the decimal value at a specific position in the matrix (identified by its row and column indices) based on the provided examples.
    Examples:
    """
    known_template = "Input: Row Index={0}, Column Index={1}\nAnswer: ${2}$\n"
    query_template = "Input: Row Index={0}, Column Index={1}\nAnswer: "
    label_template = "${0}$"
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj",
        "lm_head"
    ]
    train_model(config, template, known_template, query_template, label_template, target_modules)
