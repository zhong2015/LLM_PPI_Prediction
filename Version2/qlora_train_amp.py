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
    这里所使用的model是一个LlamaForCausalLM类对象，其详解可参见如下网址：
    https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/llama#transformers.LlamaForCausalLM
    """
    model = AutoModelForCausalLM.from_pretrained(config.pre_model, quantization_config=nf4_config, device_map="auto", trust_remote_code=True)
    print(f'Base model:\n{model}')
    model = peft.prepare_model_for_kbit_training(model)

    if config.checkpointing_flag:
        model.gradient_checkpointing_enable()  # 开启模型的梯度检查点
        model.enable_input_require_grads()  # 必须确保同时设置了model.enable_input_require_grads()和model.gradient_checkpointing_enable()
        # 关闭缓存以兼容gradient checkpointing，从而压缩模型所需的内存
        model.config.use_cache = False

    train_dataloader, val_dataloader = get_dataloader(config.trainfile_path, config.valfile_path, template,
                                                      known_template, query_template, label_template, tokenizer,
                                                      config.train_joint_symbol, config.inference_joint_symbol,
                                                      config.inference_data_num, config.train_ratio, config.entry_size,
                                                      config.batch_size)
    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    # 指定总的训练步数，它会被学习率调度器用来确定学习率的变化规律，确保学习率在整个训练过程中得以合理地调节
    max_train_steps = config.epochs * num_update_steps_per_epoch
    # 预热阶段的训练步数
    warm_steps = int(config.warmup_ratio * max_train_steps)

    """
    PEFT的介绍可参见如下网址：
    https://huggingface.co/docs/peft/v0.15.0/en/tutorial/peft_model_config#peft-configurations-and-models

    函数LoraConfig()的详解可参见如下网址：
    https://huggingface.co/docs/peft/package_reference/lora

    target_modules这个参数的设置是参见如下网址的：
    https://blog.csdn.net/weixin_44826203/article/details/129733930
    由该网址可知，这样的设置是选择了model中的q_proj和v_proj模块（即是attention中的q和v的部分）做LORA。
    """
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
        """
        函数get_peft_model()的详解可参见如下网址：
        https://huggingface.co/docs/peft/v0.15.0/en/package_reference/peft_model#peft.get_peft_model
        """
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
        """
        出处1：https://huggingface.co/docs/transformers/main/en/peft
        出处2：https://huggingface.co/docs/transformers/main/en/main_classes/peft#transformers.integrations.PeftAdapterMixin.enable_adapters
        如果想兼顾device_map="auto"，那么adaLora只能通过add_adapter()来进行实现（如果adaLora要通过peft.get_peft_model()进行实现的话
        那么上面model定义时device_map的值就不能是"auto"而要指定一个GPU，比如"cuda:0"）。
        注意，第一次执行时会报关于low_cpu_mem_usage的错误，此时我们点击进入\peft\mapping.py这个文件中，
        将"peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)"修
        改为"peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name)"即可
        """
        model.add_adapter(lora_config, adapter_name=config.lora_type)
    print(f'LoRA model:\n{model}')

    """
    值得注意的是optimizer的定义必须要放在peft.get_peft_model(model, lora_config)的后面，否则会报错！
    """
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
    print('开始训练：')
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
            """
            config.gradient_accumulation_flag=True开启梯度累计的方式来节省显存
            可以看出，梯度累计的方式实际上就是只有在特定step时才会更新模型参数和学习率
            """
            if config.gradient_accumulation_flag:
                if cur_step % config.gradient_accumulation_steps == 0 or cur_step == len(train_dataloader):
                    """
                    由于此时使用了梯度累计，故这里使用梯度裁剪，来避免梯度爆炸
                    """
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
                print("开始跑验证集了.....")
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
    print('训练结束')


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