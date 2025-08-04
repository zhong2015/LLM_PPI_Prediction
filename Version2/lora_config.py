# coding=utf-8

import torch

class ProjectConfig(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  
        self.pre_model = r'D:\downloaded_LLM\metamath-mistral-7B'
        self.trainfile_path = r'C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_train.txt'
        self.valfile_path = r'C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_val.txt'
        self.testfile_path = r'C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_test.txt'
        self.train_ratio = 0.8
        self.entry_size = 80
        self.batch_size = 1
        self.train_joint_symbol = ""
        self.inference_joint_symbol = ","
        self.inference_data_num = 80
        self.lora_type = "lora"
        self.lora_task_type = "CAUSAL_LM"
        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 5e-2
        self.lora_bias = "none"
        self.init_lora_weights = True
        self.adalora_init_warmup_ratio = 0.02
        self.adalora_final_warmup_ratio = 0.06
        self.learning_rate = 1e-4
        self.weight_decay = 1e-3
        self.warmup_ratio = 5e-4
        self.epochs = 2
        self.checkpointing_flag = True
        self.gradient_accumulation_flag = False
        self.gradient_accumulation_steps = 4
        self.max_grad_norm = 2.0
        self.logging_steps = 20
        self.valid_steps = 800
        self.non_save_count = 2
        self.save_dir = r'D:\LLM\LLM-Codes\LLM_SHDI\checkpoints'
