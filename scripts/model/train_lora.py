import os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

B = os.environ["B"]
MODE = os.environ["MODE"]   # patch or rewrite
MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_LEN = 3072              # longest patch row is ~1.8k tokens, longest rewrite row ~3.2k
N_VAL = 300                 # val rows for the val loss
out_dir = f"{B}/checkpoints/{MODE}"

tok = AutoTokenizer.from_pretrained(MODEL)


# ---------- data: tokens + labels ----------

# one row -> input_ids and labels
# labels[i] = the token the model should predict at position i. -100 means no loss there,
# so every prompt token gets -100 and the loss is only on the reply (and its <|im_end|> stop token)
def encode(msgs):
    # prompt = system + user turn + "<|im_start|>assistant\n"
    # full   = prompt + reply + "<|im_end|>\n"
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=False)
    full = tok.apply_chat_template(msgs, tokenize=False)
    assert full.startswith(prompt)
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if tok.decode(ids[-1:]) == "\n":   # drop the newline after <|im_end|>, the reply should end at <|im_end|>
        ids = ids[:-1]

    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    return dict(input_ids=ids, labels=labels)

# rows of one split; limit = take only that many, spread over t1 / t2 / t3
def load(split, limit=None):
    rows = [json.loads(l) for l in open(f"{B}/data/sft/{MODE}_{split}.jsonl")]
    if limit:
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    data = [encode(r["messages"]) for r in rows]
    kept = [d for d in data if len(d["input_ids"]) <= MAX_LEN]
    print(f"{split}: {len(kept)} of {len(data)} rows fit in {MAX_LEN} tokens")
    return kept

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)                 # no loss on padding
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad)   # padding is not attended to
    return dict(input_ids=torch.tensor(input_ids), labels=torch.tensor(labels), attention_mask=torch.tensor(attention_mask))


if __name__ == "__main__":
    train, val = load("train"), load("val", N_VAL)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    per_gpu, batch = 4, 32
    accumulation = max(1, batch // (per_gpu * n_gpus))

    # loss: the model's own cross-entropy on the labels, i.e. the mean over the reply tokens of
    #   -log p(token | all tokens before it)   (positions with label -100 are left out)
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        optim="adamw_torch_fused", adam_beta1=0.9, adam_beta2=0.999, weight_decay=0.0, max_grad_norm=1.0,
        # learning rate: linear warmup over the first 3% of steps (27 of 870) up to 2e-4, then cosine decay to 0
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
        per_device_train_batch_size=per_gpu, per_device_eval_batch_size=4, gradient_accumulation_steps=accumulation,
        bf16=True,
        group_by_length=True,   # rows of similar length go in the same batch, less padding
        logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=200,
        save_total_limit=2, report_to="none",
        ddp_find_unused_parameters=False, dataloader_num_workers=2, remove_unused_columns=False, seed=0)

    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate)
    trainer.train(resume_from_checkpoint=os.environ.get("RESUME") or None)

    trainer.save_model(f"{out_dir}/final")
    if trainer.is_world_process_zero():
        json.dump(trainer.state.log_history, open(f"{out_dir}/log_history.json", "w"), indent=1)
