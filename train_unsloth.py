from __future__ import annotations
import argparse,os
from pathlib import Path
from PIL import Image

def build_dataset(root:Path):
    rows=[]
    instruction="Classificeer dit genormaliseerde melkpoederfilter uitsluitend als A, B, C of D. Beoordeel de scorched particles. Antwoord exact met één hoofdletter."
    for label in "ABCD":
        for p in sorted((root/label).glob("*.png")):
            with Image.open(p) as im:
                rows.append({"messages":[{"role":"user","content":[{"type":"text","text":instruction},{"type":"image","image":im.convert("RGB")}]},{"role":"assistant","content":[{"type":"text","text":label}]}]})
    if not rows: raise RuntimeError("Geen PNG-referenties gevonden.")
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset-dir",default="dataset/genormaliseerd"); ap.add_argument("--output-dir",default="outputs/melkpoeder-reinheid-lora")
    ap.add_argument("--hub-model-id",default=os.getenv("HF_MODEL_REPO_ID","")); ap.add_argument("--epochs",type=float,default=3)
    ap.add_argument("--batch-size",type=int,default=1); ap.add_argument("--grad-accumulation",type=int,default=8); ap.add_argument("--learning-rate",type=float,default=2e-4)
    a=ap.parse_args()
    from transformers import set_seed
    from unsloth import FastVisionModel,is_bf16_supported
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig,SFTTrainer
    set_seed(3407); data=build_dataset(Path(a.dataset_dir))
    model,tokenizer=FastVisionModel.from_pretrained("unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",load_in_4bit=True,use_gradient_checkpointing="unsloth",max_seq_length=2048)
    model=FastVisionModel.get_peft_model(model,finetune_vision_layers=True,finetune_language_layers=True,finetune_attention_modules=True,finetune_mlp_modules=True,r=16,lora_alpha=16,lora_dropout=0,bias="none",random_state=3407,target_modules="all-linear",modules_to_save=["lm_head","embed_tokens"])
    FastVisionModel.for_training(model)
    collator=UnslothVisionDataCollator(model,tokenizer,resize="min",train_on_responses_only=True,completion_only_loss=True)
    trainer=SFTTrainer(model=model,tokenizer=tokenizer,data_collator=collator,train_dataset=data,args=SFTConfig(
        per_device_train_batch_size=a.batch_size,gradient_accumulation_steps=a.grad_accumulation,num_train_epochs=a.epochs,learning_rate=a.learning_rate,
        warmup_ratio=.05,logging_steps=1,save_strategy="epoch",save_total_limit=2,optim="adamw_8bit",weight_decay=.01,lr_scheduler_type="cosine",
        fp16=not is_bf16_supported(),bf16=is_bf16_supported(),seed=3407,output_dir=a.output_dir,report_to="none",remove_unused_columns=False,
        dataset_text_field="",dataset_kwargs={"skip_prepare_dataset":True},max_seq_length=2048))
    trainer.train(); model.save_pretrained(a.output_dir); tokenizer.save_pretrained(a.output_dir)
    if a.hub_model_id:
        model.push_to_hub(a.hub_model_id,token=os.getenv("HF_TOKEN")); tokenizer.push_to_hub(a.hub_model_id,token=os.getenv("HF_TOKEN"))
if __name__=="__main__": main()
