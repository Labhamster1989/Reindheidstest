from __future__ import annotations
import hashlib,json,os,re,shutil,threading,uuid
from collections import Counter
from pathlib import Path
from typing import Any
import gradio as gr
import numpy as np
import torch
from PIL import Image
from preprocessing import normalize_filter_image

BASE=Path(__file__).resolve().parent
DATA=BASE/"dataset"/"genormaliseerd"
CACHE=BASE/"dataset"/".cache"
MANIFEST=BASE/"dataset"/"manifest.json"
CLASSES=("A","B","C","D")
K=int(os.getenv("K_NEIGHBORS","7"))
SHOW_CONF=os.getenv("SHOW_CONFIDENCE","true").lower() in {"1","true","yes"}
CLIP_ID=os.getenv("CLIP_MODEL_ID","openai/clip-vit-base-patch32")
HF_DATASET=os.getenv("HF_DATASET_REPO_ID","").strip()
HF_TOKEN=os.getenv("HF_TOKEN","").strip()
ADAPTER=os.getenv("HF_VL_ADAPTER_ID","").strip()
for c in CLASSES:(DATA/c).mkdir(parents=True,exist_ok=True)
CACHE.mkdir(parents=True,exist_ok=True)
LOCK=threading.RLock()

def _safe(name):
    stem=re.sub(r"[^A-Za-z0-9._-]+","_",Path(name).stem).strip("._") or "filter"
    return f"{stem}_{uuid.uuid4().hex[:10]}.png"

def _manifest():
    if not MANIFEST.exists():return []
    try:return json.loads(MANIFEST.read_text(encoding="utf8"))
    except:return []

def _write_manifest(x):
    tmp=MANIFEST.with_suffix(".tmp"); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf8"); tmp.replace(MANIFEST)

def _hf():
    if not(HF_DATASET and HF_TOKEN):return None
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)

def sync_dataset():
    api=_hf()
    if not api:return
    try: files=api.list_repo_files(repo_id=HF_DATASET,repo_type="dataset")
    except Exception:return
    from huggingface_hub import hf_hub_download
    for c in CLASSES:
        for f in files:
            if f.startswith(f"genormaliseerd/{c}/") and f.endswith(".png"):
                dst=DATA/c/Path(f).name
                if dst.exists():continue
                try:shutil.copy2(hf_hub_download(repo_id=HF_DATASET,filename=f,repo_type="dataset",token=HF_TOKEN),dst)
                except Exception:pass

def hf_upload(p,c):
    api=_hf()
    if not api:return
    api.create_repo(repo_id=HF_DATASET,repo_type="dataset",exist_ok=True)
    api.upload_file(path_or_fileobj=str(p),path_in_repo=f"genormaliseerd/{c}/{p.name}",repo_id=HF_DATASET,repo_type="dataset",commit_message=f"Add {c} reference")

def hf_delete(p,c):
    api=_hf()
    if not api:return
    try:api.delete_file(path_in_repo=f"genormaliseerd/{c}/{p.name}",repo_id=HF_DATASET,repo_type="dataset",commit_message=f"Delete {c} reference")
    except Exception:pass

class Matcher:
    def __init__(self):
        from transformers import CLIPModel,CLIPProcessor
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.processor=CLIPProcessor.from_pretrained(CLIP_ID)
        self.model=CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
        self.emb=None;self.labels=[];self.paths=[];self.reload()
    @torch.inference_mode()
    def encode(self,ims):
        x=self.processor(images=ims,return_tensors="pt",padding=True)
        x={k:v.to(self.device) for k,v in x.items()}
        f=self.model.get_image_features(**x); f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        return f.cpu().numpy().astype("float32")
    def entries(self):return [(p,c) for c in CLASSES for p in sorted((DATA/c).glob("*.png"))]
    def reload(self):
        es=self.entries()
        if not es:self.emb=np.empty((0,512),np.float32);self.labels=[];self.paths=[];return
        ims=[];kept=[]
        for p,c in es:
            try:
                with Image.open(p) as im:ims.append(im.convert("RGB"))
                kept.append((p,c))
            except Exception:pass
        self.emb=np.concatenate([self.encode(ims[i:i+16]) for i in range(0,len(ims),16)])
        self.labels=[c for _,c in kept];self.paths=[str(p) for p,_ in kept]
    def predict(self,im):
        if len(self.emb)==0:raise RuntimeError("Voeg eerst referentiefoto's toe in Database Beheer.")
        q=self.encode([im])[0]; sims=self.emb@q;k=min(K,len(sims));idx=np.argsort(sims)[::-1][:k]
        ss=sims[idx]; labs=[self.labels[i] for i in idx]; weights=np.maximum(ss-ss.min()+1e-4,1e-4)
        scores=Counter()
        for lab,w in zip(labs,weights):scores[lab]+=float(w)
        lab,score=scores.most_common(1)[0]
        return lab,float(score/max(scores.total(),1e-8))

class Qwen:
    def __init__(self,adapter):
        from peft import PeftModel
        from transformers import AutoProcessor
        from unsloth import FastVisionModel
        base=os.getenv("QWEN_BASE_MODEL","unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
        self.model,self.processor=FastVisionModel.from_pretrained(base,load_in_4bit=True)
        self.model=PeftModel.from_pretrained(self.model,adapter);FastVisionModel.for_inference(self.model)
        try:self.processor=AutoProcessor.from_pretrained(adapter)
        except Exception:pass
        self.model.eval()
    def predict(self,im):
        prompt="Classificeer dit genormaliseerde melkpoederfilter uitsluitend als A, B, C of D. Beoordeel de scorched particles. Antwoord exact met één hoofdletter."
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        txt=self.processor.apply_chat_template(msg,add_generation_prompt=True,tokenize=False)
        x=self.processor(text=txt,images=im,padding=True,return_tensors="pt")
        dev=next(self.model.parameters()).device;x={k:v.to(dev) if hasattr(v,"to") else v for k,v in x.items()}
        with torch.inference_mode(): y=self.model.generate(**x,max_new_tokens=4,do_sample=False)
        y=y[:,x["input_ids"].shape[-1]:]
        out=self.processor.batch_decode(y,skip_special_tokens=True)[0].upper()
        m=re.search(r"\b([ABCD])\b",out)
        if not m:raise RuntimeError(f"Ongeldige modeluitvoer: {out}")
        return m.group(1)

try:sync_dataset()
except Exception as e:print("Dataset sync:",e)
try:MATCHER=Matcher()
except Exception as e:MATCHER=None;print("CLIP niet geladen:",e)
try:QWEN=Qwen(ADAPTER) if ADAPTER else None
except Exception as e:QWEN=None;print("Qwen niet geladen, fallback actief:",e)

def analyze(image):
    if image is None:return None,"—"
    im=normalize_filter_image(image,512)
    with LOCK:
        if QWEN:
            label=QWEN.predict(im); result=f"Klasse: {label}"
        else:
            if MATCHER is None:raise gr.Error("Het fallback-model kon niet worden geladen.")
            label,conf=MATCHER.predict(im); result=f"Klasse: {label} ({round(conf*100)}%)" if SHOW_CONF else f"Klasse: {label}"
    return im,result

def gallery():
    return [(str(p),c) for c in CLASSES for p in sorted((DATA/c).glob("*.png"))]

def choices():
    return [f"{c} | {p}" for p,c in gallery()]

def save_reference(image,c):
    if image is None:raise gr.Error("Kies eerst een foto.")
    if c not in CLASSES:raise gr.Error("Kies A, B, C of D.")
    im=normalize_filter_image(image,512); name=_safe("referentie"); p=DATA/c/name; im.save(p,"PNG",optimize=True)
    m=_manifest();m.append({"id":name,"class":c,"path":str(p.relative_to(BASE)),"size":512});_write_manifest(m)
    try:hf_upload(p,c)
    except Exception as e:print("HF upload:",e)
    if MATCHER:MATCHER.reload()
    return im,gr.update(value=gallery())

def refresh():
    g=gallery();ch=[f"{c} | {p}" for p,c in g]
    return g,gr.update(choices=ch,value=ch[0] if ch else None)

def delete_reference(sel):
    if not sel:raise gr.Error("Selecteer een foto.")
    c,path=sel.split(" | ",1);p=Path(path)
    if c not in CLASSES or p.parent.name!=c or not p.exists():raise gr.Error("Ongeldige selectie.")
    p.unlink()
    _write_manifest([x for x in _manifest() if x.get("path")!=str(p.relative_to(BASE))])
    try:hf_delete(p,c)
    except Exception:pass
    if MATCHER:MATCHER.reload()
    return refresh()

CSS=".result textarea{font-size:42px!important;font-weight:800!important;text-align:center!important}.preview img{max-height:320px;object-fit:contain}@media(max-width:700px){.result textarea{font-size:34px!important}}"
with gr.Blocks(title="Melkpoeder Reinheidstest",css=CSS,theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Melkpoeder Reinheidstest")
    with gr.Tab("Analyse / Test"):
        with gr.Row():
            inp=gr.Image(label="Filterfoto",sources=["upload","webcam"],type="pil",image_mode="RGB",height=360)
            prev=gr.Image(label="Genormaliseerd filter",interactive=False,elem_classes="preview")
        btn=gr.Button("Analyseer",variant="primary",size="lg")
        out=gr.Textbox(label="Resultaat",value="—",interactive=False,elem_classes="result",max_lines=1)
        btn.click(analyze,inp,[prev,out])
        inp.change(lambda:(None,"—"),outputs=[prev,out])
    with gr.Tab("Database Beheer"):
        with gr.Row():
            ref=gr.Image(label="Nieuwe referentiefoto",sources=["upload","webcam"],type="pil")
            refprev=gr.Image(label="Genormaliseerd",interactive=False)
        cls=gr.Dropdown(list(CLASSES),value="A",label="Klasse")
        gal=gr.Gallery(value=gallery(),columns=4,height="auto",label="Genormaliseerde database")
        save=gr.Button("Normaliseren + opslaan",variant="primary")
        save.click(save_reference,[ref,cls],[refprev,gal])
        sel=gr.Dropdown(choices=choices(),label="Te verwijderen bestand")
        dele=gr.Button("Verwijder geselecteerde foto",variant="stop");refreshbtn=gr.Button("Vernieuw overzicht")
        dele.click(delete_reference,sel,[gal,sel]);refreshbtn.click(refresh,outputs=[gal,sel])
if __name__=="__main__":demo.launch(server_name="0.0.0.0",server_port=int(os.getenv("PORT","7860")))
