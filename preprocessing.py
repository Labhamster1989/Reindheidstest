from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import cv2, numpy as np
from PIL import Image

ImageInput=Union[Image.Image,np.ndarray,str,Path,bytes]

@dataclass(frozen=True)
class Circle:
    x:int; y:int; radius:int

def _to_bgr(x:ImageInput)->np.ndarray:
    if isinstance(x,Image.Image): return cv2.cvtColor(np.array(x.convert("RGB")),cv2.COLOR_RGB2BGR)
    if isinstance(x,(str,Path)):
        im=cv2.imread(str(x))
        if im is None: raise ValueError(f"Kon afbeelding niet lezen: {x}")
        return im
    if isinstance(x,bytes):
        im=cv2.imdecode(np.frombuffer(x,np.uint8),cv2.IMREAD_COLOR)
        if im is None: raise ValueError("Ongeldige afbeeldingsbytes.")
        return im
    if isinstance(x,np.ndarray):
        im=np.clip(x,0,255).astype(np.uint8) if x.dtype!=np.uint8 else x
        if im.ndim==2: return cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        if im.shape[2]==4: return cv2.cvtColor(im,cv2.COLOR_RGBA2BGR)
        return cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    raise TypeError(type(x))

def _score(g:np.ndarray,c:Circle)->float:
    h,w=g.shape; yy,xx=np.ogrid[:h,:w]; d=np.sqrt((xx-c.x)**2+(yy-c.y)**2)
    ring=(d>=c.radius*.88)&(d<=c.radius*1.1); inner=d<=c.radius*.92
    edge=float(np.mean(cv2.Canny(g,60,140)[ring])) if ring.any() else 0
    std=float(np.std(g[inner])) if inner.any() else 0
    return 2*c.radius/(min(h,w)/2)+.006*edge+.002*min(std,100)-.9*abs(c.x-w/2)/(w/2)-.9*abs(c.y-h/2)/(h/2)

def _hough(g):
    m=min(g.shape); c=cv2.HoughCircles(cv2.GaussianBlur(g,(9,9),1.5),cv2.HOUGH_GRADIENT,1.2,
        minDist=max(20,int(m*.35)),param1=120,param2=32,minRadius=max(20,int(m*.25)),maxRadius=max(25,int(m*.44)))
    if c is None:return None
    cs=[Circle(round(x),round(y),round(r)) for x,y,r in np.squeeze(c,axis=0)]
    return max(cs,key=lambda z:_score(g,z)) if cs else None

def _contour(g):
    m=min(g.shape); e=cv2.Canny(cv2.GaussianBlur(g,(7,7),1.2),40,120)
    e=cv2.morphologyEx(e,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    contours,_=cv2.findContours(e,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best=None
    for q in contours:
        area=cv2.contourArea(q)
        if area<.16*np.pi*(m/2)**2: continue
        (x,y),r=cv2.minEnclosingCircle(q)
        if not m*.28<r<m*.60: continue
        p=cv2.arcLength(q,True)
        if p<=0:continue
        circ=4*np.pi*area/(p*p)
        score=4*circ+2*r/(m/2)-np.hypot(x-g.shape[1]/2,y-g.shape[0]/2)/(m/2)
        if best is None or score>best[0]: best=(score,Circle(round(x),round(y),round(r)))
    return best[1] if best else None

def detect_filter_circle(image:ImageInput)->Circle:
    b=_to_bgr(image); g=cv2.medianBlur(cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),5)
    c=_hough(g) or _contour(g)
    if c:return c
    h,w=g.shape; return Circle(w//2,h//2,int(.38*min(h,w)))

def _crop(b,c):
    h,w=b.shape[:2]; r=c.radius; x0=max(0,c.x-r); y0=max(0,c.y-r); x1=min(w,c.x+r); y1=min(h,c.y+r)
    out=b[y0:y1,x0:x1].copy(); cx,cy=c.x-x0,c.y-y0
    mask=np.zeros(out.shape[:2],np.uint8); cv2.circle(mask,(cx,cy),min(r,cx,cy,out.shape[1]-cx,out.shape[0]-cy),255,-1)
    return np.where(mask[...,None]==255,out,255)

def normalize_filter_image(image_input:ImageInput,output_size:int=512)->Image.Image:
    b=_crop(_to_bgr(image_input),detect_filter_circle(image_input))
    lab=cv2.cvtColor(b,cv2.COLOR_BGR2LAB); l,a,bb=cv2.split(lab)
    l=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
    b=cv2.cvtColor(cv2.merge((l,a,bb)),cv2.COLOR_LAB2BGR)
    b=cv2.resize(b,(output_size,output_size),interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(b,cv2.COLOR_BGR2RGB)).convert("RGB")
