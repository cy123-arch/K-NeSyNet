from __future__ import annotations
import numpy as np
from PIL import Image

def tissue_fraction(rgb: np.ndarray, white_threshold=0.85):
    x=rgb.astype(np.float32)/255.0
    # H&E tissue is typically darker and more saturated than background.
    sat=(x.max(2)-x.min(2))
    dark=x.mean(2) < white_threshold
    return float(np.mean(dark & (sat > 0.05)))

def _od(rgb):
    return -np.log((rgb.astype(np.float32)+1.0)/256.0)

def macenko_normalize(rgb: np.ndarray, reference_stain=None, alpha=1.0, beta=0.15):
    """Macenko-style stain normalization for RGB H&E tiles.

    This implementation performs optical-density filtering, SVD stain-axis estimation,
    percentile angle selection, concentration estimation, and reconstruction against a
    fixed reference stain matrix. It is deterministic for a given tile/reference.
    """
    if reference_stain is None:
        reference_stain=np.array([[0.650,0.704],[0.072,0.990],[0.286,0.105]],dtype=np.float32)
    shape=rgb.shape; od=_od(rgb).reshape(-1,3)
    keep=np.all(od > beta,axis=1)
    if keep.sum() < 10: return rgb
    odhat=od[keep]
    _,_,v=np.linalg.svd(odhat,full_matrices=False)
    vec=v[:2].T
    proj=odhat @ vec
    phi=np.arctan2(proj[:,1],proj[:,0])
    minphi=np.percentile(phi,alpha); maxphi=np.percentile(phi,100-alpha)
    vmin=vec @ np.array([np.cos(minphi),np.sin(minphi)])
    vmax=vec @ np.array([np.cos(maxphi),np.sin(maxphi)])
    he=np.stack([vmin,vmax],axis=1)
    if he[0,0] < he[0,1]: he=he[:,::-1]
    c=np.linalg.lstsq(he,od.T,rcond=None)[0]
    maxc=np.percentile(c,99,axis=1)
    refmax=np.array([1.9705,1.0308],dtype=np.float32)
    c=c*(refmax[:,None]/np.maximum(maxc[:,None],1e-6))
    out=255*np.exp(-(reference_stain @ c))
    return np.clip(out.T.reshape(shape),0,255).astype(np.uint8)

def iter_svs_tiles(path, tile_size=224, magnification=20, max_tiles=256, min_tissue_fraction=0.35):
    import openslide
    slide=openslide.OpenSlide(str(path))
    objective=float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, magnification))
    desired_downsample=max(objective/float(magnification),1.0)
    level=min(range(slide.level_count),key=lambda i:abs(slide.level_downsamples[i]-desired_downsample))
    ds=float(slide.level_downsamples[level])
    w,h=slide.level_dimensions[level]
    yielded=0
    for y in range(0,max(1,h-tile_size+1),tile_size):
        for x in range(0,max(1,w-tile_size+1),tile_size):
            tile=slide.read_region((int(x*ds),int(y*ds)),level,(tile_size,tile_size)).convert('RGB')
            arr=np.asarray(tile)
            if tissue_fraction(arr) < min_tissue_fraction: continue
            arr=macenko_normalize(arr)
            yield Image.fromarray(arr)
            yielded += 1
            if yielded >= max_tiles: return
