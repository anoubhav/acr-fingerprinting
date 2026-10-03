"""FAISS-only storage for the frozen temporal-vote ACR matcher.

This system variant preserves descriptors, search settings and vote semantics.
It does not retain a second reference-vector matrix, per-row content codes or
float64 clocks. Original physical frame indices plus track boundaries suffice.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


class _VectorShape:
    def __init__(self,rows,dimension):
        self.shape=(int(rows),int(dimension))
    def __len__(self):
        return self.shape[0]


class _ContentLookup:
    def __init__(self,ends):
        self.ends=ends
    def __getitem__(self,rows):
        return np.searchsorted(self.ends,rows,side="right").astype(np.int32)


class _TimeLookup:
    def __init__(self,frames,hop,sample_rate,origin):
        self.frames,self.hop,self.sample_rate,self.origin=frames,hop,sample_rate,origin
    def __getitem__(self,rows):
        # Match the original integer-multiply/divide clock ordering exactly.
        return self.frames[rows].astype(np.float64)*self.hop/self.sample_rate+self.origin


class CompactTemporalIndex:
    def __init__(self,faiss_index,frame_indices,track_ends,content_ids,config,factor=8):
        if not isinstance(factor,int) or factor<1:
            raise ValueError("factor must be a positive integer")
        frames=np.asarray(frame_indices)
        if frames.ndim!=1 or not np.issubdtype(frames.dtype,np.integer) or np.any(frames<0) or np.any(frames>np.iinfo(np.uint32).max):
            raise ValueError("Physical frame indices must be unsigned32-bit integers")
        if faiss_index.ntotal<1 or faiss_index.d!=config.output_dim:
            raise ValueError("Compact index is empty or differs from configured dimension")
        self._faiss=faiss_index
        self.frame_indices=np.ascontiguousarray(frames,dtype=np.uint32)
        self.track_ends=np.asarray(track_ends,dtype=np.int64)
        self.content_ids=list(map(str,content_ids))
        self.config,self.factor=config,int(factor)
        if len(self.frame_indices)!=faiss_index.ntotal or len(self.track_ends)!=len(self.content_ids):
            raise ValueError("Compact index metadata length mismatch")
        if not len(self.track_ends) or self.track_ends[-1]!=faiss_index.ntotal or np.any(np.diff(self.track_ends)<0):
            raise ValueError("Invalid track boundaries")
        self.vectors=_VectorShape(faiss_index.ntotal,faiss_index.d)
        self.content_codes=_ContentLookup(self.track_ends)
        self.times=_TimeLookup(self.frame_indices,config.hop_length,config.sample_rate,config.first_center_seconds)

    def search(self,queries,k=5,block_size=32768):
        q=np.ascontiguousarray(queries,dtype=np.float32)
        if q.ndim!=2 or q.shape[1]!=self._faiss.d or not np.isfinite(q).all():
            raise ValueError("Invalid compact-index query")
        if k<1:
            raise ValueError("k must be positive")
        return self._faiss.search(q,min(int(k),self._faiss.ntotal))

    def match(self,queries,query_times,**kwargs):
        # Reuse the frozen implementation rather than rewrite vote aggregation.
        from acr_fp import ExactIndex
        return ExactIndex.match(self,queries,query_times,**kwargs)

    def metadata_bytes(self):
        return self.frame_indices.nbytes+self.track_ends.nbytes+sum(len(x.encode()) for x in self.content_ids)

    def save_metadata(self,path):
        from dataclasses import asdict
        np.savez_compressed(path,frame_indices=self.frame_indices,track_ends=self.track_ends,
            content_ids=np.asarray(self.content_ids),metadata=np.array(json.dumps({"config":asdict(self.config),"factor":self.factor})))

    @classmethod
    def load(cls,index_path,metadata_path,nprobe=16):
        """Load the FAISS store and small physical-clock metadata without copies."""
        import faiss
        from acr_fp import FingerprintConfig
        index=faiss.read_index(str(index_path))
        if hasattr(index,"nprobe"):index.nprobe=int(nprobe)
        with np.load(metadata_path,allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]))
            return cls(index,z["frame_indices"],z["track_ends"],z["content_ids"],
                FingerprintConfig(**meta["config"]),int(meta["factor"]))


def streaming_enroll(gallery,model,cache,faiss_index,factor=8):
    """Select original grid rows before projecting; add one content at a time."""
    from run_ablations import ref_path
    import hashlib
    if not isinstance(factor,int) or factor<1:
        raise ValueError("factor must be a positive integer")
    if faiss_index.ntotal or not faiss_index.is_trained or faiss_index.d!=model.config.output_dim:
        raise ValueError("Enrollment requires an empty trained index of matching dimension")
    if not gallery or len({str(r['reference_id']) for r in gallery})!=len(gallery):
        raise ValueError("Enrollment requires distinct reference identifiers")
    frames,ends,names=[],[],[]
    count=0
    descriptor_digest=hashlib.sha256()
    c=model.config
    for reference in gallery:
        with np.load(ref_path(cache,reference,c),allow_pickle=False) as z:
            raw,times=z["features"],z["times"]
        grid=np.rint((times-c.first_center_seconds)/(c.hop_length/c.sample_rate)).astype(np.int64)
        selected=np.flatnonzero(grid%factor==0)
        if np.any(grid[selected]<0) or np.any(grid[selected]>np.iinfo(np.uint32).max):
            raise ValueError("Physical frame index exceeds compact encoding")
        vectors=np.ascontiguousarray(model.transform(raw[selected]),dtype=np.float32)
        faiss_index.add(vectors)
        descriptor_digest.update(memoryview(vectors).cast("B"))
        count+=len(vectors)
        frames.append(grid[selected].astype(np.uint32));ends.append(count);names.append(reference["reference_id"])
    return CompactTemporalIndex(faiss_index,np.concatenate(frames),ends,names,c,factor),descriptor_digest.hexdigest()
