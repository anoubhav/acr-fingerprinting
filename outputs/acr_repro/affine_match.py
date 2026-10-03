"""Optional NN-derived affine-time sensitivity; no annotated tempo is used.

This is a newly specified matcher, not recovered production code. The original
unit-slope matcher remains the main baseline and is included as a candidate.
"""
from __future__ import annotations
import numpy as np


def _pose_scores(scales, offsets, qi, reference_times, query_times, distances, tolerance):
    residual = reference_times[None,:] - scales[:,None]*query_times[qi][None,:] - offsets[:,None]
    inlier = np.abs(residual) <= tolerance+1e-12
    rows = len(query_times)
    flat = np.full(len(scales)*rows,np.inf,dtype=np.float64)
    positions = (np.arange(len(scales))[:,None]*rows+qi[None,:]).ravel()
    np.minimum.at(flat,positions,np.where(inlier,distances[None,:],np.inf).ravel())
    best = flat.reshape(len(scales),rows)
    valid = np.isfinite(best)
    counts = valid.sum(axis=1)
    means = np.divide(np.where(valid,best,0).sum(axis=1),counts,
                      out=np.full(len(scales),np.inf),where=counts>0)
    return counts,means,inlier


def match_affine(index,queries,query_times,top_k=5,tolerance_sec=.12,
                 min_scale=.5,max_scale=2.,proposals=64,min_pair_span_sec=1.,
                 seed=20261002,limit=1):
    """Estimate time scale from same-content nearest-neighbour correspondences.

    Fixed pair proposals use query-time separation >=1s and scales in[.5,2].
    Each query contributes at most one vote. A final least-squares refinement
    uses only inlier nearest-neighbour times and is retained if it improves the
    votes/distance ranking. The old unit-slope candidate can never be removed.
    """
    qt=np.asarray(query_times,dtype=np.float64)
    if not len(qt):
        return []
    if not 0<min_scale<=1<=max_scale or proposals<1:
        raise ValueError("Invalid affine matching range/proposals")
    unit=index.match(queries,qt,top_k=top_k,tolerance_sec=tolerance_sec,limit=len(index.content_ids))
    best_by_content={row["content_id"]:{**row,"time_scale":1.,"matcher":"unit_candidate"} for row in unit}
    distance,ids=index.search(queries,top_k)
    qi=np.repeat(np.arange(len(qt)),ids.shape[1])
    ri,di=ids.ravel(),distance.ravel()
    valid=(ri>=0)&(ri<len(index.vectors))&np.isfinite(di)
    qi,ri,di=qi[valid],ri[valid],di[valid]
    for code in np.unique(index.content_codes[ri]):
        mask=index.content_codes[ri]==code
        qindices,rt,ds=qi[mask],index.times[ri[mask]],di[mask]
        if len(np.unique(qindices))<3 or np.ptp(qt[qindices])<min_pair_span_sec:
            continue
        rng=np.random.default_rng(seed+int(code))
        # The fixed limit also bounds work for very popular candidate contents.
        left=rng.integers(0,len(qindices),proposals*12)
        right=rng.integers(0,len(qindices),proposals*12)
        delta=qt[qindices[right]]-qt[qindices[left]]
        possible=np.abs(delta)>=min_pair_span_sec
        left,right,delta=left[possible],right[possible],delta[possible]
        scales=(rt[right]-rt[left])/delta
        possible=(scales>=min_scale)&(scales<=max_scale)
        left,right,scales=left[possible][:proposals],right[possible][:proposals],scales[possible][:proposals]
        if not len(scales):
            continue
        offsets=rt[left]-scales*qt[qindices[left]]
        count,means,inlier=_pose_scores(scales,offsets,qindices,rt,qt,ds,tolerance_sec)
        order=np.lexsort((np.abs(scales-1),means,-count))
        # Refine only the highest-ranked proposed pose, using one reference
        # correspondence per query row. No ground-truth source time is read.
        first=int(order[0])
        selected={}
        for j in np.flatnonzero(inlier[first]):
            old=selected.get(int(qindices[j]))
            if old is None or ds[j]<ds[old]:
                selected[int(qindices[j])]=j
        chosen=np.array(list(selected.values()),dtype=int)
        if len(chosen)>=3 and np.ptp(qt[qindices[chosen]])>=min_pair_span_sec:
            x,y=qt[qindices[chosen]],rt[chosen]
            centered=x-x.mean()
            refined_scale=float(np.sum(centered*(y-y.mean()))/np.sum(centered*centered))
            if min_scale<=refined_scale<=max_scale:
                refined_offset=float(y.mean()-refined_scale*x.mean())
                scales=np.r_[scales,refined_scale]
                offsets=np.r_[offsets,refined_offset]
                count,means,inlier=_pose_scores(scales,offsets,qindices,rt,qt,ds,tolerance_sec)
                order=np.lexsort((np.abs(scales-1),means,-count))
        best=int(order[0]);votes=int(count[best]);mean_d=float(means[best])
        cid=index.content_ids[int(code)]
        candidate={"content_id":cid,"votes":votes,"vote_fraction":votes/len(qt),
            "mean_distance":mean_d,"offset_sec":float(offsets[best]),
            "time_scale":float(scales[best]),"score":votes/len(qt)/(1+mean_d),
            "query_fingerprints":len(qt),"matcher":"NN_affine_estimate"}
        old=best_by_content.get(cid)
        if old is None or (votes,-mean_d)>(old["votes"],-old["mean_distance"]):
            best_by_content[cid]=candidate
    hits=list(best_by_content.values())
    hits.sort(key=lambda row:(-row["votes"],row["mean_distance"],abs(row["time_scale"]-1),row["content_id"]))
    return hits[:limit]
