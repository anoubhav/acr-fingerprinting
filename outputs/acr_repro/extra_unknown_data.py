"""Acquire deterministic, gallery-absent FMA tracks by authenticated HTTPS ranges.

The official archive is not downloaded whole: selected member CRCs and local
SHA256 digests are checked. Do not claim the whole-archive SHA1 was verified.
Selection and decoding validity are independent of fingerprint predictions.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import threading
import zipfile
import zlib
import requests
import numpy as np

URL="https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
EXPECTED_SIZE=23825005356
PUBLISHED_SHA1="c67b69ea232021025fca9231fc1c7c1a063ab50b"

class HTTPRange(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.position=url,size,0
        self.session=requests.Session()
        self.cache_start,self.cache=-1,b""
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        if whence==0: self.position=offset
        elif whence==1: self.position+=offset
        elif whence==2: self.position=self.size+offset
        else: raise ValueError("Invalid seek")
        if self.position<0: raise ValueError("Negative seek")
        return self.position
    def read(self,n=-1):
        if n<0: n=self.size-self.position
        n=min(n,self.size-self.position)
        if n<=0:return b""
        end=self.position+n
        if not self.cache_start<=self.position or end>self.cache_start+len(self.cache):
            request_end=min(self.size,self.position+max(n,65536))-1
            response=self.session.get(self.url,headers={"Range":f"bytes={self.position}-{request_end}"},timeout=90)
            response.raise_for_status()
            expected=f"bytes {self.position}-{request_end}/{self.size}"
            if response.status_code!=206 or response.headers.get("Content-Range")!=expected:
                raise ValueError("Provider did not return the exact requested archive range")
            self.cache_start,self.cache=self.position,response.content
            if len(self.cache)!=request_end-self.position+1:raise ValueError("Short range read")
        data=self.cache[self.position-self.cache_start:end-self.cache_start]
        self.position=end
        return data

def metadata_rows(metadata_zip):
    with zipfile.ZipFile(metadata_zip) as z:
        with io.TextIOWrapper(z.open("fma_metadata/tracks.csv"),encoding="utf-8") as f:
            reader=csv.reader(f)
            top,bottom=next(reader),next(reader)
            next(reader)
            keys=list(zip(top,bottom))
            out=[]
            for row in reader:
                m=dict(zip(keys,row))
                if m[("set","subset")] not in ("small","medium"):continue
                out.append({"reference_id":f"{int(row[0]):06d}",
                    "artist_id":m[("artist","id")],"artist_name":m[("artist","name")],
                    "title":m[("track","title")],"license":m[("track","license")],
                    "subset":m[("set","subset")]})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",required=True,type=Path)
    p.add_argument("--exclude-protocols",required=True,nargs="+",type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--count",type=int,default=5000)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args()
    excluded={r["reference_id"] for f in a.exclude_protocols for r in json.loads(f.read_text())["references"]}
    rows=[r for r in metadata_rows(a.metadata) if r["reference_id"] not in excluded]
    rows.sort(key=lambda r:hashlib.sha256(("acr-extra-unknown-v1:"+r["reference_id"]).encode()).digest())
    a.output.mkdir(parents=True,exist_ok=True)
    reader=HTTPRange(URL,EXPECTED_SIZE)
    with zipfile.ZipFile(reader) as z:
        members={Path(x.filename).stem:x for x in z.infolist() if x.filename.endswith(".mp3")}
    candidates=rows[:a.count+100]
    state=threading.local()
    def download(row):
        rid=row["reference_id"]
        if not hasattr(state,"zip"):
            state.zip=zipfile.ZipFile(HTTPRange(URL,EXPECTED_SIZE))
        member=members[rid]
        if PurePosixPath(member.filename).is_absolute() or ".." in PurePosixPath(member.filename).parts or member.file_size>20_000_000:
            raise ValueError("Unsafe archive member")
        path=a.output/"audio"/f"{rid}.mp3"
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            data=state.zip.read(member.filename) # zipfile checks the member CRC.
            path.write_bytes(data)
        else:
            if path.stat().st_size!=member.file_size:raise ValueError("Cached member size mismatch")
        if zlib.crc32(path.read_bytes())!=member.CRC:
            raise ValueError(f"Cached archive member CRC mismatch: {rid}")
        decoded=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-f","f32le","-ar","8000","-ac","1","pipe:1"],
                               check=False,capture_output=True)
        duration=len(decoded.stdout)/4/8000
        if decoded.returncode!=0 or duration<5:
            return dict(row,valid=False,decoded_duration_s=duration,
                exclusion_reason="decode failure or less than 5 seconds of audio",
                decoder_warning=decoded.stderr.decode(errors="replace")[:500])
        return dict(row,valid=True,path=str(path.resolve()),decoded_duration_s=duration,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),member_crc32=member.CRC,
            decoder_warning=decoded.stderr.decode(errors="replace")[:500])
    valid,invalid=[],[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,row in enumerate(pool.map(download,candidates)):
            (valid if row["valid"] else invalid).append(row)
            if (i+1)%250==0:print(f"Extended unknown acquisition {i+1}/{len(candidates)}",flush=True)
    if len(valid)<a.count:raise ValueError("Insufficient valid source tracks")
    selected=valid[:a.count]
    # First 20 percent calibrate; the remainder are held-out unknown test sources.
    ncal=round(.2*a.count)
    for i,row in enumerate(selected):row["role"]="calibration_unknown" if i<ncal else "test_unknown"
    result={"source_url":URL,"archive_size":EXPECTED_SIZE,"published_archive_sha1":PUBLISHED_SHA1,
        "whole_archive_hash_verified":False,"verification":"HTTPS range content bounds, ZIP CRC32, per-member SHA256",
        "selection_seed":"acr-extra-unknown-v1","excluded_source_ids":sorted(excluded),
        "selected_count":len(selected),"calibration_count":ncal,"test_count":len(selected)-ncal,
        "invalid_candidates":invalid,"tracks":selected}
    (a.output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("tracks","excluded_source_ids","invalid_candidates")},indent=2))

if __name__=="__main__":main()
