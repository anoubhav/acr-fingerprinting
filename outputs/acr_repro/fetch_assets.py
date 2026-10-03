"""Fetch the exact public assets used in this study; no proprietary media."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,tarfile,zipfile
from pathlib import Path,PurePosixPath
import requests
from benchmark_data import safe_extract

ASSETS={
 'pex_small':('https://pexafbtpublic.blob.core.windows.net/audio-fingerprinting-benchmark-toolkit/pexafb_hard_small.zip','pexafb_hard_small.zip','sha256','bced04c226099c0d35630d2bd925248d6ed40367374733bb9c7e7ba6758fe3d4'),
 'pex_medium':('https://pexafbtpublic.blob.core.windows.net/audio-fingerprinting-benchmark-toolkit/pexafb_hard_medium.zip','pexafb_hard_medium.zip','sha256','9efafec1e5bc695784b6ef673d31dbc130f884d3dd2522a4050a05e3874de650'),
 'librispeech':('https://openslr.trmal.net/resources/12/test-clean.tar.gz','librispeech_test_clean.tar.gz','md5','32fa31d27d2e1cad72775fee3f4849a9'),
 'fma_metadata':('https://os.unil.cloud.switch.ch/fma/fma_metadata.zip','fma_metadata.zip','sha1','f0df49ffe5f2a6008d7dc83c6915b31835dfe733'),
 'nmfp':('https://zenodo.org/records/15719945/files/nmfp-triplet.zip?download=1','nmfp-triplet.zip','md5','ee8a3358fc5e5cdd09d6d2245d395021'),
 'peaknet':('https://zenodo.org/records/15782389/files/peaknetfp_checkpoints.tar.gz?download=1','peaknetfp_checkpoints.tar.gz','md5','df23e1556a206fd72bfa99d0724d241f')}
REPOS={
 'audfprint':('https://github.com/dpwe/audfprint.git','cb03ba99feafd41b8874307f0f4e808a6ce34362','benchmarks/audfprint'),
 'nmfp':('https://github.com/raraz15/neural-music-fp.git','15c6f3bcdf6a6da1daddfe47a1ffa5a0d22deadc','upstream/neural-music-fp'),
 'peaknet':('https://github.com/guillemcortes/peaknetfp.git','d55071a644f20fd6448fb8ed33c0c115769fb7a5','upstream/peaknetfp'),
 'kapre':('https://github.com/keunwoochoi/kapre.git','ae8f5077ec9071e5529992702885de03c3321941','upstream/kapre')}

def digest(path,algorithm):
    with path.open('rb') as f:return hashlib.file_digest(f,algorithm).hexdigest()

def fetch(name,work):
    url,filename,algorithm,expected=ASSETS[name]
    if name=='nmfp':dest=work/'models/nmfp'
    elif name=='peaknet':dest=work/'models/peaknetfp'
    else:dest=work/'benchmarks'
    dest.mkdir(parents=True,exist_ok=True);archive=dest/filename
    if not archive.exists():
        temporary=archive.with_suffix(archive.suffix+'.partial')
        with requests.get(url,stream=True,timeout=120) as response:
            response.raise_for_status()
            with temporary.open('wb') as f:
                for chunk in response.iter_content(1024*1024):f.write(chunk)
        temporary.replace(archive)
    actual=digest(archive,algorithm)
    if actual!=expected:raise ValueError(f'Checksum mismatch for {name}: {actual}')
    if name.startswith('pex_'):
        safe_extract(archive,dest)
    elif name=='nmfp':safe_extract(archive,dest)
    elif name in ('librispeech','peaknet'):
        extract_to=dest/'librispeech' if name=='librispeech' else dest
        extract_to.mkdir(exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                p=PurePosixPath(member.name)
                if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk() or member.isdev():
                    raise ValueError('Unsafe TAR member')
            tar.extractall(extract_to,filter='data')
    # Metadata is read directly as a ZIP by the acquisition/audit scripts.
    print(json.dumps({'asset':name,'path':str(archive),'hash_algorithm':algorithm,'verified_hash':actual}))

def clone(name,work):
    url,revision,relative=REPOS[name];dest=work/relative
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone',url,str(dest)],check=True)
    subprocess.run(['git','-C',str(dest),'checkout',revision],check=True)
    actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
    if actual!=revision:raise ValueError('Wrong upstream revision')
    print(json.dumps({'repo':name,'path':str(dest),'revision':actual}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=Path('work'))
    p.add_argument('--assets',nargs='+',choices=list(ASSETS),default=['pex_small','pex_medium','fma_metadata'])
    p.add_argument('--repos',nargs='*',choices=list(REPOS),default=['audfprint'])
    a=p.parse_args()
    for name in a.assets:fetch(name,a.work)
    for name in a.repos:clone(name,a.work)

if __name__=='__main__':main()
