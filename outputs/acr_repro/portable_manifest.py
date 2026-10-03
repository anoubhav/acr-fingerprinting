"""Export relative provenance paths, or materialize them in a new workspace."""
from __future__ import annotations
import argparse,json
from pathlib import Path

PATH_KEYS={'path','root','cache_dir','cache_path','upstream_path','model_dir','source_path',
           'input_path','output_path','reference_cache','query_cache','protocol_path',
           'pcm_path','audio_path','audfprint_audio','result_path','model_path',
           'pca_model_path','profile_path','inputs_path','manifest_path','index_path'}

def portable(value,workspace):
    prefix=str(Path(workspace).resolve())
    return _portable(value,prefix)

def _portable(value,prefix):
    # Resolve the workspace once, rather than once per JSON scalar in complete
    # prediction files. This leaves the exported values unchanged.
    if isinstance(value,dict):return {k:_portable(v,prefix) for k,v in value.items()}
    if isinstance(value,list):return [_portable(x,prefix) for x in value]
    if isinstance(value,str):
        value=value.replace(prefix+'/','')
        if value==prefix:return '.'
        # Runtime package provenance outside the task is descriptive, not an
        # executable input path. Do not disclose a local account name.
        if value.startswith('/Users/'):
            parts=value.split('/')
            if len(parts)>2:value='/'.join(parts[:2]+['USER']+parts[3:])
    return value

def materialize(value,workspace,key=None):
    if isinstance(value,dict):return {k:materialize(v,workspace,k) for k,v in value.items()}
    if isinstance(value,list):return [materialize(x,workspace,key) for x in value]
    if isinstance(value,str) and key in PATH_KEYS and not value.startswith(('http://','https://','/')):
        return str((Path(workspace)/value).resolve())
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--workspace',type=Path,default=Path('.'));p.add_argument('--materialize',action='store_true')
    a=p.parse_args();value=json.loads(a.input.read_text())
    out=materialize(value,a.workspace) if a.materialize else portable(value,a.workspace)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
