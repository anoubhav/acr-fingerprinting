"""Freeze a common calibration-only PCM subset for warm cost measurements."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from neural_baseline import centered_query

def prepare(protocol,output,count=50):
    manifest=json.loads(Path(protocol).read_text())
    candidates=[q for q in manifest['queries'] if q.get('evaluate',True) and q['role']=='calibration_known']
    candidates.sort(key=lambda q:hashlib.sha256(('acr-common-profiling-v1:'+q['query_id']).encode()).digest())
    selected=[centered_query(q,5) for q in candidates[:count]]
    waveforms=[]
    for q in selected:
        raw=subprocess.run(['ffmpeg','-v','error','-i',q['path'],'-f','f32le','-ac','1','-ar','8000','pipe:1'],check=True,capture_output=True).stdout
        audio=np.frombuffer(raw,dtype='<f4')
        start=round(q['start_s']*8000); length=round(q['duration_s']*8000)
        wave=audio[start:start+length].copy()
        if length!=40000 or len(wave)!=40000:raise ValueError('Profiling requires a complete5s crop')
        waveforms.append(wave)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    np.save(output/'audio.npy',np.stack(waveforms))
    result={'selection_seed':'acr-common-profiling-v1','selection_pool':'147 calibration-known crops',
            'input_count':len(selected),'sample_rate':8000,'duration_s':5,
            'audio_path':str((output/'audio.npy').resolve()),
            'audio_sha256':hashlib.sha256((output/'audio.npy').read_bytes()).hexdigest(),
            'protocol_sha256':hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
            'queries':selected,'decoder':'FFmpeg8k mono float32; nearest-sample crop rounding'}
    (output/'inputs.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('protocol',type=Path);p.add_argument('output',type=Path);p.add_argument('--count',type=int,default=50)
    a=p.parse_args();r=prepare(a.protocol,a.output,a.count);print(r['audio_path'],r['audio_sha256'])
