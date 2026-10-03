"""Sparse physical-clock conformance, independent of benchmark outcomes."""
import unittest
import numpy as np
import faiss
from neural_density_controls import PhysicalPeakNetIndex
from peaknet_baseline import PeakNetSequenceIndex

class SparsePeakClockTests(unittest.TestCase):
    def setUp(self):
        faiss.omp_set_num_threads(1)

    def index(self, vectors, hop, prefix):
        # Orthogonal/negative prefixes cannot enter exact query neighbors.
        earlier=np.zeros((prefix,vectors.shape[1]),dtype=np.float32)
        earlier[:,0]=-1
        allvectors=np.concatenate([earlier,vectors])
        names=['unrelated','target'] if prefix else ['target']
        starts=np.array([0,prefix],dtype=np.int64) if prefix else np.array([0],dtype=np.int64)
        ends=np.array([prefix,prefix+len(vectors)],dtype=np.int64) if prefix else np.array([len(vectors)],dtype=np.int64)
        index=PhysicalPeakNetIndex(names,allvectors,starts,ends,hop)
        index.audit_times=np.concatenate([np.arange(prefix)*hop,np.arange(len(vectors))*hop])
        index.build_index()
        return index

    def test_sparse_half_grid_ties_ignore_unrelated_prefix(self):
        cosines=np.full(80,.1,dtype=np.float32)
        cosines[16:25]=[.1,.8,.5,.9,1.,.4,.3,.6,.2]
        vectors=np.zeros((80,4),dtype=np.float32)
        vectors[:,0]=cosines;vectors[:,1]=np.sqrt(1-cosines**2)
        query=np.zeros((9,4),dtype=np.float32);query[:,0]=1
        times=np.arange(9)*.5
        for hop in [1.,2.]:
            hits=[self.index(vectors,hop,p).search_times(query,times,1) for p in [0,1,2,3]]
            first=hits[0]
            for hit in hits:
                for key in ['reference_id','start_s','score','estimated_reference_time_scale','unique_reference_evidence_frames']:
                    self.assertEqual(hit[key],first[key])
                self.assertEqual(hit['reference_id'],'target')
                self.assertEqual(hit['estimated_reference_time_scale'],1.)

    def test_sparse_physical_localization_and_scale_ignore_prefix(self):
        rng=np.random.default_rng(13)
        vectors=rng.normal(size=(80,128)).astype(np.float32)
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        times=np.arange(9)*.5
        for hop in [1.,2.]:
            for scale in [.75,1.,1.5]:
                origin=20*hop
                frame=np.rint((origin+times*scale)/hop).astype(int)
                hits=[self.index(vectors,hop,p).search_times(vectors[frame],times,1) for p in [0,1,2,3]]
                for hit in hits:
                    self.assertEqual(hit['reference_id'],'target')
                    self.assertEqual(hit['start_s'],origin)
                    self.assertEqual(hit['score'],hits[0]['score'])
                    self.assertEqual(hit['estimated_reference_time_scale'],hits[0]['estimated_reference_time_scale'])
                    self.assertGreaterEqual(hit['estimated_reference_time_scale'],.5)
                    self.assertLessEqual(hit['estimated_reference_time_scale'],2.)

    def test_full_density_matches_native_adapter(self):
        rng=np.random.default_rng(42)
        vectors=rng.normal(size=(80,128)).astype(np.float32)
        vectors/=np.linalg.norm(vectors,axis=1,keepdims=True)
        times=np.arange(9)*.5
        for prefix in [0,1,2,3]:
            for scale in [.75,1.,1.5]:
                frame=np.rint(20+times*scale/.5).astype(int)
                index=self.index(vectors,.5,prefix)
                native=PeakNetSequenceIndex(index.reference_ids,index.embeddings,index.starts,index.ends,.5)
                native._index=index._index
                old=native.search(vectors[frame],1)
                new=index.search_times(vectors[frame],times,1)
                for key in ['reference_id','start_s','score','estimated_reference_time_scale']:
                    self.assertEqual(old[key],new[key])

    def test_native_inverse_scale_two_half_ties_ignore_prefix(self):
        # Even query rows identify distinct target rows, fixing inverse scale2.
        # Odd rows identify a separate complete distractor. An unrelated prefix
        # never appears in any nearest neighbor and must not change the winner.
        dimension=128
        target=np.zeros((80,dimension),dtype=np.float32)
        for row in range(80):
            cosine=.7 if row%2==0 else .95
            target[row,0]=cosine
            target[row,row+1]=np.sqrt(1-cosine*cosine)
        query=np.zeros((9,dimension),dtype=np.float32)
        for row in range(9):
            if row%2==0:query[row]=target[20+row//2]
            else:
                query[row,0]=np.sqrt(1-.01**2)
                query[row,100+row]=.01
        distractor=np.zeros((80,dimension),dtype=np.float32)
        distractor[:,0]=-1
        for row in range(9):
            if row%2==0:distractor[row,0]=1
            else:distractor[row]=query[row]
        hits=[]
        for prefix in [0,1,2,3]:
            earlier=np.zeros((prefix,dimension),dtype=np.float32)
            earlier[:,0]=-1
            embeddings=np.concatenate([earlier,target,distractor])
            names=['unrelated','target','distractor'] if prefix else ['target','distractor']
            starts=np.array([0,prefix,prefix+80]) if prefix else np.array([0,80])
            ends=np.array([prefix,prefix+80,prefix+160]) if prefix else np.array([80,160])
            index=PeakNetSequenceIndex(names,embeddings,starts,ends,.5)
            index.build_index()
            _,neighbors=index._index.search(query,1)
            self.assertTrue(np.all(neighbors>=prefix))
            pairs={row:int(values[0]) for row,values in enumerate(neighbors) if prefix<=values[0]<prefix+80}
            q=np.array(list(pairs));r=np.array(list(pairs.values()))
            qdiff=q[:,None]-q;rdiff=r[:,None]-r
            ratio=np.divide(qdiff,rdiff,out=np.zeros_like(qdiff,dtype=float),where=rdiff!=0)
            self.assertEqual(float(ratio[ratio!=0].mean()),2.)
            hits.append(index.search(query,1))
        for hit in hits:
            for key in ['reference_id','start_s','score','estimated_reference_time_scale']:
                self.assertEqual(hit[key],hits[0][key])

        # The other endpoint, inverse0.5, uses integer grid steps and is a
        # contrast control: all exact target embeddings localize at10 seconds.
        fast_query=target[20+2*np.arange(9)]
        for prefix in [0,1,2,3]:
            physical=self.index(target,.5,prefix)
            index=PeakNetSequenceIndex(physical.reference_ids,physical.embeddings,
                physical.starts,physical.ends,.5)
            index._index=physical._index
            hit=index.search(fast_query,1)
            self.assertEqual(hit['reference_id'],'target')
            self.assertEqual(hit['start_s'],10.)
            self.assertEqual(hit['estimated_reference_time_scale'],2.)

if __name__=='__main__':unittest.main()
