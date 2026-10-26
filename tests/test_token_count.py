import base64,copy,tempfile,unittest
from pathlib import Path
from afterimage.providers import Remote,json_output

class CountTransportTest(unittest.TestCase):
    def test_large_claude_count_chunks_keep_all_appearances(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'large.png';p.write_bytes(b'x'*3000)
            remote=Remote(());remote.count_payload_limit_bytes=40000;remote.bind_file('anthropic','a',p)
            payload={'model':'candidate','thinking':{'type':'adaptive'},'messages':[{'role':'user','content':[{'type':'text','text':'prompt'}]+[{'type':'image','source':{'type':'file','file_id':'a'}} for _ in range(20)]}]}
            calls=[]
            def request(provider,endpoint,data):
                self.assertLess(len(data),40000);calls.append(json.loads(data));return {'input_tokens':123},None
            remote.request=request
            count=remote.count('anthropic',payload)
            self.assertGreater(len(calls),1);self.assertEqual(count,123*len(calls))
            self.assertEqual(sum(len(c['messages'][0]['content'])-1 for c in calls),20)
            self.assertEqual(len(payload['messages'][0]['content']),21)
            self.assertEqual(remote.last_count_metadata['mode'],'sum_of_chunk_counts_estimate')
    def test_openai_inline_exact_bytes_without_upload_or_ledger_bloat(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.png';p.write_bytes(b'PNG_A');h=hashlib.sha256(p.read_bytes()).hexdigest()
            remote=Remote(())
            remote.request=lambda *args: self.fail('Should not call the Files API')
            ref=remote.upload('openai',p,h);remote.bind_file('openai',ref,p)
            source={'input':[{'content':[{'type':'input_image','file_id':ref,'detail':'high'}]*2}]}
            materialized=remote.materialize('openai',source)
            for b in materialized['input'][0]['content']:
                self.assertEqual(base64.b64decode(b['image_url'].split(',')[1]),b'PNG_A')
                self.assertNotIn('file_id',b)
            self.assertEqual(source['input'][0]['content'][0]['file_id'],ref)
    def test_only_single_complete_fence_is_normalized(self):
        self.assertEqual(json_output('```json\n{"description":"runs"}\n```'),('{"description":"runs"}','single_json_fence_removed'))
        text='Here is my reasoning.\n```json\n{}\n```'
        self.assertEqual(json_output(text),(text,None))
    def test_claude_counts_exact_bytes_and_duplicate_order_without_mutating_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a.png';a.write_bytes(b'PNG_A')
            b=Path(tmp)/'b.png';b.write_bytes(b'PNG_B')
            remote=Remote(())
            remote.bind_file('anthropic','a',a);remote.bind_file('anthropic','b',b)
            payload={'model':'candidate','thinking':{'type':'adaptive'},'messages':[{'role':'user','content':[{'type':'text','text':'instructions'}]+[{'type':'image','source':{'type':'file','file_id':x}} for x in ['a','b','a']]}]}
            original=copy.deepcopy(payload)
            import json
            observed=[]
            def request(provider,endpoint,data):
                observed.append(json.loads(data));return {'input_tokens':123},None
            remote.request=request
            self.assertEqual(remote.count('anthropic',payload),123)
            blocks=observed[0]['messages'][0]['content'][1:]
            self.assertEqual([base64.b64decode(x['source']['data']) for x in blocks],[b'PNG_A',b'PNG_B',b'PNG_A'])
            self.assertEqual(payload,original)

if __name__=='__main__':unittest.main()
