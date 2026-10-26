import base64,copy,json,socket,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from afterimage.common import PACKAGE,CONDITIONS,write,read,sha,digest
from afterimage import experiment as e

class FixtureAPI:
    def __init__(self):self.calls=0;self.uploads=0;self.payloads=[];self.fail=False
    def upload(self,provider,path,h):self.uploads+=1;return 'fixture-'+h
    def count(self,provider,payload):return 1000
    def infer(self,provider,payload):
        self.calls+=1;self.payloads.append(payload)
        if self.fail:raise TimeoutError('fixture')
        content=payload.get('input',payload.get('messages'))[0]['content'];prompt=content[0]['text']
        grade='REFERENCE AND CANDIDATE DATA:' in prompt
        text=json.dumps({'score':4,'reason':'Fixture only.','needs_human_review':False} if grade else {'description':'A person walks through the scene.'})
        if provider=='openai':return {'id':'fixture','model':payload['model'],'status':'completed','service_tier':'default','usage':{'input_tokens':1000,'output_tokens':100},'output':[{'type':'message','content':[{'type':'output_text','text':text}]}]},'fixture'
        return {'id':'fixture','model':payload['model'],'stop_reason':'end_turn','usage':{'input_tokens':1000,'output_tokens':100,'service_tier':'standard'},'content':[{'type':'text','text':text}]},'fixture'

class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.data=self.root/'inputs';self.data.mkdir()
        image=self.data/'a.png';image.write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII='))
        i={'path':'a.png','sha256':sha(image),'frame_id':0}
        write(self.data/'manifest.json',{'videos':[{'video_id':'fixture_video','dataset':'kisa','source_sha256':'fixture','M':2,'J':1,'conditions':{c:[i]*(4 if c in CONDITIONS[:2] else 3) for c in CONDITIONS}}]})
        self.config=PACKAGE/'configs/experiment.json';self.plan=self.root/'plan';self.answers=self.root/'answers';self.ledger=self.root/'ledger.json';self.fake=FixtureAPI()
        self.socket=patch.object(socket,'socket',side_effect=AssertionError('Network forbidden'));self.socket.start()
    def tearDown(self):self.socket.stop();self.temp.cleanup()
    def prep(self,repeats=1):return e.prepare(self.data,self.plan,self.config,repeats)
    def run_api(self):return e.execute(self.plan,self.data,self.answers,self.ledger,{'openai':100,'anthropic':100},read(self.plan/'plan.json')['plan_sha256'],self.fake)
    def test_end_to_end_inference_grade_aggregate_and_resume(self):
        p=self.prep();self.assertEqual(p['requests'],16);self.assertEqual(self.run_api()['status'],'completed')
        self.assertEqual(self.fake.calls,16);self.run_api();self.assertEqual(self.fake.calls,16)
        for payload in self.fake.payloads:self.assertNotIn('reference',json.dumps(payload).lower())
        gt=self.root/'gt.json';write(gt,{'status':'fixture_only','items':[{'video_id':'fixture_video','task':t,'reference':'A person walks.'} for t in ['GT1','GT2']]})
        plan=self.root/'score_plan';s=e.prepare_scores(self.answers,gt,plan,self.config)
        scores=self.root/'scores';r=e.execute(plan,None,scores,self.ledger,{'openai':100},s['plan_sha256'],self.fake)
        self.assertEqual(r['status'],'completed');self.assertEqual(self.fake.calls,32)
        for payload in self.fake.payloads[16:]:
            text=json.dumps(payload);self.assertNotIn('candidate_model',text);self.assertNotIn('candidate_provider',text);self.assertNotIn('fixture_video',text)
        result=e.summarize(scores,self.root/'summary');self.assertEqual(result['individual_scores'],16)
        self.assertTrue(all(g['mean_score']==4 for g in read(self.root/'summary/summary.json')['groups']))
        self.assertGreater(r['cumulative_usd']['openai'],.05)
        from afterimage.metrics import export
        metrics = self.root/'metrics'
        self.assertEqual(export(self.data,self.answers,metrics,scores=scores)['scores'],16)
        measured=read(metrics/'summary.json')
        self.assertEqual(measured['counts'][0]['pooled_VLM_input_reduction'],.25)
        self.assertIsNone(measured['counts'][0]['N'])
        # A validly hashed score file must still refer to the exact answer bytes.
        idx=read(scores/'index.json');item=idx['items'][0];path=scores/item['path'];bad=read(path)
        bad['request']['candidate_record_sha256']='another-answer';write(path,bad)
        item['sha256']=sha(path);write(scores/'index.json',idx)
        with self.assertRaisesRegex(ValueError,'different answer'):
            export(self.data,self.answers,self.root/'bad_metrics',scores=scores)
    def test_dry_run_no_credentials_or_provider(self):
        self.prep()
        with patch.object(e,'Remote',side_effect=AssertionError('Provider must not initialize')):
            self.assertEqual(e.execute(self.plan,self.data,self.answers,self.ledger,{})['API_calls'],0)
    def test_budget_no_inference(self):
        self.prep();r=e.execute(self.plan,self.data,self.answers,self.ledger,{'openai':.000001,'anthropic':.000001},read(self.plan/'plan.json')['plan_sha256'],self.fake)
        self.assertEqual(r['status'],'budget_stopped');self.assertEqual(self.fake.calls,0)
    def test_unknown_not_retried(self):
        self.prep();self.fake.fail=True
        with self.assertRaises(RuntimeError):self.run_api()
        self.fake.fail=False
        with self.assertRaises(ValueError):self.run_api()
        self.assertEqual(self.fake.calls,1)
    def test_changed_input_blocked(self):
        self.prep();(self.data/'a.png').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.run_api()
        self.assertEqual(self.fake.calls,0)
    def test_repeat_jobs_not_deduplicated(self):
        self.assertEqual(self.prep(3)['requests'],48)
        self.run_api();self.assertEqual(self.fake.calls,48);self.assertEqual(self.fake.uploads,2)
    def test_missing_gt_stops_before_scoring(self):
        self.prep();self.run_api();gt=self.root/'gt.json';write(gt,{'status':'fixture','items':[]})
        with self.assertRaises(ValueError):e.prepare_scores(self.answers,gt,self.root/'score_plan',self.config)

if __name__=='__main__':unittest.main(verbosity=2)
