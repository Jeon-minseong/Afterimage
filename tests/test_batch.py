import json, tempfile, unittest
from pathlib import Path
from test_pipeline import FixtureAPI
import test_pipeline as pipeline_fixture
from afterimage import batch
from afterimage.common import read, write, canonical, sha


class FakeBatch(FixtureAPI):
    def __init__(self):
        super().__init__();self.batches={};self.uploaded=None;self.submissions=0;self.raise_submit=False
    def upload_batch(self,path):self.uploaded=[json.loads(x) for x in Path(path).read_text().splitlines()];return 'file-input'
    def request(self,provider,endpoint,data):
        self.submissions+=1
        if self.raise_submit:raise RuntimeError('ambiguous simulated submission')
        body=json.loads(data);requests=self.uploaded if provider=='openai' else body['requests'];bid='batch_'+str(self.submissions)
        rows=[]
        for r in requests:
            raw,_=self.infer(provider,r.get('body',r.get('params')))
            rows.append({'custom_id':r['custom_id'],**({'response':{'status_code':200,'request_id':'fixture','body':raw},'error':None}
                        if provider=='openai' else {'result':{'type':'succeeded','message':raw}})})
        self.batches[bid]=(provider,rows)
        return {'id':bid},'fixture'
    def get_bytes(self,provider,endpoint):
        if endpoint.endswith('/content'):
            bid=endpoint.split('/')[-2].replace('output-','')
            return b''.join(canonical(r)+b'\n' for r in reversed(self.batches[bid][1]))
        if endpoint.endswith('/results'):
            bid=endpoint.split('/')[-2]
            return b''.join(canonical(r)+b'\n' for r in reversed(self.batches[bid][1]))
        bid=endpoint.split('/')[-1]
        return canonical({'id':bid,'status':'completed','output_file_id':'output-'+bid} if provider=='openai' else {'id':bid,'processing_status':'ended'})


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.fixture=pipeline_fixture.PipelineTest();self.fixture.setUp();self.f=self.fixture;self.f.prep(3)
        write(self.f.ledger,{'requests':{'prior':{'provider':'openai','status':'completed','cost_usd':.1}},'uploads':{}})
    def tearDown(self):self.f.tearDown()
    def test_both_providers_reverse_results_repeat_integrity_and_idempotence(self):
        for provider in ['openai','anthropic']:
            fake=FakeBatch();pack=self.f.root/provider
            meta=batch.prepare(self.f.plan,self.f.data,pack,provider,adapter=fake,max_requests=8)
            self.assertEqual(meta['requests'],24)
            batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake)
            count=fake.submissions
            batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake)
            self.assertEqual(fake.submissions,count)
            result=batch.collect(pack,self.f.plan,self.f.data,self.f.answers,self.f.ledger,adapter=fake)
            self.assertEqual(result['issues'],0)
            self.assertEqual(batch.collect(pack,self.f.plan,self.f.data,self.f.answers,self.f.ledger,adapter=fake)['newly_collected'],0)
        idx=read(self.f.answers/'index.json');self.assertTrue(idx['complete']);self.assertEqual(len(idx['items']),48)
        for item in idx['items']:
            r=read(self.f.answers/item['path']);self.assertEqual(r['pricing_multiplier'],.5)
            self.assertIn(r['request']['repeat'],[1,2,3]);self.assertEqual(r['status'],'completed')
    def test_budget_blocks_entire_chunk_before_submission(self):
        fake=FakeBatch();pack=self.f.root/'pack';meta=batch.prepare(self.f.plan,self.f.data,pack,'openai',adapter=fake)
        with self.assertRaisesRegex(ValueError,'budget'):
            batch.submit(pack,self.f.ledger,.1001,meta['pack_sha256'],adapter=fake)
        self.assertEqual(fake.submissions,0);self.assertIsNone(fake.uploaded)
    def test_unknown_submission_keeps_reservation_and_never_resubmits(self):
        fake=FakeBatch();pack=self.f.root/'pack';meta=batch.prepare(self.f.plan,self.f.data,pack,'openai',adapter=fake)
        fake.raise_submit=True
        with self.assertRaises(RuntimeError):batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake)
        with self.assertRaisesRegex(ValueError,'Unresolved'):batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake)
        self.assertEqual(fake.submissions,1)
        self.assertTrue(any(x['status']=='batch_submitting_unknown' and x['reserved_usd']>0 for x in read(self.f.ledger)['requests'].values()))
    def test_tampered_batch_body_rejected(self):
        fake=FakeBatch();pack=self.f.root/'pack';meta=batch.prepare(self.f.plan,self.f.data,pack,'anthropic',adapter=fake)
        (pack/'part_001.jsonl').write_text('{}')
        with self.assertRaisesRegex(ValueError,'body changed'):batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake)
        self.assertEqual(fake.submissions,0)
    def test_watch_collects_without_submitting_new_work(self):
        fake=FakeBatch();packs=[]
        for provider in ['openai','anthropic']:
            pack=self.f.root/provider;meta=batch.prepare(self.f.plan,self.f.data,pack,provider,adapter=fake)
            batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake);packs.append(pack)
        before=fake.submissions
        result=batch.watch(packs,self.f.plan,self.f.data,self.f.answers,self.f.ledger,adapter=fake)
        self.assertEqual(result['status'],'completed');self.assertEqual(result['completed_requests'],48)
        self.assertEqual(before,fake.submissions)

    def test_subset_packs_share_plan_without_duplicate_submission(self):
        jobs=[json.loads(x) for x in (self.f.plan/'requests.jsonl').read_text(encoding='utf-8').splitlines()]
        ids=[j['job_id'] for j in jobs if j['provider']=='anthropic']
        fake=FakeBatch()
        for n, subset in enumerate([ids[:12],ids[12:]]):
            pack=self.f.root/('subset'+str(n))
            meta=batch.prepare(self.f.plan,self.f.data,pack,'anthropic',adapter=fake,job_ids=subset,max_bytes=240_000_000)
            self.assertEqual(meta['requests'],12)
            batch.submit(pack,self.f.ledger,100,meta['pack_sha256'],adapter=fake)
            result=batch.collect(pack,self.f.plan,self.f.data,self.f.answers,self.f.ledger,adapter=fake)
            self.assertEqual(result['issues'],0)
        self.assertEqual(read(self.f.answers/'index.json')['completed_requests'],24)
        overlap=self.f.root/'overlap'
        meta=batch.prepare(self.f.plan,self.f.data,overlap,'anthropic',adapter=fake,job_ids=ids[:1])
        with self.assertRaisesRegex(ValueError,'already reserved'):
            batch.submit(overlap,self.f.ledger,100,meta['pack_sha256'],adapter=fake)

    def test_unknown_subset_and_provider_specific_size_rejected(self):
        fake=FakeBatch()
        with self.assertRaisesRegex(ValueError,'Unknown selected'):
            batch.prepare(self.f.plan,self.f.data,self.f.root/'bad','anthropic',adapter=fake,job_ids=['unknown'])
        with self.assertRaisesRegex(ValueError,'Unsafe chunk'):
            batch.prepare(self.f.plan,self.f.data,self.f.root/'big','openai',adapter=fake,max_bytes=240_000_000)


if __name__=='__main__':unittest.main()
