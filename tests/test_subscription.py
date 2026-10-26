import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from afterimage import subscription as s
from afterimage.common import sha

class SubscriptionTest(unittest.TestCase):
    def test_duplicate_bytes_remain_separate_ordered_slots(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'a.png').write_bytes(b'a');(r/'b.png').write_bytes(b'b');w=r/'w';w.mkdir()
            j={'model':'gpt-6.1-sol','images':[{'path':p,'sha256':sha(r/p)} for p in ['a.png','b.png','a.png']]}
            slots=s.stage_inputs(j,r,w)
            self.assertEqual([x['sha256'] for x in slots],[sha(r/'a.png'),sha(r/'b.png'),sha(r/'a.png')])
            args=s.command('codex',j,{'effort':'medium'},w,r/'out.json')
            self.assertEqual([args[i+1] for i,x in enumerate(args) if x=='-i'],['i0000.png','i0001.png','i0002.png'])
            self.assertNotIn('resume',args)
    def test_no_paid_key_fallback(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'fixture','CODEX_API_KEY':'fixture','ANTHROPIC_API_KEY':'fixture'}):
            self.assertTrue(all(k not in s.clean_env() for k in ['OPENAI_API_KEY','CODEX_API_KEY','ANTHROPIC_API_KEY']))
    def test_tools_and_extra_sessions_rejected(self):
        text=json.dumps({'description':'A person walks.'})
        events=[{'type':'thread.started','thread_id':'a'},
                {'type':'item.completed','item':{'type':'agent_message','text':text}},
                {'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}}]
        self.assertEqual(s.parse_events(events,text,'GT1')[0],'a')
        with self.assertRaises(ValueError):s.parse_events(events+[{'type':'item.completed','item':{'type':'command_execution'}}],text,'GT1')
        with self.assertRaises(ValueError):s.parse_events(events+[{'type':'thread.started','thread_id':'b'}],text,'GT1')
    def test_invalid_output_not_repaired(self):
        events=[{'type':'thread.started','thread_id':'a'},{'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':5}}]
        with self.assertRaises(ValueError):s.parse_events(events,'unsupported prose','GT1')
    def test_dry_run_never_spawns(self):
        with patch.object(s,'load_plan',return_value=({'plan_sha256':'x'},{'processing_mode':'codex_chatgpt_subscription'},[{'provider':'openai'}])),patch.object(s.subprocess,'run',side_effect=AssertionError('must not spawn')):
            result=s.execute('plan','inputs','out')
            self.assertEqual(result['mode'],'dry_run')

if __name__=='__main__':unittest.main()
