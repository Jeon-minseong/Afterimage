import argparse,json
from pathlib import Path
from .common import PACKAGE

def main():
    p=argparse.ArgumentParser(description='Afterimage reproducibility pipeline. API runs default to dry-run.')
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--videos',required=True,type=Path);g.add_argument('--weights',required=True,type=Path);g.add_argument('--out',required=True,type=Path);g.add_argument('--ids',nargs='+');g.add_argument('--dataset',default='custom')
    i=sub.add_parser('import-fixed');i.add_argument('--project',required=True,type=Path);i.add_argument('--out',required=True,type=Path);i.add_argument('--ids',nargs='+')
    v=sub.add_parser('validate');v.add_argument('--inputs',required=True,type=Path)
    a=sub.add_parser('prepare-inference');a.add_argument('--inputs',required=True,type=Path);a.add_argument('--out',required=True,type=Path);a.add_argument('--config',type=Path,default=PACKAGE/'configs/experiment.json');a.add_argument('--repeats',type=int,choices=[1,3]);a.add_argument('--ids',nargs='+')
    s=sub.add_parser('prepare-scoring');s.add_argument('--answers',required=True,type=Path);s.add_argument('--gt',type=Path,default=PACKAGE/'gt/references.json');s.add_argument('--out',required=True,type=Path);s.add_argument('--config',type=Path,default=PACKAGE/'configs/experiment.json');s.add_argument('--allow-partial',action='store_true')
    r=sub.add_parser('run');r.add_argument('--plan',required=True,type=Path);r.add_argument('--inputs',type=Path);r.add_argument('--out',required=True,type=Path);r.add_argument('--ledger',type=Path,default=PACKAGE/'runs/budget.json');r.add_argument('--execute',action='store_true');r.add_argument('--approve-plan');r.add_argument('--budget-openai',type=float,default=0);r.add_argument('--budget-anthropic',type=float,default=0)
    t=sub.add_parser('summarize');t.add_argument('--scores',required=True,type=Path);t.add_argument('--out',required=True,type=Path)
    m=sub.add_parser('metrics');m.add_argument('--inputs',required=True,type=Path);m.add_argument('--answers',required=True,type=Path);m.add_argument('--scores',type=Path);m.add_argument('--frame-counts',type=Path);m.add_argument('--out',required=True,type=Path)
    args=vars(p.parse_args());command=args.pop('command')
    if command=='generate':
        from .generate import generate
        result=generate(**args)
    elif command=='import-fixed':
        from .generate import import_fixed
        result=import_fixed(**args)
    elif command=='validate':
        from .common import validate_inputs
        result=validate_inputs(args['inputs'])[1]
    elif command=='metrics':
        from .metrics import export
        result=export(**args)
    else:
        from . import experiment as e
        if command=='prepare-inference':args['config_path']=args.pop('config');result=e.prepare(**args)
        elif command=='prepare-scoring':args['config_path']=args.pop('config');result=e.prepare_scores(**args)
        elif command=='summarize':result=e.summarize(**args)
        else:
            live=args.pop('execute');approved=args.pop('approve_plan')
            if live and not approved:p.error('--execute also requires --approve-plan HASH')
            caps={'openai':args.pop('budget_openai'),'anthropic':args.pop('budget_anthropic')}
            args['ledger_path']=args.pop('ledger');result=e.execute(**args,caps=caps,approved_sha=approved if live else None)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
