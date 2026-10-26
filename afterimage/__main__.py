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
    b=sub.add_parser('prepare-batch');b.add_argument('--plan',required=True,type=Path);b.add_argument('--inputs',type=Path);b.add_argument('--out',required=True,type=Path);b.add_argument('--provider',required=True,choices=['openai','anthropic'])
    b=sub.add_parser('submit-batch');b.add_argument('--pack',required=True,type=Path);b.add_argument('--ledger',required=True,type=Path);b.add_argument('--budget',required=True,type=float);b.add_argument('--execute',action='store_true');b.add_argument('--approve-pack');b.add_argument('--limit',type=int)
    b=sub.add_parser('collect-batch');b.add_argument('--pack',required=True,type=Path);b.add_argument('--plan',required=True,type=Path);b.add_argument('--inputs',type=Path);b.add_argument('--out',required=True,type=Path);b.add_argument('--ledger',required=True,type=Path)
    b=sub.add_parser('watch-batches');b.add_argument('--packs',required=True,nargs='+',type=Path);b.add_argument('--plan',required=True,type=Path);b.add_argument('--inputs',type=Path);b.add_argument('--out',required=True,type=Path);b.add_argument('--ledger',required=True,type=Path);b.add_argument('--interval',type=int,default=60);b.add_argument('--hours',type=float,default=25)
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
    elif command in ('prepare-batch','submit-batch','collect-batch','watch-batches'):
        from . import batch
        if command=='prepare-batch':result=batch.prepare(**args)
        elif command=='collect-batch':args['ledger_path']=args.pop('ledger');result=batch.collect(**args)
        elif command=='watch-batches':args['ledger_path']=args.pop('ledger');result=batch.watch(**args)
        else:
            execute=args.pop('execute');approved=args.pop('approve_pack')
            if execute:
                if not approved:p.error('--execute requires --approve-pack HASH')
                args['ledger_path']=args.pop('ledger');args['cap']=args.pop('budget');result=batch.submit(**args,approved=approved)
            else:
                pack=batch.load_pack(args['pack']);result={'mode':'dry_run','requests':pack['requests'],'reserved_usd':pack['reserved_usd'],'pack_sha256':pack['pack_sha256'],'API_calls':0}
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
