#!/usr/bin/env python3
"""Patch cloned whc-dt1d experiment configs to the v6 unified baseline policy.

This patch is explicit, deterministic, and copied into every task output so the
executed configuration remains auditable. It does not modify the DT1D method.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import yaml

CNN_ORDER=['linear','bitfit','ssf','dt1d','conv_r4','bam','residual','lora_conv','sidetune','full']
CNN_METHODS={
'linear': {'label':'Linear probing','args':{'tuning_method':'linear','freeze_backbone':True}},
'bitfit': {'label':'BitFit','args':{'tuning_method':'bitfit','freeze_backbone':True,'bitfit_train_head':True}},
'ssf': {'label':'SSF','args':{'tuning_method':'ssf','freeze_backbone':True,'ssf_init_scale':1.0,'ssf_init_shift':0.0}},
'dt1d': {'label':'DT1D-Adapter','args':{'tuning_method':'dt1d','freeze_backbone':True,'dt_cache_kernel':True}},
'conv_r4': {'label':'Conv-Adapter (r=4)','args':{'tuning_method':'conv','freeze_backbone':True,'adapt_size':4,'kernel_size':3,'adapt_scale':1.0}},
'bam': {'label':'BAM-Tuning','args':{'tuning_method':'bam','freeze_backbone':True,'bam_reduction':16,'bam_dilation':4,'bam_gate_init':0.0,'bam_use_bn':True,'bam_insert':'stage','bam_stages':'1,2,3,4'}},
'residual': {'label':'Residual Adapters','args':{'tuning_method':'residual','freeze_backbone':True,'ra_mode':'parallel','ra_reduction':16,'ra_norm':'bn','ra_act':'relu','ra_gate_init':0.0,'ra_stages':'1,2,3,4'}},
'lora_conv': {'label':'LoRA-Conv','args':{'tuning_method':'lora_conv','freeze_backbone':True,'lora_r':8,'lora_alpha':16.0,'lora_target':'all'}},
'sidetune': {'label':'Side-Tuning','args':{'tuning_method':'sidetune','freeze_backbone':True,'sidetune_alpha':0.5,'sidetune_learn_alpha':True,'sidetune_width':64,'sidetune_depth':3}},
'full': {'label':'Full fine-tuning','args':{'tuning_method':'full','freeze_backbone':False}},
}
DENSE_ORDER=['linear','bitfit','ssf','dt1d','conv_adapter','bam','residual_adapter','full']
DENSE_METHODS={
'linear': {'label':'Linear probing','args':{'tuning_method':'linear'}},
'bitfit': {'label':'BitFit','args':{'tuning_method':'bitfit'}},
'ssf': {'label':'SSF','args':{'tuning_method':'ssf'}},
'dt1d': {'label':'DT1D-Adapter','args':{'tuning_method':'dt1d','cache_dt1d':True}},
'conv_adapter': {'label':'Conv-Adapter','args':{'tuning_method':'conv_adapter','adapter_reduction':4}},
'bam': {'label':'BAM','args':{'tuning_method':'bam','adapter_reduction':16}},
'residual_adapter': {'label':'Residual Adapter','args':{'tuning_method':'residual_adapter','adapter_reduction':16}},
'full': {'label':'Full fine-tuning','args':{'tuning_method':'full'}},
}

def patch(path:Path, family:str):
    d=yaml.safe_load(path.read_text())
    if family=='cnn': methods,order=CNN_METHODS,CNN_ORDER
    else: methods,order=DENSE_METHODS,DENSE_ORDER
    old=copy.deepcopy(d.get('methods',{}))
    # Prefer an existing local method definition when present; otherwise use
    # the unified canonical definition above.
    d['methods']={k:copy.deepcopy(old.get(k, methods[k])) for k in order}
    d['method_order']=list(order)
    d.setdefault('v6_unified_baseline_policy',{})
    d['v6_unified_baseline_policy']={'family':family,'order':order,'note':'legacy extra completed baselines may remain in the manuscript, but all new/missing v6 runs use this common core set'}
    path.write_text(yaml.safe_dump(d,sort_keys=False))
    return d

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--family',choices=['cnn','dense'],required=True); ap.add_argument('paths',nargs='+',type=Path); ns=ap.parse_args()
    for p in ns.paths:
        d=patch(p,ns.family); print(json.dumps({'patched':str(p),'family':ns.family,'method_order':d['method_order']},indent=2))
if __name__=='__main__': main()
