#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,math
from collections import Counter,defaultdict
from pathlib import Path

BASE=Path(__file__).resolve().parent
V12=BASE.parent/'have_anchor_L1L2_hierarchical_calibrated_v12'/'run_v12.py'
spec=importlib.util.spec_from_file_location('v12',V12); v12=importlib.util.module_from_spec(spec); spec.loader.exec_module(v12)
v11=v12.v11
V12_OUT=V12.parent/'outputs_v12_test1000'

def safe(a,b): return a/b if b else 0.0

def collect_calibration(rows,old_th):
    items=defaultdict(list); support=Counter()
    for row in rows:
        truth=v11.true_meds_norm(row['case']); support.update(truth)
        for x in row['scored']:
            m=v11.ranker.normalize_drug_name(x['medication']); items[m].append((x['ml_probability'],m in truth))
    thresholds={}; audit=[]
    for med,vals in items.items():
        pos=support[med]
        if pos<10: thresholds[med]=old_th.get(med,.34); continue
        old=old_th.get(med,.34)
        old_tp=sum(y and p>=old for p,y in vals); old_fp=sum((not y) and p>=old for p,y in vals); old_fn=pos-old_tp
        old_prec=safe(old_tp,old_tp+old_fp); old_rec=safe(old_tp,pos)
        if old_prec<.40 and old_fp>=50: group='high_false_positive'; beta=.65
        elif old_rec<.35 and pos>=20 and old_tp>=5 and old_prec>=.15: group='low_recall'; beta=1.75
        else: group='balanced'; beta=1.0
        best=(-1,old,{})
        for th_i in range(6,81,2):
            th=th_i/100; tp=sum(y and p>=th for p,y in vals); fp=sum((not y) and p>=th for p,y in vals); fn=pos-tp
            prec=safe(tp,tp+fp); rec=safe(tp,pos); b2=beta*beta; fbeta=safe((1+b2)*prec*rec,b2*prec+rec)
            # Avoid buying recall with unusably low precision.
            utility=fbeta if prec>=.16 or group!='low_recall' else fbeta*.5
            if utility>best[0]: best=(utility,th,{'tp':tp,'fp':fp,'fn':fn,'precision':prec,'recall':rec,'fbeta':fbeta})
        shrink=pos/(pos+60)
        proposed=old+shrink*(best[1]-old)
        if group=='high_false_positive': proposed=min(old+.12,max(old,proposed))
        elif group=='low_recall': proposed=max(old-.12,min(old,proposed))
        proposed=max(.06,min(.80,proposed)); thresholds[med]=proposed
        audit.append({'medication':med,'group':group,'validation_support':pos,'old_threshold':old,'new_threshold':proposed,'old_precision':old_prec,'old_recall':old_rec,'optimized_threshold':best[1],**best[2]})
    return thresholds,audit

def select(row,med_to_class,med_th,cls_th,count_w,ddi_pairs,coadmin,base_penalty,count_offset,max_extra,calibration_strength):
    case=row['case']; target=max(1,min(8,v12.predict_count(case,count_w)+count_offset)); min_k=max(1,target-1); max_k=min(8,target+max_extra)
    class_max=defaultdict(float)
    for x in row['scored']:
        m=v11.ranker.normalize_drug_name(x['medication']); c=med_to_class.get(m,'unknown'); class_max[c]=max(class_max[c],x['ml_probability'])
    active={c for c,p in class_max.items() if p>=cls_th.get(c,.34)}
    ranked=[]
    for x in row['scored']:
        m=v11.ranker.normalize_drug_name(x['medication']); c=med_to_class.get(m,'unknown'); p=x['ml_probability']; th=med_th.get(m,.34); cth=cls_th.get(c,.34)
        effective_th=.34+calibration_strength*(th-.34)
        calibrated=p-effective_th+.34+.12*(class_max[c]-cth)
        ranked.append({**x,'norm':m,'class':c,'threshold':effective_th,'calibrated_score':calibrated})
    ranked.sort(key=lambda x:(-x['calibrated_score'],x['original_rank']))
    selected=[]; norms=[]; class_n=Counter(); deferred=[]
    for x in ranked:
        m=x['norm']; c=x['class']; p=x['ml_probability']
        if c not in active or class_n[c]>=v12.CLASS_CAPS.get(c,2) or p<x['threshold']: continue
        confs=[v12.pair_conf(m,prev,coadmin) for prev in norms if tuple(sorted((m,prev))) in ddi_pairs]
        penalty=sum(base_penalty*(1-min(.8,conf/.25)) for conf in confs); effective=x['calibrated_score']-penalty
        item={**x,'effective_probability':effective,'ddi_hits':len(confs)}
        if confs and effective < .34: deferred.append(item); continue
        selected.append(item); norms.append(m); class_n[c]+=1
        if len(selected)>=max_k: break
    if len(selected)<min_k:
        chosen=set(norms); pool=[]
        for x in ranked:
            m=x['norm']; c=x['class']
            if m in chosen or class_n[c]>=v12.CLASS_CAPS.get(c,2): continue
            confs=[v12.pair_conf(m,prev,coadmin) for prev in norms if tuple(sorted((m,prev))) in ddi_pairs]
            if confs and max(confs)<.10: continue
            penalty=sum(base_penalty*(1-min(.8,conf/.25)) for conf in confs)
            pool.append({**x,'effective_probability':x['calibrated_score']-penalty,'ddi_hits':len(confs)})
        pool.sort(key=lambda x:(-x['effective_probability'],x['original_rank']))
        for x in pool:
            m=x['norm']; c=x['class']
            if m in chosen or class_n[c]>=v12.CLASS_CAPS.get(c,2): continue
            selected.append(x); norms.append(m); chosen.add(m); class_n[c]+=1
            if len(selected)>=min_k: break
    meds=[x['medication'] for x in selected]
    scores={x['norm']:x['effective_probability'] for x in selected}
    return meds,scores,target

def fast_set_metrics(rows,selector):
    vals=Counter(); js=[]; fs=[]
    for row in rows:
        truth=v11.true_meds_norm(row['case']); pred={v11.ranker.normalize_drug_name(x) for x in selector(row)[0]}; tp=len(truth&pred); fp=len(pred-truth); fn=len(truth-pred)
        j=safe(tp,len(truth|pred)); f=safe(2*tp,2*tp+fp+fn); js.append(j); fs.append(f); vals['tp']+=tp; vals['fp']+=fp; vals['fn']+=fn
    return {'jaccard':sum(js)/len(js),'f1':sum(fs)/len(fs),'precision':safe(vals['tp'],vals['tp']+vals['fp']),'recall':safe(vals['tp'],vals['tp']+vals['fn'])}

def full_evaluate(rows,selector,ctx,med_to_class,vocab):
    preds=[]; labels=[]; probs=[]; count_err=0; sizes=Counter()
    for row in rows:
        pred,scores,target=selector(row); truth=v11.true_meds_norm(row['case']); count_err+=abs(target-len(truth)); sizes[len(pred)]+=1
        preds.append({'case':row['case'],'pred':pred,'scores':scores})
        raw={v11.ranker.normalize_drug_name(x['medication']):x['ml_probability'] for x in row['scored']}
        for med in sorted(vocab): labels.append(1 if med in truth else 0); probs.append(raw.get(med,0.0))
    return {'selection_metrics':v11.metric_summary_from_predictions(preds,ctx,med_to_class),'drug_level_micro_prauc':v11.ranker.average_precision(labels,probs),'count_mae':count_err/len(rows),'prediction_size_distribution':dict(sizes)}

def main():
    out=BASE/'outputs_v13_test1000'; out.mkdir(exist_ok=True)
    args=v11.parse_args(); args.candidate_depth_predict=60; args.max_candidate_meds=60
    ctx=v11.load_context(args)
    weights=json.loads((V12_OUT/'reranker_weights.json').read_text()); count_w=json.loads((V12_OUT/'count_model_weights.json').read_text()); old_th=json.loads((V12_OUT/'medication_thresholds.json').read_text()); cls_th=json.loads((V12_OUT/'class_thresholds.json').read_text()); coadmin=json.loads((V12_OUT/'ddi_coadministration_confidence.json').read_text()); med_class=json.loads((V12_OUT/'medication_class_map.json').read_text())
    val_cases=v11.read_cases(args.validation_file,3000); test_cases=v11.read_cases(args.full_test_file,1000)
    val_rows=v12.prepare_scored(val_cases,weights,med_class,ctx,args); med_th,audit=collect_calibration(val_rows,old_th)
    # Tune count policy only on validation. Mild empirical DDI is the operating profile.
    grid=[]; best=None
    for strength in [.25,.50,.75,1.0]:
        for offset in [-1,0,1]:
            for extra in [0,1,2]:
                selector=lambda row,o=offset,e=extra,st=strength: select(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,.03,o,e,st)
                metrics=fast_set_metrics(val_rows,selector); score=metrics['jaccard']+metrics['f1']+.20*metrics['recall']+.10*metrics['precision']; item={'calibration_strength':strength,'count_offset':offset,'max_extra':extra,'score':score,**metrics}; grid.append(item)
                if best is None or score>best['score']: best=item
    test_rows=v12.prepare_scored(test_cases,weights,med_class,ctx,args)
    profiles={'empirical_mild_ddi':.03,'empirical_balanced_ddi':.08}
    results={}
    for name,pen in profiles.items():
        selector=lambda row,p=pen: select(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p,best['count_offset'],best['max_extra'],best['calibration_strength'])
        results[name]=full_evaluate(test_rows,selector,ctx,med_class,ctx['vocabulary'])
    groups=Counter(x['group'] for x in audit)
    report={'step':'asymmetric_per_drug_calibration_v13','data':{'validation':len(val_cases),'test':len(test_cases)},'reused_v12_top60_reranker':True,'calibration':{'threshold_count':len(med_th),'groups':dict(groups),'rule':'validation-derived high-FP beta=0.65; low-recall beta=1.75; otherwise beta=1.0','ranking':'drug probability minus drug threshold plus class margin'},'best_dynamic_count_config':best,'profiles':profiles,'results':results}
    for name,obj in [('v13_report.json',report),('medication_thresholds_v13.json',med_th),('threshold_calibration_audit.json',audit),('dynamic_count_validation_grid.json',grid)]: (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
