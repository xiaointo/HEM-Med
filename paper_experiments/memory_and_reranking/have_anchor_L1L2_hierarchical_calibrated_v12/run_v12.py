#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE=Path(__file__).resolve().parent
V11=BASE.parent/'have_anchor_L1L2_full_reranker_ddi_ablation_v11'/'run_full_experiment.py'
spec=importlib.util.spec_from_file_location('v11',V11); v11=importlib.util.module_from_spec(spec); spec.loader.exec_module(v11)

CLASS_CAPS={'anti_infective':2,'sedative_analgesic':2,'anticoagulant':2,'antiplatelet':2,'bronchodilator':2,'glucose_control':2,'fluid_electrolyte':2}

def count_features(case):
    age=float(case.get('age') or 0)
    sex=str(case.get('sex') or '').lower()
    hid=str(v11.ranker.hospital_id(case))
    dx=case.get('dx') or []; px=case.get('px') or []; lab=case.get('lab') or []
    return {'bias':1.0,'dx_n':min(len(dx),20)/20,'px_n':min(len(px),20)/20,'lab_n':min(len(lab),12)/12,'age':min(age,100)/100,'sex_f':1.0 if sex.startswith('f') else 0.0,'sex_m':1.0 if sex.startswith('m') else 0.0,'hospital_'+hid:1.0}

def train_count_model(cases,epochs=4,seed=1203):
    rng=random.Random(seed); w=defaultdict(float); ids=list(range(len(cases)))
    for ep in range(epochs):
        rng.shuffle(ids); lr=.025/(1+.3*ep)
        for i in ids:
            c=cases[i]; x=count_features(c); y=min(max(len(v11.true_meds_norm(c)),1),8); pred=sum(w[k]*v for k,v in x.items()); err=max(-4,min(4,y-pred))
            for k,v in x.items(): w[k]+=lr*(err*v-0.0002*w[k])
    return dict(w)

def predict_count(case,w):
    x=count_features(case); value=sum(w.get(k,0)*v for k,v in x.items())
    return max(1,min(8,int(round(value))))

def compute_ddi_coadmin(cases,ddi_pairs):
    med_n=Counter(); pair_n=Counter()
    for c in cases:
        meds=sorted(v11.true_meds_norm(c))
        med_n.update(meds)
        for i,a in enumerate(meds):
            for b in meds[i+1:]:
                pair=(a,b)
                if pair in ddi_pairs: pair_n[pair]+=1
    return {f'{a}|||{b}':n/max(1,min(med_n[a],med_n[b])) for (a,b),n in pair_n.items()}

def prepare_scored(cases,weights,med_to_class,ctx,args):
    rows=[]
    for i,case in enumerate(cases,1):
        details=v11.candidate_details(case,ctx,args)
        scored=v11.score_details(case,details,weights,med_to_class,60)
        rows.append({'case':case,'scored':[{'medication':x.get('medication'),'ml_probability':float(x.get('ml_probability',0)),'original_rank':x.get('original_rank')} for x in scored]})
        if i%1000==0: print(json.dumps({'phase':'score','cases':i,'total':len(cases)}),flush=True)
    return rows

def best_threshold(items,total_positive,default):
    if total_positive<10: return default
    best=(0,default)
    for th in [x/100 for x in range(8,76,2)]:
        tp=sum(y and p>=th for p,y in items); fp=sum((not y) and p>=th for p,y in items); fn=total_positive-tp
        f1=2*tp/max(1,2*tp+fp+fn)
        if f1>best[0]: best=(f1,th)
    shrink=total_positive/(total_positive+100)
    return max(default-.18,min(default+.18,default+shrink*(best[1]-default)))

def learn_thresholds(rows,med_to_class,global_th):
    med_items=defaultdict(list); med_pos=Counter(); cls_items=defaultdict(list); cls_pos=Counter()
    for row in rows:
        truth=v11.true_meds_norm(row['case']); true_cls={med_to_class.get(m,'unknown') for m in truth}
        med_pos.update(truth); cls_pos.update(true_cls)
        class_max=defaultdict(float)
        for x in row['scored']:
            m=v11.ranker.normalize_drug_name(x['medication']); p=x['ml_probability']; med_items[m].append((p,m in truth)); cls=med_to_class.get(m,'unknown'); class_max[cls]=max(class_max[cls],p)
        for cls,p in class_max.items(): cls_items[cls].append((p,cls in true_cls))
    med_th={m:best_threshold(items,med_pos[m],global_th) for m,items in med_items.items() if med_pos[m]>=10}
    cls_th={c:best_threshold(items,cls_pos[c],global_th) for c,items in cls_items.items() if cls_pos[c]>=10}
    return med_th,cls_th,dict(med_pos),dict(cls_pos)

def pair_conf(a,b,coadmin): return coadmin.get('|||'.join(sorted((a,b))),0.0)

def select(row,med_to_class,med_th,cls_th,count_w,ddi_pairs,coadmin,base_penalty):
    scored=row['scored']; case=row['case']; target=predict_count(case,count_w); min_k=max(1,target-1); max_k=min(8,target+1)
    class_max=defaultdict(float)
    for x in scored:
        m=v11.ranker.normalize_drug_name(x['medication']); cls=med_to_class.get(m,'unknown'); class_max[cls]=max(class_max[cls],x['ml_probability'])
    active={c for c,p in class_max.items() if p>=cls_th.get(c,.34)}
    selected=[]; norms=[]; class_n=Counter(); deferred=[]
    for x in scored:
        med=x['medication']; norm=v11.ranker.normalize_drug_name(med); cls=med_to_class.get(norm,'unknown'); p=x['ml_probability']
        if cls not in active or class_n[cls]>=CLASS_CAPS.get(cls,2) or p<med_th.get(norm,.34): continue
        hits=[]
        for prev in norms:
            pair=tuple(sorted((norm,prev)))
            if pair in ddi_pairs: hits.append(pair_conf(norm,prev,coadmin))
        # Frequent true co-administration reduces, but never removes, the safety penalty.
        penalty=sum(base_penalty*(1-min(.8,c/.25)) for c in hits)
        effective=p-penalty
        item={**x,'effective_probability':effective,'ddi_hits':len(hits),'max_coadmin_confidence':max(hits,default=0)}
        if hits and effective < med_th.get(norm,.34): deferred.append(item); continue
        selected.append(item); norms.append(norm); class_n[cls]+=1
        if len(selected)>=max_k: break
    if len(selected)<min_k:
        chosen=set(norms)
        pool=[]
        for x in scored:
            norm=v11.ranker.normalize_drug_name(x['medication']); cls=med_to_class.get(norm,'unknown')
            if norm in chosen or class_n[cls]>=CLASS_CAPS.get(cls,2): continue
            confs=[pair_conf(norm,p,coadmin) for p in norms if tuple(sorted((norm,p))) in ddi_pairs]
            # Backfill only non-DDI or empirically common co-administration pairs.
            if confs and max(confs)<.10: continue
            penalty=sum(base_penalty*(1-min(.8,c/.25)) for c in confs)
            pool.append({**x,'effective_probability':x['ml_probability']-penalty,'ddi_hits':len(confs),'max_coadmin_confidence':max(confs,default=0)})
        pool.sort(key=lambda x:(-x['effective_probability'],x['original_rank']))
        for x in pool:
            norm=v11.ranker.normalize_drug_name(x['medication']); cls=med_to_class.get(norm,'unknown')
            if norm in chosen or class_n[cls]>=CLASS_CAPS.get(cls,2): continue
            selected.append(x); norms.append(norm); chosen.add(norm); class_n[cls]+=1
            if len(selected)>=min_k: break
    meds=[x['medication'] for x in selected]
    scores={v11.ranker.normalize_drug_name(x['medication']):x['effective_probability'] for x in selected}
    return meds,scores,target

def evaluate(rows,med_to_class,med_th,cls_th,count_w,ctx,coadmin,penalty):
    preds=[]; count_abs=0; size=Counter(); candidate_hit=candidate_true=all_true=0
    for row in rows:
        truth=v11.true_meds_norm(row['case']); cand={v11.ranker.normalize_drug_name(x['medication']) for x in row['scored']}; candidate_hit+=len(truth&cand); candidate_true+=len(truth); all_true+=truth<=cand
        pred,scores,target=select(row,med_to_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,penalty)
        count_abs+=abs(target-len(truth)); size[len(pred)]+=1; preds.append({'case':row['case'],'pred':pred,'scores':scores})
    metrics=v11.metric_summary_from_predictions(preds,ctx,med_to_class)
    return {'metrics':metrics,'candidate_recall_top60':candidate_hit/max(1,candidate_true),'cases_all_true_in_top60':all_true,'count_mae':count_abs/len(rows),'prediction_size_distribution':dict(size)}

def main():
    args=v11.parse_args(); args.candidate_depth_train=60; args.candidate_depth_predict=60; args.max_candidate_meds=60; args.epochs=1
    out=BASE/'outputs_v12_test1000'; out.mkdir(exist_ok=True)
    ctx=v11.load_context(args)
    train=v11.read_cases(args.train_file,72436); val=v11.read_cases(args.validation_file,3000); test=v11.read_cases(args.full_test_file,1000)
    med_to_class=v11.collect_med_classes(train+val,ctx)
    count_w=train_count_model(train)
    coadmin=compute_ddi_coadmin(train,ctx['ddi_pairs'])
    weights,n_examples,n_pos=v11.train_logistic_stream(train,ctx,med_to_class,args)
    val_rows=prepare_scored(val,weights,med_to_class,ctx,args)
    # Use the v11 global operating point, then calibrate each medication and class on validation only.
    med_th,cls_th,med_support,cls_support=learn_thresholds(val_rows,med_to_class,.34)
    test_rows=prepare_scored(test,weights,med_to_class,ctx,args)
    profiles={'no_selection_ddi':0.0,'empirical_mild_ddi':.03,'empirical_balanced_ddi':.08}
    results={name:evaluate(test_rows,med_to_class,med_th,cls_th,count_w,ctx,coadmin,p) for name,p in profiles.items()}
    report={'step':'hierarchical_calibrated_reranker_v12','data':{'train':len(train),'validation':len(val),'test':len(test)},'training':{'examples':n_examples,'positive_examples':n_pos,'candidate_depth':60,'hard_negative_ranks':'31-60'},'features':{'dynamic_count_predictor':True,'per_medication_thresholds':len(med_th),'per_class_thresholds':len(cls_th),'hierarchical_class_to_drug_gate':True,'empirical_ddi_coadministration_pairs':len(coadmin),'ddi_backfill_rule':'no DDI or training coadministration confidence >= 0.10'},'profiles':profiles,'results':results}
    for name,obj in [('v12_report.json',report),('reranker_weights.json',weights),('count_model_weights.json',count_w),('medication_thresholds.json',med_th),('class_thresholds.json',cls_th),('ddi_coadministration_confidence.json',coadmin),('medication_class_map.json',med_to_class)]: (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
