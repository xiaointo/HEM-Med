#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, math, copy
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
V13_1 = BASE.parent / 'have_anchor_L1L2_refined_v13_1' / 'run_v13_1.py'
V13 = BASE.parent / 'have_anchor_L1L2_asymmetric_calibrated_v13' / 'run_v13.py'
V12_OUT = BASE.parent / 'have_anchor_L1L2_hierarchical_calibrated_v12' / 'outputs_v12_test1000'
OUT = BASE / 'outputs_v13_3_full_test'
OUT.mkdir(parents=True, exist_ok=True)

spec1 = importlib.util.spec_from_file_location('v13_1', V13_1)
v13_1 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(v13_1)
v12 = v13_1.v12
v11 = v13_1.v11

COUNT_POLICIES = [
    {'name': 'offset1_extra0', 'count_offset': 1, 'max_extra': 0, 'calibration_strength': 0.55},
    {'name': 'offset1_extra1', 'count_offset': 1, 'max_extra': 1, 'calibration_strength': 0.55},
    {'name': 'offset0_extra1', 'count_offset': 0, 'max_extra': 1, 'calibration_strength': 0.55},
    {'name': 'offset0_extra2', 'count_offset': 0, 'max_extra': 2, 'calibration_strength': 0.55},
]


def safe(a,b): return a/b if b else 0.0

def sigmoid(x):
    if x >= 35: return 1.0
    if x <= -35: return 0.0
    return 1.0/(1.0+math.exp(-x))

def logit(p):
    p=max(1e-5,min(1-1e-5,float(p)))
    return math.log(p/(1-p))

def collect_platt_data(rows):
    data=defaultdict(list); support=Counter(); candidates=Counter()
    for row in rows:
        truth=v11.true_meds_norm(row['case']); support.update(truth)
        for x in row['scored']:
            m=v11.ranker.normalize_drug_name(x['medication'])
            y=1 if m in truth else 0
            data[m].append((logit(x['ml_probability']), y))
            candidates[m]+=1
    return data, support, candidates

def fit_per_drug_platt(rows):
    data, support, candidates = collect_platt_data(rows)
    params={}; audit=[]
    # Global fallback.
    all_vals=[v for vals in data.values() for v in vals]
    global_pos=sum(y for _,y in all_vals); global_neg=len(all_vals)-global_pos
    for med, vals in data.items():
        pos=sum(y for _,y in vals); neg=len(vals)-pos
        if pos < 8 or neg < 20:
            params[med]={'a':1.0,'b':0.0,'mode':'identity','pos':pos,'neg':neg}
            continue
        a=1.0; b=0.0
        # More regularization for low-support meds, less for common meds.
        l2=0.015 + 1.5/(pos+neg)
        lr=0.035
        n=len(vals)
        for _ in range(180):
            ga=0.0; gb=0.0
            for x,y in vals:
                pr=sigmoid(a*x+b)
                err=pr-y
                ga += err*x
                gb += err
            ga = ga/n + l2*(a-1.0)
            gb = gb/n + l2*b
            a -= lr*ga
            b -= lr*gb
            a=max(0.35,min(2.8,a)); b=max(-3.5,min(3.5,b))
        params[med]={'a':a,'b':b,'mode':'platt','pos':pos,'neg':neg}
        # validation AP contribution is hard to isolate; store simple Brier improvement.
        before=sum((sigmoid(x)-y)**2 for x,y in vals)/n
        after=sum((sigmoid(a*x+b)-y)**2 for x,y in vals)/n
        audit.append({'medication':med,'pos':pos,'neg':neg,'a':a,'b':b,'brier_before':before,'brier_after':after,'delta':before-after})
    return params, audit

def clone_with_calibrated_scores(rows, params, strength):
    new=[]
    for row in rows:
        r={'case':row['case'], 'scored':[]}
        for x in row['scored']:
            m=v11.ranker.normalize_drug_name(x['medication'])
            par=params.get(m, {'a':1.0,'b':0.0})
            raw=float(x['ml_probability'])
            cal=sigmoid(par.get('a',1.0)*logit(raw)+par.get('b',0.0))
            # Blend to avoid overfitting validation calibration.
            p=(1-strength)*raw + strength*cal
            y=dict(x)
            y['raw_ml_probability']=raw
            y['ml_probability']=round(max(0.0,min(1.0,p)), 6)
            y['calibrated_probability']=y['ml_probability']
            y['platt_a']=par.get('a',1.0); y['platt_b']=par.get('b',0.0)
            r['scored'].append(y)
        # Preserve score order by calibrated probability, but keep original_rank as prior tie-break.
        r['scored'].sort(key=lambda z:(-z['ml_probability'], z['original_rank']))
        for i,z in enumerate(r['scored'],1):
            z['calibrated_rank']=i
        new.append(r)
    return new

def drug_level_prauc(rows, vocab):
    labels=[]; probs=[]
    for row in rows:
        truth=v11.true_meds_norm(row['case'])
        raw={v11.ranker.normalize_drug_name(x['medication']):x['ml_probability'] for x in row['scored']}
        for med in sorted(vocab):
            labels.append(1 if med in truth else 0)
            probs.append(raw.get(med,0.0))
    return v11.ranker.average_precision(labels, probs)

def tune_strength(val_rows, params, ctx):
    grid=[]; best=None
    for strength in [0.0,0.25,0.5,0.75,1.0]:
        rows=clone_with_calibrated_scores(val_rows, params, strength)
        prauc=drug_level_prauc(rows, ctx['vocabulary'])
        grid.append({'strength':strength,'drug_level_micro_prauc':prauc})
        if best is None or prauc>best['drug_level_micro_prauc']:
            best=grid[-1]
    return best, grid

def main():
    args=v11.parse_args(); args.candidate_depth_predict=60; args.max_candidate_meds=60
    ctx=v11.load_context(args)
    weights=json.loads((V12_OUT/'reranker_weights.json').read_text())
    count_w=json.loads((V12_OUT/'count_model_weights.json').read_text())
    old_th=json.loads((V12_OUT/'medication_thresholds.json').read_text())
    cls_th=json.loads((V12_OUT/'class_thresholds.json').read_text())
    coadmin=json.loads((V12_OUT/'ddi_coadministration_confidence.json').read_text())
    med_class=json.loads((V12_OUT/'medication_class_map.json').read_text())
    train_cases=v11.read_cases(args.train_file, None)
    valid_cases=v11.read_cases(args.validation_file, None)
    val_cases=train_cases + valid_cases
    test_cases=v11.read_cases(args.full_test_file, None)
    print(json.dumps({'phase':'prepare_val','cases':len(val_cases)},ensure_ascii=False))
    val_rows_raw=v12.prepare_scored(val_cases,weights,med_class,ctx,args)
    params, platt_audit = fit_per_drug_platt(val_rows_raw)
    best_strength, strength_grid = tune_strength(val_rows_raw, params, ctx)
    print(json.dumps({'best_score_calibration_strength':best_strength},ensure_ascii=False))
    val_rows=clone_with_calibrated_scores(val_rows_raw, params, best_strength['strength'])
    med_th, th_audit = v13_1.collect_calibration_v13_1(val_rows, old_th)
    grid=[]; best_policy=None
    for policy in COUNT_POLICIES:
        selector=lambda row,p=policy: v13_1.select_v13_1(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p)
        metrics=v13_1.fast_set_metrics(val_rows,selector)
        score=metrics['jaccard']+metrics['f1']+.16*metrics['recall']+.14*metrics['precision']
        item={**policy,'score':score,**metrics}; grid.append(item)
        if best_policy is None or score>best_policy['score']:
            best_policy=item
    print(json.dumps({'best_validation_policy':best_policy},ensure_ascii=False))
    print(json.dumps({'phase':'prepare_test','cases':len(test_cases)},ensure_ascii=False))
    test_rows_raw=v12.prepare_scored(test_cases,weights,med_class,ctx,args)
    test_rows=clone_with_calibrated_scores(test_rows_raw, params, best_strength['strength'])
    results={}
    eval_policies=[]; seen=set()
    for p in [best_policy]+COUNT_POLICIES:
        if p['name'] not in seen:
            eval_policies.append(p); seen.add(p['name'])
    for policy in eval_policies:
        selector=lambda row,p=policy: v13_1.select_v13_1(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p)
        results[policy['name']]=v13_1.full_evaluate(test_rows,selector,ctx,med_class,ctx['vocabulary'])
        print(json.dumps({'policy':policy['name'],'exact_drug':results[policy['name']]['selection_metrics']['exact_drug'],
                          'drug_level_micro_prauc':results[policy['name']]['drug_level_micro_prauc'],
                          'count_mae':results[policy['name']]['count_mae'],
                          'sizes':results[policy['name']]['prediction_size_distribution']},ensure_ascii=False))
    best_saj=max(eval_policies, key=lambda p: results[p['name']]['selection_metrics']['exact_drug']['safety_adjusted_jaccard'])
    audit_selector=lambda row,audit=False,p=best_saj: v13_1.select_v13_1(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p,audit=audit)
    v13_1.write_error_audit(test_rows,audit_selector,med_class,OUT/'per_case_error_audit.jsonl')
    report={'step':'v13_3_full_train_score_calibrated_top60',
            'data':{'train_for_calibration_and_threshold':len(val_cases),'train_file':str(args.train_file),'validation_file':str(args.validation_file),'test':len(test_cases),'test_file':str(args.full_test_file)},
            'changes':['train calibration/thresholds on full train+validation split','evaluate on full held-out test split','reuse v13.1 Top60 selection framework','fit per-drug Platt calibration on validation scored candidates','blend raw and calibrated probabilities before threshold tuning and selection','retune per-drug thresholds after score calibration','retune count policy'],
            'score_calibration':{'best_strength':best_strength,'strength_grid':strength_grid},
            'best_validation_policy':best_policy,'best_saj_policy':best_saj,
            'validation_grid':grid,'results':results}
    for name,obj in [('v13_3_full_report.json',report),('platt_parameters_v13_3_full.json',params),('platt_audit_v13_3_full.json',platt_audit),('medication_thresholds_v13_3_full.json',med_th),('threshold_calibration_audit_v13_3_full.json',th_audit)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'written':str(OUT/'v13_3_full_report.json'),'audit':str(OUT/'per_case_error_audit.jsonl')},ensure_ascii=False))

if __name__=='__main__': main()
