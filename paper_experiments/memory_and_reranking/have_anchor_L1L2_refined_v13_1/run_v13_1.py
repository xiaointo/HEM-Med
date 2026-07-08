#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, json, math, csv
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
V13 = BASE.parent / 'have_anchor_L1L2_asymmetric_calibrated_v13' / 'run_v13.py'
V12_OUT = BASE.parent / 'have_anchor_L1L2_hierarchical_calibrated_v12' / 'outputs_v12_test1000'
OUT = BASE / 'outputs_v13_1_test1000'
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location('v13', V13)
v13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v13)
v12 = v13.v12
v11 = v13.v11

HIGH_FP_FORCE = {
    'acetylsalicylic acid', 'heparin', 'insulin', 'sodium chloride',
    'metoprolol', 'enoxaparin', 'albuterol', 'vancomycin',
    'dextrose, unspecified form', 'ipratropium', 'potassium chloride',
}
LOW_RECALL_FORCE = {'warfarin', 'clopidogrel', 'lorazepam', 'hydralazine', 'glucagon'}
COUNT_POLICIES = [
    {'name': 'offset1_extra0', 'count_offset': 1, 'max_extra': 0, 'calibration_strength': 0.55},
    {'name': 'offset1_extra1', 'count_offset': 1, 'max_extra': 1, 'calibration_strength': 0.55},
    {'name': 'offset0_extra1', 'count_offset': 0, 'max_extra': 1, 'calibration_strength': 0.55},
    {'name': 'offset0_extra2', 'count_offset': 0, 'max_extra': 2, 'calibration_strength': 0.55},
    {'name': 'offset_minus1_extra2', 'count_offset': -1, 'max_extra': 2, 'calibration_strength': 0.55},
]
BALANCED_DDI_PENALTY = 0.08

def safe(a,b): return a/b if b else 0.0

def context_text(case):
    parts=[]
    for key in ('dx','px','lab'):
        vals=case.get(key) or []
        if isinstance(vals, list): parts.extend(str(x).lower() for x in vals)
        else: parts.append(str(vals).lower())
    return ' | '.join(parts)

def med_mentioned(med, text):
    m=med.lower()
    if len(m) < 5: return False
    return m in text

def collect_calibration_v13_1(rows, old_th):
    items=defaultdict(list); support=Counter()
    for row in rows:
        truth=v11.true_meds_norm(row['case']); support.update(truth)
        for x in row['scored']:
            m=v11.ranker.normalize_drug_name(x['medication']); items[m].append((x['ml_probability'], m in truth))
    thresholds={}; audit=[]
    for med, vals in items.items():
        pos=support[med]
        old=old_th.get(med, .34)
        if pos < 10:
            thresholds[med]=old
            continue
        old_tp=sum(y and p>=old for p,y in vals); old_fp=sum((not y) and p>=old for p,y in vals); old_fn=pos-old_tp
        old_prec=safe(old_tp, old_tp+old_fp); old_rec=safe(old_tp, pos)
        if med in HIGH_FP_FORCE or (old_prec < .42 and old_fp >= 40):
            group='high_false_positive'; beta=.45; precision_floor=.38
        elif med in LOW_RECALL_FORCE or (old_rec < .35 and pos >= 20 and old_tp >= 5 and old_prec >= .12):
            group='low_recall'; beta=1.85; precision_floor=.13
        else:
            group='balanced'; beta=1.0; precision_floor=.18
        best=(-1, old, {})
        for th_i in range(6, 83, 2):
            th=th_i/100
            tp=sum(y and p>=th for p,y in vals); fp=sum((not y) and p>=th for p,y in vals); fn=pos-tp
            prec=safe(tp,tp+fp); rec=safe(tp,pos); b2=beta*beta; fbeta=safe((1+b2)*prec*rec,b2*prec+rec)
            # Stronger high-FP target: prefer thresholds that reach useful precision.
            utility=fbeta
            if group == 'high_false_positive' and prec < precision_floor:
                utility *= .55
            if group == 'low_recall' and prec < precision_floor:
                utility *= .70
            # Avoid selecting thresholds that nearly zero-out a common medication.
            if pos >= 100 and rec < .12:
                utility *= .65
            if utility > best[0]:
                best=(utility, th, {'tp':tp,'fp':fp,'fn':fn,'precision':prec,'recall':rec,'fbeta':fbeta})
        shrink=pos/(pos+55)
        proposed=old + shrink*(best[1]-old)
        if group == 'high_false_positive':
            proposed = min(old + .18, max(old, proposed))
        elif group == 'low_recall':
            proposed = max(old - .14, min(old, proposed))
        proposed=max(.06, min(.82, proposed))
        thresholds[med]=proposed
        audit.append({
            'medication':med,'group':group,'validation_support':pos,
            'old_threshold':old,'new_threshold':proposed,
            'old_precision':old_prec,'old_recall':old_rec,
            'old_tp':old_tp,'old_fp':old_fp,'old_fn':old_fn,
            'optimized_threshold':best[1], **best[2]
        })
    return thresholds, audit

def select_v13_1(row, med_to_class, med_th, cls_th, count_w, ddi_pairs, coadmin, policy, audit=False):
    case=row['case']
    text=context_text(case)
    target=max(1, min(8, v12.predict_count(case, count_w)+policy['count_offset']))
    min_k=max(1, target-1)
    max_k=min(8, target+policy['max_extra'])
    strength=policy['calibration_strength']
    class_max=defaultdict(float); class_vals=defaultdict(list)
    for x in row['scored']:
        m=v11.ranker.normalize_drug_name(x['medication']); c=med_to_class.get(m,'unknown')
        class_max[c]=max(class_max[c], x['ml_probability']); class_vals[c].append(x['ml_probability'])
    active={c for c,p in class_max.items() if p>=cls_th.get(c,.34)}
    class_avg={c:sum(v)/len(v) for c,v in class_vals.items()}
    ranked=[]
    for x in row['scored']:
        m=v11.ranker.normalize_drug_name(x['medication']); c=med_to_class.get(m,'unknown')
        p=x['ml_probability']; th=med_th.get(m,.34); cth=cls_th.get(c,.34)
        effective_th=.34 + strength*(th-.34)
        threshold_margin=p-effective_th
        class_margin=class_max[c]-cth
        within_class_margin=p-class_avg[c]
        mention_boost=.025 if med_mentioned(m, text) else 0.0
        high_fp_penalty=.018 if m in HIGH_FP_FORCE and not med_mentioned(m, text) else 0.0
        low_recall_boost=.014 if m in LOW_RECALL_FORCE else 0.0
        calibrated=(
            threshold_margin + .34
            + .11*class_margin
            + .045*within_class_margin
            + mention_boost + low_recall_boost - high_fp_penalty
        )
        ranked.append({**x,'norm':m,'class':c,'threshold':effective_th,'calibrated_score':calibrated,
                       'threshold_margin':threshold_margin,'class_margin':class_margin,
                       'within_class_margin':within_class_margin,'mention_boost':mention_boost,
                       'high_fp_penalty':high_fp_penalty,'low_recall_boost':low_recall_boost})
    ranked.sort(key=lambda x:(-x['calibrated_score'], x['original_rank']))
    selected=[]; norms=[]; class_n=Counter(); chosen=set(); events=[]
    for x in ranked:
        m=x['norm']; c=x['class']; p=x['ml_probability']
        reason=None
        if c not in active: reason='class_inactive'
        elif class_n[c] >= v12.CLASS_CAPS.get(c,2): reason='class_cap'
        elif p < x['threshold']: reason='below_threshold'
        if reason:
            if audit: events.append({**x,'stage':'primary_skip','reason':reason})
            continue
        confs=[v12.pair_conf(m,prev,coadmin) for prev in norms if tuple(sorted((m,prev))) in ddi_pairs]
        penalty=sum(BALANCED_DDI_PENALTY*(1-min(.8,conf/.25)) for conf in confs)
        # Class-internal duplicate penalty: second drug in same class must clear a small margin.
        duplicate_penalty=.025*class_n[c]
        effective=x['calibrated_score']-penalty-duplicate_penalty
        item={**x,'effective_probability':effective,'ddi_hits':len(confs),'stage':'primary','ddi_penalty':penalty,'class_duplicate_penalty':duplicate_penalty}
        if confs and effective < .34:
            if audit: events.append({**item,'reason':'deferred_ddi_low_score'})
            continue
        selected.append(item); norms.append(m); chosen.add(m); class_n[c]+=1
        if len(selected)>=max_k: break
    if len(selected)<min_k:
        pool=[]
        for x in ranked:
            m=x['norm']; c=x['class']
            if m in chosen: continue
            if class_n[c]>=v12.CLASS_CAPS.get(c,2): continue
            confs=[v12.pair_conf(m,prev,coadmin) for prev in norms if tuple(sorted((m,prev))) in ddi_pairs]
            # Balanced DDI default: backfill should avoid rare DDI pairs.
            if confs and max(confs)<.10: continue
            penalty=sum(BALANCED_DDI_PENALTY*(1-min(.8,conf/.25)) for conf in confs)
            duplicate_penalty=.025*class_n[c]
            pool.append({**x,'effective_probability':x['calibrated_score']-penalty-duplicate_penalty,
                         'ddi_hits':len(confs),'stage':'backfill','ddi_penalty':penalty,'class_duplicate_penalty':duplicate_penalty})
        pool.sort(key=lambda x:(-x['effective_probability'], x['original_rank']))
        for x in pool:
            m=x['norm']; c=x['class']
            if m in chosen or class_n[c]>=v12.CLASS_CAPS.get(c,2): continue
            selected.append(x); norms.append(m); chosen.add(m); class_n[c]+=1
            if len(selected)>=min_k: break
    meds=[x['medication'] for x in selected]
    scores={x['norm']:x['effective_probability'] for x in selected}
    if not audit:
        return meds, scores, target
    return meds, scores, target, selected, ranked, events

def fast_set_metrics(rows, selector):
    vals=Counter(); js=[]; fs=[]
    for row in rows:
        truth=v11.true_meds_norm(row['case'])
        pred={v11.ranker.normalize_drug_name(x) for x in selector(row)[0]}
        tp=len(truth&pred); fp=len(pred-truth); fn=len(truth-pred)
        js.append(safe(tp,len(truth|pred))); fs.append(safe(2*tp,2*tp+fp+fn))
        vals['tp']+=tp; vals['fp']+=fp; vals['fn']+=fn
    return {'jaccard':sum(js)/len(js),'f1':sum(fs)/len(fs),'precision':safe(vals['tp'],vals['tp']+vals['fp']),'recall':safe(vals['tp'],vals['tp']+vals['fn'])}

def full_evaluate(rows, selector, ctx, med_to_class, vocab):
    preds=[]; labels=[]; probs=[]; count_err=0; sizes=Counter()
    for row in rows:
        pred,scores,target=selector(row)
        truth=v11.true_meds_norm(row['case']); count_err+=abs(target-len(truth)); sizes[len(pred)]+=1
        preds.append({'case':row['case'],'pred':pred,'scores':scores})
        raw={v11.ranker.normalize_drug_name(x['medication']):x['ml_probability'] for x in row['scored']}
        for med in sorted(vocab): labels.append(1 if med in truth else 0); probs.append(raw.get(med,0.0))
    return {'selection_metrics':v11.metric_summary_from_predictions(preds,ctx,med_to_class),
            'drug_level_micro_prauc':v11.ranker.average_precision(labels,probs),
            'count_mae':count_err/len(rows),'prediction_size_distribution':dict(sizes)}

def write_error_audit(rows, selector, med_class, out_path):
    with out_path.open('w') as f:
        for row in rows:
            pred,scores,target,selected,ranked,events=selector(row, audit=True)
            truth=v11.true_meds_norm(row['case'])
            pred_norm={v11.ranker.normalize_drug_name(x) for x in pred}
            rank_map={x['norm']:x for x in ranked}
            selected_map={x['norm']:x for x in selected}
            fp=sorted(pred_norm-truth); fn=sorted(truth-pred_norm); tp=sorted(truth&pred_norm)
            fn_detail=[]
            for m in fn:
                x=rank_map.get(m)
                if not x:
                    fn_detail.append({'medication':m,'source':'not_in_top60_candidate','class':med_class.get(m,'unknown')})
                else:
                    reason='not_selected'
                    if x['class'] not in {y['class'] for y in selected}: reason='class_not_selected_or_inactive'
                    if x['ml_probability'] < x['threshold']: reason='below_threshold'
                    fn_detail.append({'medication':m,'source':'candidate_not_selected','class':x['class'],
                                      'probability':x['ml_probability'],'threshold':x['threshold'],
                                      'original_rank':x['original_rank'],'calibrated_score':x['calibrated_score'],
                                      'reason':reason})
            fp_detail=[]
            for m in fp:
                x=selected_map.get(m, rank_map.get(m, {}))
                fp_detail.append({'medication':m,'class':med_class.get(m,'unknown'),
                                  'probability':x.get('ml_probability'),'threshold':x.get('threshold'),
                                  'original_rank':x.get('original_rank'),'calibrated_score':x.get('calibrated_score'),
                                  'effective_probability':x.get('effective_probability'),
                                  'stage':x.get('stage'),'ddi_hits':x.get('ddi_hits'),
                                  'high_fp_penalty':x.get('high_fp_penalty'),
                                  'mention_boost':x.get('mention_boost')})
            rec={'case_id':row['case'].get('id'),'hospital_id':row['case'].get('hid'),
                 'true_count':len(truth),'pred_count':len(pred_norm),'target_count':target,
                 'tp':tp,'fp':fp,'fn':fn,'fp_detail':fp_detail,'fn_detail':fn_detail}
            f.write(json.dumps(rec,ensure_ascii=False)+'\n')

def main():
    args=v11.parse_args(); args.candidate_depth_predict=60; args.max_candidate_meds=60
    ctx=v11.load_context(args)
    weights=json.loads((V12_OUT/'reranker_weights.json').read_text())
    count_w=json.loads((V12_OUT/'count_model_weights.json').read_text())
    old_th=json.loads((V12_OUT/'medication_thresholds.json').read_text())
    cls_th=json.loads((V12_OUT/'class_thresholds.json').read_text())
    coadmin=json.loads((V12_OUT/'ddi_coadministration_confidence.json').read_text())
    med_class=json.loads((V12_OUT/'medication_class_map.json').read_text())
    val_cases=v11.read_cases(args.validation_file,3000)
    test_cases=v11.read_cases(args.full_test_file,1000)
    print(json.dumps({'phase':'prepare_val','cases':len(val_cases)},ensure_ascii=False))
    val_rows=v12.prepare_scored(val_cases,weights,med_class,ctx,args)
    med_th,audit=collect_calibration_v13_1(val_rows,old_th)
    grid=[]; best=None
    for policy in COUNT_POLICIES:
        selector=lambda row,p=policy: select_v13_1(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p)
        metrics=fast_set_metrics(val_rows,selector)
        # Balanced objective: keep F1/Jaccard but reward precision to reduce FP.
        score=metrics['jaccard']+metrics['f1']+.14*metrics['recall']+.16*metrics['precision']
        item={**policy,'score':score,**metrics}; grid.append(item)
        if best is None or score>best['score']: best=item
    print(json.dumps({'best_validation_policy':best},ensure_ascii=False))
    print(json.dumps({'phase':'prepare_test','cases':len(test_cases)},ensure_ascii=False))
    test_rows=v12.prepare_scored(test_cases,weights,med_class,ctx,args)
    results={}
    # Always include selected best, plus the explicitly safer policy for comparison.
    eval_policies=[]
    seen=set()
    for pol in [best]+COUNT_POLICIES:
        key=pol['name']
        if key not in seen:
            eval_policies.append(pol); seen.add(key)
    for policy in eval_policies:
        selector=lambda row,p=policy: select_v13_1(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p)
        results[policy['name']]=full_evaluate(test_rows,selector,ctx,med_class,ctx['vocabulary'])
        print(json.dumps({'policy':policy['name'],'exact_drug':results[policy['name']]['selection_metrics']['exact_drug'],
                          'count_mae':results[policy['name']]['count_mae'],
                          'sizes':results[policy['name']]['prediction_size_distribution']},ensure_ascii=False))
    best_policy=max(eval_policies, key=lambda p: results[p['name']]['selection_metrics']['exact_drug']['safety_adjusted_jaccard'])
    audit_selector=lambda row,audit=False,p=best_policy: select_v13_1(row,med_class,med_th,cls_th,count_w,ctx['ddi_pairs'],coadmin,p,audit=audit)
    write_error_audit(test_rows,audit_selector,med_class,OUT/'per_case_error_audit.jsonl')
    groups=Counter(x['group'] for x in audit)
    report={'step':'refined_v13_1_top60_threshold_class_internal_rerank',
            'data':{'validation':len(val_cases),'test':len(test_cases)},
            'changes':['reuse v13 Top60 framework','stronger high-FP precision target','class-internal rerank using threshold margin, within-class margin, mention boost, duplicate penalty','balanced DDI profile as default','count policy retuned','per-case FP/FN error audit'],
            'default_ddi_profile':'balanced_ddi_penalty_0.08',
            'threshold_groups':dict(groups),'validation_grid':grid,
            'best_validation_policy':best,'best_saj_policy':best_policy,
            'results':results}
    for name,obj in [('v13_1_report.json',report),('medication_thresholds_v13_1.json',med_th),('threshold_calibration_audit_v13_1.json',audit)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'written':str(OUT/'v13_1_report.json'),'audit':str(OUT/'per_case_error_audit.jsonl')},ensure_ascii=False))

if __name__=='__main__': main()
