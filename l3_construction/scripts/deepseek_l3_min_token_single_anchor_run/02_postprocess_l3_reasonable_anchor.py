#!/usr/bin/env python3
"""Postprocess DeepSeek L3 single-anchor output with conservative local rules.

L3 is a drug-conditioned patient-context attribution memory.

Offline construction direction:
    medication -> anchor

For each medication in one ICU stay, the model/script selects the most clinically
reasonable anchor from the patient's existing dx / px / lab candidate set.

The selected anchor is not a confirmed prescription reason. It is not a
ground-truth indication. It is only candidate patient-context support for the
medication. If no clinically reasonable anchor exists in the provided candidate
set, anchor=[] is allowed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
TOP_FIELDS=['stay_id','hospital_id','medication','medication_function','anchor','attribution_hypothesis']
ANCHOR_FIELDS={'anchor_id','anchor_type','anchor_string'}
FORBIDDEN=[
 'appears consistent with','associated with','linked to','management of','therapy','standard',
 'commonly used','often used','often requires','prescribed for','confirmed indication','caused by',
 'definitely treated','treatment of','for retrieval support'
]
FUNCTION_PHRASE={
 'anti_infective':'infection-related','anticoagulant':'coagulation-related','antiplatelet':'cardiovascular or thrombosis-related',
 'antiarrhythmic':'arrhythmia-related','bronchodilator':'respiratory','steroid':'inflammatory or respiratory',
 'glucose_control':'glucose-control','endocrine_thyroid':'thyroid- or endocrine-related','cardiac_medication':'cardiac',
 'antihypertensive':'cardiovascular','diuretic':'fluid-management','fluid_electrolyte':'fluid or electrolyte support',
 'acid_suppression':'gastrointestinal or acid-suppression','sedative_analgesic':'sedation or analgesia',
 'bowel_regimen':'supportive-care','analgesic_antipyretic':'analgesic or antipyretic','unknown':'uncertain'
}
MED_FUNCTION_FIX={
 'levothyroxine':'endocrine_thyroid','levothyroxine sodium':'endocrine_thyroid','thyroxine':'endocrine_thyroid',
 'lisinopril':'antihypertensive','acetaminophen':'analgesic_antipyretic','glucagon':'glucose_control',
 'dextrose, unspecified form':'glucose_control','dextrose':'glucose_control','d-glucose':'glucose_control',
 'esomeprazole':'acid_suppression','famotidine':'acid_suppression','furosemide':'diuretic','spironolactone':'diuretic',
 'carvedilol':'antihypertensive','metoprolol':'antihypertensive','hydralazine':'antihypertensive','nitroglycerin':'antihypertensive'
}
STATINS={'atorvastatin','simvastatin','pravastatin','rosuvastatin','lovastatin'}

COUNT_RULES={
 'glucagon_hyperglycemia_removed':'num_glucagon_hyperglycemia_removed',
 'metronidazole_non_gi_anchor_removed':'num_metronidazole_non_gi_anchor_removed',
 'anticoagulant_bad_anchor_removed':'num_anticoagulant_bad_anchor_removed',
 'sedative_bad_anchor_removed':'num_sedative_bad_anchor_removed',
 'statin_heart_failure_anchor_removed':'num_statin_heart_failure_anchor_removed',
 'acid_suppression_sepsis_anchor_removed':'num_acid_suppression_sepsis_anchor_removed',
}


def rel(p:Path)->str:
    try:return str(p.relative_to(ROOT))
    except ValueError:return str(p)

def low(s:Any)->str:return str(s or '').casefold().strip()
def has(text:str,terms:list[str]|set[str])->bool:return any(t in text for t in terms)
def exact_or_has(text:str,terms:list[str]|set[str])->bool:return has(text,terms)

def anchor_to_str(anchor:list[dict]|dict|None)->str:
    if isinstance(anchor,list): return json.dumps(anchor,ensure_ascii=False,sort_keys=True)
    if isinstance(anchor,dict): return json.dumps(anchor,ensure_ascii=False,sort_keys=True)
    return ''

def normalize_medication_function(medication:str, medication_function:str)->tuple[str,list[dict]]:
    fixed=MED_FUNCTION_FIX.get(low(medication), medication_function or 'unknown')
    events=[]
    if fixed!=medication_function:
        events.append({'action':'function_fixed','rule_name':'medication_function_fix','reason':f'{medication} -> {fixed}'})
    return fixed,events

def summarize_anchor(anchor_string:str)->str:
    t=low(anchor_string)
    mapping={
      'copd':'COPD','glucose_high':'high glucose','glucose_low':'low glucose','potassium_high':'high potassium','potassium_low':'low potassium',
      'wbc_high':'elevated WBC','lactate_high':'elevated lactate','heart failure':'heart failure','coronary artery disease':'coronary artery disease',
      'myocardial infarction':'myocardial infarction','r/o myocardial ischemia':'possible myocardial ischemia','atrial fibrillation':'atrial fibrillation',
      'sepsis':'sepsis','pneumonia':'pneumonia','wound infection':'wound infection','urinary tract infection':'urinary tract infection',
      'rheumatoid arthritis':'rheumatoid arthritis','acute respiratory failure':'acute respiratory failure','normal saline administration':'normal saline administration',
      'volume resuscitation':'volume resuscitation','gastrointestinal bleeding':'gastrointestinal bleeding','sinus tachycardia':'sinus tachycardia',
      'hyperglycemia':'hyperglycemia','hypoglycemia':'hypoglycemia'
    }
    if t in mapping:return mapping[t]
    for k,v in mapping.items():
        if k in t:return v
    return anchor_string.strip()

def is_reasonable_anchor_for_medication(medication:str, medication_function:str, anchor:dict)->tuple[bool,bool,str]:
    med=low(medication); fn=low(medication_function); typ=low(anchor.get('anchor_type')); txt=low(anchor.get('anchor_string'))
    weak=False
    if not txt or typ not in {'diagnosis','procedure','lab'}: return False,False,'invalid_anchor_format'
    # medication-specific overrides
    if 'metronidazole' in med:
        allow=['intra-abdominal infection','peritonitis','colitis','diarrhea due to infection','c. difficile','clostridioides difficile','clostridium difficile','abscess','bowel ischemia','ischemic bowel','cholecystitis','anaerobic']
        if has(txt,allow): return True,False,'metronidazole_gi_or_anaerobic_anchor'
        return False,False,'metronidazole_non_gi_anchor_removed'
    if 'glucagon' in med and has(txt,{'hyperglycemia','glucose_high','potassium_high'}): return False,False,'glucagon_hyperglycemia_removed'
    if med in STATINS and txt=='heart failure': return False,False,'statin_heart_failure_anchor_removed'
    if fn=='acid_suppression' and txt in {'sepsis','septic shock'}: return False,False,'acid_suppression_sepsis_anchor_removed'

    if fn=='anti_infective':
        allow=['sepsis','septic shock','pneumonia','urinary tract infection','wound infection','infected pressure ulcer','diarrhea due to infection','intra-abdominal infection','peritonitis','colitis','bacteremia','cellulitis','abscess','cholecystitis','meningitis','skin infection','bone and joint infection','soft tissue infection']
        proc=['blood culture','sputum culture','urine culture','wound culture','bronchoscopy','culture']
        bad=['coronary artery disease','atrial fibrillation','heart failure','pleural effusion','pneumothorax','glucose_high','glucose_low','sodium_high','sodium_low','potassium_high','potassium_low','platelet_low','hemoglobin_low','creatinine_high']
        if has(txt,bad): return False,False,'anti_infective_bad_anchor_removed'
        if typ=='lab' and txt in {'wbc_high','lactate_high'}: return True,True,'anti_infective_weak_lab_anchor'
        return (has(txt,allow) or (typ=='procedure' and has(txt,proc))),False,'anti_infective_anchor_rule'
    if fn=='vasopressor':
        if has(txt,['coronary artery disease','atrial fibrillation','creatinine_high','wbc_high','glucose_high','glucose_low','sodium_high','sodium_low','potassium_low','hemoglobin_low','platelet_low']): return False,False,'vasopressor_bad_anchor_removed'
        if txt=='lactate_high': return True,True,'vasopressor_weak_lactate_anchor'
        return has(txt,['shock','septic shock','cardiac arrest','hypotension']),False,'vasopressor_anchor_rule'
    if fn=='anticoagulant':
        if has(txt,['gastrointestinal bleeding','bleeding','platelet_low','coagulopathy','hemoglobin_low','anemia','aortic dissection','heart failure','sepsis','pneumonia','mechanical ventilation','wbc_high','sodium_high','sodium_low','glucose_high','glucose_low','potassium_low','pneumothorax','pleural effusion','compression boots']): return False,False,'anticoagulant_bad_anchor_removed'
        if 'warfarin' in med and has(txt,['myocardial infarction','acute coronary syndrome','r/o myocardial ischemia']): return False,False,'anticoagulant_bad_anchor_removed'
        if med in {'heparin','enoxaparin'} and has(txt,['myocardial infarction','acute coronary syndrome','r/o myocardial ischemia']): return True,True,'heparin_enoxaparin_weak_acs_anchor'
        return has(txt,['atrial fibrillation','atrial flutter','dvt','deep vein thrombosis','pulmonary embolism','embolism','thrombosis','hypercoagulable state']),False,'anticoagulant_anchor_rule'
    if fn=='antiplatelet':
        if has(txt,['sepsis','pneumonia','gastrointestinal bleeding','bleeding','hemoglobin_low','platelet_low','glucose_high','glucose_low','wbc_high','sodium_high','sodium_low','atrial fibrillation']): return False,False,'antiplatelet_bad_anchor_removed'
        return has(txt,['myocardial infarction','acute coronary syndrome','coronary artery disease','coronary','myocardial ischemia','r/o myocardial ischemia','ischemic stroke','cerebral infarct','stent','s/p ptca']),False,'antiplatelet_anchor_rule'
    if fn=='antiarrhythmic':
        if has(txt,['hemoglobin_low','platelet_low','glucose_high','glucose_low','sodium_high','sodium_low','wbc_high','sepsis','pneumonia']): return False,False,'antiarrhythmic_bad_anchor_removed'
        return has(txt,['atrial fibrillation','atrial flutter','ventricular tachycardia','ventricular fibrillation','arrhythmia','bradycardia','tachycardia','sinus tachycardia','cardiac arrest']),False,'antiarrhythmic_anchor_rule'
    if fn=='bronchodilator':
        if has(txt,['wbc_high','glucose_high','glucose_low','sodium_high','sodium_low','potassium_low','platelet_low','hemoglobin_low','coronary artery disease','atrial fibrillation']): return False,False,'bronchodilator_bad_anchor_removed'
        return has(txt,['copd','asthma','bronchospasm','respiratory failure','acute respiratory failure','pneumonia','mechanical ventilation','intubation']),False,'bronchodilator_anchor_rule'
    if fn=='steroid':
        if has(txt,['wbc_high','glucose_high','glucose_low','sodium_high','sodium_low','platelet_low','hemoglobin_low','coronary artery disease','atrial fibrillation']): return False,False,'steroid_bad_anchor_removed'
        return has(txt,['copd','asthma','bronchospasm','respiratory failure','acute respiratory failure','septic shock','shock','pneumonia','meningitis','transplant','autoimmune','inflammatory','rheumatoid arthritis']),False,'steroid_anchor_rule'
    if fn=='glucose_control':
        if has(txt,['potassium_low','sodium_high','sodium_low','creatinine_high','wbc_high','lactate_high']): return False,False,'glucose_control_bad_anchor_removed'
        if 'dextrose' in med or 'd-glucose' in med: return has(txt,['glucose_low','hypoglycemia','potassium_high','hyperkalemia']),False,'dextrose_anchor_rule'
        if 'glucagon' in med: return has(txt,['glucose_low','hypoglycemia']),False,'glucagon_anchor_rule'
        return has(txt,['glucose_high','hyperglycemia','glucose_low','hypoglycemia','potassium_high','hyperkalemia']),False,'glucose_control_anchor_rule'
    if fn=='endocrine_thyroid':
        if has(txt,['heart failure','coronary artery disease','atrial fibrillation','arrhythmia','pneumonia','sepsis','renal failure','pleural effusion','pneumothorax','glucose_high','glucose_low','wbc_high','platelet_low','hemoglobin_low']): return False,False,'endocrine_thyroid_bad_anchor_removed'
        return has(txt,['hypothyroidism','thyroid','myxedema','endocrine|thyroid','thyroid disease']),False,'endocrine_thyroid_anchor_rule'
    if fn=='cardiac_medication':
        if med in STATINS:
            if has(txt,['heart failure','pleural effusion','pneumothorax','sepsis','pneumonia']): return False,False,'statin_heart_failure_anchor_removed' if txt=='heart failure' else 'statin_bad_anchor_removed'
            return has(txt,['coronary artery disease','myocardial infarction','acute coronary syndrome','myocardial ischemia','r/o myocardial ischemia','hyperlipidemia','ischemic stroke','atherosclerosis']),False,'statin_anchor_rule'
        return has(txt,['heart failure','atrial fibrillation','arrhythmia']),False,'cardiac_medication_anchor_rule'
    if fn=='antihypertensive':
        return has(txt,['hypertension','myocardial infarction','acute coronary syndrome','r/o myocardial ischemia','sinus tachycardia','tachycardia','heart failure','cardiomyopathy','shock','cardiac arrest','atrial fibrillation']),False,'antihypertensive_anchor_rule'
    if fn=='diuretic':
        if has(txt,['myocardial infarction']) and not has(txt,['heart failure','edema','fluid overload','volume overload','pulmonary edema']): return False,False,'diuretic_mi_alone_removed'
        return has(txt,['heart failure','fluid overload','volume overload','pulmonary edema','edema','renal failure with volume overload']),False,'diuretic_anchor_rule'
    if fn=='fluid_electrolyte':
        if 'potassium chloride' in med: return txt=='potassium_low',False,'potassium_chloride_anchor_rule'
        if 'sodium chloride' in med: return has(txt,['sodium_low','dehydration','shock','volume resuscitation','normal saline administration','fluid resuscitation']),False,'sodium_chloride_anchor_rule'
        if 'lactated' in med: return has(txt,['shock','volume resuscitation','normal saline administration']),False,'lactated_ringers_anchor_rule'
        if 'magnesium' in med: return txt=='magnesium_low',False,'magnesium_anchor_rule'
        if 'calcium' in med: return txt=='calcium_low',False,'calcium_anchor_rule'
        return False,False,'fluid_electrolyte_unmatched_anchor_removed'
    if fn=='acid_suppression': return has(txt,['gastrointestinal bleeding','gi bleeding','stress ulcer','stress ulcer prophylaxis','upper gi bleed','ulcer']),False,'acid_suppression_anchor_rule'
    if fn=='sedative_analgesic':
        if has(txt,['cardiology consultation','dementia','extremity','coronary artery disease','hypertension','wbc_high','glucose_high']) or txt in {'shock','sepsis'}: return False,False,'sedative_bad_anchor_removed'
        return has(txt,['mechanical ventilation','intubation','procedure pain','surgery','postoperative pain','pain','agitation','delirium with agitation','sedation']),False,'sedative_anchor_rule'
    if fn=='bowel_regimen': return has(txt,['constipation','ileus','bowel regimen']),False,'bowel_regimen_anchor_rule'
    if fn=='analgesic_antipyretic': return has(txt,['fever','pain','headache','postoperative pain']),False,'analgesic_antipyretic_anchor_rule'
    return False,False,'unknown_function_anchor_removed'

def rewrite_attribution_hypothesis(medication:str, medication_function:str, anchor:list, weak:bool)->str:
    if not anchor:
        return f'{medication} has no reliable patient-context anchor after validation. This record remains unattributed; candidate attribution is not confirmed prescription intent.'
    summary=summarize_anchor(str(anchor[0].get('anchor_string','')))
    phrase=FUNCTION_PHRASE.get(medication_function,'uncertain')
    if weak:
        return f'{medication} weakly reflects a clinically relevant patient-context anchor involving {summary}. This is a weak candidate attribution, not confirmed prescription intent.'
    return f'{medication} plausibly reflects a {phrase} patient-context pattern involving {summary}. This is a candidate attribution, not confirmed prescription intent.'

def clean_l3_record(record:dict)->tuple[dict,list[dict]]:
    events=[]
    before_fn=str(record.get('medication_function',''))
    med=str(record.get('medication',''))
    fixed_fn,fn_events=normalize_medication_function(med,before_fn)
    events.extend(fn_events)
    anchor_before=record.get('anchor') if isinstance(record.get('anchor'),list) else []
    anchor=(anchor_before[:1] if anchor_before else [])
    weak=False
    if anchor:
        keep,weak,reason=is_reasonable_anchor_for_medication(med,fixed_fn,anchor[0])
        if keep:
            events.append({'action':'anchor_kept','rule_name':reason,'reason':reason})
        else:
            events.append({'action':'anchor_removed_unreasonable','rule_name':reason,'reason':reason})
            anchor=[]
    else:
        events.append({'action':'record_unattributed','rule_name':'no_anchor_input','reason':'input_anchor_empty'})
    hyp=rewrite_attribution_hypothesis(med,fixed_fn,anchor,weak)
    events.append({'action':'hypothesis_rewritten','rule_name':'rewrite_from_cleaned_anchor','reason':'do_not_trust_original_llm_hypothesis'})
    cleaned={
        'stay_id':str(record.get('stay_id','')),
        'hospital_id':str(record.get('hospital_id','')),
        'medication':med,
        'medication_function':fixed_fn,
        'anchor':anchor,
        'attribution_hypothesis':hyp,
    }
    return cleaned,events

def forbidden_hits(text:str)->list[str]:
    l=low(text); return [x for x in FORBIDDEN if x in l]

def validate_schema(records:list[dict])->tuple[bool,bool,bool,int,int]:
    top_ok=True; anchor_ok=True; max_anchor=0; bad=0; too_many=0
    for obj in records:
        if set(obj)!=set(TOP_FIELDS): top_ok=False; bad+=1
        if not isinstance(obj.get('anchor'),list): anchor_ok=False; bad+=1; continue
        max_anchor=max(max_anchor,len(obj['anchor']))
        if len(obj['anchor'])>1: too_many+=1
        for a in obj['anchor']:
            if set(a)!=ANCHOR_FIELDS or a.get('anchor_type') not in {'diagnosis','procedure','lab'}:
                anchor_ok=False; bad+=1
    return top_ok,anchor_ok,(too_many==0),max_anchor,bad

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-file',required=True); ap.add_argument('--output-file',required=True); ap.add_argument('--audit-file',required=True); ap.add_argument('--report-file',required=True)
    ap.add_argument('--high-risk-file',required=True); ap.add_argument('--null-anchor-file',required=True); ap.add_argument('--unsafe-hypothesis-file',required=True)
    args=ap.parse_args()
    paths={k:Path(v) for k,v in vars(args).items() if k.endswith('_file') or k in {'input_file','output_file','audit_file','report_file'}}
    for key,path in paths.items():
        if key!='input_file': path.parent.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in Path(args.input_file).open(encoding='utf-8') if line.strip()]
    cleaned=[]; audit=[]; high=[]; nulls=[]; unsafe=[]
    report={
      'step':'postprocess_l3_reasonable_anchor','input_file':rel(Path(args.input_file)),'output_file':rel(Path(args.output_file)),
      'num_records_input':len(rows),'num_records_output':0,'num_records_with_anchor_before':0,'num_records_with_anchor_after':0,
      'num_records_without_anchor_before':0,'num_records_without_anchor_after':0,'num_anchor_kept':0,'num_anchor_removed_unreasonable':0,
      'num_function_fixed':0,'num_hypothesis_rewritten':0,'num_records_unattributed':0,
      'num_glucagon_hyperglycemia_removed':0,'num_metronidazole_non_gi_anchor_removed':0,'num_anticoagulant_bad_anchor_removed':0,
      'num_sedative_bad_anchor_removed':0,'num_statin_heart_failure_anchor_removed':0,'num_acid_suppression_sepsis_anchor_removed':0,
      'schema_unchanged':True,'top_level_fields_strict':True,'anchor_format_unchanged':True,'max_anchor_per_record':1,
      'used_api':False,'used_llm':False,'used_condition_group':False,'modified_original_l3_file':False,'modified_stage1_files':False,
      'l3_definition':'L3 is a drug-conditioned patient-context attribution memory. Offline construction direction: medication -> anchor. The selected anchor is candidate patient-context support, not confirmed prescription reason or ground-truth indication. anchor=[] is allowed.',
      'num_forbidden_phrase_hits_after':0,'final_pass':False
    }
    for rec in rows:
        report['num_records_with_anchor_before']+=1 if rec.get('anchor') else 0
        report['num_records_without_anchor_before']+=0 if rec.get('anchor') else 1
        clean,events=clean_l3_record(rec); cleaned.append(clean)
        report['num_records_with_anchor_after']+=1 if clean.get('anchor') else 0
        report['num_records_without_anchor_after']+=0 if clean.get('anchor') else 1
        if not clean.get('anchor'):
            report['num_records_unattributed']+=1; nulls.append(clean)
        hits=forbidden_hits(clean.get('attribution_hypothesis',''))
        if hits:
            report['num_forbidden_phrase_hits_after']+=len(hits); unsafe.append({'record':clean,'hits':hits})
        for e in events:
            action=e['action']
            if action=='anchor_kept': report['num_anchor_kept']+=1
            if action=='anchor_removed_unreasonable':
                report['num_anchor_removed_unreasonable']+=1; high.append({'record_before':rec,'record_after':clean,'reason':e['reason']})
                key=COUNT_RULES.get(e['rule_name'])
                if key: report[key]+=1
            if action=='function_fixed': report['num_function_fixed']+=1
            if action=='hypothesis_rewritten': report['num_hypothesis_rewritten']+=1
            audit.append({
              'stay_id':rec.get('stay_id',''),'hospital_id':rec.get('hospital_id',''),'medication':rec.get('medication',''),
              'medication_function_before':rec.get('medication_function',''),'medication_function_after':clean.get('medication_function',''),
              'anchor_before':anchor_to_str(rec.get('anchor')),'anchor_after':anchor_to_str(clean.get('anchor')),
              'weak_before':False,'weak_after':('weakly reflects' in clean.get('attribution_hypothesis','')),
              'action':action,'rule_name':e.get('rule_name',''),'reason':e.get('reason',''),
              'hypothesis_before':rec.get('attribution_hypothesis',''),'hypothesis_after':clean.get('attribution_hypothesis','')
            })
    Path(args.output_file).write_text(''.join(json.dumps({k:r[k] for k in TOP_FIELDS},ensure_ascii=False,allow_nan=False)+'\n' for r in cleaned),encoding='utf-8')
    fieldnames=['stay_id','hospital_id','medication','medication_function_before','medication_function_after','anchor_before','anchor_after','weak_before','weak_after','action','rule_name','reason','hypothesis_before','hypothesis_after']
    with Path(args.audit_file).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(audit)
    Path(args.high_risk_file).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in high),encoding='utf-8')
    Path(args.null_anchor_file).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in nulls),encoding='utf-8')
    Path(args.unsafe_hypothesis_file).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in unsafe),encoding='utf-8')
    top_ok,anchor_ok,one_ok,max_anchor,bad=validate_schema(cleaned)
    report['num_records_output']=len(cleaned); report['top_level_fields_strict']=top_ok; report['anchor_format_unchanged']=anchor_ok; report['max_anchor_per_record']=max_anchor
    report['schema_unchanged']=top_ok and anchor_ok and one_ok and bad==0
    report['final_pass']=report['schema_unchanged'] and report['num_forbidden_phrase_hits_after']==0 and len(cleaned)==len(rows)
    Path(args.report_file).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__': main()
