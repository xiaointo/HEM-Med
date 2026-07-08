#!/usr/bin/env python3
"""Recover compact L3 from existing DeepSeek raw responses without API calls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
FINAL_FIELDS=['stay_id','hospital_id','medication','medication_function','anchor','attribution_hypothesis']
NULL_STRINGS={'','null','none','nil','n/a','na','no anchor','no_anchor'}
FUNCTION_PHRASE={
 'anti_infective':'infection-related','anticoagulant':'coagulation-related','antiplatelet':'cardiovascular or thrombosis-related',
 'antiarrhythmic':'arrhythmia-related','antihypertensive':'cardiovascular','cardiac_medication':'cardiac','steroid':'inflammatory or respiratory',
 'vasopressor':'hemodynamic support','bronchodilator':'respiratory','glucose_control':'glucose-control','sedative_analgesic':'sedation or analgesia',
 'fluid_electrolyte':'fluid or electrolyte support','acid_suppression':'gastrointestinal or acid-suppression','nutrition':'nutrition-support',
 'bowel_regimen':'supportive-care','diuretic':'fluid-management','endocrine_thyroid':'thyroid- or endocrine-related','analgesic_antipyretic':'analgesic or antipyretic','unknown':'uncertain'
}


def rel(p:Path)->str:
    try:return str(p.relative_to(ROOT))
    except ValueError:return str(p)

def load_jsonl(path:Path)->list[dict[str,Any]]:
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]

def strip_fence(s:str)->str:
    m=re.match(r'^```(?:json)?\s*(.*?)\s*```$',str(s).strip(),re.S|re.I)
    return m.group(1) if m else str(s).strip()

def norm_med_list(case:dict[str,Any])->list[tuple[str,str]]:
    out=[]
    for item in case.get('med') or []:
        if isinstance(item,(list,tuple)) and len(item)>=2: out.append((str(item[0]).strip(),str(item[1]).strip()))
        elif isinstance(item,dict): out.append((str(item.get('medication') or item.get('m') or '').strip(),str(item.get('medication_function') or item.get('f') or 'unknown').strip()))
    return [(m,f or 'unknown') for m,f in out if m]

def load_input_cases(path:Path)->dict[str,dict[str,Any]]:
    cases={}
    for row in load_jsonl(path):
        sid=str(row.get('id') or row.get('stay_id') or row.get('patientunitstayid') or '')
        if sid: cases[sid]=row
    return cases

def med_function_map(case:dict[str,Any])->dict[str,str]:
    return {m.casefold():f for m,f in norm_med_list(case)}

def target_from_raw(raw:dict[str,Any], case:dict[str,Any])->list[tuple[str,str]]:
    targets=[]
    for item in raw.get('target_medications') or []:
        if isinstance(item,str) and '|' in item:
            m,f=item.split('|',1); targets.append((m.strip(),f.strip() or 'unknown'))
    if targets: return targets
    return norm_med_list(case)[:10]

def parse_response(raw:dict[str,Any])->dict[str,Any]|None:
    parsed=raw.get('parsed_response')
    if isinstance(parsed,dict): return parsed
    text=raw.get('raw_response','')
    try:return json.loads(strip_fence(text))
    except Exception:return None

def normalize_anchor_id(v:Any)->str|None|list:
    if v is None: return None
    if isinstance(v,list): return v
    if isinstance(v,dict): return v
    s=str(v).strip()
    return None if s.casefold() in NULL_STRINGS else s

def summarize_anchor(anchor_string:str)->str:
    t=str(anchor_string or '').casefold().strip()
    mapping={'copd':'COPD','glucose_high':'high glucose','glucose_low':'low glucose','potassium_high':'high potassium','potassium_low':'low potassium','wbc_high':'elevated WBC','lactate_high':'elevated lactate','heart failure':'heart failure','coronary artery disease':'coronary artery disease','myocardial infarction':'myocardial infarction','r/o myocardial ischemia':'possible myocardial ischemia','atrial fibrillation':'atrial fibrillation','sepsis':'sepsis','pneumonia':'pneumonia','wound infection':'wound infection','urinary tract infection':'urinary tract infection','acute respiratory failure':'acute respiratory failure','normal saline administration':'normal saline administration','volume resuscitation':'volume resuscitation','gastrointestinal bleeding':'gastrointestinal bleeding','sinus tachycardia':'sinus tachycardia','hyperglycemia':'hyperglycemia','hypoglycemia':'hypoglycemia'}
    if t in mapping:return mapping[t]
    for k,v in mapping.items():
        if k in t:return v
    return str(anchor_string).strip()

def rewrite_hypothesis(medication:str, medication_function:str, anchor:list[dict[str,Any]])->str:
    if not anchor:
        return f'{medication} has no reliable patient-context anchor after validation. This record remains unattributed; candidate attribution is not confirmed prescription intent.'
    phrase=FUNCTION_PHRASE.get(medication_function,'uncertain')
    summary=summarize_anchor(anchor[0].get('anchor_string',''))
    return f'{medication} is plausibly associated with a {phrase} patient-context pattern involving {summary}. This is a candidate attribution, not confirmed prescription intent.'

def restore_anchor(anchor_id:Any,mapping:dict[str,dict[str,Any]],raw:dict[str,Any],medication:str,illegal_rows:list[dict[str,Any]])->list[dict[str,Any]]:
    aid=normalize_anchor_id(anchor_id)
    if aid is None: return []
    if isinstance(aid,(list,dict)):
        illegal_rows.append({'prompt_id':raw.get('prompt_id'),'stay_id':raw.get('stay_id'),'medication':medication,'anchor_id':aid,'reason':'anchor_id_not_scalar'})
        return []
    if aid not in mapping:
        illegal_rows.append({'prompt_id':raw.get('prompt_id'),'stay_id':raw.get('stay_id'),'medication':medication,'anchor_id':aid,'reason':'anchor_id_not_in_case_map'})
        return []
    a=mapping[aid]
    return [{'anchor_id':aid,'anchor_type':a.get('anchor_type') or a.get('type'),'anchor_string':a.get('anchor_string') or a.get('text')}]

def find_llm_record(rows:list[Any],target_med:str,seen:set[int])->tuple[dict[str,Any]|None,int|None]:
    t=target_med.casefold()
    for i,row in enumerate(rows):
        if i in seen or not isinstance(row,dict): continue
        med=str(row.get('medication') or row.get('m') or '').strip()
        if '|' in med: med=med.split('|',1)[0].strip()
        if med.casefold()==t:
            return row,i
    return None,None

def write_jsonl(path:Path,rows:list[Any])->None:
    path.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rows),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--raw-responses',required=True);ap.add_argument('--input-file',required=True);ap.add_argument('--output-file',required=True)
    ap.add_argument('--schema-error-file',required=True);ap.add_argument('--illegal-anchor-file',required=True);ap.add_argument('--null-anchor-file',required=True);ap.add_argument('--report-file',required=True)
    args=ap.parse_args()
    raw_rows=load_jsonl(Path(args.raw_responses)); cases=load_input_cases(Path(args.input_file))
    out=[]; schema=[]; illegal=[]; nulls=[]
    for raw in raw_rows:
        sid=str(raw.get('stay_id') or '')
        case=cases.get(sid)
        if not case:
            schema.append({'prompt_id':raw.get('prompt_id'),'stay_id':sid,'reason':'input_case_not_found'}); continue
        parsed=parse_response(raw)
        if not isinstance(parsed,dict):
            schema.append({'prompt_id':raw.get('prompt_id'),'stay_id':sid,'reason':'parsed_response_not_object'}); continue
        rows=parsed.get('records')
        if not isinstance(rows,list):
            schema.append({'prompt_id':raw.get('prompt_id'),'stay_id':sid,'reason':'records_not_list','parsed_response':parsed}); rows=[]
        target=target_from_raw(raw,case); fn_map=med_function_map(case); mapping=raw.get('anchor_id_map') or {}
        seen=set()
        for med,fn_raw in target:
            fn=fn_raw or fn_map.get(med.casefold(),'unknown')
            row,idx=find_llm_record(rows,med,seen)
            if idx is not None: seen.add(idx)
            if row is None:
                anchor=[]
            else:
                anchor_id=row.get('anchor_id', row.get('a'))
                anchor=restore_anchor(anchor_id,mapping,raw,med,illegal)
            rec={'stay_id':str(case.get('id') or sid),'hospital_id':str(case.get('hid') or raw.get('hospital_id') or ''),'medication':med,'medication_function':fn,'anchor':anchor,'attribution_hypothesis':rewrite_hypothesis(med,fn,anchor)}
            out.append({k:rec[k] for k in FINAL_FIELDS})
            if not anchor: nulls.append(rec)
        for i,row in enumerate(rows):
            if i not in seen and isinstance(row,dict) and (row.get('medication') or row.get('m')):
                schema.append({'prompt_id':raw.get('prompt_id'),'stay_id':sid,'reason':'unknown_or_extra_medication_record','record':row})
    write_jsonl(Path(args.output_file),out); write_jsonl(Path(args.schema_error_file),schema); write_jsonl(Path(args.illegal_anchor_file),illegal); write_jsonl(Path(args.null_anchor_file),nulls)
    report={'step':'recover_l3_from_raw_responses','used_api':False,'used_llm':False,'recovered_from_raw_responses':True,'raw_responses_file':rel(Path(args.raw_responses)),'input_file':rel(Path(args.input_file)),'output_file':rel(Path(args.output_file)),'num_raw_cases':len(raw_rows),'num_records_recovered':len(out),'num_schema_error_records':len(schema),'num_illegal_anchor_records':len(illegal),'num_null_anchor_records':len(nulls),'modified_original_l3_file':False,'modified_stage1_files':False,'used_condition_group':False}
    Path(args.report_file).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__': main()
