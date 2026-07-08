#!/usr/bin/env python3
"""Step 3.4: real DeepSeek min-token, single-anchor L3 generation using AsyncOpenAI."""
from __future__ import annotations

import argparse
import asyncio
import csv
import importlib.util
import httpx
import json
import os
import re
import runpy
import sqlite3
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
except Exception:
    APIConnectionError = APIStatusError = APITimeoutError = RateLimitError = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
BUILDER_PATH = ROOT / 'scripts/deepseek_l3_test20_dx_px_nonempty_min_token_prompt_audit/01_build_min_token_prompts_test20.py'
# Load the helper builder without modifying the helper file itself.
# The helper is loaded from the local L3 construction tree.
# patch that path in memory only for this process.
builder = types.ModuleType('min_token_builder')
_builder_src = BUILDER_PATH.read_text(encoding='utf-8')
exec(compile(_builder_src, str(BUILDER_PATH), 'exec'), builder.__dict__)
normalizer = builder.base

DEFAULT_INPUT = ROOT / 'outputs/deepseek_l3_minimal_case_inputs/stay_level_all_minimal_for_llm_dx_px_nonempty.jsonl'
DEFAULT_OUT = ROOT / 'outputs/deepseek_l3_min_token_single_anchor_run'
PLACEHOLDER = 'YOUR_DEEPSEEK_API_KEY_HERE'
PROMPT_VERSION = 'v2_no_medication_procedure_anchor'
INPUT_PRICE = 0.14
OUTPUT_PRICE = 0.28
USD_RMB = 7.2
MAX_CONSECUTIVE_FAILURES = 20
RETRY_DELAYS = [2, 5, 10]

UNSAFE = [
    'was prescribed for', 'was given to treat', 'was used to treat', 'the reason is',
    'because the patient had', 'the indication is definitely', 'for shock', 'for sepsis',
    'for infection', 'treatment for',
]
FINAL_FIELDS = ['stay_id', 'hospital_id', 'medication', 'medication_function', 'anchor', 'attribution_hypothesis']
RECORD_FIELDS = {'medication', 'medication_function', 'anchor_id', 'attribution_hypothesis'}

SYSTEM_PROMPT_V2 = """You are a clinical data attribution assistant for ICU medication recommendation research.
eICU does not state true prescribing intent.
For each medication, choose exactly one best anchor_id from D/P/L IDs in the given context, or null if no suitable anchor exists.
Do not invent anchors or medications.
Prefer diagnosis anchors first, then non-drug clinical support procedure anchors, then clinically relevant lab abnormality anchors.
Do not choose medication-administration procedure anchors when the anchor text mainly names the same medication, another medication, or a medication class.
Valid procedure anchors are non-drug clinical support procedures such as mechanical ventilation, dialysis, vasopressor therapy, blood transfusion, oxygen/respiratory support, or volume resuscitation.
Use vasopressor therapy only as a hemodynamic-support context, not as a medication-administration anchor for a specific vasopressor drug.
If only medication-administration procedure anchors are available, use null.
Do not force a weak diagnosis anchor just to avoid null.
Use cautious association wording, not causal treatment claims.
Return JSON only."""

TASK_V2 = """For each target medication, return one record.
Medication must come from target_medications.
For each medication, select exactly one best anchor_id from context IDs, or null if no suitable anchor exists.
anchor_id must be one D/P/L ID from context, or null.
Do not output anchor strings, anchor types, full anchor objects, or multiple anchor IDs.
Default anchor priority: diagnosis > non-drug clinical support procedure > clinically relevant lab abnormality > null.
For glucose-control, electrolyte, and lab-driven supportive medications, clinically relevant lab abnormalities may be selected before procedures.
If a diagnosis is present but only weakly related to the target medication, do not select it merely because diagnosis has higher priority.
Do not select a procedure anchor if its text contains the current target medication name.
Do not select a procedure anchor if its text contains another drug name and does not describe a non-drug clinical support state.
Avoid medication-procedure anchors such as therapeutic antibacterials, vancomycin, piperacillin/tazobactam, carbapenem, glucocorticoid administration, bronchodilator, stress ulcer treatment, famotidine, pantoprazole, antihyperlipidemic agent, hmg-coa reductase inhibitor, insulin / glucose, analgesics, sedatives, or medication administration.
Valid procedure anchors include mechanical ventilation, dialysis, vasopressor therapy, blood transfusion, oxygen support, respiratory support, volume resuscitation, and other non-drug clinical support procedures.
If the only plausible anchors are medication-procedure anchors, return anchor_id null.
Do not claim true prescribing intent.
Avoid: was prescribed for, was given to treat, was used to treat, because the patient had, the reason is, for shock, for sepsis, for infection, treatment for.
Use cautious wording: is associated with, appears consistent with, may reflect, is plausibly linked to.
Mention exact prescribing intent is not directly available from eICU.
Return JSON only."""

MEDICATION_PROCEDURE_KEYWORDS = [
    'medications', 'medication administration', 'therapeutic antibacterials',
    'bronchodilator', 'stress ulcer prophylaxis', 'stress ulcer treatment',
    'famotidine', 'pantoprazole', 'analgesics', 'sedatives', 'hmg-coa',
    'atorvastatin', 'antihyperlipidemic', 'glucocorticoid', 'glucocorticoids',
    'insulin / glucose', 'insulin', 'vancomycin', 'piperacillin', 'tazobactam',
    'carbapenem', 'macrolide', 'azithromycin', 'therapeutic antibacterials',
]
VALID_SUPPORT_PROCEDURE_KEYWORDS = [
    'mechanical ventilation', 'dialysis', 'vasopressor therapy', 'blood transfusion',
    'cardioversion', 'cardiology consultation', 'surgery', 'oxygen',
    'respiratory support', 'volume resuscitation', 'intubation', 'central venous catheter',
]
BAD_MEDICATION_ANCHOR_KEYWORDS = MEDICATION_PROCEDURE_KEYWORDS
NULL_STRINGS = {'', 'null', 'none', 'nil', 'na', 'n/a', 'no anchor', 'no_anchor'}


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _split_context_item(item: str) -> tuple[str, str] | None:
    m = re.match(r'^([DPL]\d+):\s*(.+)$', str(item).strip())
    if not m:
        return None
    return m.group(1), m.group(2).strip()


def _last_path_piece(text: str) -> str:
    parts = [x.strip() for x in str(text).split('|') if x.strip()]
    return parts[-1] if parts else str(text).strip()


def clean_diagnosis_for_prompt(item: str) -> str:
    parsed = _split_context_item(item)
    if not parsed:
        return item
    anchor_id, text = parsed
    return f'{anchor_id}: {_last_path_piece(text)}'


def is_medication_procedure_px(text: str) -> bool:
    lower = str(text).casefold()
    has_med_keyword = any(k in lower for k in MEDICATION_PROCEDURE_KEYWORDS)
    has_support_keyword = any(k in lower for k in VALID_SUPPORT_PROCEDURE_KEYWORDS)
    return has_med_keyword and not has_support_keyword


def clean_procedure_for_prompt(item: str) -> str | None:
    parsed = _split_context_item(item)
    if not parsed:
        return item
    anchor_id, text = parsed
    if is_medication_procedure_px(text):
        return None
    lower = text.casefold()
    for keyword in VALID_SUPPORT_PROCEDURE_KEYWORDS:
        if keyword in lower:
            return f'{anchor_id}: {keyword}'
    return f'{anchor_id}: {_last_path_piece(text)}'


def clean_context_for_prompt(ctx_raw: dict[str, list[str]], args: argparse.Namespace) -> dict[str, list[str]]:
    dx = dedupe_keep_order([clean_diagnosis_for_prompt(x) for x in ctx_raw.get('dx', [])])
    px = []
    for item in ctx_raw.get('px', []):
        cleaned = clean_procedure_for_prompt(item)
        if cleaned:
            px.append(cleaned)
    lab = dedupe_keep_order(list(ctx_raw.get('lab', [])))[:args.max_lab_items]
    return {'dx': dx, 'px': dedupe_keep_order(px), 'lab': lab}


def normalize_llm_record_before_validation(record: dict[str, Any]) -> dict[str, Any]:
    record = dict(record)
    med = str(record.get('medication', '')).strip()
    fn = str(record.get('medication_function', '')).strip()
    if '|' in med:
        left, right = med.split('|', 1)
        record['medication'] = left.strip()
        if not fn:
            record['medication_function'] = right.strip()
    if 'anchor_ids' in record:
        anchor_ids = record.get('anchor_ids')
        if anchor_ids in (None, ''):
            record['anchor_id'] = None
        elif isinstance(anchor_ids, list):
            if len(anchor_ids) == 0:
                record['anchor_id'] = None
            elif len(anchor_ids) == 1:
                record['anchor_id'] = anchor_ids[0]
            else:
                record['anchor_id'] = anchor_ids
        else:
            record['anchor_id'] = anchor_ids
        record.pop('anchor_ids', None)
    anchor_id = record.get('anchor_id')
    if isinstance(anchor_id, str) and anchor_id.strip().casefold() in NULL_STRINGS:
        record['anchor_id'] = None
    return record


def sanitize_hypothesis(hyp: str) -> tuple[str, list[str]]:
    original = str(hyp or '').strip()
    lower = original.casefold()
    hits = [x for x in UNSAFE if x in lower]
    if not hits:
        return original, []
    cleaned = original
    replacements = {
        'was prescribed for': 'is plausibly associated with',
        'was given to treat': 'is plausibly associated with',
        'was used to treat': 'is plausibly associated with',
        'the reason is': 'the candidate association involves',
        'because the patient had': 'in the patient context of',
        'the indication is definitely': 'the candidate attribution may involve',
        'for shock': 'with shock',
        'for sepsis': 'with sepsis',
        'for infection': 'with infection',
        'treatment for': 'candidate association with',
    }
    for bad, good in replacements.items():
        cleaned = re.sub(re.escape(bad), good, cleaned, flags=re.I)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if 'candidate attribution' not in cleaned.casefold():
        cleaned = cleaned.rstrip('.') + '. This is a candidate attribution, not confirmed prescription intent.'
    return cleaned, hits


def is_bad_medication_procedure_anchor(anchor: dict[str, Any], medication: str) -> bool:
    if str(anchor.get('anchor_type', '')).casefold() != 'procedure':
        return False
    text = str(anchor.get('anchor_string', '')).casefold()
    med = str(medication or '').casefold().strip()
    has_support_keyword = any(k in text for k in VALID_SUPPORT_PROCEDURE_KEYWORDS)
    if med and med in text and not has_support_keyword:
        return True
    return any(k in text for k in BAD_MEDICATION_ANCHOR_KEYWORDS) and not has_support_keyword


def load_deepseek_api_key() -> tuple[str, str]:
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if key and key != PLACEHOLDER:
        return key, 'env:DEEPSEEK_API_KEY'
    key_file = ROOT / 'configs' / 'private' / 'deepseek_key.py'
    if key_file.exists():
        try:
            ns = runpy.run_path(str(key_file))
            key = str(ns.get('DEEPSEEK_API_KEY', '')).strip()
            if key and key != PLACEHOLDER:
                return key, 'configs/private/deepseek_key.py'
        except Exception as e:
            return '', f'key_file_read_error:{type(e).__name__}'
    return '', 'not_found'


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_path(v: str | Path) -> Path:
    p = Path(v)
    return p if p.is_absolute() else ROOT / p


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def parse_bool(v: Any) -> bool:
    return str(v).strip().casefold() in {'1', 'true', 'yes', 'y'}


def strip_fence(s: str) -> str:
    m = re.match(r'^```(?:json)?\s*(.*?)\s*```$', s.strip(), re.S | re.I)
    return m.group(1) if m else s.strip()


def append_jsonl(p: Path, obj: Any) -> None:
    with p.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')


def init_files(paths: dict[str, Path]) -> None:
    for key, path in paths.items():
        if key not in {'report', 'checkpoint', 'tokens'}:
            path.touch(exist_ok=True)
    if not paths['tokens'].exists():
        with paths['tokens'].open('w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow([
                'prompt_id', 'stay_id', 'hospital_id', 'num_target_meds', 'num_context_ids',
                'prompt_chars', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'api_status'
            ])


def init_checkpoint(p: Path) -> None:
    sql = (
        'CREATE TABLE IF NOT EXISTS cases('
        'prompt_id TEXT PRIMARY KEY, stay_id TEXT, status TEXT, '
        'response_saved INTEGER, error_message TEXT, updated_at TEXT)'
    )
    with sqlite3.connect(p) as c:
        c.execute(sql)


def checkpoint_status(p: Path, pid: str) -> tuple[str | None, int]:
    with sqlite3.connect(p) as c:
        row = c.execute('SELECT status,response_saved FROM cases WHERE prompt_id=?', (pid,)).fetchone()
    return row if row else (None, 0)


def set_checkpoint(p: Path, pid: str, stay: str, status: str, saved: bool, error: str = '') -> None:
    sql = (
        'INSERT INTO cases VALUES(?,?,?,?,?,?) '
        'ON CONFLICT(prompt_id) DO UPDATE SET '
        'status=excluded.status,response_saved=excluded.response_saved,'
        'error_message=excluded.error_message,updated_at=excluded.updated_at'
    )
    with sqlite3.connect(p) as c:
        c.execute(sql, (pid, stay, status, int(saved), error[:2000], now()))


def output_paths(out: Path, tag: str) -> dict[str, Path]:
    return {
        'l3': out / f'L3_deepseek_generated_{tag}.jsonl',
        'raw': out / f'raw_responses_{tag}.jsonl',
        'failed': out / f'failed_records_{tag}.jsonl',
        'schema': out / f'schema_error_records_{tag}.jsonl',
        'illegal': out / f'illegal_anchor_records_{tag}.jsonl',
        'unsafe': out / f'unsafe_causal_language_records_{tag}.jsonl',
        'null': out / f'null_anchor_records_{tag}.jsonl',
        'medproc_removed': out / f'medication_procedure_anchor_removed_{tag}.jsonl',
        'tokens': out / f'token_usage_{tag}.csv',
        'report': out / f'run_report_{tag}.json',
        'checkpoint': out / f'checkpoint_{tag}.sqlite',
    }


def write_token(p: Path, row: list[Any]) -> None:
    with p.open('a', encoding='utf-8', newline='') as f:
        csv.writer(f).writerow(row)


def response_content_and_usage(response: Any) -> tuple[str, dict[str, Any]]:
    content = response.choices[0].message.content or ''
    usage_obj = getattr(response, 'usage', None)
    usage: dict[str, Any] = {}
    if usage_obj is not None:
        usage = {
            'prompt_tokens': int(getattr(usage_obj, 'prompt_tokens', 0) or 0),
            'completion_tokens': int(getattr(usage_obj, 'completion_tokens', 0) or 0),
            'total_tokens': int(getattr(usage_obj, 'total_tokens', 0) or 0),
        }
    return str(content), usage


def retryable_error_message(e: Exception) -> str:
    text = repr(e)
    lower = text.casefold()
    tokens = ['429', 'rate limit', 'timeout', 'connection error', 'apiconnectionerror', 'apitimeouterror', 'ratelimiterror']
    retryable_by_text = any(t in lower for t in tokens)
    retryable_by_type = False
    if RateLimitError is not None and isinstance(e, RateLimitError):
        retryable_by_type = True
    if APITimeoutError is not None and isinstance(e, APITimeoutError):
        retryable_by_type = True
    if APIConnectionError is not None and isinstance(e, APIConnectionError):
        retryable_by_type = True
    if APIStatusError is not None and isinstance(e, APIStatusError) and getattr(e, 'status_code', None) in {429, 500, 502, 503, 504}:
        retryable_by_type = True
    return text if (retryable_by_text or retryable_by_type) else text


def build_case_prompt(case: dict[str, Any], args: argparse.Namespace, pid: str) -> dict[str, Any]:
    ctx_raw = builder.build_context(case)
    ctx = clean_context_for_prompt(ctx_raw, args)
    mapping = builder.parse_context(ctx)
    target = [f"{m['medication']}|{m['medication_function']}" for m in normalizer.choose_meds(case)[:args.max_target_meds_per_case]]
    user_obj = {
        'case': {'stay_id': case['stay_id'], 'hospital_id': case['hospital_id'], 'age': case['age'], 'sex': case['sex']},
        'context': ctx,
        'target_medications': target,
        'task': TASK_V2,
        'output': builder.OUTPUT,
    }
    user_prompt = json.dumps(user_obj, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    return {
        'prompt_id': pid,
        'case': case,
        'context': ctx,
        'mapping': mapping,
        'target': target,
        'user_prompt': user_prompt,
        'prompt_chars': len(SYSTEM_PROMPT_V2) + len(user_prompt),
    }


async def call_deepseek_one_case(client: Any, case: dict[str, Any], semaphore: asyncio.Semaphore, args: argparse.Namespace, pid: str) -> dict[str, Any]:
    prompt = build_case_prompt(case, args, pid)
    async with semaphore:
        last_error = ''
        for attempt_index in range(args.max_retries + 1):
            attempt_no = attempt_index + 1
            try:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT_V2},
                        {'role': 'user', 'content': prompt['user_prompt']},
                    ],
                    temperature=args.temperature,
                    stream=False,
                )
                raw, usage = response_content_and_usage(response)
                parsed = json.loads(strip_fence(raw))
                return {'ok': True, 'raw': raw, 'parsed': parsed, 'usage': usage, 'api_calls': attempt_no, **prompt}
            except Exception as e:
                last_error = retryable_error_message(e)
                if attempt_index < args.max_retries:
                    await asyncio.sleep(RETRY_DELAYS[min(attempt_index, len(RETRY_DELAYS) - 1)])
        return {'ok': False, 'error': last_error or 'DeepSeek request failed', 'api_calls': args.max_retries + 1, **prompt}


async def run_async_batch(items_to_process: list[tuple[str, dict[str, Any]]], args: argparse.Namespace, api_key: str):
    if AsyncOpenAI is None:
        raise RuntimeError('openai package is not installed; AsyncOpenAI is unavailable')
    # Disable environment proxy inheritance so AsyncOpenAI/httpx does not require socksio
    # when the shell has SOCKS proxy variables set.
    http_client = httpx.AsyncClient(trust_env=False)
    client = AsyncOpenAI(api_key=api_key, base_url='https://api.deepseek.com', http_client=http_client)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [asyncio.create_task(call_deepseek_one_case(client, case, semaphore, args, pid)) for pid, case in items_to_process]
    try:
        for future in asyncio.as_completed(tasks):
            yield await future
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await client.close()


def validate_response(parsed: Any, case: dict[str, Any], target: list[str], mapping: dict[str, dict[str, str]], pid: str, paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {'valid': 0, 'schema': 0, 'illegal': 0, 'unsafe': 0, 'null': 0, 'medproc_removed': 0}
    valid: list[dict[str, Any]] = []

    def schema(reason: str, record: Any = None) -> None:
        stats['schema'] += 1
        append_jsonl(paths['schema'], {'prompt_id': pid, 'stay_id': case['stay_id'], 'reason': reason, 'record': record})

    if not isinstance(parsed, dict):
        schema('response_not_object', parsed)
        return valid, stats
    extra_top = set(parsed) - {'stay_id', 'hospital_id', 'records'}
    if extra_top:
        schema('unexpected_top_level_fields', sorted(extra_top))
    if str(parsed.get('stay_id', '')) != case['stay_id'] or str(parsed.get('hospital_id', '')) != case['hospital_id']:
        schema('stay_or_hospital_id_mismatch')
    rows = parsed.get('records')
    if not isinstance(rows, list):
        schema('records_not_list', rows)
        return valid, stats

    targets = {(x.split('|', 1)[0].casefold(), x.split('|', 1)[1]): x.split('|', 1) for x in target}
    seen: set[tuple[str, str]] = set()
    for record in rows:
        if not isinstance(record, dict):
            schema('record_not_object', record)
            continue
        record = normalize_llm_record_before_validation(record)
        extras = set(record) - RECORD_FIELDS
        missing = RECORD_FIELDS - set(record)
        if 'anchor_ids' in record:
            schema('anchor_ids_array_forbidden', record)
            continue
        if extras or missing:
            schema('record_fields_invalid', {'extras': sorted(extras), 'missing': sorted(missing), 'record': record})
            continue
        med = str(record['medication']).strip()
        fn = str(record['medication_function']).strip()
        key = (med.casefold(), fn)
        if key not in targets or key in seen:
            schema('unknown_or_duplicate_target_medication', record)
            continue
        seen.add(key)
        canonical_med, canonical_fn = targets[key]
        anchor_id = record['anchor_id']
        if isinstance(anchor_id, (list, dict, tuple)) or (anchor_id is not None and not isinstance(anchor_id, str)):
            schema('anchor_id_not_string_or_null', record)
            continue
        if anchor_id in {'', None}:
            anchor = []
        elif anchor_id not in mapping:
            stats['illegal'] += 1
            append_jsonl(paths['illegal'], {'prompt_id': pid, 'stay_id': case['stay_id'], 'medication': canonical_med, 'anchor_id': anchor_id})
            continue
        else:
            anchor = [{'anchor_id': anchor_id, 'anchor_type': mapping[anchor_id]['anchor_type'], 'anchor_string': mapping[anchor_id]['anchor_string']}]
        if anchor and is_bad_medication_procedure_anchor(anchor[0], canonical_med):
            stats['medproc_removed'] += 1
            append_jsonl(paths['medproc_removed'], {
                'prompt_id': pid,
                'stay_id': case['stay_id'],
                'medication': canonical_med,
                'original_anchor': anchor[0],
                'action': 'set_anchor_empty',
            })
            anchor = []
        if len(anchor) > 1:
            schema('restored_anchor_length_gt_1', record)
            continue
        hyp = str(record['attribution_hypothesis']).strip()
        if not hyp:
            schema('empty_attribution_hypothesis', record)
            continue
        hyp, hits = sanitize_hypothesis(hyp)
        if hits:
            stats['unsafe'] += 1
            append_jsonl(paths['unsafe'], {
                'prompt_id': pid,
                'stay_id': case['stay_id'],
                'medication': canonical_med,
                'hits': hits,
                'original_hypothesis': record.get('attribution_hypothesis', ''),
                'sanitized_hypothesis': hyp,
                'repaired': True,
            })
        final = {
            'stay_id': case['stay_id'],
            'hospital_id': case['hospital_id'],
            'medication': canonical_med,
            'medication_function': canonical_fn,
            'anchor': anchor,
            'attribution_hypothesis': hyp,
        }
        if list(final) != FINAL_FIELDS:
            schema('final_fields_invalid', final)
            continue
        valid.append(final)
        stats['valid'] += 1
        if not anchor:
            stats['null'] += 1
            append_jsonl(paths['null'], final)

    for key, (med, fn) in targets.items():
        if key not in seen:
            schema('missing_target_medication_record', {'medication': med, 'medication_function': fn})
    return valid, stats


def medication_procedure_anchor_check(path: Path) -> dict[str, int]:
    counts = {
        'anchor_contains_current_medication_count': 0,
        'medication_administration_anchor_count': 0,
        'therapeutic_antibacterials_anchor_count': 0,
        'glucocorticoid_administration_anchor_count': 0,
        'bronchodilator_medication_anchor_count': 0,
        'stress_ulcer_medication_anchor_count': 0,
        'antihyperlipidemic_medication_anchor_count': 0,
    }
    if not path.exists():
        return counts
    for line in path.open(encoding='utf-8'):
        if not line.strip():
            continue
        row = json.loads(line)
        anchors = row.get('anchor') or []
        if not anchors:
            continue
        text = str(anchors[0].get('anchor_string', '')).casefold()
        med = str(row.get('medication', '')).casefold().strip()
        if med and med in text:
            counts['anchor_contains_current_medication_count'] += 1
        if 'medications' in text or 'medication administration' in text:
            counts['medication_administration_anchor_count'] += 1
        if 'therapeutic antibacterials' in text:
            counts['therapeutic_antibacterials_anchor_count'] += 1
        if 'glucocorticoid administration' in text:
            counts['glucocorticoid_administration_anchor_count'] += 1
        if 'bronchodilator' in text:
            counts['bronchodilator_medication_anchor_count'] += 1
        if any(x in text for x in ('stress ulcer treatment', 'famotidine', 'pantoprazole')):
            counts['stress_ulcer_medication_anchor_count'] += 1
        if any(x in text for x in ('antihyperlipidemic', 'hmg-coa', 'atorvastatin')):
            counts['antihyperlipidemic_medication_anchor_count'] += 1
    return counts


def base_report(args: argparse.Namespace, tag: str, seen: int) -> dict[str, Any]:
    return {
        'experiment_name': 'deepseek_l3_min_token_single_anchor_run',
        'run_tag': tag,
        'prompt_version': PROMPT_VERSION,
        'input_file': rel(project_path(args.input_file)),
        'output_dir': rel(project_path(args.output_dir)),
        'model': args.model,
        'temperature': args.temperature,
        'offset': args.offset,
        'limit': args.limit,
        'concurrency': args.concurrency,
        'parallel_api_call_enabled': args.concurrency > 1,
        'async_client': 'AsyncOpenAI',
        'run_mode': tag,
        'api_called': False,
        'num_input_cases_seen': seen,
        'num_cases_attempted': 0,
        'num_cases_success': 0,
        'num_cases_failed': 0,
        'num_l3_records_generated': 0,
        'valid_l3_records': 0,
        'schema_error_records': 0,
        'illegal_anchor_records': 0,
        'unsafe_causal_language_records': 0,
        'null_anchor_records': 0,
        'medication_procedure_anchor_removed_records': 0,
        'stopped_early_due_to_consecutive_failures': False,
        'max_consecutive_failures': MAX_CONSECUTIVE_FAILURES,
        'medication_procedure_anchor_check': medication_procedure_anchor_check(Path('__missing__')),
        'single_anchor_check': {
            'single_anchor_required': True,
            'anchor_id_field_used': True,
            'anchor_ids_array_forbidden': True,
            'max_one_anchor_per_medication': True,
            'restored_anchor_length_lte_1': True,
        },
        'format_check': {
            'no_condition_group': True,
            'no_candidate_anchors': True,
            'no_allowed_anchor_ids': True,
            'target_medications_pipe_format': True,
            'final_l3_no_raw_prompt_or_response': True,
        },
        'token_usage_summary': {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'mean_prompt_tokens': 0,
            'mean_completion_tokens': 0,
            'mean_total_tokens': 0,
        },
        'cost_estimate': {
            'input_price_per_million': INPUT_PRICE,
            'output_price_per_million': OUTPUT_PRICE,
            'estimated_cost_usd': 0,
            'estimated_cost_rmb_rough': 0,
        },
        'input_quality_check': {
            'used_dx_px_nonempty_file': True,
            'old_empty_context_file_used': False,
            'condition_group_used': False,
        },
        'async_calling': {
            'uses_async_openai': True,
            'uses_asyncio_semaphore': True,
            'stream': False,
            'worker_writes_files': False,
            'main_coroutine_writes_outputs_and_checkpoint': True,
        },
        'cleaning_check': {
            'input_context_filter_enabled': True,
            'diagnosis_path_compressed': True,
            'procedure_path_compressed': True,
            'medication_procedure_px_filtered': True,
            'pipe_medication_repair_enabled': True,
            'string_null_anchor_repair_enabled': True,
            'unsafe_hypothesis_sanitization_enabled': True,
            'bad_medication_procedure_anchor_removed': True,
        },
        'warnings': [],
    }


def write_report(p: Path, r: dict[str, Any]) -> None:
    p.write_text(json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def select_cases(args: argparse.Namespace) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    inp = project_path(args.input_file)
    cases: list[tuple[str, dict[str, Any]]] = []
    seen = 0
    eligible = 0
    with inp.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            case = normalizer.normalize(json.loads(line))
            if not case['dx'] or not case['px'] or not case['med']:
                continue
            if eligible < args.offset:
                eligible += 1
                continue
            if args.limit is not None and len(cases) >= args.limit:
                break
            absolute_index = eligible + 1
            pid = f'case_{absolute_index:06d}'
            cases.append((pid, case))
            eligible += 1
    return cases, seen


async def main_async(args: argparse.Namespace) -> None:
    out = project_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.run_tag or (f'offset{args.offset}_limit{args.limit}' if args.limit is not None else f'offset{args.offset}_full')
    paths = output_paths(out, tag)
    init_files(paths)
    init_checkpoint(paths['checkpoint'])

    cases, seen = select_cases(args)
    report = base_report(args, tag, seen)
    api_key, api_key_source = load_deepseek_api_key()
    report['api_key_source'] = api_key_source
    print(f'DeepSeek API key source: {api_key_source}')
    print(f'DeepSeek API key detected: {"yes" if api_key else "no"}')
    if not api_key:
        report['reason'] = 'DEEPSEEK_API_KEY not found. Set environment variable or fill configs/private/deepseek_key.py'
        report['warnings'].append('No API call was made. Edit configs/private/deepseek_key.py or set DEEPSEEK_API_KEY.')
        write_report(paths['report'], report)
        print('DeepSeek API key not configured. Edit configs/private/deepseek_key.py.')
        return

    if parse_bool(args.dry_run):
        report['api_called'] = False
        report['warnings'].append('Dry run requested; API not called.')
        write_report(paths['report'], report)
        return

    items_to_process: list[tuple[str, dict[str, Any]]] = []
    for pid, case in cases:
        st, saved = checkpoint_status(paths['checkpoint'], pid)
        if args.resume and st == 'success' and saved:
            continue
        items_to_process.append((pid, case))

    report['api_called'] = True
    usage_rows: list[tuple[int, int, int]] = []
    consecutive_failures = 0

    target_total_for_progress = len(items_to_process)
    async for result in run_async_batch(items_to_process, args, api_key):
        case = result['case']
        pid = result['prompt_id']
        target = result['target']
        mapping = result['mapping']
        report['num_cases_attempted'] += 1

        if not result.get('ok'):
            consecutive_failures += 1
            error = str(result.get('error', 'unknown error'))
            report['num_cases_failed'] += 1
            append_jsonl(paths['failed'], {'prompt_id': pid, 'stay_id': case['stay_id'], 'hospital_id': case['hospital_id'], 'error': error, 'updated_at': now()})
            set_checkpoint(paths['checkpoint'], pid, case['stay_id'], 'failed', False, error)
            write_token(paths['tokens'], [pid, case['stay_id'], case['hospital_id'], len(target), len(mapping), result['prompt_chars'], 0, 0, 0, 'failed'])
            if report['num_cases_attempted'] % 50 == 0:
                print(f"Progress: {report['num_cases_attempted']}/{target_total_for_progress} completed, success={report['num_cases_success']}, failed={report['num_cases_failed']}")
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                report['stopped_early_due_to_consecutive_failures'] = True
                report['warnings'].append(f'Stopped current batch after {MAX_CONSECUTIVE_FAILURES} consecutive failed API calls.')
                break
            continue

        consecutive_failures = 0
        usage = result.get('usage') or {}
        parsed = result['parsed']
        raw = result['raw']
        append_jsonl(paths['raw'], {
            'prompt_id': pid,
            'stay_id': case['stay_id'],
            'hospital_id': case['hospital_id'],
            'prompt_version': PROMPT_VERSION,
            'system_prompt': SYSTEM_PROMPT_V2,
            'user_prompt': result['user_prompt'],
            'raw_response': raw,
            'parsed_response': parsed,
            'anchor_id_map': mapping,
            'target_medications': target,
            'usage': usage,
            'api_status': 'success',
            'api_calls': result.get('api_calls', 1),
        })
        valid, stats = validate_response(parsed, case, target, mapping, pid, paths)
        for record in valid:
            append_jsonl(paths['l3'], record)
        report['num_cases_success'] += 1
        report['valid_l3_records'] += stats['valid']
        report['schema_error_records'] += stats['schema']
        report['illegal_anchor_records'] += stats['illegal']
        report['unsafe_causal_language_records'] += stats['unsafe']
        report['null_anchor_records'] += stats['null']
        report['medication_procedure_anchor_removed_records'] += stats.get('medproc_removed', 0)
        set_checkpoint(paths['checkpoint'], pid, case['stay_id'], 'success', True)

        pt = int(usage.get('prompt_tokens') or 0)
        ct = int(usage.get('completion_tokens') or 0)
        tt = int(usage.get('total_tokens') or pt + ct)
        write_token(paths['tokens'], [pid, case['stay_id'], case['hospital_id'], len(target), len(mapping), result['prompt_chars'], pt, ct, tt, 'success'])
        usage_rows.append((pt, ct, tt))
        if report['num_cases_attempted'] % 50 == 0:
            print(f"Progress: {report['num_cases_attempted']}/{target_total_for_progress} completed, success={report['num_cases_success']}, failed={report['num_cases_failed']}")

    report['num_l3_records_generated'] = report['valid_l3_records']
    report['medication_procedure_anchor_check'] = medication_procedure_anchor_check(paths['l3'])
    n = len(usage_rows)
    tp = sum(x[0] for x in usage_rows)
    tc = sum(x[1] for x in usage_rows)
    tt = sum(x[2] for x in usage_rows)
    report['token_usage_summary'] = {
        'total_prompt_tokens': tp,
        'total_completion_tokens': tc,
        'total_tokens': tt,
        'mean_prompt_tokens': tp / n if n else 0,
        'mean_completion_tokens': tc / n if n else 0,
        'mean_total_tokens': tt / n if n else 0,
    }
    usd = tp / 1e6 * INPUT_PRICE + tc / 1e6 * OUTPUT_PRICE
    report['cost_estimate'].update({'estimated_cost_usd': usd, 'estimated_cost_rmb_rough': usd * USD_RMB})
    write_report(paths['report'], report)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', default=str(DEFAULT_INPUT))
    ap.add_argument('--output-dir', default=str(DEFAULT_OUT))
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--temperature', type=float, default=0)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--run-tag', default='')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-target-meds-per-case', type=int, default=10)
    ap.add_argument('--max-lab-items', type=int, default=20)
    ap.add_argument('--dry-run', default='false')
    ap.add_argument('--max-retries', type=int, default=3)
    args = ap.parse_args()
    if args.offset < 0:
        raise ValueError('--offset must be >= 0')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be >= 1')
    if args.concurrency < 1:
        raise ValueError('--concurrency must be >= 1')
    if args.max_retries < 1:
        raise ValueError('--max-retries must be >= 1')
    return args


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == '__main__':
    main()
