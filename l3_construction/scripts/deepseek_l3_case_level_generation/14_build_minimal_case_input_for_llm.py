#!/usr/bin/env python3
"""Build a compact, deterministic LLM input JSONL from Stage 1 stay records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_NAME = "stay_level_all_minimal_for_llm.jsonl"
REPORT_NAME = "minimal_case_input_report.json"
SAMPLE_NAME = "minimal_case_input_pretty_sample.json"
ALLOWED_FIELDS = {"id", "hid", "age", "sex", "allergy", "dx", "px", "lab", "med"}
FORBIDDEN_STRINGS = {
    "med_mapping_details", "raw_drugname", "clean_name", "drugbank_ids",
    "drugbank_names", "drugbank_info", "smiles", "atc_codes", "rxcui",
    "match_method", "mapping_confidence", "meds_mapped_drugbank_names",
    "med_drugbank_ids", "meds_standardized_names", "meds_unmapped",
    "meds_discarded",
}

DX_SIMPLIFY = {
    "cardiovascular|arrhythmias|atrial fibrillation|with hemodynamic compromise": "atrial fibrillation",
    "cardiovascular|arrhythmias|atrial fibrillation|with rapid ventricular response": "atrial fibrillation",
    "cardiovascular|arrhythmias|atrial fibrillation": "atrial fibrillation",
    "cardiovascular|chest pain / ashd|coronary artery disease|known": "coronary artery disease",
    "cardiovascular|chest pain / ashd|coronary artery disease": "coronary artery disease",
    "cardiovascular|chest pain / ashd|acute coronary syndrome|unstable angina": "acute coronary syndrome",
    "cardiovascular|chest pain / ashd|acute coronary syndrome": "acute coronary syndrome",
    "pulmonary|pulmonary infections|pneumonia": "pneumonia",
    "pulmonary|respiratory failure|acute respiratory distress": "respiratory failure",
    "renal|disorder of kidney|chronic kidney disease": "chronic kidney disease",
    "renal|disorder of kidney|esrd end stage renal disease": "ESRD",
    "gastrointestinal|gi bleeding / pud|gi bleeding": "gastrointestinal bleeding",
    "hematology|white blood cell disorders|leukocytosis": "leukocytosis",
    "hematology|coagulation disorders|coagulopathy": "coagulopathy",
    "cardiovascular|ventricular disorders|congestive heart failure": "heart failure",
}

PX_SIMPLIFY = {
    "mechanical ventilation": "mechanical ventilation",
    "intubation": "intubation",
    "central venous catheter": "central venous catheter",
    "infectious diseases|cultures / immuno-assays|cultures|sputum": "sputum culture",
    "infectious diseases|cultures / immuno-assays|cultures|blood": "blood culture",
    "cardiovascular|vascular disorders|vte prophylaxis|compression boots": "compression boots",
    "pulmonary|radiologic procedures / bronchoscopy|bronchoscopy|with bronchial alveolar lavage": "bronchoscopy",
}

FUNCTION_HINT_MAP = {
    "vasopressor_inotrope": "vasopressor",
    "cardiovascular_anticoagulant": "anticoagulant",
    "steroid_endocrine": "steroid",
    "sedative_analgesic": "sedative_analgesic",
    "anti_infective": "anti_infective",
    "fluid_electrolyte_nutrition": "fluid_electrolyte",
    "gi_supportive": "acid_suppression",
    "line_flush": "other_supportive",
    "topical_or_care_product": "other_supportive",
    "topical_local": "other_supportive",
    "vaccine_prevention": "other_supportive",
    "local_anesthetic": "local_anesthetic",
}

FUNCTION_RULES = {
    "anti_infective": [
        "vancomycin", "cefepime", "ceftriaxone", "ceftazidime", "piperacillin",
        "tazobactam", "piperacillin/tazobactam", "piperacillin|tazobactam",
        "meropenem", "imipenem", "aztreonam", "levofloxacin", "ciprofloxacin",
        "moxifloxacin", "metronidazole", "linezolid", "daptomycin", "gentamicin",
        "tobramycin", "amikacin", "azithromycin", "clindamycin", "nafcillin",
        "ampicillin", "cefazolin", "fluconazole", "acyclovir",
    ],
    "vasopressor": [
        "norepinephrine", "epinephrine", "vasopressin", "dopamine",
        "phenylephrine", "dobutamine", "midodrine",
    ],
    "anticoagulant": [
        "heparin", "enoxaparin", "warfarin", "argatroban", "bivalirudin",
        "fondaparinux",
    ],
    "antiplatelet": ["aspirin", "acetylsalicylic acid", "clopidogrel", "ticagrelor"],
    "antiarrhythmic": ["amiodarone", "lidocaine", "procainamide", "sotalol", "adenosine"],
    "steroid": ["hydrocortisone", "methylprednisolone", "prednisone", "dexamethasone", "medrol"],
    "bronchodilator": ["albuterol", "ipratropium", "levalbuterol", "tiotropium"],
    "glucose_control": ["insulin", "glucose", "d-glucose", "dextrose", "glucagon"],
    "sedative_analgesic": [
        "propofol", "midazolam", "lorazepam", "diazepam", "dexmedetomidine",
        "fentanyl", "morphine", "hydromorphone", "oxycodone", "hydrocodone",
        "tramadol", "ketamine",
    ],
    "diuretic": ["furosemide", "bumetanide", "torsemide", "metolazone", "spironolactone"],
    "antihypertensive": [
        "nicardipine", "nitroprusside", "nitroglycerin", "labetalol", "metoprolol",
        "carvedilol", "diltiazem", "verapamil", "enalapril", "captopril",
        "hydralazine", "clonidine",
    ],
    "cardiac_medication": [
        "digoxin", "milrinone", "esmolol", "simvastatin", "atorvastatin",
        "pravastatin", "rosuvastatin", "levothyroxine",
    ],
    "fluid_electrolyte": [
        "sodium chloride", "potassium chloride", "magnesium sulfate", "calcium gluconate",
        "calcium chloride", "lactated ringers", "sodium bicarbonate",
        "monopotassium phosphate", "sodium phosphate", "sodium phosphate, monobasic",
        "albumin human",
    ],
    "acid_suppression": ["pantoprazole", "omeprazole", "famotidine", "ranitidine"],
    "bowel_regimen": [
        "docusate", "bisacodyl", "senna", "sennosides", "polyethylene glycol",
        "lactulose", "magnesium hydroxide",
    ],
    "nutrition": ["thiamine", "folic acid", "multivitamin", "parenteral nutrition"],
}

FUNCTION_PRIORITY = [
    "anti_infective", "vasopressor", "anticoagulant", "antiplatelet",
    "antiarrhythmic", "steroid", "cardiac_medication", "bronchodilator",
    "antihypertensive", "glucose_control", "sedative_analgesic", "diuretic",
    "fluid_electrolyte", "acid_suppression", "bowel_regimen", "nutrition",
    "local_anesthetic", "other_supportive", "unknown",
]
FUNCTION_RANK = {name: index for index, name in enumerate(FUNCTION_PRIORITY)}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def norm(value: Any) -> str:
    return clean_text(value).casefold()


def dedupe(values: Iterable[str]) -> Tuple[List[str], int]:
    output: List[str] = []
    seen = set()
    duplicates = 0
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append(text)
    return output, duplicates


def extract_context_text(item: Any, keys: Sequence[str]) -> str:
    if isinstance(item, str):
        return clean_text(item)
    if isinstance(item, dict):
        for key in keys:
            text = clean_text(item.get(key))
            if text:
                return text
    return ""


def extract_context(items: Any, keys: Sequence[str], simplify: Dict[str, str]) -> Tuple[List[str], int, int]:
    if not isinstance(items, list):
        return [], 0, 0
    raw = []
    for item in items:
        text = extract_context_text(item, keys)
        if text:
            raw.append(simplify.get(text.casefold(), text))
    unique, duplicates = dedupe(raw)
    return unique, len(raw), duplicates


def normalize_age(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    text = clean_text(value)
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text or None


def normalize_sex(value: Any) -> str:
    value_norm = norm(value)
    if value_norm in {"female", "f"}:
        return "F"
    if value_norm in {"male", "m"}:
        return "M"
    return "U"


def extract_allergy(value: Any) -> List[str]:
    values = value if isinstance(value, list) else ([] if value in (None, "") else [value])
    result = []
    for item in values:
        if isinstance(item, dict):
            text = extract_context_text(item, ("normalized", "raw", "text", "name"))
        else:
            text = clean_text(item)
        if text:
            result.append(text)
    return dedupe(result)[0]


def list_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return [clean_text(item) for item in value if not isinstance(item, (dict, list)) and clean_text(item)]
    if isinstance(value, str) and clean_text(value):
        return [clean_text(value)]
    return []


def medication_names_from_item(item: Any) -> List[str]:
    if isinstance(item, str):
        return [clean_text(item)] if clean_text(item) else []
    if not isinstance(item, dict):
        return []
    for key in ("standardized_name", "drugbank_names", "clean_name"):
        value = item.get(key)
        names = list_strings(value)
        if names:
            return names
    return []


def medication_aliases_from_item(item: Any) -> List[str]:
    """Return local-only aliases for metadata matching; aliases are never emitted."""
    if isinstance(item, str):
        return [clean_text(item)] if clean_text(item) else []
    if not isinstance(item, dict):
        return []
    aliases: List[str] = []
    for key in ("standardized_name", "drugbank_names", "clean_name", "raw_drugname"):
        aliases.extend(list_strings(item.get(key)))
    return dedupe(aliases)[0]


def is_removed_med_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    return (
        bool(item.get("exclude_from_core_recommendation"))
        or bool(item.get("is_discarded"))
        or norm(item.get("action")) == "discard"
    )


def canonical_function(value: Any) -> str:
    value_norm = norm(value).replace(" ", "_")
    if value_norm in FUNCTION_RANK:
        return value_norm
    return FUNCTION_HINT_MAP.get(value_norm, "")


def infer_function(name: str, metadata: Sequence[Dict[str, Any]], top_functions: Dict[str, Any]) -> str:
    name_norm = norm(name)
    for key, value in top_functions.items():
        if norm(key) == name_norm:
            function = canonical_function(value)
            if function:
                return function
    for item in metadata:
        function = canonical_function(item.get("medication_function"))
        if function:
            return function
    for item in metadata:
        function = canonical_function(item.get("function_hint"))
        if function:
            return function
    for function, terms in FUNCTION_RULES.items():
        if any(term in name_norm for term in terms):
            return function
    return "unknown"


def build_medications(stay: Dict[str, Any], max_med: int, stats: Counter) -> Tuple[List[List[str]], int]:
    metadata_by_name: Dict[str, List[Dict[str, Any]]] = {}
    removed_names = set()

    metadata_sources = []
    for key in ("meds", "med_mapping_details"):
        value = stay.get(key)
        if isinstance(value, list):
            metadata_sources.extend(item for item in value if isinstance(item, dict))
    for item in metadata_sources:
        names = medication_aliases_from_item(item)
        for name in names:
            metadata_by_name.setdefault(norm(name), []).append(item)
            if is_removed_med_item(item):
                removed_names.add(norm(name))

    discarded = stay.get("meds_discarded")
    if isinstance(discarded, list):
        for item in discarded:
            names = medication_aliases_from_item(item)
            removed_names.update(norm(name) for name in names if clean_text(name))

    source: List[Any] = []
    standardized = stay.get("meds_standardized_names")
    meds = stay.get("meds")
    mapped = stay.get("meds_mapped_drugbank_names")
    details = stay.get("med_mapping_details")
    if isinstance(standardized, list) and standardized:
        source = standardized
    elif isinstance(meds, list) and meds:
        source = meds
    elif isinstance(mapped, list) and mapped:
        source = mapped
    elif isinstance(details, list):
        source = details

    candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
    for item in source:
        names = medication_names_from_item(item)
        if not names:
            stats["empty_meds_removed"] += 1
            continue
        item_metadata = [item] if isinstance(item, dict) else []
        for name in names:
            name = clean_text(name)
            if not name:
                stats["empty_meds_removed"] += 1
                continue
            metadata = metadata_by_name.get(norm(name), []) + item_metadata
            if norm(name) in removed_names or is_removed_med_item(item):
                stats["discarded_meds_removed"] += 1
                continue
            candidates.append((name, metadata))

    before = len(candidates)
    unique: List[Tuple[str, List[Dict[str, Any]]]] = []
    seen = set()
    for name, metadata in candidates:
        key = norm(name)
        if key in seen:
            stats["duplicate_meds_removed"] += 1
            continue
        seen.add(key)
        unique.append((name, metadata))

    top_functions = stay.get("medication_functions")
    if not isinstance(top_functions, dict):
        top_functions = {}
    classified = [
        (index, name, infer_function(name, metadata, top_functions))
        for index, (name, metadata) in enumerate(unique)
    ]
    classified.sort(key=lambda row: (FUNCTION_RANK.get(row[2], FUNCTION_RANK["unknown"]), row[0]))
    if len(classified) > max_med:
        stats["med_truncated"] += 1
    return [[name, function] for _, name, function in classified[:max_med]], before


def validate_minimal(obj: Dict[str, Any]) -> None:
    if not set(obj).issubset(ALLOWED_FIELDS):
        raise ValueError(f"minimal record has forbidden fields: {sorted(set(obj) - ALLOWED_FIELDS)}")
    if "allergy" in obj and not obj["allergy"]:
        raise ValueError("empty allergy must be omitted")
    if obj.get("sex") not in {"F", "M", "U"}:
        raise ValueError("sex must be F, M, or U")
    for key in ("dx", "px", "lab", "med"):
        if not isinstance(obj.get(key), list):
            raise ValueError(f"{key} must be a list")
    for medication in obj["med"]:
        if not (isinstance(medication, list) and len(medication) == 2 and all(isinstance(x, str) for x in medication)):
            raise ValueError("each med item must be a two-string list")
    serialized = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    hits = [term for term in FORBIDDEN_STRINGS if term in serialized]
    if hits:
        raise ValueError(f"minimal record contains forbidden strings: {hits}")


def safe_average(total: int, count: int) -> float:
    return round(total / count, 6) if count else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/stage1_preprocess_final/stay_level_all_detailed.jsonl")
    parser.add_argument("--output-dir", default="outputs/deepseek_l3_minimal_case_inputs")
    parser.add_argument("--max-dx", type=int, default=12)
    parser.add_argument("--max-px", type=int, default=8)
    parser.add_argument("--max-lab", type=int, default=8)
    parser.add_argument("--max-med", type=int, default=5)
    args = parser.parse_args()

    for name in ("max_dx", "max_px", "max_lab", "max_med"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    input_path = resolve_path(args.input).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    protected_dir = (PROJECT_ROOT / "outputs/stage1_preprocess_final").resolve()
    if output_dir == protected_dir or protected_dir in output_dir.parents:
        raise ValueError("output-dir must not point inside outputs/stage1_preprocess_final/")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    if output_path.resolve() == input_path:
        raise ValueError("minimal output must not overwrite the Stage 1 input")

    input_stat_before = input_path.stat()
    input_hash_before = sha256_file(input_path)
    temp_output = output_dir / f".{OUTPUT_NAME}.tmp"
    counters: Counter = Counter()
    sums: Counter = Counter()
    warnings: List[str] = []
    pretty: List[Dict[str, Any]] = []

    try:
        with input_path.open("r", encoding="utf-8") as source, temp_output.open("w", encoding="utf-8") as target:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                counters["input_records"] += 1
                try:
                    stay = json.loads(line)
                except json.JSONDecodeError as exc:
                    counters["bad_json_lines"] += 1
                    warnings.append(f"bad JSON line {line_number}: {exc.msg}")
                    continue
                if not isinstance(stay, dict):
                    counters["bad_json_lines"] += 1
                    warnings.append(f"non-object JSON line {line_number}")
                    continue

                dx_all, dx_before, dx_duplicates = extract_context(stay.get("diags"), ("normalized", "raw"), DX_SIMPLIFY)
                px_all, px_before, _ = extract_context(stay.get("procs"), ("normalized", "raw", "text"), PX_SIMPLIFY)
                lab_all, lab_before, lab_duplicates = extract_context(stay.get("labs"), ("trigger",), {})
                counters["duplicate_dx_removed"] += dx_duplicates
                counters["duplicate_labs_removed"] += lab_duplicates
                if len(dx_all) > args.max_dx:
                    counters["dx_truncated"] += 1
                if len(px_all) > args.max_px:
                    counters["px_truncated"] += 1
                if len(lab_all) > args.max_lab:
                    counters["lab_truncated"] += 1
                dx = dx_all[:args.max_dx]
                px = px_all[:args.max_px]
                lab = lab_all[:args.max_lab]
                med, med_before = build_medications(stay, args.max_med, counters)

                base_info = stay.get("base_info") if isinstance(stay.get("base_info"), dict) else {}
                obj: Dict[str, Any] = {
                    "id": clean_text(stay.get("patientunitstayid") or stay.get("patient_id")),
                    "hid": clean_text(stay.get("hospital_id") or stay.get("hospitalid")),
                    "age": normalize_age(base_info.get("age")),
                    "sex": normalize_sex(base_info.get("sex")),
                    "dx": dx,
                    "px": px,
                    "lab": lab,
                    "med": med,
                }
                allergy = extract_allergy(base_info.get("allergy"))
                if allergy:
                    obj["allergy"] = allergy
                validate_minimal(obj)
                target.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                counters["output_records"] += 1
                if len(pretty) < 3:
                    pretty.append(obj)

                sums["dx_before"] += dx_before
                sums["dx_after"] += len(dx)
                sums["px_before"] += px_before
                sums["px_after"] += len(px)
                sums["lab_before"] += lab_before
                sums["lab_after"] += len(lab)
                sums["med_before"] += med_before
                sums["med_after"] += len(med)
                counters["with_dx"] += bool(dx)
                counters["with_px"] += bool(px)
                counters["with_lab"] += bool(lab)
                counters["with_med"] += bool(med)
                counters["without_any_context"] += not (dx or px or lab)
                counters["without_med"] += not med
        os.replace(temp_output, output_path)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    if counters["output_records"] == 0 or output_path.stat().st_size == 0:
        raise RuntimeError("minimal output JSONL is empty")
    expected_output = counters["input_records"] - counters["bad_json_lines"]
    if counters["output_records"] != expected_output:
        raise RuntimeError("output record count does not equal valid input record count")
    if counters["bad_json_lines"]:
        warnings.append(
            f"output has {counters['output_records']} records for {counters['input_records']} non-empty input lines because {counters['bad_json_lines']} lines were invalid"
        )

    validated_lines = 0
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            validate_minimal(obj)
            validated_lines += 1
    if validated_lines != counters["output_records"]:
        raise RuntimeError("post-write validation line count mismatch")

    input_hash_after = sha256_file(input_path)
    input_stat_after = input_path.stat()
    if input_hash_before != input_hash_after or input_stat_before.st_size != input_stat_after.st_size:
        raise RuntimeError("Stage 1 input file changed during minimal build")

    with (output_dir / SAMPLE_NAME).open("w", encoding="utf-8") as handle:
        json.dump(pretty, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    count = counters["output_records"]
    report = {
        "step": "build_minimal_case_input_for_llm",
        "input_file": relative_path(input_path),
        "output_file": relative_path(output_path),
        "num_input_records": counters["input_records"],
        "num_output_records": count,
        "max_dx": args.max_dx,
        "max_px": args.max_px,
        "max_lab": args.max_lab,
        "max_med": args.max_med,
        "num_records_with_dx": counters["with_dx"],
        "num_records_with_px": counters["with_px"],
        "num_records_with_lab": counters["with_lab"],
        "num_records_with_med": counters["with_med"],
        "num_records_without_any_context": counters["without_any_context"],
        "num_records_without_med": counters["without_med"],
        "avg_dx_before": safe_average(sums["dx_before"], count),
        "avg_dx_after": safe_average(sums["dx_after"], count),
        "avg_px_before": safe_average(sums["px_before"], count),
        "avg_px_after": safe_average(sums["px_after"], count),
        "avg_lab_before": safe_average(sums["lab_before"], count),
        "avg_lab_after": safe_average(sums["lab_after"], count),
        "avg_med_before": safe_average(sums["med_before"], count),
        "avg_med_after": safe_average(sums["med_after"], count),
        "num_dx_truncated": counters["dx_truncated"],
        "num_px_truncated": counters["px_truncated"],
        "num_lab_truncated": counters["lab_truncated"],
        "num_med_truncated": counters["med_truncated"],
        "num_discarded_meds_removed": counters["discarded_meds_removed"],
        "num_empty_meds_removed": counters["empty_meds_removed"],
        "num_duplicate_meds_removed": counters["duplicate_meds_removed"],
        "num_duplicate_labs_removed": counters["duplicate_labs_removed"],
        "num_duplicate_dx_removed": counters["duplicate_dx_removed"],
        "num_bad_json_lines": counters["bad_json_lines"],
        "warnings": warnings,
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "modified_original_files": False,
        "used_llm": False,
        "used_api": False,
        "used_condition_group": False,
    }
    with (output_dir / REPORT_NAME).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    print("Minimal LLM case input build completed successfully.")


if __name__ == "__main__":
    main()
