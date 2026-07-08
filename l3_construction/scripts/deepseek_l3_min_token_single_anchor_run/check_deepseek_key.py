#!/usr/bin/env python3
from pathlib import Path
import os
import runpy

PLACEHOLDER = 'YOUR_DEEPSEEK_API_KEY_HERE'

def load_deepseek_api_key():
    key = os.getenv('DEEPSEEK_API_KEY', '').strip()
    if key and key != PLACEHOLDER:
        return key, 'env:DEEPSEEK_API_KEY'

    project_root = Path(__file__).resolve().parents[2]
    key_file = project_root / 'configs' / 'private' / 'deepseek_key.py'
    if key_file.exists():
        try:
            ns = runpy.run_path(str(key_file))
            key = str(ns.get('DEEPSEEK_API_KEY', '')).strip()
            if key and key != PLACEHOLDER:
                return key, 'configs/private/deepseek_key.py'
        except Exception as e:
            return '', f'key_file_read_error:{type(e).__name__}'

    return '', 'not_found'

key, source = load_deepseek_api_key()
print({
    'deepseek_api_key_exists': bool(key),
    'api_key_source': source,
    'key_length': len(key) if key else 0,
})
