"""Independent Draft 2020-12 validation against the repository JSON schemas."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

_SCHEMA_FILES = {
    'manifest': 'manifest.schema.json',
    'subtitles': 'subtitles.schema.json',
    'vocabulary': 'vocabulary.schema.json',
    'culture': 'culture.schema.json',
    'quizzes': 'quizzes.schema.json',
}

def validate_schemas(package: dict, schema_dir: str | Path | None = None) -> list[str]:
    """Return readable schema violations for the package's five JSON documents."""
    package_schema_dir = Path(__file__).resolve().parent / 'schemas'
    source_schema_dir = Path(__file__).resolve().parents[2] / 'schemas'
    if schema_dir:
        root = Path(schema_dir)
    elif package_schema_dir.is_dir():
        root = package_schema_dir
    else:
        root = source_schema_dir
    errors=[]
    for document,filename in _SCHEMA_FILES.items():
        schema_path=root/filename
        try:
            schema=json.loads(schema_path.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError) as exc:
            errors.append(f"cannot load {filename}: {exc}"); continue
        Draft202012Validator.check_schema(schema)
        validator=Draft202012Validator(schema)
        for error in sorted(validator.iter_errors(package.get(document,{})),key=lambda e:list(map(str,e.absolute_path))):
            path='.'.join(str(part) for part in error.absolute_path) or document
            errors.append(f"schema {filename} at {path}: {error.message}")
    return errors
