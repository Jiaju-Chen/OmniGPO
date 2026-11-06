"""Verify the declared packaging-only transformations against pinned sources."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=json.loads((ROOT/'SOURCE_PROVENANCE.json').read_text())


def test_graph_and_teacher_match_the_exact_experiment_snapshot():
    for rel in ['gigpo/core_bepo.py','gigpo/graph_value_teacher.py']:
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==MANIFEST['original_source_sha256'][rel]


def test_neural_credit_diff_is_only_import_and_explicit_fixed_core_guard():
    text=(ROOT/'gigpo/bepo_neural_credit.py').read_text()
    text=text.replace('from gigpo.ppo import compute_turn_ppo_gae',
        'from verl.trainer.ppo.core_algos import compute_turn_ppo_gae')
    guard="    if bepo_cfg.get('neural_credit_router') is not None or bepo_cfg.get('mc_direct_supported', False):\n        raise ValueError('This core release supports fixed-q routing only')\n"
    assert text.count(guard)==1
    text=text.replace(guard,'')
    assert hashlib.sha256(text.encode()).hexdigest()==MANIFEST['original_source_sha256']['gigpo/bepo_neural_credit.py']


def test_ppo_functions_are_mechanically_extracted_without_arithmetic_changes():
    tree=ast.parse((ROOT/'gigpo/ppo.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef):
            digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
            assert digest==MANIFEST['function_ast_sha256'][node.name]
