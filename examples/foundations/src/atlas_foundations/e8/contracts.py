"""Offline, inspectable contracts shared by the actual training and browser lessons."""
import hashlib
import json
import math

TEMPLATE = '### Instruction:\n{instruction}\n### Context:\n{context}\n### Response:\n'


def prompt(row):
    return TEMPLATE.format(**row)


def check_data(rows):
    ids, contexts, prompts = set(), set(), set()
    for row in rows:
        assert row['split'] in ('train', 'validation', 'test')
        assert row['id'] not in ids and prompt(row) not in prompts
        assert row['context'].strip() and row['context'] not in contexts
        assert row['response'].strip()
        ids.add(row['id']); contexts.add(row['context']); prompts.add(prompt(row))
    return True


def encode(tokenizer, row, response=None, max_length=256):
    """Completion-only, explicit EOS; reject overflow instead of silently losing supervision.

    Pythia is a base model: our ordinary-text delimiters are NOT native chat control tokens.
    Encoding prefix and completion separately defines this serialization contract.
    EOS and pad may share an ID; padding labels depend on position/attention, never that ID.
    """
    prefix = tokenizer.encode(prompt(row), add_special_tokens=False)
    answer = tokenizer.encode(row['response'] if response is None else response, add_special_tokens=False)
    assert answer, 'Empty completion'
    tokens = prefix + answer + [tokenizer.eos_token_id]
    if len(tokens) > max_length:
        raise ValueError(f"{row['id']}: {len(tokens)} tokens exceeds {max_length}; revise data/profile explicitly")
    labels = [-100] * len(prefix) + answer + [tokenizer.eos_token_id]
    return dict(input_ids=tokens, labels=labels, attention_mask=[1] * len(tokens), prefix_tokens=len(prefix))


def dpo_loss(pc, pr, rc, rr, beta=0.1):
    if beta <= 0 or not all(math.isfinite(x) for x in (pc, pr, rc, rr, beta)):
        raise ValueError('Finite log probabilities and positive beta required')
    z = beta * ((pc-pr)-(rc-rr))
    return max(0, -z) + math.log1p(math.exp(-abs(z)))


def check_preferences(rows):
    check_data(rows)
    for r in rows:
        assert r['chosen'].strip() and r['rejected'].strip()
        assert r['chosen'] != r['rejected']
        assert r['chosen'] == r['response']
    return True


def output_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_judgments(evaluation, imported):
    """Offline judge/human importer: bind labels to exact output, task, rubric, and order.

    Returns missing cases explicitly. Invalid/duplicate/stale rows fail, never vanish.
    No API access, no credentials, and no fabricated human ratings.
    """
    expected = {(r['model'], r['id']): r for r in evaluation}
    seen = set()
    for r in imported:
        key = (r['model'], r['id'])
        if key not in expected or key in seen:
            raise ValueError('Unknown or duplicate judgment')
        source = expected[key]
        if r['output_sha256'] != output_hash(source['prediction']):
            raise ValueError('Stale output fingerprint')
        if r['rubric_version'] != 'atlas-e8-v1' or not r['judge_version'].strip():
            raise ValueError('Rubric or judge version missing')
        if r['order'] not in ('AB','BA') or any(type(r[k]) is not int or not 0<=r[k]<=2 for k in ('grounding','instruction','clarity')):
            raise ValueError('Invalid order or rubric score')
        seen.add(key)
    return {'imported':len(seen),'missing':[list(k) for k in expected if k not in seen], 'complete':seen==set(expected)}
