"""Generate staged notebook views of the same verified model sources."""

import json

SPECS = {
    "retrieval-benchmark": (
        "L12 · how do ranking and candidate coverage affect quality?",
        "benchmark",
        "e6-retrieval",
        "Choose a query with an exact identifier and one that paraphrases its document. Predict BM25 versus dense ordering. Six dense candidate chunks bound reranking. Overlapping chunks are collapsed to document IDs before metrics. AP divides by all relevant documents; absent-query MAP is undefined and excluded with a denominator.",
        'print(json.dumps(result["aggregates"],indent=2))\nprint("Design threshold:",result["threshold_selection"])',
        "Find one query where reranking cannot recover a missing candidate. Move a relevant document from rank three to rank one and hand-calculate AP@3. Why are absent queries evaluated separately?",
    ),
    "grounded-retrieval": (
        "L13 · which stage caused an unsupported answer?",
        "grounded",
        "e6-grounded",
        "Compare baseline, rewriting, multi-query, dependent multi-hop, routing, and finite-choice source selection. Each prompt and search uses actual prior state, never reference answers. L13 uses author-known mechanism probes; L12 retains the held-out retrieval evaluation. A threshold override, model abstention, citation resolution, and human support judgment are different properties.",
        'print(json.dumps([{k:r[k] for k in ["id","variant","display_answer","controller_withheld","model_citations","model_calls","search_calls","scored_tokens"]} for r in result["rows"]],indent=2))',
        "Find a correct raw answer with no citation and a wrong answer whose number appears in an unrelated passage. Inspect the actual planner input that changed the retrieved entity. Explain why resolving a source ID is not entailment.",
    ),
    "openclip-retrieval": (
        "L14a · what pixels and vectors does OpenCLIP compare?",
        "vision",
        "e6-vision",
        "Three familiar public-domain/CC0 photos are mechanism probes, not a held-out vision benchmark. Inspect attribution, aspect ratio, crop, normalization, text IDs, unit vectors, actual patch projection, and the symmetric paired loss. The original weights remain frozen; no training claim.",
        'print(json.dumps({k:result[k] for k in ["embedding_shapes","cosines","bare_cosines","contrastive"]},indent=2))\nfrom IPython.display import display,Image\ndisplay(Image(filename=str(root / "public/images/e6/coffee-clip-input.png")))',
        "Find the discarded part of a non-square photo. Explain why adding a distractor caption changes softmax while leaving cosine unchanged. Trace one 32×32 patch from 3072 numbers to 768 projected values and the 50-token image sequence.",
    ),
    "blip2-captioning": (
        "L14b · how do actual BLIP-2 queries bridge vision and language?",
        "caption",
        "e6-caption",
        "Genuine Salesforce BLIP-2, Linux CPU bfloat16 inference with float32 Q-Former components and explicit CPU autocast. Full model weights and runtime buffers are resident; see measured memory in the execution record. The public checkpoint is about 14.98 GB; loading, disk access, runtime buffers, and active layers are additional. The verified heavy Linux CI profile is separate from the smaller local labs. No CUDA or 4-bit run is claimed. Inspect vision tokens, learned query count, projected query width, integer text IDs, and actual generated caption/VQA text.",
        'print(json.dumps({k:result[k] for k in ["profile","bridge_config","rows"]},indent=2))',
        "Does matching the LLM embedding width prove semantic alignment? Explain the role of learned queries and the trainable projection in the historical training process. Inspect the actual caption and identify any detail the photograph cannot establish.",
    ),
    "multimodal-chat": (
        "L14c · can a visual chatbot infer a price that is not shown?",
        "chat",
        "e6-chat",
        "The separate 256M instruction model receives real image pixels and actual generated conversation history. It is not BLIP-2. Inspect the chat template, pixel tensor, continuation-only decoding, and the image-splitting setting. Unknown pretraining overlap prevents a held-out benchmark claim.",
        'print(json.dumps(result["rows"],indent=2))',
        "Does a fluent price answer have visual support? Find the first generated answer inside the next prompt. Distinguish application-retained history from a model weight update.",
    ),
}


def notebook(eid):
    title, module, artifact, predict, inspect, exercise = SPECS[eid]

    def md(id, text):
        return {
            "cell_type": "markdown",
            "id": id,
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }

    def code(id, text):
        return {
            "cell_type": "code",
            "id": id,
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }

    setup = """import sys,json
from pathlib import Path
project=Path.cwd()
if project.name=="notebooks":project=project.parent
root=project.parents[2]
sys.path.insert(0,str(project/"src"))
corpus=json.loads((root/"examples/data/fixtures/retrieval-corpus.json").read_text())
images=json.loads((root/"examples/data/fixtures/vision-images.json").read_text())
print("Documents:",len(corpus["documents"]),"query splits:",{s:sum(q["split"]==s for q in corpus["queries"]) for s in ["design","evaluation"]})
print("Image attribution:",[(r["id"],r["credit"],r["license"]) for r in images["images"]])
"""
    cells = [
        md(
            "question",
            "# "
            + title
            + "\n\nPredict → inspect inputs → run the actual shared source → evaluate failures → reload the artifact → transfer.\n",
        ),
        md(
            "contract",
            "## Execution contract\n\nInstall `uv sync --locked --project examples/applications/retrieval` and select that Python 3.11 kernel. Start in the project or notebooks subdirectory. No hosted provider or key is used. Public safetensors/tokenizer files are pinned and checksummed. First downloads need network; cached runs can set HF_HUB_OFFLINE=1. Read README for download sizes, measured host resources, CPU placement and heavy-model memory, and optional unverified hardware configurations. Restart the kernel and Run All.\n",
        ),
        md(
            "inspect",
            "## Inspect provenance and protect evaluation\n\nDocuments/queries are original MIT fiction. Photos retain their public-domain/CC0 attribution. Corpus preprocessing and threshold choices are declared; evaluation answers never enter model prompts. L12 design queries select the threshold, while its evaluation set remains separate. L13 and vision runs are explicit mechanism probes.\n",
        ),
        code("data", setup),
        md(
            "predict",
            "## Baseline and prediction\n\n"
            + predict
            + "\n\nThe next cell invokes the complete source shown in the connected lesson. It performs the actual model calls and assertions; no copied algorithm or hidden prior state.\n",
        ),
        code("run-source", f"from atlas_retrieval.{module} import main\nmain()"),
        md(
            "evidence",
            "## Evaluate and inspect intermediate evidence\n\nRead the saved result from this run. Scores, shapes, raw calls, source IDs, and failures are preserved. Correct format, useful ranking, faithful answers, citation support, and fluent wording are separate checks.\n",
        ),
        code(
            "inspect-result",
            f"artifact_path=root/'examples/artifacts/{artifact}/results.json'\nresult=json.loads(artifact_path.read_text())\n{inspect}",
        ),
        md(
            "work",
            "## Account for work and limits\n\nGenerated tokens, input tokens, finite-label scoring tokens, model calls, search calls, or image tensor shapes appear where relevant. Timings describe one execution host, not a speed or cost leaderboard. Extra searches and summaries do not guarantee better answers. Caption/VQA output can be truncated or unsupported.\n",
        ),
        code(
            "reload",
            'reloaded=json.loads(artifact_path.read_text())\nassert reloaded==result\nprint("Saved result reload checked; see the source for index/embedding reload assertions where applicable.")',
        ),
        md(
            "transfer",
            "## Practice and transfer\n\n"
            + exercise
            + "\n\nReport template: input IDs; baseline; changed component; prediction; original outputs; intermediate evidence; metric denominators; calls/tokens/resources; failure explanation; next test. Use fresh queries/images for a new generalization claim rather than selecting on these displayed results.\n\n<details><summary>Solution guidance</summary>\n\nTrace actual source IDs and tensor/answer boundaries before judging the final output. A downstream stage cannot restore evidence that was removed upstream. Use an independent verifier or human proposition judgment for semantic claims; a valid schema, similarity score, or citation marker does not establish truth.\n\n</details>\n",
        ),
        md(
            "verification",
            "## Verification boundary\n\nNotebooks and CLI import the same source. Run notebooks before final CLI --record commands, because timing-bearing artifacts are overwritten. Author execution/review is distinct from independent teaching acceptance and beginner feedback. No book pages or artwork are redistributed.\n",
        ),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3.11 (atlas retrieval)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "cells": cells,
    }
