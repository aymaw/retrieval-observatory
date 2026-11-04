import argparse
import json
import time
from pathlib import Path

import numpy as np

from .core import Index, answer, ranking_metrics, verify_citation
from .data import corpus, queries
from .visual import save, table


def run(out, quick=False):
    start = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    docs = corpus()
    index = Index(docs)
    dev = queries("development")
    # Cache the top hit's development cosine: threshold search never needs to refit or rerank.
    dev_scores = [
        (q, bool(q["relevant"]), index.search(q["query"], q["station"])[0]["word"]) for q in dev
    ]
    candidates = []
    for c in np.linspace(0, 1, 41):
        pos = [s >= c for q, positive, s in dev_scores if positive]
        neg = [s < c for q, positive, s in dev_scores if not positive]
        candidates.append(((np.mean(pos) + np.mean(neg)) / 2, float(c)))
    cutoff = max(candidates)[1]
    test = queries("test")
    if quick:
        test = test[:30]
    records = []
    summary = []
    for mode in ["bm25", "word", "character", "hybrid"]:
        mode_rows = []
        for q in test:
            hits = index.search(q["query"], q["station"], mode=mode)
            ids = [h["document"].id for h in hits]
            metrics = ranking_metrics(ids, q["relevant"]) if q["relevant"] else None
            a = answer(index, q["query"], q["station"], cutoff, mode)
            citation = a["citation"]
            integrity = (
                verify_citation(citation, next(d for d in docs if d.id == citation["document_id"]))
                if citation
                else None
            )
            row = {
                **q,
                "mode": mode,
                "ranking": ids,
                "retrieval": metrics,
                **a,
                "citation_integrity": integrity,
                "correct_evidence": bool(citation and citation["document_id"] in q["relevant"]),
            }
            records.append(row)
            mode_rows.append(row)
        for kind in ["literal", "typo", "paraphrase"]:
            selected = [r for r in mode_rows if r["kind"] == kind]
            if selected:
                summary.append(
                    {
                        "method": mode,
                        "slice": kind,
                        "n": len(selected),
                        **{
                            k: round(float(np.mean([r["retrieval"][k] for r in selected])), 4)
                            for k in ["recall_at_5", "mrr_at_5", "ndcg_at_5"]
                        },
                        "correct_evidence_rate": round(
                            float(np.mean([r["correct_evidence"] for r in selected])), 4
                        ),
                    }
                )
    unsupported = [r for r in records if r["mode"] == "hybrid" and not r["relevant"]]
    answered = [r for r in records if r["mode"] == "hybrid" and not r["abstained"]]
    false_answer_rate = (
        float(np.mean([not r["abstained"] for r in unsupported])) if unsupported else None
    )
    payload = {
        "quick": quick,
        "documents": len(docs),
        "development_queries": len(dev),
        "test_queries": len(test),
        "cutoff": cutoff,
        "summary": summary,
        "records": records,
        "hybrid_unsupported_false_answer_rate": false_answer_rate,
        "answered_citation_integrity": float(np.mean([r["citation_integrity"] for r in answered]))
        if answered
        else None,
        "seconds": time.perf_counter() - start,
        "corpus": [d.__dict__ for d in docs],
    }
    body = (
        '<div class="card"><label>Inspect a held-out query <select id="query"></select>'
        '</label><div id="detail"></div></div><h2>Ranking ablations</h2>'
        + table(
            summary, ["method", "slice", "n", "recall_at_5", "mrr_at_5", "correct_evidence_rate"]
        )
        + "<p>Hybrid unsupported false-answer rate: "
        + str(false_answer_rate)
        + ". Citation integrity verifies source identity and exact offsets; it does not "
        "prove the quote answers the question. Character features help spelling overlap "
        "but are not learned semantic embeddings. Unfamiliar paraphrases remain a hard slice.</p>"
    )
    script = """
    const qs=DATA.records.filter(r=>r.mode==='hybrid'),select=document.getElementById('query');
    qs.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${r.station} · ${r.kind} · ${r.query}`;select.append(o)});
    function draw(){const r=qs[Number(select.value)],box=document.getElementById('detail');box.replaceChildren();
      for(const [label,value] of [['Outcome',r.abstained?'ABSTAIN':'QUOTE RETURNED'],['Correct evidence',r.correct_evidence],
          ['Word cosine',r.evidence_score.toFixed(3)],['Relevant documents',r.relevant.join(', ')||'none'],
          ['Top five',r.ranking.join(' → ')],['Citation',r.citation?`${r.citation.document_id} [${r.citation.start}:${r.citation.end}] sha256:${r.citation.sha256}`:'none'],
          ['Quote',r.citation?.quote||'No quote returned']]){const p=document.createElement('p');p.textContent=label+': '+value;box.append(p)}
    }select.onchange=draw;draw();
    """
    save(
        out,
        payload,
        "Retrieval Observatory",
        "Inspect every ranking and citation in a fictional orbital-station manual search. "
        "Compare BM25, word TF-IDF, character TF-IDF and reciprocal rank fusion.",
        body,
        script,
    )
    preview = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 520">',
        '<rect width="840" height="520" fill="#101f36"/>',
        '<text x="24" y="35" fill="#e9f0ff" font-family="sans-serif" font-size="22">Held-out retrieval MRR@5 — higher is better</text>',
    ]
    for i, r in enumerate(summary):
        y = 60 + i * 35
        preview.append(
            f'<text x="24" y="{y + 16}" fill="#e9f0ff" font-family="sans-serif">{r["method"]} / {r["slice"]}</text>'
        )
        preview.append(
            f'<rect x="240" y="{y}" width="{r["mrr_at_5"] * 460}" height="22" fill="#66e4cc"/>'
        )
        preview.append(
            f'<text x="730" y="{y + 16}" fill="#e9f0ff" font-family="sans-serif">{r["mrr_at_5"]}</text>'
        )
    (out / "preview.svg").write_text("".join(preview) + "</svg>")
    print(
        json.dumps({"summary": summary, "cutoff": cutoff, "false_answer_rate": false_answer_rate})
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/reference")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    run(a.out, a.quick)
