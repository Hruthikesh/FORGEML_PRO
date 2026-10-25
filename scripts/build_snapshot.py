"""Build a portable, read-only dashboard from actual local experiment artifacts.

This is an offline artifact viewer, not a backend-connected application. GET
requests made by the shared dashboard are resolved from embedded, unchanged
artifact JSON. Mutations and inference are disabled and never simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(root: Path, artifact_dir: Path | None = None) -> Path:
    root = root.resolve()
    artifact_dir = artifact_dir or root / "artifacts"
    frontend = root / "frontend"

    def artifact(name: str, default=None):
        path = artifact_dir / name
        if not path.is_file():
            if default is not None:
                return default
            raise FileNotFoundError(f"Required executed artifact missing: {path}")
        return json.loads(path.read_text(encoding="utf-8-sig"))

    report = artifact("report.json")
    # Preserve actual experiment values; no invented fallback metrics.
    mapping = {
        "/api/report": report,
        "/api/examples": {"examples": artifact("examples.json", []), "mode": report.get("mode")},
        "/api/graph": report.get("graph", {}),
        "/api/monitoring": artifact("replay.json", report.get("monitoring", {})),
        "/api/governance": artifact("registry.json", report.get("governance", {})),
        "/api/explanations": artifact("explanations.json", report.get("explanation", {})),
    }
    payload = json.dumps(mapping, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    bootstrap = """
<script id="snapshot-artifacts" type="application/json">__PAYLOAD__</script>
<script>
'use strict';
const snapshotArtifacts=JSON.parse(document.getElementById('snapshot-artifacts').textContent);
window.fetch=async function(path,options={}) {
 if(options.method && options.method.toUpperCase()!=='GET') throw new Error('Offline artifact snapshot: inference and lifecycle actions are unavailable. Start the local API to use these controls.');
 const key=String(path);
 if(!(key in snapshotArtifacts)) throw new Error('Offline artifact snapshot: this resource was not embedded: '+key);
 return {ok:true,status:200,json:async()=>structuredClone(snapshotArtifacts[key])};
};
function applySnapshotState(){
 const connection=document.querySelector('#connection');
 if(connection && connection.textContent!=='Offline artifact viewer')connection.textContent='Offline artifact viewer';
 for(const button of document.querySelectorAll('#predict-form button,#replay,[data-govern]')) {
  if(!button.disabled)button.disabled=true;
  if(button.title!=='Unavailable in the read-only artifact snapshot')button.title='Unavailable in the read-only artifact snapshot';
 }
 const result=document.querySelector('#prediction-result');
 if(result && !result.dataset.snapshotNote){
  result.dataset.snapshotNote='true';
  result.innerHTML='<div class="result-placeholder">Inference is unavailable in this offline artifact snapshot.<br>Start the local API to score transactions.<br>Recorded SHAP artifacts below remain inspectable.</div>';
 }
}
new MutationObserver(applySnapshotState).observe(document.body,{childList:true,subtree:true});
applySnapshotState();
</script>
""".replace("__PAYLOAD__", payload)
    css = (frontend / "styles.css").read_text(encoding="utf-8-sig")
    js = (frontend / "app.js").read_text(encoding="utf-8-sig").replace("</script", "<\\/script")
    html = (frontend / "index.html").read_text(encoding="utf-8-sig")
    html = html.replace(
        "<title>ForgeML Pro · Fraud detection dashboard</title>", "<title>ForgeML Pro · Offline artifact snapshot</title>"
    )
    html = html.replace(
        '<link rel="stylesheet" href="/static/styles.css">',
        "<style>"
        + css
        + "\n.snapshot-banner{padding:13px 25px;border-bottom:1px solid #74613b;background:#352b18;color:#f0ce83;font-size:11px;line-height:1.7;letter-spacing:.3px}.snapshot-banner strong{font-weight:700}</style>",
    )
    html = html.replace(
        "<main>",
        '<main><div class="snapshot-banner" role="note"><strong>OFFLINE ARTIFACT SNAPSHOT</strong> · Inference and actions unavailable. Read-only view of executed local experiment artifacts; no backend connection.</div>',
    )
    html = html.replace('<script src="/static/app.js" defer></script>', bootstrap + "<script>" + js + "</script>")
    target = frontend / "snapshot.html"
    target.write_text(html, encoding="utf-8")
    print(f"Built {target} ({target.stat().st_size:,} bytes) from actual experiment artifacts.")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    build(args.root, args.artifacts)
