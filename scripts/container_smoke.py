import json
import time
import urllib.request

for attempt in range(30):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/ready", timeout=2) as response:
            assert response.status == 200
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("Container did not become ready")
with urllib.request.urlopen("http://127.0.0.1:8000/api/examples") as response:
    payload = json.load(response)
examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
transaction = examples[0].get("transaction", examples[0])
request = urllib.request.Request(
    "http://127.0.0.1:8000/predict", data=json.dumps(transaction).encode(), headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(request) as response:
    prediction = json.load(response)
assert 0 <= prediction["fraud_probability"] <= 1
print("Container readiness and inference verified")
