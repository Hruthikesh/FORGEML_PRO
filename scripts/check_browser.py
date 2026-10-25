"""Exercise the running dashboard and capture its actual rendered pages.

Run after starting the API: python scripts/check_browser.py
"""
import base64
import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

output = Path("browser-results")
output.mkdir(exist_ok=True)
with sync_playwright() as playwright:
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto("http://127.0.0.1:8000", wait_until="networkidle")
    expect(page.locator("#connection")).to_have_text("Local API connected")
    expect(page.locator("#mode")).to_contain_text("SYNTHETIC")
    for view in ["overview", "laboratory", "prediction", "graph", "monitoring", "governance"]:
        page.locator(f'nav button[data-view="{view}"]').click()
        expect(page.locator("#view")).not_to_be_empty()
        expect(page.locator("#error")).to_be_hidden()
        if view == "prediction":
            examples = page.request.get("http://127.0.0.1:8000/api/examples").json()["examples"]
            page.locator("#example-select").select_option("0")
            expect(page.locator('[name="TransactionID"]')).to_have_value(str(examples[0]["TransactionID"]))
            page.locator('[name="card1"]').fill("")
            with page.expect_request(lambda request: request.url.endswith("/predict")) as sent:
                page.locator("#predict-form button").click()
            assert sent.value.post_data_json["TransactionID"] == examples[0]["TransactionID"]
            assert sent.value.post_data_json["card1"] is None
            expect(page.locator("#prediction-result")).to_contain_text("%", timeout=30000)
            expect(page.locator("#example-select")).to_have_value("0")
            expect(page.locator("#error")).to_be_hidden()
        if view in ["overview", "prediction"]:
            page.screenshot(path=str(output / f"dashboard-{view}.jpg"), type="jpeg", quality=85, full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator('nav button[data-view="overview"]').click()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "Mobile horizontal overflow"
    assert not errors, errors
    browser.close()
print("Browser checks passed: six views, live prediction, optional input and mobile layout")
# Export capture bytes for retrieving through CI logs when artifact downloads are unavailable.
if os.environ.get("FORGEML_EXPORT_SCREENSHOTS") == "1":
    for screenshot in sorted(output.glob("*.jpg")):
        print(f"FORGEML_SCREENSHOT {screenshot.name} {base64.b64encode(screenshot.read_bytes()).decode()}")
