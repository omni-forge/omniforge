#!/usr/bin/env python3
"""FastAPI server for OmniForge streaming inference.

Source: ZIP1 (primary) with ZIP2/ZIP6 style improvements (lifespan, better UI).
Uses Server-Sent Events for streaming token generation.
"""

import json
import time
from typing import Generator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

import config
from inference import generate_tokens, load_model_and_tokenizer


app = FastAPI(title="OmniForge Inference Server")


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", description="Prompt text")
    max_new_tokens: int = Field(default=config.DEFAULT_MAX_NEW_TOKENS, ge=1, le=2048)
    temperature: float = Field(default=config.DEFAULT_TEMPERATURE, ge=0.01, le=5.0)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=0, le=config.VOCAB_SIZE)
    top_p: float = Field(default=config.DEFAULT_TOP_P, ge=0.01, le=1.0)


@app.on_event("startup")
def startup() -> None:
    load_model_and_tokenizer()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OmniForge Inference Server</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0a0f; color: #e0e0e0; font-family: system-ui, -apple-system, sans-serif; padding: 2rem; display: flex; justify-content: center; }
    .container { max-width: 860px; width: 100%; }
    h1 { color: #f59e0b; font-size: 1.75rem; font-weight: 700; letter-spacing: .05em; margin-bottom: .25rem; }
    .subtitle { color: #6b7280; font-size: .75rem; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 2rem; }
    .card { background: #111118; border: 1px solid #1f2937; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    label { display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; color: #9ca3af; margin-bottom: .5rem; }
    textarea { width: 100%; height: 120px; background: #000; border: 1px solid #1f2937; border-radius: 4px; color: #e0e0e0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .8125rem; padding: .75rem; resize: vertical; outline: none; }
    textarea:focus { border-color: #f59e0b; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem; }
    .slider-row { display: flex; align-items: center; gap: .75rem; }
    input[type=range] { flex: 1; accent-color: #f59e0b; }
    .val { color: #f59e0b; font-size: .8125rem; min-width: 2.5rem; text-align: right; }
    button { width: 100%; padding: .75rem; background: #f59e0b; color: #000; border: none; border-radius: 6px; font-weight: 600; font-size: .875rem; text-transform: uppercase; letter-spacing: .08em; cursor: pointer; margin-top: 1rem; transition: all .15s; }
    button:hover { background: #d97706; }
    button:disabled { opacity: .4; cursor: not-allowed; }
    .output-header { display: flex; justify-content: space-between; margin-bottom: .75rem; }
    pre { background: #000; border: 1px solid #1f2937; border-radius: 4px; padding: 1rem; min-height: 200px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .8125rem; line-height: 1.6; color: #34d399; white-space: pre-wrap; word-break: break-word; }
    #tps { color: #6b7280; font-size: .75rem; }
    .cursor { display: inline-block; width: 8px; height: 16px; background: #f59e0b; animation: blink .8s infinite; vertical-align: middle; margin-left: 2px; }
    @keyframes blink { 0%,50% { opacity:1; } 51%,100% { opacity:0; } }
  </style>
</head>
<body>
<div class="container">
  <h1>OmniForge</h1>
  <div class="subtitle">125M Parameter Code Language Model</div>
  <div class="card">
    <label>Prompt</label>
    <textarea id="prompt" placeholder="Enter a code prompt...">def fibonacci(n):</textarea>
    <div class="grid">
      <div>
        <label>Temperature</label>
        <div class="slider-row">
          <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="0.8">
          <span class="val" id="tempValue">0.80</span>
        </div>
      </div>
      <div>
        <label>Max Tokens</label>
        <div class="slider-row">
          <input type="range" id="maxTokens" min="16" max="1024" step="16" value="256">
          <span class="val" id="maxValue">256</span>
        </div>
      </div>
    </div>
    <button id="submitBtn">Generate</button>
  </div>
  <div class="card">
    <div class="output-header">
      <span>Generated Output</span>
      <span id="tps">— tok/s</span>
    </div>
    <pre id="output"><span class="cursor"></span></pre>
  </div>
</div>
<script>
const promptEl = document.getElementById('prompt');
const outputEl = document.getElementById('output');
const tempEl = document.getElementById('temperature');
const maxEl = document.getElementById('maxTokens');
const tempVal = document.getElementById('tempValue');
const maxVal = document.getElementById('maxValue');
const tpsEl = document.getElementById('tps');
const btn = document.getElementById('submitBtn');

tempEl.oninput = () => tempVal.textContent = parseFloat(tempEl.value).toFixed(2);
maxEl.oninput = () => maxVal.textContent = maxEl.value;

btn.onclick = async () => {
  outputEl.textContent = '';
  tpsEl.textContent = 'Generating...';
  btn.disabled = true;
  let tokenCount = 0;
  const startTime = performance.now();
  try {
    const resp = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        prompt: promptEl.value,
        max_new_tokens: parseInt(maxEl.value),
        temperature: parseFloat(tempEl.value),
        top_k: 50,
        top_p: 0.95
      })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const events = buf.split('\\n\\n');
      buf = events.pop();
      for (const ev of events) {
        if (!ev.startsWith('data: ')) continue;
        const data = JSON.parse(ev.slice(6));
        if (data.done) {
          tpsEl.textContent = data.tokens_per_second.toFixed(2) + ' tok/s';
        } else if (data.token !== undefined) {
          outputEl.textContent += data.token;
          tokenCount++;
        }
      }
    }
  } catch(e) {
    outputEl.textContent = 'Error: ' + e.message;
    tpsEl.textContent = 'Error';
  }
  btn.disabled = false;
};
</script>
</body>
</html>"""


@app.post("/generate")
def generate_endpoint(request: GenerateRequest) -> StreamingResponse:
    def event_stream() -> Generator[str, None, None]:
        start = time.time()
        count = 0
        for token in generate_tokens(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            strategy="top_p",
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        ):
            count += 1
            yield f"data: {json.dumps({'token': token})}\n\n"
        elapsed = max(time.time() - start, 1e-9)
        yield f"data: {json.dumps({'done': True, 'tokens': count, 'tokens_per_second': count / elapsed})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)