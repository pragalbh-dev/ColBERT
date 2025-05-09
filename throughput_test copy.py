

import asyncio
import httpx
import pandas as pd
import time
import argparse
from tqdm.asyncio import tqdm_asyncio


# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="sample_collection.tsv", help="CSV file with a column named 'prompt'")
parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent requests")
parser.add_argument("--max_tokens", type=int, default=10, help="Max completion tokens")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
parser.add_argument("--url", type=str, default="http://localhost:8000/v1/chat/completions", help="Inference URL")
parser.add_argument("--guided", nargs="+", default=["enterprise", "non enterprise"], help="Guided choices")
parser.add_argument("--api_key", type=str, default="sk-fake-key", help="API key for OpenAI-compatible endpoint")
args = parser.parse_args()


# --- Load Prompts ---
df=pd.read_csv(args.csv,header=None,index_col=None,sep='\t')
df.columns=['doc_id','factsheet']
df['len']=df.factsheet.apply(lambda x:len(x.split(' ')))
df=df[df['len']<1024]
prompts = df["factsheet"].iloc[:100].dropna().tolist()


# --- Async Inference Call ---
async def call_model(prompt, client):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": args.max_tokens,
        "extra_body": {
            "guided_choice": args.guided,
            "top_k": 1
        }
    }

    start_time = time.time()
    try:
        response = await client.post(args.url, json=payload, timeout=30.0)
        latency = time.time() - start_time
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {"latency": latency, "response": content}
        else:
            return {"latency": latency, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"latency": None, "error": str(e)}


# --- Benchmark Function ---
async def run_benchmark():
    results = []
    connector = httpx.AsyncHTTPTransport(retries=2)
    headers = {
        "Authorization": f"Bearer {args.api_key}"
    }

    async with httpx.AsyncClient(transport=connector, headers=headers) as client:
        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_call(prompt):
            async with sem:
                return await call_model(prompt, client)

        tasks = [bounded_call(prompt) for prompt in prompts]
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Running benchmark"):
            result = await future
            results.append(result)

    return results


# --- Main ---
if __name__ == "__main__":
    print(f"Starting benchmark: {len(prompts)} prompts with concurrency={args.concurrency}\n")
    start = time.time()
    results = asyncio.run(run_benchmark())
    duration = time.time() - start

    valid_latencies = [r["latency"] for r in results if r.get("latency") is not None]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
    throughput = len(valid_latencies) / duration if duration > 0 else 0

    print("\n--- Benchmark Results ---")
    print(f"Total Requests: {len(prompts)}")
    print(f"Total Time: {duration:.2f} sec")
    print(f"Throughput: {throughput:.2f} req/sec")
    print(f"Average Latency: {avg_latency:.3f} sec")

    # Save results
    pd.DataFrame(results).to_csv("async_benchmark_results.csv", index=False)
