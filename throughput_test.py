import asyncio
import httpx
import pandas as pd
import time
import argparse
from tqdm.asyncio import tqdm_asyncio
import os


# --- Argument Parser ---
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default="sample_collection.tsv", help="CSV file with a column named 'prompt'")
parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent requests")
parser.add_argument("--max_tokens", type=int, default=10, help="Max completion tokens")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Model name")
parser.add_argument("--url", type=str, default="http://localhost:8000/v1/chat/completions", help="Inference URL")
parser.add_argument("--guided", nargs="+", default=["enterprise", "non enterprise"], help="Guided choices")
parser.add_argument("--api_key", type=str, default="sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86", help="API key for OpenAI-compatible endpoint")
args = parser.parse_args()


# --- Load Prompts ---
df=pd.read_csv(args.csv,header=None,index_col=None,sep='\t')
df.columns=['doc_id','factsheet']
df['len']=df.factsheet.apply(lambda x:len(x.split(' ')))
df=df[df['len']<1024]
prompts = df["factsheet"].iloc[:100].dropna().tolist()


# --- Async Inference Call ---
async def call_model(prompt, client, prompt_id):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "extra_body": {
            "guided_choice": args.guided,
            "top_k": 1
        }
    }

    start_time = time.time()
    error_detail = None
    response_text = None
    status_code = None
    
    try:
        response = await client.post(args.url, json=payload, timeout=30.0)
        status_code = response.status_code
        
        if response.status_code == 200:
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            status = "success"
        else:
            error_detail = response.text[:500]  # Capture response text but limit length
            status = "error"
    except Exception as e:
        error_detail = str(e)
        status = "error"
    
    latency = time.time() - start_time
    
    return {
        "prompt_id": prompt_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latency": latency,
        "status": status,
        "status_code": status_code,
        "error_detail": error_detail,
        "response": response_text,
        "prompt_length": len(prompt),
        "model": args.model,
        "concurrency": args.concurrency
    }


# --- Benchmark Function ---
async def run_benchmark():
    results = []
    connector = httpx.AsyncHTTPTransport(retries=2)
    headers = {
        "Authorization": f"Bearer {args.api_key}"
    }

    async with httpx.AsyncClient(transport=connector, headers=headers) as client:
        sem = asyncio.Semaphore(args.concurrency)

        async def bounded_call(prompt, prompt_id):
            async with sem:
                return await call_model(prompt, client, prompt_id)

        tasks = [bounded_call(prompt, i) for i, prompt in enumerate(prompts)]
        for future in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Running benchmark"):
            result = await future
            results.append(result)

    return results


# --- Main ---
if __name__ == "__main__":
    # Create output filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.csv"
    
    print(f"Starting benchmark: {len(prompts)} prompts with concurrency={args.concurrency}\n")
    start = time.time()
    results = asyncio.run(run_benchmark())
    duration = time.time() - start

    valid_latencies = [r["latency"] for r in results if r.get("latency") is not None]
    avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
    throughput = len(valid_latencies) / duration if duration > 0 else 0
    
    # Count successes and errors
    success_count = sum(1 for r in results if r.get("status") == "success")
    error_count = len(results) - success_count

    # Analyze error patterns
    error_types = {}
    for r in results:
        if r.get("status") == "error":
            error_type = str(r.get("error_detail", "Unknown"))[:50]  # Use first 50 chars as error type
            error_types[error_type] = error_types.get(error_type, 0) + 1

    print("\n--- Benchmark Results ---")
    print(f"Total Requests: {len(prompts)}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {error_count}")
    print(f"Total Time: {duration:.2f} sec")
    print(f"Throughput: {throughput:.2f} req/sec")
    print(f"Average Latency: {avg_latency:.3f} sec")
    
    if valid_latencies:
        p50 = sorted(valid_latencies)[len(valid_latencies)//2]
        p90 = sorted(valid_latencies)[int(len(valid_latencies)*0.9)]
        p99 = sorted(valid_latencies)[int(len(valid_latencies)*0.99)]
        print(f"P50 Latency: {p50:.3f} sec")
        print(f"P90 Latency: {p90:.3f} sec")
        print(f"P99 Latency: {p99:.3f} sec")
    
    if error_count > 0:
        print("\n--- Error Analysis ---")
        for error_type, count in error_types.items():
            print(f"{count} requests failed with: {error_type}")

    # Create benchmark summary with metadata
    benchmark_summary = {
        "benchmark_id": timestamp,
        "model": args.model,
        "concurrency": args.concurrency,
        "total_requests": len(prompts),
        "successful_requests": success_count,
        "failed_requests": error_count,
        "total_time_seconds": duration,
        "throughput_req_per_sec": throughput,
        "avg_latency_seconds": avg_latency,
        "p50_latency_seconds": p50 if valid_latencies else None,
        "p90_latency_seconds": p90 if valid_latencies else None,
        "p99_latency_seconds": p99 if valid_latencies else None
    }
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Save summary to a central log file for comparing multiple benchmark runs
    summary_file = "benchmark_history.csv"
    summary_df = pd.DataFrame([benchmark_summary])
    
    # Append or create the summary file
    if os.path.exists(summary_file):
        summary_df.to_csv(summary_file, mode='a', header=False, index=False)
    else:
        summary_df.to_csv(summary_file, index=False)

    print(f"\nDetailed results saved to: {output_file}")
    print(f"Benchmark summary added to: {summary_file}")
