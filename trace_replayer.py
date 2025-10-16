#!/usr/bin/env python3
"""
Script to replay GMI trace using vLLM benchmark serving.
Converts hash_ids to synthetic prompts and replays the trace with original timing.
"""

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the benchmarks directory to the path
sys.path.append(str(Path(__file__).parent / "benchmarks"))

from benchmark_dataset import SampleRequest
from benchmark_serving import benchmark, calculate_metrics
from backend_request_func import get_tokenizer, ASYNC_REQUEST_FUNCS
from benchmark_serving import RequestFuncInput


class GMITraceDataset:
    """Custom dataset that generates requests based on GMI trace."""
    
    def __init__(self, trace_file: str, start_time: float = 0, duration: float = float('inf')):
        self.trace_file = trace_file
        self.start_time = start_time
        self.duration = duration
        self.requests = []
        self.load_trace()
    
    def load_trace(self):
        """Load and filter trace data."""
        # for GMI trace, we should first sort the trace by timestamp
        # and turn the absolute timestamps to relative timestamps (start from 0)
        raw_entries = []
        with open(self.trace_file, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    raw_entries.append(entry)
        if not raw_entries:
            raise ValueError(f"No entries found in trace file: {self.trace_file}")
        
        raw_entries.sort(key = lambda x : x['timestamp'])
        first_timestamp = int(raw_entries[0]['timestamp'])
        print(f"First timestamp: {first_timestamp}")

        # GMI trace is in nanoseconds, we need to convert it to seconds
        for entry in raw_entries:
            entry['relative_timestamp'] = (int(entry['timestamp']) - first_timestamp) / 1_000_000_000.0
        
        started = False
        for entry in raw_entries:
            timestamp = entry['relative_timestamp']
            if timestamp >= self.start_time and timestamp <= self.start_time + self.duration:
                started = True
                self.requests.append(entry)
            elif started:
                # early exit since we have already found the end of the time window
                break
        
        print(f"Loaded {len(self.requests)} requests from trace")
    
    def generate_synthetic_prompt(self, hash_ids: List[int], target_length: int, tokenizer) -> str:
        """Generate synthetic prompt based on hash_ids and target length."""
        # Use hash_ids as seed for deterministic generation
        import random
        import numpy as np
        
        # Create a deterministic seed from hash_ids
        seed = sum(hash_ids) % (2**31)
        random.seed(seed)
        np.random.seed(seed)
        
        vocab_size = tokenizer.vocab_size
        
        # Generate tokens to approximately match target length
        # Account for potential compression during encode/decode
        buffer_factor = 1.2
        initial_tokens = int(target_length * buffer_factor)
        
        # Use hash_ids to influence token generation
        base_offset = hash_ids[0] if hash_ids else 0
        token_ids = []
        
        chunk_size = 32
        num_chunks = initial_tokens // chunk_size

        for i in range(num_chunks):
            # Create pseudo-random but deterministic token sequence
            # token_id = (base_offset + i + sum(hash_ids[i % len(hash_ids):i % len(hash_ids) + 3])) % vocab_size
            start_hash_idx = hash_ids[i % len(hash_ids)]
            for j in range(chunk_size):
                token_id = (base_offset + i + start_hash_idx + j) % vocab_size
                token_ids.append(token_id)
        
        # Decode to text and re-encode to get actual length
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        final_tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Truncate or pad to target length
        if len(final_tokens) > target_length:
            final_tokens = final_tokens[:target_length]
        elif len(final_tokens) < target_length:
            # Pad with more deterministic tokens
            needed = target_length - len(final_tokens)
            padding = [(base_offset + len(final_tokens) + i) % vocab_size for i in range(needed)]
            final_tokens.extend(padding)
        
        # Final decode
        return tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    def to_sample_requests_with_timing(self, tokenizer) -> List[Dict[str, Any]]:
        """Convert trace entries to SampleRequest objects with original timestamps."""
        sample_requests = []
        print("started converting to sample requests with timing")
        
        # Calculate relative timestamps from the first request
        if not self.requests:
            return sample_requests
            
        first_timestamp = self.requests[0]['relative_timestamp']
        
        for entry in self.requests:
            input_length = entry['input_length']
            output_length = entry['output_length']
            hash_ids = entry['hash_ids']
            timestamp = entry['relative_timestamp']
            
            # Generate synthetic prompt
            prompt = self.generate_synthetic_prompt(hash_ids, input_length, tokenizer)
            
            # Verify actual prompt length
            actual_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            
            sample_request = SampleRequest(
                prompt=prompt,
                prompt_len=actual_length,
                expected_output_len=output_length,
            )
            
            # Add timing information
            sample_requests.append({
                'request': sample_request,
                'timestamp': timestamp,
                'relative_time': timestamp - first_timestamp,
                'original_entry': entry
            })
        print("finished converting to sample requests with timing")
        return sample_requests


async def send_timed_request(request_data: Dict[str, Any], request_func, api_url: str, 
                           model_id: str, model_name: str, logprobs, ignore_eos: bool,
                           extra_body: Dict[str, Any]):
    """Send a single request at the appropriate time."""
    
    request = request_data['request']
    
    # Create the request input
    request_func_input = RequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=request.prompt,
        api_url=api_url,
        prompt_len=request.prompt_len,
        output_len=request.expected_output_len,
        logprobs=logprobs,
        multi_modal_content=None,
        ignore_eos=ignore_eos,
        extra_body=extra_body,
    )

    
    # Send the request
    start_time = time.time()
    output = await request_func(request_func_input=request_func_input)
    end_time = time.time()
    
    # Add timing info to the output
    output.request_start_time = start_time
    output.request_end_time = end_time
    output.original_timestamp = request_data['timestamp']
    output.relative_time = request_data['relative_time']
    
    return output


async def replay_trace_with_timing(args):
    """Replay trace with original timing preserved."""
    
    # Load tokenizer
    tokenizer = get_tokenizer(
        args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Create dataset from trace
    dataset = GMITraceDataset(
        trace_file=args.trace_file,
        start_time=args.start_time,
        duration=args.duration
    )
    
    # Convert to sample requests with timing
    timed_requests = dataset.to_sample_requests_with_timing(tokenizer)
    
    if not timed_requests:
        print("No requests found in the specified time window!")
        return
    
    print(f"Generated {len(timed_requests)} timed requests")
    print(f"Time span: {timed_requests[0]['relative_time']:.2f}s to {timed_requests[-1]['relative_time']:.2f}s")
    
    # Set API URL
    if args.base_url:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = args.base_url
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    
    # Get request function
    if args.backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[args.backend]
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    
    # Test with first request
    print("Testing connection with first request...")
    test_output = await send_timed_request(
        timed_requests[0], request_func, api_url, args.model, args.model,
        args.logprobs, args.ignore_eos, {"temperature": 0.0}
    )
    
    if not test_output.success:
        raise ValueError(f"Test request failed: {test_output.error}")
    
    print("Connection test successful. Starting timed replay...")
    
    # Schedule requests according to their original timestamps
    tasks = []
    start_time = time.time()
    
    async def schedule_request(request_data):
        # Wait until the appropriate time to send this request
        delay = request_data['relative_time']
        if args.time_scale != 1.0:
            delay *= args.time_scale
            
        if delay > 0:
            await asyncio.sleep(delay)
        
        return await send_timed_request(
            request_data, request_func, api_url, args.model, args.model,
            args.logprobs, args.ignore_eos, {"temperature": 0.0}
        )
    
    # Create all scheduled tasks
    for request_data in timed_requests:
        task = asyncio.create_task(schedule_request(request_data))
        tasks.append(task)
    
    print(f"Scheduled {len(tasks)} requests. Starting replay...")
    
    # Wait for all requests to complete
    outputs = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful outputs
    successful_outputs = []
    failed_count = 0
    
    for output in outputs:
        if isinstance(output, Exception):
            print(f"Request failed with exception: {output}")
            failed_count += 1
        elif hasattr(output, 'success') and output.success:
            successful_outputs.append(output)
        else:
            failed_count += 1
    
    actual_duration = time.time() - start_time
    
    print(f"Replay completed in {actual_duration:.2f}s")
    print(f"Successful requests: {len(successful_outputs)}")
    print(f"Failed requests: {failed_count}")
    
    # Calculate metrics
    input_requests = [req_data['request'] for req_data in timed_requests]
    
    if successful_outputs:
        metrics, _ = calculate_metrics(
            input_requests=input_requests[:len(successful_outputs)],
            outputs=successful_outputs,
            dur_s=actual_duration,
            tokenizer=tokenizer,
            selected_percentile_metrics=["ttft", "tpot", "itl"],
            selected_percentiles=[50.0, 90.0, 95.0, 99.0],
            goodput_config_dict={}
        )
        
        result = {
            'metrics': dataclasses.asdict(metrics),
            'successful_requests': len(successful_outputs),
            'failed_requests': failed_count,
            'total_requests': len(timed_requests),
            'actual_duration': actual_duration,
            'original_time_span': timed_requests[-1]['relative_time'],
            'time_scale': args.time_scale
        }
        
        # Save results if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {args.output_file}")
        
        return result


async def replay_trace(args):
    """Main function to replay the GMI trace."""
    
    if args.preserve_timing:
        return await replay_trace_with_timing(args)
    
    # Original implementation for non-timed replay
    # Load tokenizer
    tokenizer = get_tokenizer(
        args.model,
        tokenizer_mode=args.tokenizer_mode,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Create dataset from trace
    dataset = GMITraceDataset(
        trace_file=args.trace_file,
        start_time=args.start_time,
        duration=args.duration
    )
    
    # Convert to sample requests
    input_requests = [req_data['request'] for req_data in dataset.to_sample_requests_with_timing(tokenizer)]
    
    if not input_requests:
        print("No requests found in the specified time window!")
        return
    
    print(f"Generated {len(input_requests)} sample requests")
    print(f"Input lengths: {[req.prompt_len for req in input_requests[:5]]}... (showing first 5)")
    print(f"Output lengths: {[req.expected_output_len for req in input_requests[:5]]}... (showing first 5)")
    
    # Set API URL
    if args.base_url:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = args.base_url
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"
    
    # Run benchmark
    result = await benchmark(
        backend=args.backend,
        api_url=api_url,
        base_url=base_url,
        model_id=args.model,
        model_name=args.model,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=args.logprobs,
        request_rate=args.request_rate,
        burstiness=args.burstiness,
        disable_tqdm=args.disable_tqdm,
        profile=False,
        selected_percentile_metrics=["ttft", "tpot", "itl"],
        selected_percentiles=[50.0, 90.0, 95.0, 99.0],
        ignore_eos=args.ignore_eos,
        goodput_config_dict={},
        max_concurrency=args.max_concurrency,
        lora_modules=None,
        extra_body={"temperature": 0.0},
    )
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Replay GMI trace with vLLM benchmark")
    
    # Trace file options
    parser.add_argument("--trace-file", type=str, required=True,
                       help="Path to GMI trace JSONL file")
    parser.add_argument("--start-time", type=float, default=0,
                       help="Start time in seconds (default: 0)")
    parser.add_argument("--duration", type=float, default=60,
                       help="Duration to replay in seconds (default: 60)")
    
    # Timing options
    parser.add_argument("--preserve-timing", action="store_true",
                       help="Preserve original timestamps from the trace (default: False)")
    parser.add_argument("--time-scale", type=float, default=1.0,
                       help="Scale factor for time intervals (1.0 = real-time, 0.5 = 2x faster, 2.0 = 2x slower)")
    
    # vLLM server options
    parser.add_argument("--backend", type=str, default="vllm",
                       choices=["vllm", "openai-chat"],
                       help="Backend to use")
    parser.add_argument("--base-url", type=str, default=None,
                       help="Base URL for the API server")
    parser.add_argument("--host", type=str, default="localhost",
                       help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--endpoint", type=str, default="/v1/completions",
                       help="API endpoint")
    
    # Model options
    parser.add_argument("--model", type=str, required=True,
                       help="Model name")
    parser.add_argument("--tokenizer-mode", type=str, default="auto",
                       choices=["auto", "slow"],
                       help="Tokenizer mode")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code")
    
    # Benchmark options (only used when preserve-timing is False)
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                       help="Request rate (requests per second) - ignored if --preserve-timing is used")
    parser.add_argument("--burstiness", type=float, default=1.0,
                       help="Burstiness factor - ignored if --preserve-timing is used")
    parser.add_argument("--max-concurrency", type=int, default=None,
                       help="Maximum concurrent requests - ignored if --preserve-timing is used")
    parser.add_argument("--logprobs", type=int, default=None,
                       help="Number of logprobs to return")
    parser.add_argument("--ignore-eos", action="store_true",
                       help="Ignore EOS tokens")
    parser.add_argument("--disable-tqdm", action="store_true",
                       help="Disable progress bar")
    
    # Output options
    parser.add_argument("--output-file", type=str, default=None,
                       help="File to save benchmark results")
    
    args = parser.parse_args()
    
    # Run the replay
    asyncio.run(replay_trace(args))


if __name__ == "__main__":
    main() 