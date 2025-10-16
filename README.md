# Trace Analysis and Replying

Place `trace_replayer.py` into `vllm/benchmarks`

Examples: 
```bash
python vllm/benchmarks/replay_gmi_trace.py --trace-file gmi_trace_2_cloned.jsonl --start-time 0 --duration 120 --host localhost --port 8000 --model Qwen/Qwen3-30B-A3B --output-file gmi_results.json --request-rate 8
```