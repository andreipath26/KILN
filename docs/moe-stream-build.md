# MoE Streaming Build — Session Note

## What was built

The `moe-stream` branch of `github.com/andreipath26/kiln-llama` was
cloned to `/tmp/llama-stream` and built on the Dell Latitude 7490.
Result: a working `llama` binary with the streaming flags present.

## Flags confirmed present

- `--moe-stream`
- `--moe-stream-cache <NG|Ns>`
- `--moe-stream-io-threads N`
- `--moe-stream-direct`
- `-ncmoe` / `--n-cpu-moe N`
- `--spec-draft-n-cpu-moe` / `-ncmoed`

## Flags confirmed absent

- `--moe-stream-window` — not in the help output. This is the buggy
  feature noted in the fix list (issue 2). Its absence is good.

## Build command

    cd /tmp && git clone --branch moe-stream \
        https://github.com/andreipath26/kiln-llama.git llama-stream
    cd llama-stream
    cmake -B build -DGGML_NATIVE=ON -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
    cmake --build build --config Release -j4 --target llama-app

## Not tested

No MoE model was downloaded. Data budget is tight. The next test
requires Qwen3-30B-A3B at Q2_K (~11 GB) or Q4_K_M (18.5 GB). Do not
download until the data budget allows.

## Next steps when data allows

1. Download Qwen3-30B-A3B at Q2_K.
2. Run `llama cli -m <model> --moe-stream --moe-stream-cache 8 -p "hi" -n 5`
   and measure tok/s.
3. If it works, integrate the streaming path into KILN as
   `cpu_moe_streamed` (Phase 5.4).
4. Apply the fix list in roadmap Section 15.6.
