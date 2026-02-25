# Instantiate a docker container with command 
docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
  -e http_proxy=http://proxy-dmz.intel.com:912 \
  -e https_proxy=http://proxy-dmz.intel.com:912 \
  -e HTTP_PROXY=http://proxy-dmz.intel.com:912 \
  -e HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
  -e no_proxy=localhost,127.0.0.1,0.0.0.0 \
  -e NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
  -e HF_TOKEN=${HF_READ_TOKEN} \
  --name=vllm-test \
  --device /dev/dri:/dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /dev/dri/by-path:/dev/dri/by-path \
  -v /root/dkorat/vllm-bench/:/root/vllm-bench \
  --entrypoint= intel/vllm:0.14.1-xpu /bin/bash

# Run command
docker exec -it vllm-test bash
# in 2 separate terminals to enter container environments for the server and the client respectively.

# Launch Server in the Server Environment
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve Qwen/Qwen3-4B-Thinking-2507 \
  --dtype=bfloat16 \
  --enforce-eager \
  --port 8000 \
  --block-size 64 \
  --gpu-memory-util 0.9 \
  --no-enable-prefix-caching \
  --disable-sliding-window \
  --disable-log-requests \
  --max-num-batched-tokens=8192 \
  --max-model-len 2048 \
  -tp=4
#   --quantization fp8

# Launch benchmarking Client in the Client Environment
export no_proxy=localhost,127.0.0.1,0.0.0.0
export NO_PROXY=localhost,127.0.0.1,0.0.0.0
vllm bench serve \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --trust-remote-code \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --ignore-eos \
  --max-concurrency 2 \
  --num-prompts 4 \
  --save-result --result-filename bench-results_gpt-oss-20b-tp=4-quant=no.json
