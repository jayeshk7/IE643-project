# sets up the demo on a new gpu machine: a venv, the base model, both adapters and the test examples from HF
# B=/some/big/dir bash setup_demo.sh   (uv has to be on PATH or in $B/bin, no HF login needed)
# everything goes inside $B, nothing in the home folder. about 22 GB, or about 8 GB if HF_HUB_CACHE
# already has Qwen2.5-Coder-7B-Instruct
# to start the demo afterwards, from scripts/interface/:
#   export B=/some/big/dir HF_HOME=$B/hf HF_HUB_CACHE=$B/hf_cache XDG_CACHE_HOME=$B/cache TMPDIR=$B/tmp
#   CUDA_VISIBLE_DEVICES=0 $B/venv/bin/python app.py   and tunnel port 7860 (PORT=... for another one)
set -e
export PATH=$B/bin:$PATH HF_HOME=$B/hf HF_HUB_CACHE=${HF_HUB_CACHE:-$B/hf_cache} XDG_CACHE_HOME=$B/cache \
    UV_CACHE_DIR=$B/cache/uv UV_PYTHON_INSTALL_DIR=$B/python TMPDIR=$B/tmp
mkdir -p $B/tmp
uv venv $B/venv --python 3.11
uv pip install --python $B/venv/bin/python torch==2.8.0 transformers==4.57.1 peft==0.17.1 accelerate==1.10.1 \
    cadquery==2.8.0 cadquery-ocp==7.9.3.1.1 trimesh==5.1.1 scipy gradio==6.17.3 "huggingface_hub[cli]"
uv cache clean
$B/venv/bin/hf download Qwen/Qwen2.5-Coder-7B-Instruct
$B/venv/bin/hf download jajamken/DLTP_project --include "patch_lora/*" "rewrite_lora/*" --local-dir $B/hf_model
$B/venv/bin/hf download jajamken/DLTP_project --repo-type dataset --include "data/test/*" --local-dir $B/hf_data
# the paths app.py expects
mkdir -p $B/checkpoints/patch $B/checkpoints/rewrite $B/data/final
ln -sfn $B/hf_model/patch_lora $B/checkpoints/patch/final
ln -sfn $B/hf_model/rewrite_lora $B/checkpoints/rewrite/final
ln -sfn $B/hf_data/data/test $B/data/final/test
echo "demo ready"
