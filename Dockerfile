# L2 image: nano-stack/engine:sglang (t4b, replaces upstream Dockerfile — t4 §6.1)
#
# Build context = STACK ROOT (C4 R4):   docker build -f mini-sglang/Dockerfile -t nano-stack/engine:sglang .
# Base singleton (C4): upstream nvidia/cuda:12.8.1-devel + uv venv rejected.
#
# Compiler toolchain five-piece (t4a spike verdict §3): pip nvcc wheel is NOT a
# toolkit — crt/ headers + nvvm/ (cicc) come from NVIDIA apt debs pinned to the
# same 13.0.48-1 as the nvcc wheel; libcuda stub resolves at runtime via the
# nvidia container runtime (dangling symlink at build time is intentional).
FROM nano-stack/base:torch-cu13-py312
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG NVIDIA_REPO=https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu2404/x86_64

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends g++ curl \
    && rm -rf /var/lib/apt/lists/*

# crt/ + nvvm/ extraction (wheel flattens crt into include/ top-level but wrappers
# still #include "crt/*.h"; CUDA 13 splits nvcc and nvvm into two debs)
RUN curl -fsSL ${NVIDIA_REPO}/cuda-crt-13-0_13.0.48-1_amd64.deb -o /tmp/crt.deb \
 && curl -fsSL ${NVIDIA_REPO}/libnvvm-13-0_13.0.48-1_amd64.deb -o /tmp/nvvm.deb \
 && dpkg-deb -x /tmp/crt.deb /tmp/x && dpkg-deb -x /tmp/nvvm.deb /tmp/y \
 && cp -r /tmp/x/usr/local/cuda-13.0/targets/x86_64-linux/include/crt /opt/crt \
 && cp -r /tmp/y/usr/local/cuda-13.0/nvvm /opt/nvvm \
 && rm -rf /tmp/x /tmp/y /tmp/crt.deb /tmp/nvvm.deb

RUN pip install --break-system-packages --index-url ${PIP_INDEX} \
    apache-tvm-ffi==0.1.14.post1 pyzmq msgpack msgspec tqdm ninja \
    flashinfer-python==0.7.0.post1 torch-c-dlpack-ext==0.1.5 \
    "transformers>=4.56.0,<=4.57.3" nvidia-cuda-nvcc==13.0.48

# C2 server deps (t4d): FastAPI frontend + prometheus /metrics + interactive shell
RUN pip install --break-system-packages --index-url ${PIP_INDEX} \
    fastapi "uvicorn[standard]" prometheus_client prompt_toolkit

# synthesize CUDA_HOME layout + dev-symlink names the wheel omits
RUN NV=/usr/local/lib/python3.12/dist-packages/nvidia/cu13 \
 && cp -r /opt/crt "$NV/include/crt" && cp -r /opt/nvvm "$NV/nvvm" \
 && ln -s libcudart.so.13 "$NV/lib/libcudart.so" \
 && ln -s "$NV/lib" "$NV/lib64" \
 && rm -f /usr/local/cuda && ln -s "$NV" /usr/local/cuda \
 && rm -rf /opt/crt /opt/nvvm \
 && mkdir -p "$NV/lib/stubs" \
 && ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$NV/lib/stubs/libcuda.so"

ENV CUDA_HOME=/usr/local/cuda

# JIT cache singleton (C5: large deps shared via host .cache/ mount)
ENV TVM_FFI_CACHE_DIR=/opt/cache/tvm-ffi FLASHINFER_WORKSPACE_BASE=/opt/cache/flashinfer

COPY mini-sglang/ /opt/mini-sglang/
RUN pip install --break-system-packages --no-deps /opt/mini-sglang
