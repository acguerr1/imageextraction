FROM python:3.10.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    libicu-dev \
    libleptonica-dev \
    libcairo2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    liblcms2-dev \
    libnss3-dev \
    libopenjp2-7-dev \
    libboost-all-dev \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
COPY ./src /app/src
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python /app/src/install_pkgs.py

# Create necessary cache directories
RUN mkdir -p /root/.cache/torch/checkpoints /root/.cache/layoutparser/model_zoo/PubLayNet /root/.paddleocr/whl

# Download and extract LayoutParser model
# RUN cd /root/.cache/layoutparser/model_zoo/PubLayNet && \
#     wget https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar && \
#     tar xf ppyolov2_r50vd_dcn_365e_publaynet.tar && \
#     rm ppyolov2_r50vd_dcn_365e_publaynet.tar && \
#     test -f inference.pdiparams && \
#     test -f inference.pdiparams.info && \
#     test -f inference.pdmodel

RUN mkdir -p /root/.cache/layoutparser/model_zoo/PubLayNet && \
    cd /root/.cache/layoutparser/model_zoo/PubLayNet && \
    wget https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar && \
    tar -xf ppyolov2_r50vd_dcn_365e_publaynet.tar && \
    mv ppyolov2_r50vd_dcn_365e_publaynet/* . && \
    rm ppyolov2_r50vd_dcn_365e_publaynet.tar ppyolov2_r50vd_dcn_365e_publaynet -rf && \
    ls -la && \
    test -f inference.pdiparams && \
    test -f inference.pdiparams.info && \
    test -f inference.pdmodel

# Set environment variables
ENV LAYOUTPARSER_CACHE_DIR=/root/.cache/layoutparser
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy source files for initialization
COPY ./src /app/src
COPY ./scripts /app/scripts

# Initialize PaddleOCR and LayoutParser models
RUN python -c "import sys; sys.path.append('/app/src'); \
    from init_layoutparser_model import initialize_model; \
    from paddleocr import PaddleOCR; \
    ocr = PaddleOCR(use_angle_cls=True, lang='en'); \
    model = initialize_model()"

# Copy remaining application files
COPY . .

# Start building the final image
FROM python:3.10.12-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libicu-dev \
    libleptonica-dev \
    libcairo2-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libglib2.0-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    liblcms2-dev \
    libnss3-dev \
    libopenjp2-7-dev \
    libboost-all-dev \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy files from builder stage
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.paddleocr /root/.paddleocr
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /app /app

# Set environment variables
ENV PYTHONPATH=/app/src:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV LAYOUTPARSER_CACHE_DIR=/root/.cache/layoutparser

# Configure library paths
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local.conf && ldconfig

ENTRYPOINT ["python", "main.py"]
