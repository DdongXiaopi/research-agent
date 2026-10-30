ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG APT_MIRROR=http://deb.debian.org
RUN sed -i "s|http://deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && printf 'Acquire::By-Hash "false";\nAcquire::ForceIPv4 "true";\nAcquire::http::Timeout "20";\nAcquire::Retries "1";\n' > /etc/apt/apt.conf.d/99-rsi-build \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-matplotlib texlive-latex-base texlive-latex-recommended fonts-dejavu-core fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/rsi-worker \
    && dpkg-query -W > /opt/rsi-worker/packages.txt
COPY worker.py worker_contract.py /opt/rsi-worker/
ENV HOME=/work MPLCONFIGDIR=/work/mplconfig MPLBACKEND=Agg \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    openin_any=p openout_any=p TEXMFOUTPUT=/work
USER 65532:65532
WORKDIR /work
ENTRYPOINT ["/usr/bin/python3", "/opt/rsi-worker/worker.py"]
