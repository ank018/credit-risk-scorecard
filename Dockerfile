# Serving image for the credit risk scorecard API.
#
# Only the scoring path is installed - no xgboost, shap, optuna or plotting.
# The challenger and the analysis scripts are research artefacts and have no
# business in a production image.
#
# optbinning is required despite serving needing only `transform`, because the
# fitted BinningProcess is unpickled here. Reimplementing the bin edges to avoid
# the dependency would break the guarantee that a served application passes
# through identical logic to a development one, which is the point.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Build tools are needed for some wheels but not at runtime; installed and
# removed in one layer so they do not reach the final image.
COPY requirements-api.txt .
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements-api.txt \
 && apt-get purge -y build-essential \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

# Artefacts and code. Copied after the dependency layer so a code change does
# not invalidate the pip install cache.
COPY models/ ./models/
COPY src/features.py src/config.py ./src/
COPY app/ ./app/

# Serving as root is an unnecessary risk in a container that accepts network
# input.
RUN useradd --create-home --uid 10001 scorecard \
 && chown -R scorecard:scorecard /srv
USER scorecard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

# PORT is read at runtime because most managed platforms assign it rather than
# letting the container choose.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
