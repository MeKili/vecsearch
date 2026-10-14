"""FastAPI service: create named indexes, query them, inspect and delete them."""

from __future__ import annotations

import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vecsearch import __version__
from vecsearch.core import Metric
from vecsearch.index.base import Index
from vecsearch.index.registry import ALGORITHMS, create_index

MAX_VECTORS = int(os.environ.get("VECSEARCH_MAX_VECTORS", "100000"))

app = FastAPI(title="vecsearch", version=__version__)
_indexes: dict[str, Index] = {}


class CreateRequest(BaseModel):
    algorithm: str = "flat"
    metric: Metric = Metric.L2
    params: dict[str, int] = Field(default_factory=dict)
    ids: list[str] = Field(min_length=1)
    vectors: list[list[float]] = Field(min_length=1)


class SearchRequest(BaseModel):
    vector: list[float] = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=1000)


class HitOut(BaseModel):
    id: str
    distance: float


class SearchResponse(BaseModel):
    hits: list[HitOut]
    scanned_fraction: float


class IndexInfo(BaseModel):
    name: str
    algorithm: str
    metric: Metric
    size: int
    dim: int
    params: dict[str, int]


def _info(name: str, index: Index) -> IndexInfo:
    return IndexInfo(
        name=name,
        algorithm=index.algorithm,
        metric=index.metric,
        size=len(index),
        dim=index.dim,
        params=index.params,
    )


def _get(name: str) -> Index:
    index = _indexes.get(name)
    if index is None:
        raise HTTPException(status_code=404, detail=f"no index named {name!r}")
    return index


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/algorithms")
def algorithms() -> dict[str, list[str]]:
    return {"algorithms": list(ALGORITHMS)}


@app.get("/indexes")
def list_indexes() -> list[IndexInfo]:
    return [_info(name, idx) for name, idx in sorted(_indexes.items())]


@app.put("/indexes/{name}", response_model=IndexInfo)
def create(name: str, req: CreateRequest) -> IndexInfo:
    if len(req.ids) > MAX_VECTORS:
        raise HTTPException(status_code=413, detail=f"at most {MAX_VECTORS} vectors are accepted")
    try:
        index = create_index(req.algorithm, req.metric, **req.params)
        index.build(req.ids, np.asarray(req.vectors, dtype=np.float64))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _indexes[name] = index
    return _info(name, index)


@app.post("/indexes/{name}/search", response_model=SearchResponse)
def search(name: str, req: SearchRequest) -> SearchResponse:
    index = _get(name)
    try:
        hits = index.search(np.asarray(req.vector, dtype=np.float64), req.k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return SearchResponse(
        hits=[HitOut(id=h.id, distance=h.distance) for h in hits],
        scanned_fraction=index.scanned_fraction(),
    )


@app.delete("/indexes/{name}", status_code=204)
def delete(name: str) -> None:
    _get(name)
    del _indexes[name]
