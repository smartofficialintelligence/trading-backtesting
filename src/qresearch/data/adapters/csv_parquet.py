"""Local CSV and Parquet bar adapter.

Provider-agnostic on purpose: it is the path used by synthetic fixtures, by exported
vendor files, and by anything else that lands on disk before a licensed provider is
chosen (DEVELOPMENT_PLAN.md sec. 10). A real provider adapter reuses the helpers in
:mod:`qresearch.data.adapters.base` and differs only in how it fetches bytes and which
:class:`~qresearch.data.manifests.TimestampLabel` it declares.

Source timestamps must carry a UTC offset. A naive column is rejected rather than assumed
to be UTC, because "assumed UTC" is how an exchange-local export becomes a dataset that
is wrong by a whole number of hours without anything looking broken.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import polars as pl

from qresearch.data.adapters.base import (
    IngestRequest,
    NormalizedBatch,
    apply_duplicate_policy,
    derive_bar_bounds,
    resolve_symbols,
)
from qresearch.data.manifests import SourceRef
from qresearch.ids import file_hash_full
from qresearch.time import now_utc

_REQUIRED_NUMERIC = ("open", "high", "low", "close", "volume")


class LocalFileBarAdapter:
    """Reads bars from one local ``.csv``, ``.csv.gz``, or ``.parquet`` file."""

    def read(self, request: IngestRequest) -> NormalizedBatch:
        path = Path(request.uri.removeprefix("file://"))
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")

        frame = self._read_raw(path)
        mapping = request.mapping
        self._require_columns(frame, request)

        frame = frame.with_columns(
            self._as_utc(frame, mapping.timestamp).alias("_source_ts"),
        )
        frame = derive_bar_bounds(
            frame,
            bar_size=request.bar_size,
            label=request.policy.timestamp_label,
            timestamp_column="_source_ts",
        )

        # Resolve symbols at bar_start: the instrument identity of a bar is the identity
        # that was in force when the bar happened.
        frame, dropped = resolve_symbols(
            frame, request, symbol_column=mapping.symbol, at_column="bar_start"
        )

        frame = frame.with_columns(
            pl.lit(request.bar_size).alias("bar_size"),
            self._availability(frame, request).alias("available_at"),
            *[
                pl.col(getattr(mapping, name)).cast(pl.Float64).alias(name)
                for name in _REQUIRED_NUMERIC
            ],
            self._optional(frame, mapping.vwap, pl.Float64).alias("vwap"),
            self._optional(frame, mapping.trade_count, pl.Int64).alias("trade_count"),
            pl.lit(request.source_name).alias("source"),
            self._optional(frame, mapping.source_key, pl.String).alias("source_key"),
            self._optional(
                frame, mapping.revision, pl.Int32, default=request.source_revision
            ).alias("revision"),
            pl.lit(now_utc()).cast(pl.Datetime("us", "UTC")).alias("ingested_at"),
        )

        frame = apply_duplicate_policy(frame, request.policy.duplicates)
        frame = frame.select(
            "instrument_id",
            "bar_size",
            "bar_start",
            "bar_end",
            "available_at",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
            "source",
            "source_key",
            "revision",
            "ingested_at",
        ).sort(["instrument_id", "bar_start", "revision"])

        return NormalizedBatch(
            frame=frame,
            source=SourceRef(
                provider=request.source_name,
                feed=request.feed,
                uri=path.resolve().as_uri(),
                content_sha256=file_hash_full(path),
                source_revision=request.source_revision,
            ),
            dropped_unmapped_symbols=dropped,
        )

    @staticmethod
    def _read_raw(path: Path) -> pl.DataFrame:
        if path.suffix == ".parquet":
            return pl.read_parquet(path)
        if path.suffixes[-2:] == [".csv", ".gz"] or path.suffix == ".csv":
            return pl.read_csv(path, try_parse_dates=False)
        raise ValueError(
            f"unsupported source extension {path.suffix!r}; expected .csv, .csv.gz, or .parquet"
        )

    @staticmethod
    def _require_columns(frame: pl.DataFrame, request: IngestRequest) -> None:
        mapping = request.mapping
        required = [
            mapping.timestamp,
            mapping.symbol,
            *(getattr(mapping, n) for n in _REQUIRED_NUMERIC),
        ]
        optional = [
            mapping.vwap,
            mapping.trade_count,
            mapping.source_key,
            mapping.revision,
            mapping.available_at,
        ]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError(
                f"source file is missing required columns {missing}; present columns are "
                f"{sorted(frame.columns)}"
            )
        absent_optional = [c for c in optional if c is not None and c not in frame.columns]
        if absent_optional:
            raise ValueError(
                f"column mapping names {absent_optional}, which the source does not contain; "
                "leave the mapping entry as None if the source has no such column"
            )

    @staticmethod
    def _as_utc(frame: pl.DataFrame, column: str) -> pl.Expr:
        """Parse a timestamp column, requiring an explicit UTC offset."""
        dtype = frame.schema[column]
        if isinstance(dtype, pl.Datetime):
            if dtype.time_zone is None:
                raise ValueError(
                    f"source timestamp column {column!r} is timezone-naive; attach an "
                    "offset at export time. Assuming UTC here would silently shift the "
                    "entire dataset if the export was exchange-local."
                )
            return pl.col(column).dt.convert_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
        if dtype == pl.String:
            parsed = pl.col(column).str.to_datetime(time_unit="us", time_zone="UTC")
            sample = frame.get_column(column).drop_nulls().head(1).to_list()
            if sample and not _has_offset(str(sample[0])):
                raise ValueError(
                    f"source timestamp column {column!r} has no UTC offset (example: "
                    f"{sample[0]!r}); an offset or trailing 'Z' is required"
                )
            return parsed
        raise ValueError(
            f"source timestamp column {column!r} has type {dtype}; expected a datetime or "
            "an ISO-8601 string with an offset"
        )

    @staticmethod
    def _availability(frame: pl.DataFrame, request: IngestRequest) -> pl.Expr:
        """Provider-published availability when present, otherwise bar_end + latency."""
        column = request.mapping.available_at
        if column is not None:
            return LocalFileBarAdapter._as_utc(frame, column)
        latency = request.policy.publication_latency // _dt.timedelta(microseconds=1)
        return pl.col("bar_end") + pl.duration(microseconds=latency)

    @staticmethod
    def _optional(
        frame: pl.DataFrame,
        column: str | None,
        dtype: pl.DataType | type[pl.DataType],
        *,
        default: object = None,
    ) -> pl.Expr:
        if column is None:
            return pl.lit(default, dtype=dtype)
        return pl.col(column).cast(dtype)


def _has_offset(text: str) -> bool:
    tail = text[10:]  # skip the date portion, where '-' is a separator
    return text.endswith(("Z", "z")) or "+" in tail or "-" in tail
