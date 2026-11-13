"""PySpark/SQL rollup of raw geo-tagged logs into a cell x time-block exposure table."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession

from .config import DATA_DIR, SimConfig

log = logging.getLogger(__name__)

# The core rollup is deliberately real SQL executed by Spark SQL against the
# temp views `events` and `mileage`, which already carry cell_id / time_block.
_ROLLUP_BODY = """
WITH mileage_agg AS (
    SELECT
        cell_id,
        time_block,
        SUM(miles)  AS exposure_miles,
        COUNT(*)    AS n_segments
    FROM mileage
    GROUP BY cell_id, time_block
),
event_agg AS (
    SELECT
        cell_id,
        time_block,
        COUNT(*) AS event_count,
        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) AS critical_count
    FROM events
    GROUP BY cell_id, time_block
)
SELECT
    COALESCE(m.cell_id, e.cell_id)           AS cell_id,
    COALESCE(m.time_block, e.time_block)     AS time_block,
    m.exposure_miles                         AS exposure_miles,
    COALESCE(e.event_count, 0)               AS event_count,
    COALESCE(e.critical_count, 0)            AS critical_count,
    COALESCE(m.n_segments, 0)                AS n_segments
FROM mileage_agg m
FULL OUTER JOIN event_agg e
    ON m.cell_id = e.cell_id AND m.time_block = e.time_block
"""

ROLLUP_SQL = _ROLLUP_BODY + """
WHERE m.exposure_miles IS NOT NULL AND m.exposure_miles >= 1.0
"""

# Companion query: exactly the rows the filter above discards, for auditing.
DROPPED_SQL = _ROLLUP_BODY + """
WHERE m.exposure_miles IS NULL OR m.exposure_miles < 1.0
"""


def get_spark(app_name: str = "bstr-aggregate") -> SparkSession:
    """Local[*] SparkSession tuned for a laptop, with quiet logs."""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _binning_sql(cfg: SimConfig, table: str) -> str:
    """Arithmetic (x, y) -> cell mapping and time derivation, done inside Spark SQL."""
    return f"""
    SELECT
        *,
        CAST(FLOOR(y / {cfg.CELL_KM}) AS INT) AS row_idx,
        CAST(FLOOR(x / {cfg.CELL_KM}) AS INT) AS col_idx,
        CAST(FLOOR(y / {cfg.CELL_KM}) * {cfg.N_COLS}
             + FLOOR(x / {cfg.CELL_KM}) AS INT) AS cell_id,
        CAST(HOUR(TO_TIMESTAMP(timestamp)) AS INT) AS time_block,
        CAST(DATEDIFF(TO_DATE(timestamp), DATE '2026-01-01') AS INT) AS day_index
    FROM raw_{table}
    """


def prepare_views(spark: SparkSession, data_dir: Path, cfg: SimConfig) -> None:
    """Register `raw_*`, then binned `events` / `mileage` temp views."""
    spark.read.parquet(str(Path(data_dir) / "raw_events.parquet")).createOrReplaceTempView(
        "raw_events"
    )
    spark.read.parquet(str(Path(data_dir) / "raw_mileage.parquet")).createOrReplaceTempView(
        "raw_mileage"
    )
    spark.sql(_binning_sql(cfg, "events")).createOrReplaceTempView("events")
    spark.sql(_binning_sql(cfg, "mileage")).createOrReplaceTempView("mileage")


def validate_exposure_table(
    df: pd.DataFrame,
    raw_event_count: int | None = None,
    dropped_events: int | None = None,
) -> dict:
    """Assert the structural invariants of the rollup and audit event conservation."""
    assert not df.isna().any().any(), "exposure table contains nulls"
    assert (df["exposure_miles"] > 0).all(), "non-positive exposure"
    assert (df["event_count"] >= df["critical_count"]).all(), "critical_count exceeds event_count"
    assert (df["critical_count"] >= 0).all(), "negative critical_count"
    assert (df["event_count"] >= 0).all(), "negative event_count"
    assert not df.duplicated(["cell_id", "time_block"]).any(), "duplicate cell/time rows"

    audit = {"rows": int(len(df)), "events": int(df["event_count"].sum())}
    if raw_event_count is not None:
        dropped = int(raw_event_count - df["event_count"].sum())
        assert dropped >= 0, "exposure table contains more events than the raw logs"
        if dropped_events is not None:
            assert dropped == dropped_events, (
                f"event conservation failed: {dropped} dropped vs {dropped_events} documented"
            )
        audit["dropped_events"] = dropped
    return audit


def run_aggregation(
    cfg: SimConfig,
    data_dir: Path = DATA_DIR,
    out_dir: Path = DATA_DIR,
    verbose: bool = True,
) -> dict:
    """Run the whole raw-log -> exposure-table pipeline and write both parquet tables."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = get_spark("bstr-aggregate")
    try:
        prepare_views(spark, data_dir, cfg)
        raw_event_count = spark.table("events").count()
        raw_mileage_count = spark.table("mileage").count()

        rollup = spark.sql(ROLLUP_SQL)
        dropped_row = spark.sql(
            "SELECT COUNT(*) AS pairs, "
            "COALESCE(SUM(event_count), 0) AS events, "
            "COALESCE(SUM(exposure_miles), 0.0) AS miles FROM (" + DROPPED_SQL + ") d"
        ).collect()[0]
        dropped_pairs = int(dropped_row["pairs"])
        dropped_events = int(dropped_row["events"])
        dropped_miles = float(dropped_row["miles"])

        truth = (
            spark.read.parquet(str(data_dir / "ground_truth.parquet"))
            .select("cell_id", "intersection_density", "pedestrian_density", "region_id")
            .dropDuplicates(["cell_id"])
        )
        rollup = rollup.join(truth, on="cell_id", how="left")
        rollup.createOrReplaceTempView("exposure")

        exposure_pd = (
            rollup.select(
                "cell_id",
                "time_block",
                "exposure_miles",
                "event_count",
                "critical_count",
                "n_segments",
                "intersection_density",
                "pedestrian_density",
                "region_id",
            )
            .orderBy("cell_id", "time_block")
            .toPandas()
        )
        exposure_pd["event_count"] = exposure_pd["event_count"].astype(int)
        exposure_pd["critical_count"] = exposure_pd["critical_count"].astype(int)
        exposure_pd["n_segments"] = exposure_pd["n_segments"].astype(int)
        exposure_pd["region_id"] = exposure_pd["region_id"].astype(int)

        region_pd = spark.sql(
            """
            SELECT
                region_id,
                time_block,
                SUM(exposure_miles) AS exposure_miles,
                SUM(event_count)    AS event_count,
                SUM(critical_count) AS critical_count,
                COUNT(*)            AS n_cells
            FROM exposure
            GROUP BY region_id, time_block
            ORDER BY region_id, time_block
            """
        ).toPandas()
        region_pd["exposure_miles"] = region_pd["exposure_miles"].astype(float)
        region_pd["event_count"] = region_pd["event_count"].astype(int)
        region_pd["critical_count"] = region_pd["critical_count"].astype(int)
        region_pd["n_cells"] = region_pd["n_cells"].astype(int)
        region_pd["rate_per_1k_miles"] = (
            1000.0 * region_pd["event_count"] / region_pd["exposure_miles"]
        )
    finally:
        spark.stop()

    validate_exposure_table(
        exposure_pd, raw_event_count=raw_event_count, dropped_events=dropped_events
    )
    exposure_pd.to_parquet(out_dir / "exposure_table.parquet", index=False)
    region_pd.to_parquet(out_dir / "region_table.parquet", index=False)

    total_exposure = float(exposure_pd["exposure_miles"].sum())
    total_events = int(exposure_pd["event_count"].sum())
    by_cell = (
        exposure_pd.groupby("cell_id", as_index=False)
        .agg(
            event_count=("event_count", "sum"),
            exposure_miles=("exposure_miles", "sum"),
            region_id=("region_id", "first"),
        )
        .assign(rate_per_1k_miles=lambda d: 1000.0 * d["event_count"] / d["exposure_miles"])
        .sort_values("rate_per_1k_miles", ascending=False)
        .head(5)
    )

    stats = {
        "raw_event_rows": int(raw_event_count),
        "raw_mileage_rows": int(raw_mileage_count),
        "output_rows": int(len(exposure_pd)),
        "dropped_pairs": dropped_pairs,
        "dropped_events": dropped_events,
        "dropped_miles": dropped_miles,
        "total_exposure_miles": total_exposure,
        "total_events": total_events,
        "overall_rate_per_1k_miles": 1000.0 * total_events / total_exposure,
        "top5_cells": by_cell,
    }
    if verbose:
        log.info(
            "rolled up %d event rows and %d mileage rows into %d cell x block rows; "
            "dropped %d pairs (exposure < 1.0 mile) carrying %d events / %.2f miles",
            raw_event_count,
            raw_mileage_count,
            len(exposure_pd),
            dropped_pairs,
            dropped_events,
            dropped_miles,
        )
    return {"exposure": exposure_pd, "region": region_pd, "stats": stats}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = SimConfig()
    result = run_aggregation(cfg)
    stats = result["stats"]
    print(f"input event rows          : {stats['raw_event_rows']:,}")
    print(f"input mileage rows        : {stats['raw_mileage_rows']:,}")
    print(f"output cell x block rows  : {stats['output_rows']:,}")
    print(f"dropped (cell, block) rows: {stats['dropped_pairs']:,}")
    print(f"total exposure miles      : {stats['total_exposure_miles']:,.1f}")
    print(f"observed events / 1k miles: {stats['overall_rate_per_1k_miles']:.4f}")
    print("top-5 highest-rate cells  :")
    print(stats["top5_cells"].to_string(index=False))
    print("validation                : passed")
