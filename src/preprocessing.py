from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql.functions import col as spark_col
from pyspark.sql.functions import concat_ws, length, regexp_replace, trim, when


def clean_text_col(df: DataFrame, col: str = "text") -> DataFrame:
    cleaned = df.filter(spark_col(col).isNotNull())
    cleaned = cleaned.withColumn(col, trim(regexp_replace(spark_col(col), r"\s+", " ")))
    cleaned = cleaned.filter(length(spark_col(col)) > 0)

    if "title" in cleaned.columns:
        cleaned = cleaned.withColumn("title", when(spark_col("title").isNull(), "").otherwise(spark_col("title")))
        cleaned = cleaned.withColumn("title", trim(regexp_replace(spark_col("title"), r"\s+", " ")))
        cleaned = cleaned.withColumn("full_text", concat_ws(" ", spark_col("title"), spark_col(col)))
    else:
        cleaned = cleaned.withColumnRenamed(col, "full_text")

    return cleaned
