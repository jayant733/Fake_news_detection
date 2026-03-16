def load_data(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True)
