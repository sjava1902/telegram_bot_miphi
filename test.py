# Import the required modules and classes
import sparknlp
import pyspark
from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    DateMatcher,
    MultiDateMatcher
)
import pyspark.sql.functions as F

spark = sparknlp.start()
spark_context = pyspark.SparkContext.getOrCreate()
# Step 1: Transforms raw texts to `document` annotation
document_assembler = (
    DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")
)
# Step 2: Extracts one date information from text
date = (
    DateMatcher()
    .setInputCols("document") \
    .setOutputCol("date") \
    .setOutputFormat("yyyy/MM/dd")
)
# Step 3: Extracts multiple date information from text
multiDate = (
    MultiDateMatcher()
    .setInputCols("document") \
    .setOutputCol("multi_date") \
    .setOutputFormat("MM/dd/yy")
)
nlpPipeline = Pipeline(stages=[document_assembler, date, multiDate])
text_list = ["See you on next monday.",  
             "She was born on 02/03/1966.", 
             "The project started yesterday and will finish next year.", 
             "She will graduate by July 2023.", 
             "She will visit doctor tomorrow and next month again."]

# Create a dataframe
spark_df = spark.createDataFrame(text_list, StringType()).toDF("text")
# Fit the dataframe and get predictions
result = pipeline.fit(spark_df).transform(spark_df)
# Display the extracted date information in a dataframe
result.selectExpr("text","date.result as date", "multi_date.result as multi_date").show(truncate=False)