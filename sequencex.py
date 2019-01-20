## Import Packages
# Spark DF profiling and preparation
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import StringType

## Event Mapping
# Function to select Timestamp, ID, and Event Column

def event_mapping(df, ts, id, event):
    events = df.select(
                df[ts].alias("ts")
                ,df[id].alias("id")
                ,df[event].alias("event")
                )
    return events

## Event Coding
# Function to map the events into code
    
def event_coding(df, codes):
    columns = ["symbol","value"]
    lookup = spark.createDataFrame(codes,columns)
    
    event_coded = df.join(lookup,lookup["value"]==df["event"], "left_outer")
    event_coded = event_coded.drop("event","value")
    event_coded = event_coded.withColumnRenamed("symbol","event")
    
    return event_coded

## Event Sequencing  
# Function to sessionize and sequence the event in a string
    
def event_sequencing(df):
    window = Window.partitionBy(df["id"]).orderBy(df["ts"])
    event_transposed = df.withColumn("event", collect_list("event").over(window))
    event_sequence = event_transposed.groupBy(event_transposed["id"]).agg(max("event").alias("event_path"))
    event_sequence = event_sequence.withColumn("event_path",concat_ws("", event_sequence["event_path"]))
    return event_sequence

## String Compaction
# Function to compact the event sequence by removing repeating events in a row

def string_compaction(event_string):
    prev = ""
    compacted = ""
    
    for x in event_string:
        if(x != prev):
            compacted += x
            prev = x
    
    return compacted
  
# Register as UDF
    
str_compact = udf(lambda z: string_compaction(z), StringType())
