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

spark = SparkSession.builder.getOrCreate()

## Load Events

raw_logs = (spark.read
                 .format("com.databricks.spark.csv")
                 .option("delimiter", "\t")
                 .option("inferSchema","true")
                 .load("/tmp/fajar/ich_banking"))


events = raw_logs.select(
                        raw_logs["_c0"].alias("customer_skey")
                        ,raw_logs["_c1"].alias("customer_identifier")
                        ,raw_logs["_c2"].alias("customer_cookie")
                        ,raw_logs["_c3"].alias("customer_online_id")
                        ,raw_logs["_c4"].alias("customer_offline_id")
                        ,raw_logs["_c5"].alias("customer_type")
                        ,raw_logs["_c7"].alias("interaction_session_number")
                        ,to_timestamp("_c8","yyyy-MM-dd HH:mm:ss").alias("interaction_timestamp")
                        ,raw_logs["_c9"].alias("interaction_source")
                        ,raw_logs["_c10"].alias("interaction_type")
                        ,raw_logs["_c13"].alias("product_category")
                  ).orderBy(asc("interaction_timestamp"))

# Filter by exisitng customers
events = events.filter(events["product_category"] != "-1")

## Load customers data

raw_customers = (spark.read
                      .format("com.databricks.spark.csv")
                      .option("inferSchema","true")
                      .option("delimiter","\t")
                      .load("/tmp/fajar/customers"))
                      

customers = raw_customers.select(
                          raw_customers["_c1"].alias("customer_identifier")
                          ,raw_customers["_c2"].alias("firstname")
                          ,raw_customers["_c3"].alias("lastname")
                          ,raw_customers["_c4"].alias("email")
                          ,raw_customers["_c5"].alias("phone")
                          ,to_date(raw_customers["_c6"],"MM/dd/yyyy").alias("birthday")
                          ,raw_customers["_c7"].alias("streetaddress")
                          ,raw_customers["_c8"].alias("city")
                          ,raw_customers["_c9"].alias("state")
                          ,raw_customers["_c10"].alias("zipcode")
                          ,raw_customers["_c13"].alias("num_accounts")
                        )

## Select Customers who Churn

customers_churned = (spark.read
                    .format("com.databricks.spark.csv")
                    .option("inferSchema","true")
                    .option("delimiter","\t")
                    .option("header","true")
                    .load("/tmp/fajar/customers_churned_new"))

customers_churned.show()

## Tagging Target Column
customers_tagged = customers.join(customers_churned, customers["customer_identifier"] == customers_churned["cust_id"],"left_outer")
customers_tagged = customers_tagged.withColumn("churned",when(customers_tagged["cust_id"].isNull(),"N").otherwise("Y")).drop("cust_id")

## Events Mapping
events_mapped = event_mapping(events,"interaction_timestamp","customer_identifier","interaction_type")

val = [
("A","ACCOUNT_BOOKED_OFFLINE")
,("B","ACCOUNT_BOOKED_ONLINE")
,("C","ACCOUNT_CLOSED")
,("D","ADD_DIRECT_DEPOSIT")
,("E","BALANCE_TRANSFER")
,("F","BROWSE")
,("G","CALL_COMPLAINT")
,("H","CLICK")
,("I","COMPARE")
,("J","COMPLETE_APPLICATION")
,("K","ENROLL_AUTO_SAVINGS")
,("L","FEE_REVERSAL")
,("M","LINK_EXTERNAL_ACCOUNT")
,("N","LOAN_CALC")
,("O","MORTGAGE_CALC")
,("P","OLB")
,("Q","REFERRAL")
,("R","STARTS_APPLICATION")]

events_coded = event_coding(events_mapped,val)

## Events Sequencing

events_sequenced = event_sequencing(events_coded)

## Churned Events Path

events_churned = events_sequenced.filter(substring("event_path",-1,1) == "C")

events_churned_last_path = events_churned.select(events_churned["id"].alias("customer_identifier"),substring("event_path",-5,5).alias("event_path"))

cust_churned_path = events_churned_last_path.join(customers_tagged, "customer_identifier")

cust_churned_path.show()

## Not Churned Events Path

events_notchurned = events_sequenced.filter(substring(events_sequenced["event_path"],-1,1) != "C")

events_notchurned_last_path = events_notchurned.select(events_notchurned["id"].alias("customer_identifier"),substring("event_path",-5,5).alias("event_path"))

cust_notchurned_path = events_notchurned_last_path.join(customers_tagged, "customer_identifier")

cust_notchurned_path.show()

## Path to churn

path_to_churn = events_churned_last_path.groupBy("event_path").count().orderBy(desc("count"))
path_to_churn.show()

## Customer Path

cust_path = cust_churned_path.unionAll(cust_notchurned_path)
cust_path.show()

## Prepare Training Data Set

sdf_train = cust_path.select("event_path"
                            , cust_path["churned"].alias("target"))
                            
sdf_train.show()

