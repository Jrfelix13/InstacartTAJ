
import sys
sys.path.insert(0, '/var/www/html/flaskapp')
sys.path.insert(1, '/usr/local/spark-2.0.2-bin-hadoop2.7/python')
sys.path.insert(2, '/usr/local/spark-2.0.2-bin-hadoop2.7/python/lib/py4j-0.10.3-src.zip')
from pyspark import SparkContext
sc = SparkContext('local')

from flask import Flask, render_template, redirect, jsonify, request

import numpy as np
import pandas as pd
import requests
import json

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, inspect, func
from sqlalchemy.orm import aliased

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowthModel
#################################################
# Database Setup
#################################################
POSTGRES = {
    'user': 'root',
    'pw': 'FinalProject_2020',
    'db': 'final_project_db',
    'host': 'finalprojectdb.cyj4dabyex0o.us-east-2.rds.amazonaws.com',
    'port': '5432',
}
engine = create_engine('postgresql://%(user)s:\
%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES)

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)

# Save reference to the table
Aisle = Base.classes.aisle_tbl
Product = Base.classes.product_tbl
Department = Base.classes.department_tbl
Orders = Base.classes.orders_tbl
OrdersP = Base.classes.orders_product_prior
OrdersT = Base.classes.orders_product_train
ProductL= Base.classes.product_list
DepartmentL= Base.classes.dep_tbl
AisleL= Base.classes.ais_tbl

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

spark = SparkSession.builder.appName("FinalProject").getOrCreate()

#################################################
# Flask Routes
#################################################

@app.route("/")
def index():

    return render_template("index.html")

@app.route("/story/")
def story():

    return render_template("story.html")

@app.route("/about_us/")
def about_us():

    return render_template("about_us.html")

@app.route("/shop/")
def shop():

    return render_template("shop.html")

@app.route("/shop/<item1>/<item2>/<item3>")
def neworder(item1,item2=None,item3=None):
    
    session = Session(engine)

    order_items = []

    item_1 = session.query(Product.product_id).filter(Product.product_name == item1).all()
    order_items.append(item_1)
    if item2 != None:
        item_2 = session.query(Product.product_id).filter(Product.product_name == item2).all()
        order_items.append(item_2)
    if item3 != None:
        item_3 = session.query(Product.product_id).filter(Product.product_name == item3).all()
        order_items.append(item_3)
    
    order_item=[]
    for i in range(0,len(order_items)):
        order_item.append(order_items[i][0][0])
    order_item = [("neworder",order_item)]
    
    order_items_df=pd.DataFrame(order_item,columns=['OrderId','ProductID'])
    order_items = spark.createDataFrame(order_items_df)

    model=FPGrowthModel.load("model/instacart_model3/")

    data_df = model.transform(order_items).select("*").toPandas()

    predictions = data_df["prediction"][0]

    if len(predictions)>=1:
        product_final = session.query(Product.product_name).filter(Product.product_id == predictions[0]).all()
        aisle_final = session.query(Aisle.aisle).filter(Aisle.aisle_id == Product.aisle_id).filter(Product.product_id == predictions[0]).all()
        department_final = session.query(Department.department).filter(Department.department_id == Product.department_id).filter(Product.product_id == predictions[0]).all()
        result={}
        result["product"]=str(product_final[0][0])
        result["aisle"]=str(aisle_final[0][0])
        result["department"]=str(department_final[0][0])
    else:
        result={}
        result["product"]="There is not suggestion"
        result["aisle"]="NA"
        result["department"]="NA"
    
    session.close()
    return jsonify(result)


@app.route("/graph/aisle")
def aisle():
    session = Session(engine)
    results = session.query(AisleL.aisle,AisleL.count).all()
    aisle = []
    order_count = []
    for result in results:
        aisle.append(result[0])
        order_count.append(result[1])

    pie_aisle_df= pd.DataFrame({"aisle":aisle,"Total_aisle":order_count})
    pie_aisle_df.sort_values(by=["Total_aisle"],inplace=True,ascending=False)
    pie_aisle_df1 = pie_aisle_df.iloc[0:10,:]
    aisle_other= pie_aisle_df.Total_aisle.iloc[10:].sum()
    df_append = pd.DataFrame({"aisle":["Other"],"Total_aisle":[aisle_other]})
    pie_aisle_df1 = pie_aisle_df1.append(df_append,ignore_index=True)
    session.close()
    total = []
    for i in range(0,len(pie_aisle_df1)):
        dicto={}
        dicto["aisle"]=pie_aisle_df1.aisle.iloc[i]
        dicto["Total_aisle"]=int(pie_aisle_df1.Total_aisle.iloc[i])
        total.append(dicto)

    return jsonify(total)

@app.route("/graph/department")
def department():
    session = Session(engine)
    results = session.query(DepartmentL.department,DepartmentL.count).all()
    department = []
    order_count = []
    for result in results:
        department.append(result[0])
        order_count.append(result[1])

    pie_department_df= pd.DataFrame({"department":department,"Total_department":order_count})
    pie_department_df.sort_values(by=["Total_department"],inplace=True,ascending=False)
    pie_department_df1 = pie_department_df.iloc[0:10,:]
    department_other= pie_department_df.Total_department.iloc[10:].sum()
    df_append = pd.DataFrame({"department":["Other"],"Total_department":[department_other]})
    pie_department_df1 = pie_department_df1.append(df_append,ignore_index=True)


    total = []
    for i in range(0,len(pie_department_df1)):
        dicto={}
        dicto["department"]=pie_department_df1.department.iloc[i]
        dicto["Total_department"]=int(pie_department_df1.Total_department.iloc[i])
        total.append(dicto)

    session.close()
    return jsonify(total)

@app.route("/graph/product")
def product():
    session = Session(engine)
    results = session.query(ProductL.product_name,ProductL.count).all()
    product = []
    order_count = []
    for result in results:
        product.append(result[0])
        order_count.append(result[1])

    product_df= pd.DataFrame({"product_name":product,"Total_product":order_count})
    product_df.sort_values(by=["Total_product"],inplace=True,ascending=False)
    product_df1 = product_df.iloc[0:10,:]

    total = []
    for i in range(0,len(product_df1)):
        dicto={}
        dicto["product_name"]=product_df1.product_name.iloc[i]
        dicto["Total_product"]=int(product_df1.Total_product.iloc[i])
        total.append(dicto)
    session.close()
    return jsonify(total)

@app.route("/product/product_list")
def products_list():
    session = Session(engine)
    results = session.query(ProductL.product_name,ProductL.count).all()
    product = []
    order_count = []
    for result in results:
        product.append(result[0])
        order_count.append(result[1])

    product_df= pd.DataFrame({"product_name":product,"Total_product":order_count})
    product_df.sort_values(by=["Total_product"],inplace=True,ascending=False)
    product_df1 = product_df.iloc[0:25,:]

    total = []
    for i in range(0,len(product_df1)):
        dicto={}
        dicto["product_name"]=product_df1.product_name.iloc[i]
        total.append(dicto)
    session.close()
    return jsonify(total)

@app.route("/graph/heat_map")
def heat():
    session = Session(engine)
    query = f"select order_dow,order_hour_of_day,count(order_id) from orders_tbl\
        group by order_dow,order_hour_of_day"
    results = engine.execute(query).fetchall()
    total =[]
    for result in results:
        dicto = {}
        if result[0]==0:
            day = "Monday"
        elif result[0]==1:
            day = "Tuesday"
        elif result[0]==2:
            day = "Wednesday"
        elif result[0]==3:
            day = "Thursday"
        elif result[0]==4:
            day = "Friday"
        elif result[0]==5:
            day = "Saturday"
        elif result[0]==6:
            day = "Sunday"
        dicto["order_dow"]=day
        dicto["order_hour"]=result[1]
        dicto["order_count"]=result[2]
        total.append(dicto)

    return jsonify(total)


if __name__ == "__main__":
    app.run(debug=True,port=8080)
