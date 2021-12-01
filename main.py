from flask import Flask, request, render_template 
import json
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
# Flask constructor
recommendations=[]
lis = []
json_obj_recommend = ""
vall = 0
app = Flask(__name__)   
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def gfg():
   if request.method == "POST":
      vall = int(request.form.get("vall"))
      data=pd.concat(map(pd.read_csv,['reviews-1-115.csv','reviews-11265-13495.csv','reviews-4575-6805.csv','reviews-6805-9035.csv']))
      data['voted_up']=data['voted_up'].astype(int)
      a=data['appid'].unique()
      b=data['steamid'].unique()
      steam_id=pd.DataFrame(b)
      data_first=data.groupby(['appid'])['votes_up'].sum()
      data_first=data_first.reset_index()
      data_second=data.groupby(['appid'])['num_reviews'].sum()
      data_second=data_second.reset_index()
      data_third=data.groupby(['appid'])['playtime_at_review'].sum()
      data_third=data_third.reset_index()
      data_four=data.groupby(['appid'])['voted_up'].sum()
      data_four=data_four.reset_index()
      data_five=data.groupby(['appid'])['playtime_forever'].sum()
      data_five=data_five.reset_index()
      data_six=data.groupby(['steamid'])['num_games_owned'].sum()
      data_six=data_six.reset_index()
      final_data=pd.DataFrame(data_first)
      final_data['num_reviews']=data_second['num_reviews']
      final_data['playtime_at_review']=data_third['playtime_at_review']
      final_data['voted_up']=data_four['voted_up']
      final_data['playtime_forever']=data_five['playtime_forever']
      final_data_modify=final_data.iloc[:,1:]
      cosine_nn=NearestNeighbors(algorithm='brute',metric='cosine')
      game_id_fit=cosine_nn.fit(final_data_modify)
      distances, indices = cosine_nn.kneighbors(final_data_modify.iloc[vall,:].values.reshape(1,-1),n_neighbors=7)
      for i in range(0,len(distances.flatten())):
         lis.append(final_data_modify.index[indices.flatten()[i]])
      
      for i in lis:
         recommendations.append(final_data.iloc[i,:].to_dict())
      recommendations
      json_obj=json.dumps(lis)
      json_obj_recommend=json.dumps(recommendations)
      print(json_obj,json_obj_recommend)
      with open("sample.json", "w") as outfile:
         outfile.write(json_obj_recommend)
   return render_template("validation.html",recommendations= recommendations,lis=lis)

    
app.run()



  
