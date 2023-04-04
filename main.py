# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 21:03:51 2023

@author: Satoru
"""

from flask import Flask, render_template, redirect, request, flash, current_app
import numpy as np
import cv2
import os
import glob
import copy
import math
import sys
from werkzeug.utils import secure_filename

#DIR_DATA = "C:/Users/Satoru/Desktop/drop_test/matome/flt_178/cut/"
#file_list = glob.glob(DIR_DATA + "*.jpg")


UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

WIDTH = 1024 #1110
HEIGHT = 1024 #1110
     
W_CM = 4.3
H_CM = 4.3
    

app = Flask(__name__, static_folder='./templates/images')

#IMG_FOLDER = os.path.join('static', 'IMG')
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        upload_files = request.files.getlist('file')    
                
        if upload_files and allowed_file(file.filename):
            #TODO:フォルダ内の画像を削除
            file_list_past = glob.glob(UPLOAD_FOLDER + "*.jpg")
            for p in file_list_past:
                os.remove(p) 

            for file in upload_files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                       
            #file_list = os.listdir('static/IMG')
            #file_list = ['IMG/' + i for i in file_list]
            #file_list = os.listdir(UPLOAD_FOLDER)
            #file_list = [UPLOAD_FOLDER + i for i in file_list]
            file_list = glob.glob(UPLOAD_FOLDER + "*.jpg")

            area_arr = np.zeros(len(file_list), dtype=np.float64)
            area_ratio_arr = np.zeros(len(file_list), dtype=np.float64)
            num_obj_arr = np.zeros(len(file_list), dtype=np.int64)
            diameter_arr = np.zeros(len(file_list), dtype=np.int64)
            num_obj_dens_arr = np.zeros(len(file_list), dtype=np.int64)
            
            list_data_name = []
            list_trial = []
            list_box = []
            list_face = []
                        
            for f_i, f in enumerate(file_list):
                data_name = f[-12:-4]
                list_data_name.append(data_name)
                list_trial.append(data_name[:3]) #trial
                list_box.append(data_name[4:6]) #box
                list_face.append(data_name[-1]) #face
                img = cv2.imread(f)
                img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2HSV)
                # https://qiita.com/miyamotok0105/items/ce6f44064f128a580640
                img_HSV_H, img_HSV_S, img_HSV_V = cv2.split(img_HSV)
                 
                img_th = copy.copy(img_HSV_S)
                img = copy.copy(img_RGB)
                ret,thresh = cv2.threshold(img_th,12,255,cv2.THRESH_BINARY)
                
                # Extract contours
                contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
                num_objects = 0
                total_area = 0.0
                droplets_size= []
            
                if len(contours) > 0:
                    th = 5 # threshldold for removing small dots
                
                    for i in range(0, len(contours)):
                        # remove small objects
                        area = cv2.contourArea(contours[i])
                        if area > th:
                            # continue #前のif文に戻る
                            total_area += area
                            droplets_size.append(area)
                            
                            #面積(pixel)粒径(mm)に変換して0以上の値を抽出する関数
                            #引数に液滴データ
                            #A=πr2 
                            #r=√A/π
                    
                            diameter_arr = np.sqrt(area*0.001722/np.pi)*2
                            diameter_arr = diameter_arr[diameter_arr > 0]
                            cv2.polylines(img, contours[i], True, (255, 0, 0), 2)
                            rect = contours[i]
                            if hier[0,i,3] != -1:
                                x, y, w, h = cv2.boundingRect(rect)
                                cv2.rectangle(img, (x, y), (x + w, y + h), (1, 0, 0), 4)
                            
                        num_objects += 1

                    area_ratio = total_area/(WIDTH*HEIGHT)
                    num_obj_dens =  num_objects/(W_CM*H_CM)
                    area_arr[f_i] = total_area
                    num_obj_arr[f_i] = num_objects
                    area_ratio_arr[f_i] = area_ratio
                    num_obj_dens_arr[f_i] = num_obj_dens  
       
            ave_diameter = np.mean(diameter_arr)
        
            calc_diameter = np.floor(ave_diameter *100)/100
        
            max_dens = np.fmax.reduce(num_obj_dens_arr)

            calc_dens =  np.floor(max_dens * 100)/100       
     
            #階級分け
            #粒径0.35mm以下:2-3粒/cm2⇒1, 4-7粒/cm2⇒2, 8-15粒/cm2⇒3, 16-31粒/cm2⇒4, 32-63粒/cm2⇒5, 64-127粒/cm2⇒6, 128-255粒/cm2⇒7, 256粒/cm2以上⇒8
            #粒径0.36-0.75mm:0.8-1.5粒/cm2⇒1, 1.6-3.1粒/cm2⇒2, 3.2-6.3粒/cm2⇒3, 6.4-12.7粒/cm2⇒4, 12.8-25.5粒/cm2⇒5, 25.6-51.1粒/cm2⇒6, 51.2-102.3粒/cm2⇒7, 102.4粒/cm2以上⇒8
            #粒径0.76-1.25mm:0.2-0.3粒/cm2⇒1, 0.4-0.7粒/cm2⇒2, 0.8-1.5粒/cm2⇒3, 1.6-3.1粒/cm2⇒4, 3.2-6.3粒/cm2⇒5, 6.4-12.7粒/cm2⇒6, 12.8-25.5粒/cm2⇒7, 25.6粒/cm2以上⇒8
            #粒径1.26mm以上:0.05-0.09粒/cm2⇒1, 0.1-0.19粒/cm2⇒2, 0.2-0.39粒/cm2⇒3, 0.4-0.79粒/cm2⇒4, 0.8-1.59粒/cm2⇒5, 1.6-3.19粒/cm2⇒6, 3.2-6.39粒/cm2⇒7, 6.4粒/cm2以上⇒8
        
            if calc_diameter <= 0.35:
                diameter_rank = "A"
                if 2.00 <= calc_dens <=3.99:
                    density_rank = 1
                elif 4.00 <= calc_dens <=7.99:
                    density_rank = 2
                elif 8.00 <= calc_dens <=15.99:
                    density_rank = 3
                elif 16.00 <= calc_dens <=31.99:
                    density_rank = 4
                elif 32.00 <= calc_dens <=63.99:
                    density_rank = 5
                elif 64.00 <= calc_dens <=127.99:
                    density_rank = 6
                elif 128.00 <= calc_dens <=255.99:
                    density_rank = 7
                elif 256.00 <= calc_dens:
                    density_rank = 8
                else:
                    density_rank = 0
        
            elif 0.36 <= calc_diameter <= 0.75:
                diameter_rank = "B"
                if 0.80 <= calc_dens <=1.59:
                    density_rank = 1
                elif 1.60 <= calc_dens <=3.19:
                    density_rank = 2
                elif 3.20 <= calc_dens <=6.39:
                    density_rank = 3
                elif 6.40 <= calc_dens <=12.79:
                    density_rank = 4
                elif 12.80 <= calc_dens <=25.59:
                    density_rank = 5
                elif 25.60 <= calc_dens <=51.19:
                    density_rank = 6
                elif 51.20 <= calc_dens <=102.39:
                    density_rank = 7
                elif 102.40 <= calc_dens:
                    density_rank = 8
                else:
                    density_rank = 0
        
            elif 0.76 <= calc_diameter <= 1.25:
                diameter_rank = "C"
                if 0.20 <= calc_dens <=0.39:
                    density_rank = 1
                elif 0.40 <= calc_dens <=0.79:
                    density_rank = 2
                elif 0.80 <= calc_dens <=1.59:
                    density_rank = 3
                elif 1.60 <= calc_dens <=3.19:
                    density_rank = 4
                elif 3.20 <= calc_dens <=6.39:
                    density_rank = 5
                elif 6.40 <= calc_dens <=12.79:
                    density_rank = 6
                elif 12.80 <= calc_dens <=25.59:
                    density_rank = 7
                elif 25.60 <= calc_dens:
                    density_rank = 8
                else:
                    density_rank = 0
        
            else:
                diameter_rank = "D"
                if 0.05 <= calc_dens <=0.09:
                    density_rank = 1
                elif 0.10 <= calc_dens <=0.19:
                    density_rank = 2
                elif 0.20 <= calc_dens <=0.39:
                    density_rank = 3
                elif 0.40 <= calc_dens <=0.79:
                    density_rank = 4
                elif 0.80 <= calc_dens <=1.59:
                    density_rank = 5
                elif 1.60 <= calc_dens <=3.19:
                    density_rank = 6
                elif 3.20 <= calc_dens <=6.39:
                    density_rank = 7
                elif 6.40 <= calc_dens:
                    density_rank = 8
                else:
                    density_rank = 0
            
            result = f"{str(diameter_rank)}, {str(density_rank)}"
            pred_answer = "粒径区分と最大指数は " + result + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host = '0.0.0.0', port = port)
