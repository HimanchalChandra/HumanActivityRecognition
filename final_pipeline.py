import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import tensorflow as tf
import tensorflow_hub as hub
import cv2

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from utils import *

if __name__=="__main__":
    opt = parse_opts()

    #Loading the class labels
    with open("label_map.txt") as obj:
        class_names = [line.strip() for line in obj.readlines()]

    #Loading list of video
    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    #Inference using Pytorch Model
    if(opt.type == 'pyt'):

        opt.mean = get_mean()
        opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
        #Setting Sample Size
        opt.sample_size = 112
        #Setting Sample Duration
        opt.sample_duration = 16
        #Number of classes in our label list
        opt.n_classes = 400

        model = generate_model(opt)
        print('loading model {}'.format(opt.model))
        #Loading the Model
        model_data = torch.load(opt.model)
        assert opt.arch == model_data['arch']
        model.load_state_dict(model_data['state_dict'])
        model.eval()

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)

        output = []
        for input_file in input_files:
            video_path = os.path.join(opt.video_root, input_file)
            if os.path.exists(video_path):
                subprocess.call('mkdir tmp', shell=True)
                #Converting Video into frames and storing it temporarily in tmp folder
                subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(video_path),
                                shell=True)

                #Classification step"
                result = classify_video('tmp', input_file, class_names, model, opt)
                #Storing the result in list
                output.append(result)
                
                #Deleting the tmp folder
                subprocess.call('rm -rf tmp', shell=True)
            else:
                print('{} does not exist'.format(input_file))

        if os.path.exists('tmp'):
            subprocess.call('rm -rf tmp', shell=True)

        print("PRINTING THE RESULT USING PYTORCH MODEL:")
        for x,y in zip(input_files, output):
            print(x, y)
    
    #Inference using Tensorflow Model
    elif (opt.type == 'tf'):
        #Loading Tensorflow Model
        i3d = hub.load(opt.model).signatures['default']

        print("PRINTING THE RESULT USING TENSORFLOW MODEL:")

        for input_file in input_files:
            video_path = os.path.join(opt.video_root, input_file)
            if os.path.exists(video_path):
                # Preprocessing and loading the video
                sample_video = load_video(video_path)
                model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
                logits = i3d(model_input)['default'][0]
                probabilities = tf.nn.softmax(logits)
                index = np.argsort(probabilities)[::-1][:5][0]
                print(input_file, class_names[index])


