# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 21:15:55 2024

@author: P51
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0")

""" SF NET """
class SF_NET(nn.Module):
    def __init__(self, input_size, SF_NUM, test_sf_flag, model_path_sf):
        super(SF_NET, self).__init__()
        """ load nn parameter files """
        if test_sf_flag == True:
            var_dict_sf = torch.load(model_path_sf) 
        
        if test_sf_flag == False: 
            self.sf_input_layer = nn.Linear(input_size, int(input_size/2),device=device) 
            self.sf_hid1_layer = nn.Linear(int(input_size/2), int(input_size),device=device) 
            self.sf_hid2_layer = nn.Linear(int(input_size), int(input_size/2),device=device)  
            self.sf_output_layer = nn.Linear(int(input_size/2), SF_NUM*input_size,device=device)             
        else:
            self.sf_input_layer = nn.Linear(input_size, int(input_size/2),device=device) 
            self.sf_hid1_layer = nn.Linear(int(input_size/2), int(input_size),device=device) 
            self.sf_hid2_layer = nn.Linear(int(input_size), int(input_size/2),device=device)  
            self.sf_output_layer = nn.Linear(int(input_size/2), SF_NUM*input_size,device=device)   
            # load encoder and decoder
            self.sf_input_layer.weight = nn.Parameter(var_dict_sf['sf_input_layer.weight'])
            self.sf_input_layer.bias = nn.Parameter(var_dict_sf['sf_input_layer.bias'])
            self.sf_hid1_layer.weight = nn.Parameter(var_dict_sf['sf_hid1_layer.weight'])
            self.sf_hid1_layer.bias = nn.Parameter(var_dict_sf['sf_hid1_layer.bias'])
            self.sf_hid2_layer.weight = nn.Parameter(var_dict_sf['sf_hid2_layer.weight'])
            self.sf_hid2_layer.bias = nn.Parameter(var_dict_sf['sf_hid2_layer.bias'])
            self.sf_output_layer.weight = nn.Parameter(var_dict_sf['sf_output_layer.weight'])
            self.sf_output_layer.bias = nn.Parameter(var_dict_sf['sf_output_layer.bias'])
            
    def forward(self,x):
        x = F.tanh(self.sf_input_layer(x))
        x = F.tanh(self.sf_hid1_layer(x))
        x = F.tanh(self.sf_hid2_layer(x))
        x = (self.sf_output_layer(x))
        return x
""" MUSIC NET """

class MUSIC_NET(nn.Module):
    def __init__(self, input_size_sf, output_size_music, SF_NUM, test_music_flag, model_path_music):
        super(MUSIC_NET, self).__init__()
        input_size_music = input_size_sf
        """ load nn parameter files """
        if test_music_flag == True:
            var_dict_music = torch.load(model_path_music) 
            self.layer1 = nn.Linear(input_size_music, int(input_size_music/4),device=device)
            self.layer2 = nn.Linear(int(input_size_music/4), int(input_size_music/8),device=device) 
            self.layer3 = nn.Linear(int(input_size_music/8), int(input_size_music/10),device=device)
            self.layer4 = nn.Linear(int(input_size_music/10), int(input_size_music/10),device=device)
            self.layer5 = nn.Linear(int(input_size_music/10), int(input_size_music/10),device=device)
            self.layer6 = nn.Linear(int(input_size_music/10), int(input_size_music/10),device=device)
            self.layer7 = nn.Linear(int(input_size_music/10), int(input_size_music/20),device=device)
            self.layer8 = nn.Linear(int(input_size_music/20), int(output_size_music/SF_NUM),device=device)    
            self.layer1.weight = nn.Parameter(var_dict_music['layer1.weight'])
            self.layer1.bias = nn.Parameter(var_dict_music['layer1.bias'])
            self.layer2.weight = nn.Parameter(var_dict_music['layer2.weight'])
            self.layer2.bias = nn.Parameter(var_dict_music['layer2.bias'])
            self.layer3.weight = nn.Parameter(var_dict_music['layer3.weight'])
            self.layer3.bias = nn.Parameter(var_dict_music['layer3.bias'])
            self.layer4.weight = nn.Parameter(var_dict_music['layer4.weight'])
            self.layer4.bias = nn.Parameter(var_dict_music['layer4.bias'])
            self.layer5.weight = nn.Parameter(var_dict_music['layer5.weight'])
            self.layer5.bias = nn.Parameter(var_dict_music['layer5.bias'])
            self.layer6.weight = nn.Parameter(var_dict_music['layer6.weight'])
            self.layer6.bias = nn.Parameter(var_dict_music['layer6.bias'])
            self.layer7.weight = nn.Parameter(var_dict_music['layer7.weight'])
            self.layer7.bias = nn.Parameter(var_dict_music['layer7.bias'])
            self.layer8.weight = nn.Parameter(var_dict_music['layer8.weight'])
            self.layer8.bias = nn.Parameter(var_dict_music['layer8.bias'])
        else:
            self.layer1 = nn.Linear(input_size_music, int(input_size_music/4),device=device)
            self.layer2 = nn.Linear(int(input_size_music/4), int(input_size_music/8),device=device) 
            self.layer3 = nn.Linear(int(input_size_music/8), int(input_size_music/10),device=device)
            self.layer4 = nn.Linear(int(input_size_music/10), int(input_size_music/10),device=device)
            self.layer5 = nn.Linear(int(input_size_music/10), int(input_size_music/10),device=device)
            self.layer6 = nn.Linear(int(input_size_music/10), int(input_size_music/10),device=device)
            self.layer7 = nn.Linear(int(input_size_music/10), int(input_size_music/20),device=device)
            self.layer8 = nn.Linear(int(input_size_music/20), int(output_size_music/SF_NUM),device=device)       
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        return x
