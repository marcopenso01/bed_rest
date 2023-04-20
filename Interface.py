"""
Created on Tue Apr 11 09:08:20 2023

@author: Marco Penso
"""
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import pydicom
import PySimpleGUI as sg
from io import BytesIO
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import shutil
import pandas as pd

drawing=False # true if mouse is pressed
mode=True
file_types = [("(*.hdf5)","*.hdf5")]
w, h = sg.Window.get_screen_size()
color = [(255,255,0),(0,255,0),(255,0,0)]
right_click_menu = ['', ['Delete All Contour (Slice)',
                         'Delete (Slice): SAX LV Endocardial Contour',
                         'Delete (Slice): SAX LV Epicardial Contour',
                         'Delete (Slice): SAX RV Endocardial Contour',
                         'Delete All Contour (Phase)',
                         'Delete (Phase): SAX LV Endocardial Contour',
                         'Delete (Phase): SAX LV Epicardial Contour',
                         'Delete (Phase): SAX RV Endocardial Contour',
                         'Reset Default Window']]
right_click_menu2 = ['', ['Phase view',
                          'Slice view']]
menu_def = ['&File',['&Open', '&Save', 'Save &New', 'Save &Volume']],['&Help', ['&About Us...']]


def draw(img, image_binary):
    def paint_draw(event,former_x,former_y,flags,param):
        global current_former_x,current_former_y,drawing, mode
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            current_former_x,current_former_y=former_x,former_y
        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                if mode==True:
                    cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                    cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                    current_former_x = former_x
                    current_former_y = former_y
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),2)
                current_former_x = former_x
                current_former_y = former_y
        return former_x,former_y
    return paint_draw
    
def imfill(img, dim=None):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    if dim:
        img = cv2.resize(img, (dim, dim))
    img[img > 0] = 255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    return img | cv2.bitwise_not(im_floodfill)

def plot_img(grid_file, grid_mask, n_ph=0, n_sl=0, alpha=1, beta=1):
    img_o = cv2.normalize(src=grid_file[str(n_ph)][0][n_sl], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img_o = cv2.convertScaleAbs(img_o, alpha=alpha, beta=beta)
    img_mask = cv2.cvtColor(img_o, cv2.COLOR_GRAY2RGB)
    for struc in [1,2,3]:
        mask = grid_mask[str(n_ph)][0][n_sl].astype(np.uint8)
        if struc == 1:
            mask[mask!=1]=0
        elif struc == 2:
            mask[mask==1]=0
        elif struc == 3:
            mask[mask!=3]=0
        mask[mask!=0]=1
        contours_mask, _ = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(image=img_mask, contours=contours_mask, contourIdx=-1, color=color[struc-1], thickness=1, lineType=cv2.LINE_AA)
    return img_mask
     
def blank_frame():
    return sg.Frame("", [[sg.Image(key="-PHASEPLOT-",expand_x=True, expand_y=True, background_color='#404040', right_click_menu=right_click_menu2)]], pad=(5, 3), expand_x=True, expand_y=True, background_color='#404040', border_width=0)

def array_to_data(array):
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format="PNG")
        data = output.getvalue()
    return data

def resize_img(img, size):
    size = int(size)
    if size > 160:
        return cv2.resize(img, (size, size), interpolation = cv2.INTER_LINEAR)
    elif size < 160:
        return cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
    else:
        return img
    
def delete(grid_mask, n_ph=None, n_sl=None, struc=None):
    try:
        if n_sl != None:
            for i in struc:
                mask = grid_mask[str(n_ph)][0][n_sl].astype(np.uint8)
                mask[mask == i] = 0
                if struc == 3 and np.sum(mask[mask == 2]) != 0:
                    mask_MYO = mask.copy()
                    mask_MYO[mask_MYO == 1] = 0
                    mask_MYO = imfill(mask_MYO, mask.shape[0])
                    mask_MYO[mask_MYO!=0]=2
                    mask[mask==2]=0
                    mask = mask+mask_MYO
                grid_mask[str(n_ph)][0][n_sl] = mask
        else:
            for n in range(len(grid_mask[str(n_ph)][0][:])):
                for i in struc:
                    mask = grid_mask[str(n_ph)][0][n].astype(np.uint8)
                    mask[mask == i] = 0
                    grid_mask[str(n_ph)][0][n] = mask
        return grid_mask
    except:   
        pass

def update_info(window, n_ph=None, n_sl=None):
    #tx = window["-PHASE-"].get()
    #window["-PHASE-"].update(value=tx+str(n_ph))
    #tx = window["-SLICE-"].get()
    #window["-SLICE-"].update(value=tx+str(n_sl))
    if n_ph == None:
        window["-PHASE-"].update(value=str('Phase: '))
    else:
        window["-PHASE-"].update(value=str('Phase: ')+str(n_ph))
    if n_sl == None:
        window["-SLICE-"].update(value=str('Slice: '))
    else:
        window["-SLICE-"].update(value=str('Slice: ')+str(n_sl))
        
def vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph=True, n_ph=None, n_sl=None):  
    if n_ph != None:
        window["-RVvolume-"].update(value=str('RV: ')+str(round(sum(c for c in RV_vol[str(n_ph)][:]),2))+str(' ml'))
        window["-Myovolume-"].update(value=str('Myo: ')+str(round(sum(c for c in Myo_vol[str(n_ph)][:]),2))+str(' ml'))
        window["-LVvolume-"].update(value=str('LV: ')+str(round(sum(c for c in LV_vol[str(n_ph)][:]),2))+str(' ml'))
        vol_lv = []
        vol_rv = []
        for n in range(len(LV_vol)):
            vol_lv.append(round(sum(c for c in LV_vol[str(n)][:]),2))
            vol_rv.append(round(sum(c for c in RV_vol[str(n)][:]),2))
        window["-LVSV-"].update(value=str('LVSV: ')+str(round((max(vol_lv)-min(vol_lv)),2))+str(' ml'))
        window["-RVSV-"].update(value=str('RVSV: ')+str(round((max(vol_rv)-min(vol_rv)),2))+str(' ml'))
        window["-LVEF-"].update(value=str('LVEF: ')+str(round(((max(vol_lv)-min(vol_lv))/max(vol_lv)*100),2))+str(' %'))
        window["-RVEF-"].update(value=str('RVEF: ')+str(round(((max(vol_rv)-min(vol_rv))/max(vol_rv)*100),2))+str(' %'))

        RV = []
        LV = []
        MYO = []
        if flag_ph:
            for n in range(len(LV_vol)):
                LV.append(round(sum(c for c in LV_vol[str(n)][:]),3))
                RV.append(round(sum(c for c in RV_vol[str(n)][:]),3))
                MYO.append(round(sum(c for c in Myo_vol[str(n)][:]),3))
        else:
            for n in range(len(LV_vol)):
                LV.append(round(LV_vol[str(n)][n_sl],3))
                RV.append(round(RV_vol[str(n)][n_sl],3))
                MYO.append(round(Myo_vol[str(n)][n_sl],3))
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.set_size_inches(10.5, 9.5, forward=True)
        fig.set_facecolor('#404040')
        fig.subplots_adjust(hspace=0.4)
        if flag_ph:
            fig.suptitle("Phase view", color='white', fontsize=15)
            val=10
        else:
            fig.suptitle("Slice view", color='white', fontsize=15)
            val=5
        ax1.plot(LV, color='red', marker='o')
        ax1.set_facecolor('#404040')
        ax1.set_xlabel('phase', color='white')
        ax1.set_ylabel('LV [ml]', color='white')
        ax1.tick_params(axis='x', colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.set_xticks(np.arange(0, len(LV_vol)+1, 2))
        ax1.grid(axis = 'y', linewidth = 0.6)
        ax1.vlines(x=n_ph, ymin=min(LV)-val, ymax=max(LV)+val, colors='white', lw=2, alpha=0.5)
        
        ax2.plot(MYO, color='green', marker='o')
        ax2.set_facecolor('#404040')
        ax2.set_xlabel('phase', color='white')
        ax2.set_ylabel('Myo [ml]', color='white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.set_xticks(np.arange(0, len(LV_vol)+1, 2))
        ax2.grid(axis = 'y', linewidth = 0.6)
        ax2.vlines(x=n_ph, ymin=min(MYO)-val, ymax=max(MYO)+val, colors='white', lw=2, alpha=0.5)
        
        ax3.plot(RV, color='yellow', marker='o')
        ax3.set_facecolor('#404040')
        ax3.set_xlabel('phase', color='white')
        ax3.set_ylabel('RV [ml]', color='white')
        ax3.tick_params(axis='x', colors='white')
        ax3.tick_params(axis='y', colors='white')
        ax3.set_xticks(np.arange(0, len(LV_vol)+1, 2))
        ax3.grid(axis = 'y', linewidth = 0.6)
        ax3.vlines(x=n_ph, ymin=min(RV)-val, ymax=max(RV)+val, colors='white', lw=2, alpha=0.5)
        plt.close(fig)
        '''
        #plt.ioff()
        fig = plt.figure(figsize=(8,3), facecolor='#404040');
        plt.rcParams.update({'axes.facecolor':'#404040'});
        ax = fig.add_subplot(111);
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis = 'y', color = 'black', linestyle = '--', linewidth = 0.5);
        ax.plot(LV, color='red');
        ax.set_xlabel('phase')
        ax.set_ylabel('LV [ml]')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        draw_figure(window['-CANVAS1-'].TKCanvas, fig);
        
        fig = plt.figure(figsize=(8,3), facecolor='#404040');
        plt.rcParams.update({'axes.facecolor':'#404040'});
        ax = fig.add_subplot(111);
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis = 'y', color = 'black', linestyle = '--', linewidth = 0.5)
        ax.plot(MYO, color='green');
        ax.set_xlabel('phase')
        ax.set_ylabel('Myo [ml]')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        draw_figure(window['-CANVAS2-'].TKCanvas, fig);
        
        fig = plt.figure(figsize=(8,3), facecolor='#404040');
        plt.rcParams.update({'axes.facecolor':'#404040'});
        ax = fig.add_subplot(111);
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(axis = 'y', color = 'black', linestyle = '--', linewidth = 0.5)
        ax.plot(RV, color='yellow');
        ax.set_xlabel('phase')
        ax.set_ylabel('RV [ml]')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        draw_figure(window['-CANVAS3-'].TKCanvas, fig);
        '''
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = array_to_data(data)
        window["-PHASEPLOT-"].update(data=img)
        
    else:
        window["-RVvolume-"].update(value=str('RV: '))
        window["-Myovolume-"].update(value=str('Myo: '))
        window["-LVvolume-"].update(value=str('LV: '))
        window["-LVSV-"].update(value=str('LVSV: '))
        window["-RVSV-"].update(value=str('RVSV: '))
        window["-LVEF-"].update(value=str('LVEF: '))
        window["-RVEF-"].update(value=str('RVEF: '))
        window["-PHASEPLOT-"].update('')

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack()
    return figure_canvas_agg
        
def volume(LV_vol, RV_vol, Myo_vol, grid_mask, px_size, struc=None, n_ph=None, n_sl=None):
    for i in struc:
        vol = (grid_mask[str(n_ph)][0][n_sl].astype(np.uint8) == i).sum() * px_size[0] * px_size[1] * px_size[2] / 1000
        vol = round(vol, 3)
        if i==1:
            RV_vol[str(n_ph)][n_sl] = vol
        if i==2:
            Myo_vol[str(n_ph)][n_sl] = vol
        if i==3:
            LV_vol[str(n_ph)][n_sl] = vol
    return LV_vol, RV_vol, Myo_vol

def del_volume(LV_vol, RV_vol, Myo_vol, struc=None, n_ph=None, n_sl=None):
    try:
        if n_sl != None:
            for i in struc:
                if i==1:
                    RV_vol[str(n_ph)][n_sl] = 0
                if i==2:
                    Myo_vol[str(n_ph)][n_sl] = 0
                if i==3:
                    LV_vol[str(n_ph)][n_sl] = 0
        else:
            for n in range(len(LV_vol[str(n_ph)][:])):
                for i in struc:
                    if i==1:
                        RV_vol[str(n_ph)][n] = 0
                    if i==2:
                        Myo_vol[str(n_ph)][n] = 0
                    if i==3:
                        LV_vol[str(n_ph)][n] = 0
        return LV_vol, RV_vol, Myo_vol
    except:   
        pass
    
def compute_metrics(LV_vol,RV_vol,Myo_vol):
    cardiac_phase = []
    structure_names = []
    slice_number = []
    vol_list = []
    
    for ph in range(len(LV_vol)):
        for struc in ['LV', 'Myo', 'RV']:
            for sl in range(len(LV_vol[str(ph)])):
                slice_number.append(sl)
                cardiac_phase.append(ph)
                structure_names.append(struc)
                if struc == 'LV':
                    vol_list.append(LV_vol[str(ph)][sl])
                if struc == 'Myo':
                    vol_list.append(Myo_vol[str(ph)][sl])
                if struc == 'RV':
                    vol_list.append(RV_vol[str(ph)][sl])
                    
    df = pd.DataFrame({'vol': vol_list, 'phase': cardiac_phase, 'struc': structure_names,
                       'slicenumber': slice_number})  
    return df
        
def popup():
    tab1_layout = [
        [sg.Text('About this program')],
        [sg.Text('Version 1.0')],
        [sg.Text('Pysimplegui 4.60.4')],
        [sg.Text('Python 3.8.16 64-bit')],      
    ]
    tab2_layout = [
        [sg.Text('Created by Marco Penso')],
        [sg.Text('with the support of Prof. Enrico G. Caiani')],
        [sg.Text('and Francesca Righetti.')],
        [sg.Text('')],
        [sg.Text('This project is part of a collaboraton with')],
        [sg.Text('the Politecnico of Milan and')],
        [sg.Text('the European Space Agency (ESA)')],
        [sg.Text('')],
        [sg.Text('For help with errors and crashes, please write to')],
        [sg.Text('Penso: marco1.penso@mail.polimi.it')],
        [sg.Text('Caiani: enrico.caiani@polimi.it')],
        [sg.Text('Righetti: francesca.righetti@polimi.it')],
    ]
    tab3_layout = [
        [sg.Text('Copyright © 2023')],
        [sg.Text('Distributed under the terms of the')],
        [sg.Text('MIT License')],
    ]
    tabgrp = [[sg.TabGroup([[sg.Tab('Overview', tab1_layout),
                             sg.Tab('Community', tab2_layout),
                             sg.Tab('Legal', tab3_layout)]])],
              [sg.Push(), sg.Button('OK')]]
        
    return sg.Window('About Us', tabgrp, modal=True).read(close=True)
    
    
def main():
    try: 
        del data
    except: 
        pass
    sg.theme('DarkGrey4')
    flag=0
    layout_frame1 = [
        [
         sg.Text('Patient ID: ', key="-PAZ-"),
         sg.Text('Phase: ', key="-PHASE-"),
         sg.Text('Slice: ', key="-SLICE-"),
        ],
        [sg.Text('Phase info'),
         sg.VSeperator(),
         sg.Text('LV: ', key="-LVvolume-"),
         sg.Text('Myo: ', key="-Myovolume-"),
         sg.Text('RV: ', key="-RVvolume-"),
        ], 
        [sg.Text('Function:'),
         sg.VSeperator(),
         sg.Text('LVSV: ', key="-LVSV-"),
         sg.Text('LVEF: ', key="-LVEF-"),
         sg.Text('RVSV: ', key="-RVSV-"),
         sg.Text('RVEF: ', key="-RVEF-"),
        ],
        [sg.Text('_'  * 100)],
        [sg.Column([[sg.Image(key="-IMAGE-", right_click_menu=right_click_menu)]], justification='center')],
        [sg.VPush()],
        [sg.Text('_'  * 100)],
        [sg.Text("Resize"), sg.Push(), sg.Slider(range=(100, 420), orientation='h', size=(45, 15), default_value=350, resolution=10, disabled=True, enable_events=True, key="-RESIZE-", disable_number_display=True),],
        [sg.Text("Contrast"), sg.Push(), sg.Slider(range=(0.1, 3.0), orientation='h', size=(45, 15), default_value=1.0, resolution=.1, disabled=True, enable_events=True, key="-Contrast-", disable_number_display=True),],
        [sg.Text("Brightness"), sg.Push(), sg.Slider(range=(-100, 100), orientation='h', size=(45, 15), default_value=1, disabled=True, enable_events=True, key="-Brightness-", disable_number_display=True),],
        [sg.Text('_'  * 100)],
    ]

    layout_frame2 = [
        [blank_frame()],
    ]

    layout = [
        [sg.Menu(menu_def, key='menu')],
        [
        sg.VSeperator(),
        sg.Button('LV', button_color=('white', 'red'), key="-LV-", disabled=True),
        sg.Button('Myo', button_color=('white', 'green'), key="-Myo-", disabled=True),
        sg.Button('RV', button_color=('black', 'yellow'), key="-RV-", disabled=True),
        sg.VSeperator(),
        sg.Text('', key="-message-"),
        sg.Push(),
        sg.Button("Close", size=(6,1)),
        ],
        
        [sg.Frame("Frame 1", layout_frame1, size=(520, h-170)),
         sg.Frame("Frame 2", layout_frame2, size=(w/2-40, h-170), title_location=sg.TITLE_LOCATION_TOP)],
        ]
    
    window = sg.Window("SAX Interface", layout, finalize=True)
    window.bind('<Right>', '-RIGHT-')
    window.bind('<Left>', '-LEFT-')
    window.bind('<Down>', '-DOWN-')
    window.bind('<Up>', '-UP-')

    # Run the Event Loop
    n_ph=0
    n_sl=0
    while True: 
        
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == 'Open':
            filename = sg.popup_get_file('Document to open', 'Open file')
            #filename = values["-FILE-"]
            try:
                if filename.split('.')[-1] != 'hdf5':
                    sg.popup('Error!', 'Could not open file!', filename)
            except:
                pass
            try:
                data = h5py.File(filename, 'r+')
                flag=1
                flag_ph = 1
                n_phase = len(np.unique(data['phase']))
                n_slice = np.sum(data['phase'][:] == data['phase'][0])
                grid_file = {}
                grid_mask = {}
                LV_vol = {}
                RV_vol = {}
                Myo_vol = {}
                for n in range(n_phase):
                    grid_file[str(n)] = []
                    grid_mask[str(n)] = []
                    for ind in np.where(data['phase'][:] == str(n).encode("utf-8")):
                        grid_file[str(n)].append(data['img_raw'][ind])
                        grid_mask[str(n)].append(data['pred'][ind])
                for n in range(n_phase):
                    LV_vol[str(n)] = []
                    RV_vol[str(n)] = []
                    Myo_vol[str(n)] = []
                    for i in range(n_slice):
                        LV_vol[str(n)].append(0)
                        RV_vol[str(n)].append(0)
                        Myo_vol[str(n)].append(0)
                px_size = [float(data['pixel_size'][0][0].decode("utf-8")), float(data['pixel_size'][0][1].decode("utf-8")), float(data['pixel_size'][0][2].decode("utf-8"))]
                for n in range(n_phase):
                    for i in range(n_slice):
                        LV_vol,RV_vol,Myo_vol = volume(LV_vol,RV_vol,Myo_vol,grid_mask,px_size,[1,2,3],n,i)
                update_info(window, n_ph=0, n_sl=0)
                vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=0, n_sl=0)
                #tx = window["-PAZ-"].get()
                window["-PAZ-"].update(value=str('Patient: ')+data['paz'][0].decode("utf-8"))
                window["-Contrast-"].update(disabled=False)
                window["-Brightness-"].update(disabled=False)
                window["-RESIZE-"].update(disabled=False)
                window["-LV-"].update(disabled=False)
                window["-RV-"].update(disabled=False)
                window["-Myo-"].update(disabled=False)
                img = plot_img(grid_file, grid_mask, n_ph=0, n_sl=0, alpha=values["-Contrast-"], beta=values["-Brightness-"])
                img = resize_img(img, size=values["-RESIZE-"])
                img = array_to_data(img)
                window["-IMAGE-"].update(data=img)
                window["-message-"].update(value=str(''))
                
            except:   
                pass
        
        if event == '-RIGHT-':
            try:
                n_ph += 1
                if n_ph > n_phase:
                    n_ph = 0
                img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
                img = resize_img(img, size=values["-RESIZE-"])
                img = array_to_data(img)
                window["-IMAGE-"].update(data=img)
                update_info(window, n_ph=n_ph, n_sl=n_sl)
                vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
                window["-message-"].update(value=str(''))
            except:   
                pass
            
        if event == '-LEFT-':
            try:
                n_ph -= 1
                if n_ph < 0:
                    n_ph = n_phase
                img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
                img = resize_img(img, size=values["-RESIZE-"])
                img = array_to_data(img)
                window["-IMAGE-"].update(data=img)
                update_info(window, n_ph=n_ph, n_sl=n_sl)
                vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
                window["-message-"].update(value=str(''))
            except:   
                pass
        
        if event == '-UP-':
            try:
                n_sl -= 1
                if n_sl < 0:
                    n_sl = n_slice
                img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
                img = resize_img(img, size=values["-RESIZE-"])
                img = array_to_data(img)
                window["-IMAGE-"].update(data=img)
                update_info(window, n_ph=n_ph, n_sl=n_sl)
                vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
                window["-message-"].update(value=str(''))
            except:   
                pass
        
        if event == '-DOWN-':
            try:
                n_sl += 1
                if n_sl > n_slice:
                    n_sl = 0
                img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
                img = resize_img(img, size=values["-RESIZE-"])
                img = array_to_data(img)
                window["-IMAGE-"].update(data=img)
                update_info(window, n_ph=n_ph, n_sl=n_sl)
                vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
                window["-message-"].update(value=str(''))
            except:   
                pass       
        
        if event == "Close":
            try:
                data.close()
            except:
                pass
            window["-IMAGE-"].update('')
            window["-PAZ-"].update(value=str('Patient: '))
            update_info(window)
            #window["-FILE-"].update('')
            grid_file = {}
            grid_mask = {}
            LV_vol = {}
            RV_vol = {}
            Myo_vol = {}
            n_ph=0
            n_sl=0
            px_size = 0
            window["-Contrast-"].update(disabled=True, value=1.0)
            window["-Brightness-"].update(disabled=True, value=1)
            window["-RESIZE-"].update(disabled=True, value=350)
            window["-LV-"].update(disabled=True)
            window["-RV-"].update(disabled=True)
            window["-Myo-"].update(disabled=True)
            window["-message-"].update(value=str(''))
            vol_info(window, LV_vol, RV_vol, Myo_vol)
            flag = 0
        
        if event == "-Contrast-":
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
        
        if event == "-Brightness-":
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
        
        if event == "-RESIZE-":
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
        
        if event == 'Delete All Contour (Slice)':
            grid_mask = delete(grid_mask, n_ph=n_ph, n_sl=n_sl, struc=[1,2,3])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[1,2,3], n_ph=n_ph, n_sl=n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete (Slice): SAX LV Endocardial Contour':
            grid_mask = delete(grid_mask, n_ph=n_ph, n_sl=n_sl, struc=[3])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[3], n_ph=n_ph, n_sl=n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete (Slice): SAX LV Epicardial Contour':
            grid_mask = delete(grid_mask, n_ph=n_ph, n_sl=n_sl, struc=[2])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[2], n_ph=n_ph, n_sl=n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete (Slice): SAX RV Endocardial Contour':
            grid_mask = delete(grid_mask, n_ph=n_ph, n_sl=n_sl, struc=[1])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[1], n_ph=n_ph, n_sl=n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete All Contour (Phase)':
            grid_mask = delete(grid_mask, n_ph=n_ph, struc=[1,2,3])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[1,2,3], n_ph=n_ph)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete (Phase): SAX LV Endocardial Contour':
            grid_mask = delete(grid_mask, n_ph=n_ph, struc=[3])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[3], n_ph=n_ph)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete (Phase): SAX LV Epicardial Contour':
            grid_mask = delete(grid_mask, n_ph=n_ph, struc=[2])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[2], n_ph=n_ph)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == 'Delete (Phase): SAX RV Endocardial Contour':
            grid_mask = delete(grid_mask, n_ph=n_ph, struc=[1])
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol, RV_vol, Myo_vol = del_volume(LV_vol, RV_vol, Myo_vol, struc=[1], n_ph=n_ph)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
            
        if event == 'Reset Default Window':
            window["-Contrast-"].update(value=1.0)
            window["-Brightness-"].update(value=1)
            window["-RESIZE-"].update(value=350)
            window.refresh()
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=1.0, beta=1)
            img = resize_img(img, size=350)
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            
        if event == 'Phase view':
            flag_ph = 1
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
            
        if event == 'Slice view':
            flag_ph = 0
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
            
        if event == '-RV-':
            tit=['---RV Endocardial Contour---']
            img = grid_file[str(n_ph)][0][n_sl].copy()
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dim = img.shape[0]
            img = cv2.convertScaleAbs(img, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
            cv2.namedWindow(tit[0])
            cv2.setMouseCallback(tit[0],draw(img, image_binary))
            while(1):
                cv2.imshow(tit[0],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    im_out = imfill(image_binary, dim)
                    break              
            cv2.destroyAllWindows()
            im_out[im_out>0]=1
            mask_MYO = grid_mask[str(n_ph)][0][n_sl].copy()
            mask_LV = grid_mask[str(n_ph)][0][n_sl].copy()
            mask_MYO[mask_MYO != 2] = 0
            mask_LV[mask_LV != 3] = 0
            final_mask = im_out + mask_MYO + mask_LV
            m_myo = mask_MYO.copy()
            m_lv = mask_LV.copy()
            m_myo[m_myo!=0]=1
            m_lv[m_lv!=0]=1
            mm = m_myo+m_lv
            coord = np.where((mm+im_out)>1)
            for nn in range(len(coord[0])):
                final_mask[coord[0][nn],coord[1][nn]]=1
            grid_mask[str(n_ph)][0][n_sl] = final_mask
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol,RV_vol,Myo_vol = volume(LV_vol,RV_vol,Myo_vol,grid_mask,px_size,[1],n_ph,n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == '-Myo-':
            tit=['---LV Epicardial Contour---']       
            img = grid_file[str(n_ph)][0][n_sl].copy()
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dim = img.shape[0]
            img = cv2.convertScaleAbs(img, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
            cv2.namedWindow(tit[0])
            cv2.setMouseCallback(tit[0],draw(img, image_binary))
            while(1):
                cv2.imshow(tit[0],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    im_out = imfill(image_binary, dim)
                    break              
            cv2.destroyAllWindows()
            im_out[im_out>0]=2
            mask_RV = grid_mask[str(n_ph)][0][n_sl].copy()
            mask_LV = grid_mask[str(n_ph)][0][n_sl].copy()
            mask_RV[mask_RV != 1] = 0
            mask_LV[mask_LV != 3] = 0
            final_mask = im_out + mask_RV + mask_LV
            m_rv = mask_RV.copy()
            m_lv = mask_LV.copy()
            m_rv[m_rv!=0]=1
            m_lv[m_lv!=0]=1
            mm = m_rv+m_lv
            coord = np.where((mm+im_out)>2)
            for nn in range(len(coord[0])):
                final_mask[coord[0][nn],coord[1][nn]]=2
            grid_mask[str(n_ph)][0][n_sl] = final_mask
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol,RV_vol,Myo_vol = volume(LV_vol,RV_vol,Myo_vol,grid_mask,px_size,[2],n_ph,n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
        
        if event == '-LV-':
            tit=['---LV Endocardial Contour---']
            img = grid_file[str(n_ph)][0][n_sl].copy()
            img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dim = img.shape[0]
            img = cv2.convertScaleAbs(img, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            image_binary = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
            cv2.namedWindow(tit[0])
            cv2.setMouseCallback(tit[0],draw(img, image_binary))
            while(1):
                cv2.imshow(tit[0],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    im_out = imfill(image_binary, dim)
                    break              
            cv2.destroyAllWindows()
            im_out[im_out>0]=3
            mask_MYO = grid_mask[str(n_ph)][0][n_sl].copy()
            mask_RV = grid_mask[str(n_ph)][0][n_sl].copy()
            mask_MYO[mask_MYO != 2] = 0
            mask_RV[mask_RV != 1] = 0
            final_mask = im_out + mask_RV + mask_MYO
            m_rv = mask_RV.copy()
            m_myo = mask_MYO.copy()
            m_rv[m_rv!=0]=1
            m_myo[m_myo!=0]=1
            mm = m_rv+m_myo
            coord = np.where((mm+im_out)>3)
            for nn in range(len(coord[0])):
                final_mask[coord[0][nn],coord[1][nn]]=3
            grid_mask[str(n_ph)][0][n_sl] = final_mask
            img = plot_img(grid_file, grid_mask, n_ph=n_ph, n_sl=n_sl, alpha=values["-Contrast-"], beta=values["-Brightness-"])
            img = resize_img(img, size=values["-RESIZE-"])
            img = array_to_data(img)
            window["-IMAGE-"].update(data=img)
            window["-message-"].update(value=str(''))
            LV_vol,RV_vol,Myo_vol = volume(LV_vol,RV_vol,Myo_vol,grid_mask,px_size,[3],n_ph,n_sl)
            vol_info(window, LV_vol, RV_vol, Myo_vol, flag_ph, n_ph=n_ph, n_sl=n_sl)
                
        if event == 'Save':
            try: 
                data = h5py.File(filename, 'r+')
            except:   
                pass
            try:
                if flag:
                    for ph in range(n_phase):
                        for sl in range(n_slice):
                            data['pred'][np.where(data['phase'][:] == str(ph).encode("utf-8"))[0][sl]] = grid_mask[str(ph)][0][sl].astype(np.uint8)
                    data.close()
                    window["-message-"].update(value=str('Save Workspace Done'))
                else:
                    window["-message-"].update(value=str('Attention: No File Selected'))
            except:
                pass
            
        if event == 'Save New':
            try: 
                data = h5py.File(filename, 'r+')
            except:   
                pass
            try:
                if flag:
                    data.close()
                    new_path = filename.split('pred')[0]
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d_time_%H_%M_%S")
                    new_path = os.path.join(new_path, 'pred_'+dt_string+'.hdf5')
                    shutil.copy2(filename, new_path);
                    new_data = h5py.File(new_path, 'r+')
                    for ph in range(n_phase):
                        for sl in range(n_slice):
                            new_data['pred'][np.where(new_data['phase'][:] == str(ph).encode("utf-8"))[0][sl]] = grid_mask[str(ph)][0][sl].astype(np.uint8)
                    new_data.close()
                    data = h5py.File(filename, 'r+')
                    window["-message-"].update(value=str('Save New Workspace Done'))
                else:
                    window["-message-"].update(value=str('Attention: No File Selected'))
            except:
                pass
            
        if event == 'Save Volume':
            try:
                if flag:
                    new_path = filename.split('pred')[0]
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d_time_%H_%M_%S")
                    new_path = os.path.join(new_path, 'Excel_'+dt_string+'.xlsx')
                    df = compute_metrics(LV_vol,RV_vol,Myo_vol)
                    df.to_excel(new_path)
                    window["-message-"].update(value=str('Save Volume Analysis Done'))
                else:
                    window["-message-"].update(value=str('Attention: No File Selected'))
            except:
                pass
            
        if event == 'About Us...':
            popup();
            
    window.close()


if __name__ == "__main__":
    main()
