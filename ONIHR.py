print('\033[91m' +r"""                                                        
                                                        
Ȝ     ▄▀▀▀▀▄   ▄▀▀▄ ▀▄  ▄▀▀█▀▄   ▄▀▀▄ ▄▄   ▄▀▀▄▀▀▀▄    Ȝ
     █      █ █  █ █ █ █   █  █ █  █   ▄▀ █   █   █     
     █      █ ▐  █  ▀█ ▐   █  ▐ ▐  █▄▄▄█  ▐  █▀▀█▀      
Ξ    ▀▄    ▄▀   █   █      █       █   █   ▄▀    █     Ξ
       ▀▀▀▀   ▄▀   █    ▄▀▀▀▀▀▄   ▄▀  ▄▀  █     █       
              █    ▐   █       █ █   █    ▐     ▐       
Δ             ▐        ▐       ▐ ▐   ▐                 Δ
                                                         """+ '\033[0m')
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||IMPORT

from ast import Break, Return, While
from cmath import nan
from concurrent.futures import process
import ctypes
from enum import Flag
from itertools import count
from turtle import shape
import uuid
from http.client import FOUND
from pickle import FALSE
from sre_parse import FLAGS
from tkinter import ALL
from traceback import print_tb
from typing import Set
from scipy.spatial.transform import Rotation as R
from traitlets import All
from mem_edit_MOD import Process
from nbformat import read
import numpy as np
import time
import os
import math
import concurrent.futures
import psutil
import math
import warnings
import pymem
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||EVIRONEMENT
np.warnings.filterwarnings('ignore',category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||FAKE OBJ TEST

BREP = np.array(([np.array(([np.array(([np.array((0,2,1,0,3,1)),
                                        np.array((0,3,1,3,3,1)),
                                        np.array((3,3,1,3,2,1)),
                                        np.array((3,2,1,0,2,1))]
                             )),
                             np.array(([np.array((1,1,1,1,2,1)),
                                        np.array((1,2,1,2,2,1)),
                                        np.array((2,2,1,2,1,1)),
                                        np.array((2,1,1,1,1,1))]))
                             ])),

                  np.array((1,1)),
                  np.array(([np.array(([np.array((20,52,20,-20,30,50)),
                                        np.array((-20,30,50,-20,10,50)),
                                        np.array((50,40,-22,20,52,-20)),
                                        np.array((20,52,-20,20,52,20))]))])),
                  np.array((1))
                ]))
POINT = np.array((np.array((1,1,1))))
LINE = np.array(([[1,1,1,1,2,1]]))
NURBS = np.array(([[0,2,1,0,3,1,1,1,1,0,0,0,3]]))
PTC = np.array(([[0,2,1],[0,3,1],[1,1,1],[0,0,0]],[],[]))
CIR=np.array((
    np.array((0,1,0)),
    np.array((1,0,0)),
    np.array((0,1,0)),
    52,0.5,0.5
))
MESH=np.array((
    np.array((
        [2,3,5],
        [5,6,8],
        [2,3,9],
        [1,0,2]
    )),
    np.array((
        [0,1,2],
        [1,2,3]
    ))
))

#OBJECT_ARY = [10,MESH,np.nan,"ddddd",np.nan,0,-1,0,0]
OBJECT_ARY = [0,CIR,np.nan,"ddddd",np.nan,0,-1,0,0]
#OBJECT_ARY = [13,PTC,np.nan,"ddddd",np.nan,0,-1,0,0]
#OBJECT_ARY = [1,NURBS,np.nan,"ddddd",np.nan,0,-1,0,0]
#OBJECT_ARY = [9,LINE,np.nan,"ddddd",np.nan,0,-1,0,0]
#OBJECT_ARY = [14,POINT,np.nan,"ddddd",np.nan,0,-1,0,0]
#OBJECT_ARY = [12,BREP,np.nan,"ddddd",np.nan,0,-1,0,0]
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||HELPeqqq 
#print(Ecriture des OBJET [TYPE_OBJ,OBJ_GEOMETRIQUE,ID,NAME])


#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||CREDIT
def Credit ():
    print("By Enzo Arioli Bildstein\nhttps://github.com/EnzoArioliBildstein\nhttps://www.linkedin.com/in/enzo-arioli-bildstein-b304a5163/")
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||DEF
def D_RTN_VAL (addrs, offset) :
    return p.read_memory(int(addrs+offset), ctypes.c_ulonglong()).value
def D_STR_2_ASM_STR (Input) :
    Input = str(Input).encode("ascii").hex()
    Input=np.flip(np.array(list(map(list,(Input+"000000")[0:(math.ceil(len(Input)/8)*8)]))).reshape(-1,2),axis=1)
    return [''.join(row) for row in np.flip(np.stack((Input,np.zeros((np.shape(Input)[0],2),np.uint())),axis=1).reshape(-1,16),axis=1)]
def D_GOOD_HEX (INPUT) :
    return ("0000000000000000" + hex(INPUT)[2:])[-16:]
def D_IS_ADRESS_64 (input) :
    return any([all([type(input) == int and type(input) == int]) or all([all([i in [0,1,2,3,4,5,6,7,8,9,"A","B","C","D","E","F"] for i in str(input)])])])
def D_RTD_UTYPE (Shape) :
    if Shape == 2:
        Shape_Ctype = ctypes.c_ubyte()
    elif Shape == 4:
        Shape_Ctype = ctypes.c_short()
    elif Shape == 8:
        Shape_Ctype = ctypes.c_ulong()
    elif Shape == 16:
        Shape_Ctype = ctypes.c_ulonglong()
    return Shape_Ctype
def D_PRINT_FROM (a) :
    [print(("   "+hex(i))[-5:]+"     "+(a[i:i+8]).hex()) for i in range(0,len(a),8)] 
def D_PRINT_FROM_FROM (a,FROM) :
    [print(("   "+hex(i))[-5:]+"     "+(a[i:i+8]).hex()+"   "+("0000000000000000"+hex(p.read_memory(FROM+i,ctypes.c_longlong()).value)[2:])[-16:]) for i in range(0,len(a),8)] 
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||FONCTION GEO
def D_NP_TO_BY (array) :
    if array.dtype!="float64":
        array=array.astype(dtype=np.float64)
    return np.flip(array).tobytes()[::-1]
def D_CUM_DOM (p) :
    return np.cumsum([0]+[((p[0+i]-p[3+i])**2+(p[1+i]-p[4+i])**2+(p[2+i]-p[5+i])**2)**0.5 for i in range(0,np.size(p)-3,3)])
def D_DOM (p) :
    return np.sum([0]+[((p[0+i]-p[3+i])**2+(p[1+i]-p[4+i])**2+(p[2+i]-p[5+i])**2)**0.5 for i in range(0,np.size(p)-3,3)])
def D_ARRAY_COUNT (NUM) :
    return b''.join([bytes(ctypes.c_long(NUM)) for i in range (0,2)])
def D_BOND_BOX (ARRAY) :
    ARRAY = ARRAY.reshape(-1,3)
    return D_NP_TO_BY(np.concatenate((np.nanmin(ARRAY,axis=0),np.nanmax(ARRAY,axis=0))))
def D_ARRAY_COUNT_MULTI (NUM) :
    return b''.join([bytes(ctypes.c_long(i)) for i in NUM])
def D_TO_BY (array) :
    if type(array)==bytes :
        return array
    else :
        if type(array) == int :
            return ((bytes(ctypes.c_longlong(array))))
        elif type(array) == float :
            return ((bytes(ctypes.c_longdouble(array))))
        elif type(array) == list :
            if type(array[0]) == float :
                array = np.array((array)).astype(dtype=np.float64)
            else :
                array = np.array((array))
            return np.flip(array).tobytes()
        else :
            return np.flip(array).tobytes()
def D_VECTEUR_MATRIX (array) :
    array=array.reshape(-1,3)
    array = (np.abs(array)**0.5/np.abs(np.sum(array,axis=1))[:,np.newaxis]*np.sign(array))
    return b''.join([D_TO_BY(i) for i in [FLAG,0,0,3,D_ARRAY_COUNT_MULTI([int((np.size(d)-1)/3+1),d[-1]+1]),0,0,3,0,D_CUM_DOM(d[:-1])]])+D_NP_TO_BY(d[:-1]),S_SIZE,CALL_POST
def FLAG_DISPACHEUR (FLAG,LIST_ATT) :
    return FLAG[LIST_ATT[0]] + b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0' + D_ARRAY_COUNT(int(LIST_ATT[1]))
def FLAG_DISPACHEUR_NO_CALL (FLAG,LIST_ATT) :
    return FLAG[LIST_ATT[0]] + b'\x00\x00\x00\x00\x00\x00\x00\x00' + D_ARRAY_COUNT(int(LIST_ATT[1]))
def D_FAKE_FLAG (Value):
    return b''.join([b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0']*Value)
def D_rotation_matrix_from_vectors(vec1,vec2):
    a,b=(vec1 / np.linalg.norm(vec1)).reshape(3),(vec2 / np.linalg.norm(vec2)).reshape(3)
    v=np.cross(a,b)
    c=np.dot(a,b)
    s=np.linalg.norm(v)
    kmat=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    rotation_matrix=np.eye(3)+kmat+kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||FONCTION GESTION
def D_OFFSET_HEX (Adrss, Offset) :
    if type(Adrss) == int :
        if type(Offset) == int :
            Adrss = Adrss + Offset
        else :
            Adrss = Adrss  + int(Offset,16)
    else :
        if type(Offset) == int :
            Adrss = ("0000000000000000"+(hex(int(Adrss,16) + Offset))[2:])[:16]
        else :
            Adrss = ("0000000000000000"+(hex(int(Adrss,16) + int(Offset,16))[2:]))[:16]
    return Adrss
def D_NAV_PTR (Adrss) :
    print("PTR = " + Adrss)
    Command = input("CMD = ")
    if Commande == 'g' :
        Temp_Adrss =  (input("Goto Adress (Hex) = ")).upper() 
        if Temp_Adrss == lng(16) and all([i in [0,1,2,3,4,5,6,7,8,9,"A","B","C","D","E","F"] for i in Temp_Adrss]) :
            Adrss = Temp_Adrss
        elif type(Temp_Adrss) == int:
            Adrss = D_GOOD_HEX(Temp_Adrss)
        else :
            print("ERROR "+ Temp_Adrss + " is not valid")
    elif Command == "o" :
        Temp_Adrss = input("Offset from " + Adress + " = " )
        if D_IS_ADRESS_64(Adrss) :
            Adrss = ("0000000000000000"+(hex(int(Adrss,16))[2:]))[:16]
    elif Command == "c" :
        Temp_Adrss = np.array((input("Offset Down = " ) , input("Offset Up = " )))
        Copy_type  =  input("Copy adress (a for Adress) or Value (v for Value) " )
        Data_Shape = input ("Data shape = ")
        if  D_IS_ADRESS_64(Temp_Adrss[0]) and D_IS_ADRESS_64(Temp_Adrss[1]) and Copy_type in ["a","v"] and Data_Shape not in [2,4,8,16,32,64,128]:
            Temp_Adrss = [[int(Temp_Adrss,16) if Temp_Adrss != int else Temp_Adrss[i]] for i in range (0,2)]
            if np.sum(Temp_Adrss)%Data_Shape == 0 :
                User_Variable = input("Enter variable name = ")
                if len(User_Variable) == 0 :
                    print("ERROR "+ User_Varible + " is not valid")
                    locals()[User_Variable] = np.arange(Adress-Offset[0]+Data_Shape,Adress+Offset[1]-Data_Shape+Data_Shape)      
            else : 
                print("ERROR "+ Data_type + " is not valid")
        else :
            print("ERROR "+ Temp_Adrss + " or " + Copy_type +" or " + Data_type  +  " is not valid")
    elif Command == "w" :
        Free_Write_Mode = input("Free write mode (y for Yes / n for No) = ").upper()
        if Free_Write_Mode == Y :
            klm=50
        elif Free_Write_Mode == N :
            Adress_to_write = input("Variable name of adress = ")
            Value_to_write = input("Variable name of value = ")
            if np.size(Adress_to_write)!=np.size(Value_to_write) or np.size(Value_to_write) == 0:
                print("ERROR "+ Adress_to_write + " hadn't the same size of "+ Value_to_write + " or value is emply")
            else :
                k=50
        else :
            print("ERROR "+ Free_Write_Mode + " is not valid")
    elif Command == "p" :
        Print("")
    elif Command == "q" :
        print("Quit Nav Mode")
#        break
    elif Command == "h" :
        print("h for Help \n g for Goto \n q for Quit \n o for Offset \n P for Print \n W for Write \n C for Copy")
    else :
        input("h for Help")
def D_PID_LOAD ():
    Rhino_PID =[]
    for process in psutil.process_iter ():
        if process.name() == "Rhino.exe" :
            Rhino_PID.append(process.pid)
    if len(Rhino_PID) == 0 :
        print ("No Rhino found")
    elif len(Rhino_PID) == 1 :
        PID = Rhino_PID[0]
        print ("Run on Rhino PID = " + str(PID))
    else :
        print ("Rhino PID list = " + str(Rhino_PID))
        PID = int(input("PID = "))
        print ("Run on Rhino PID = " + str(PID))
    return PID
def D_Init_point(ppro,REALOAD_FLAG) :
    print('\033[95m' +"\nINITIALISTION OF VALUE\n----------------------"+ '\033[0m')
    ALL_MAP = p.list_mapped_regions()
    HEAP = ALL_MAP[:np.argmax(np.diff(np.array([i[1] for i in ALL_MAP])))+1]
    PSB_MAP=[]
    m = p.search_all_memory(ctypes.c_ulong(int("0000003D",16)),Memory_map=HEAP)
    for c in m :
        l = p.read_memory(c-8,ctypes.c_ulong()).value 
        if 32759<l and l>32762 :
            for i in HEAP :
                if (i[0]<c and c<i[1]) :
                    if i not in PSB_MAP :
                        PSB_MAP.append(i)
    for MAP in PSB_MAP[::-1] :
        PSB_POS = np.array([(p.search_all_memory(ctypes.c_ulong(i),Memory_map=[MAP])) for i in range(int((("0000000000000000" + (str(hex(MAP[0]))[2:]))[-16:][:8]),16),int((("0000000000000000" + (str(hex(MAP[1]))[2:]))[-16:][:8]),16)+1)]).flatten()
        PSB_POS = PSB_POS + ((PSB_POS-MAP[0])%8)
        PSB_POS_READ = []
        for i in range(0,np.size(PSB_POS)) :
            j =p.read_memory(int(PSB_POS[i]),ctypes.c_ulonglong()).value
            if j > MAP[0] and j < MAP[1]:
                PSB_POS_READ.append(j)
        PSB_POS = np.flip(np.stack(np.unique(np.array((PSB_POS_READ)),return_counts=True)).T)
        PSB_POS = PSB_POS[PSB_POS[:,0].argsort()][:,1]
        for i in PSB_POS.tolist():
            Temp_List =[i]
            for k in range(0,3):
                i = p.read_memory(i,ctypes.c_ulonglong()).value
                Temp_List.append(i)
            if Temp_List[3] == 4958897824200426312 :
                break
        else:
            continue
        break
    ON_RD_RH_RD,ON_TXT_RH_TXT,ON_DT_RH_DT = D_GET_ON_RD_RH_RD(ppro,Temp_List)
    REDRAW_BIN_Adrss = D_REDRAW_BIN_Adrss (ppro,ON_RD_RH_RD)
    FLAG_OBJ,FLAG_OBJ_ADRSS,OBJ_TYPE = D_FLAG_OBJ(ON_TXT_RH_TXT,ON_RD_RH_RD,ON_DT_RH_DT,Temp_List[0],REALOAD_FLAG,(Temp_List[0]+16112))
    VISUAL_FLAG =(p.read_memory(Temp_List[0]+2576,ctypes.c_longlong())).value-195152
    print ("Rhino Doc Adrss = " + str(hex(Temp_List[0])))
    print ("Rhino Doc Flag = " + str(hex(Temp_List[1])))
    print ("Rhino Obj List Adrss = " + str(hex(Temp_List[0]+16112)))
    print ("Rhino GUID List Adrss = " + str(hex(Temp_List[0]+16104)))
    print ("Visual Flag = " + hex(VISUAL_FLAG))
    print ("RDATA of opennurbs.dll = " + " to ".join([hex(i) for i in ON_RD_RH_RD[0]]))
    print ("RDATA of rhcommon_c.dll = " + " to ".join([hex(i) for i in ON_RD_RH_RD[1]]))
    print ("RDATA of rhinocore.dll = " + " to ".join([hex(i) for i in ON_RD_RH_RD[2]]))
    print ("RDATA of tl.dll = " + " to ".join([hex(i) for i in ON_RD_RH_RD[3]]))
    print ("TXT of opennurbs.dll = " + " to ".join([hex(i) for i in ON_TXT_RH_TXT[0]]))
    print ("TXT of rhcommon_c.dll = " + " to ".join([hex(i) for i in ON_TXT_RH_TXT[1]]))
    print ("TXT of rhinocore.dll = " + " to ".join([hex(i) for i in ON_TXT_RH_TXT[2]]))
    print ("TXT of tl.dll = " + " to ".join([hex(i) for i in ON_TXT_RH_TXT[3]]))
    print ("DATA of opennurbs.dll = " + " to ".join([hex(i) for i in ON_DT_RH_DT[0]]))
    print ("DATA of rhcommon_c.dll = " + " to ".join([hex(i) for i in ON_DT_RH_DT[1]]))
    print ("DATA of rhinocore.dll = " + " to ".join([hex(i) for i in ON_DT_RH_DT[2]]))
    print ("DATA of tl.dll = " + " to ".join([hex(i) for i in ON_DT_RH_DT[3]]))
    print ("REDRAW BIN Adrss = " + hex(REDRAW_BIN_Adrss))
    print('\033[95m' + "\nLIST OF FLAG\n-------------"+ '\033[0m')
    [[print("FLAG "  + ("0" + str(j))[-2:] + " OF "+ OBJ_TYPE[i][0] +" = "+ hex(FLAG_OBJ_ADRSS[i][j])) for j in range(len(FLAG_OBJ_ADRSS[i])) ]  for i in range(len(FLAG_OBJ_ADRSS))]
    OBJECT_LIST_FLAG_BYTES = [[bytes(ctypes.c_longlong(j)) for j in i]  for i in FLAG_OBJ_ADRSS]
    ATTRIBUT_FLAG = D_FIND_UNDER_OBJ (Temp_List[0],p)
    ATTRIBUT_FLAG_BYTES = [bytes(ctypes.c_longlong(int(i))) if True else np.nan for i in ATTRIBUT_FLAG[:,0].tolist()]
    print('\033[95m' + "\nLIST OF ATTRIBUT FLAG\n---------------------"+ '\033[0m')
    [print("Flag of "+i[1]+" = " +hex(int(i[0])))  for i in ATTRIBUT_FLAG.tolist()]
    return Temp_List[0],Temp_List[1],Temp_List[0]+16112,ON_RD_RH_RD,FLAG_OBJ_ADRSS,ATTRIBUT_FLAG,OBJECT_LIST_FLAG_BYTES,ATTRIBUT_FLAG_BYTES,VISUAL_FLAG,REDRAW_BIN_Adrss,ON_TXT_RH_TXT,ON_DT_RH_DT,OBJ_TYPE,Temp_List[0]+16104
def D_LIST_OBJ (RHN_OBJ_Adrss,OBJECT_LIST_FLAG,OBJECT_LIST_FLAG_IPT) :
    print('\033[95m' + "\nOBJ LIST LOOK\n-------------"+ '\033[0m')
    Head_Obj = p.read_memory(RHN_OBJ_Adrss,ctypes.c_longlong()).value
    if  int(Head_Obj) != 0 :
        OBJ_LIST_Adrss = [Head_Obj]
        Head_Obj = D_RTN_VAL(Head_Obj,128)
        while int(Head_Obj) != 0 :
            OBJ_LIST_Adrss.append(Head_Obj)
            Head_Obj = D_RTN_VAL(Head_Obj,128)
        FLAG_SET_HEADER = [i[0] for i in OBJECT_LIST_FLAG]
        for i in OBJ_LIST_Adrss :
            m =  p.read_memory(i,ctypes.c_longlong()).value
            for j in range(len(FLAG_SET_HEADER)) :
                if  m == int(FLAG_SET_HEADER[j]) :
                    print(OBJECT_LIST_FLAG_IPT[j]+ " find in " +hex(i))
                    break
        LIST_MAP_OBJ_STOR = D_OBJ_MAP_STOR(OBJ_LIST_Adrss)
        print("STORAGED IN " + str(len(LIST_MAP_OBJ_STOR)) + " MAPS MEMORY")
        [print("MAP " + str(i)+ " BETWEEN " + str(hex(LIST_MAP_OBJ_STOR[i][0]))+ " TO " +str(hex(LIST_MAP_OBJ_STOR[i][0]))) for i in range (0,len(LIST_MAP_OBJ_STOR))]
        return np.array(OBJ_LIST_Adrss), LIST_MAP_OBJ_STOR
    else :
        print("No Object draw")
        return nan,nan
def D_LOAD_TEMP (PID) :
    TEMP_LOAD,ASK_LOAD =np.zeros((14)).astype(str).tolist(),False
    TEMP_PATH =os.path.expandvars(r'%TEMP%')+"/onihr/onihr.npy"
    if os.path.exists(TEMP_PATH): 
        TEMP = np.load(TEMP_PATH,allow_pickle=True)
        if TEMP[0] == PID :
            TEMP_LOAD,ASK_LOAD = TEMP,True
    print('\033[91m' +"\nTemporary file read"+ '\033[0m') if ASK_LOAD else print('\033[91m' +"\nTemporary unread  file"+ '\033[0m')
    if ASK_LOAD :
        D_LOAD_TEMP_PRINT(TEMP_LOAD)
    return np.append(TEMP_LOAD[1:],ASK_LOAD==False)
def D_LOAD_TEMP_PRINT (TEMP_LOAD) :
    print('\033[95m' +"\nINITIALISTION OF VALUE\n----------------------"+ '\033[0m')
    print ("Rhino Doc Adrss = " + str(hex(TEMP_LOAD[1])))
    print ("Rhino Doc Flag = " + str(hex(TEMP_LOAD[2])))
    print ("Rhino Obj List Adrss = " + str(hex(TEMP_LOAD[3])))
    print ("Rhino GUID List Adrss = " + str(hex(TEMP_LOAD[13])))
    print ("Visual List Adrss = " + str(hex(p.read_memory(TEMP_LOAD[1]-21616,ctypes.c_longlong()).value)))
    print ("RDATA of opennurbs.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[4][0]]))
    print ("RDATA of rhcommon_c.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[4][1]]))
    print ("RDATA of rhinocore.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[4][2]]))
    print ("RDATA of tl.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[4][3]]))
    print ("TXT of opennurbs.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[11][0]]))
    print ("TXT of rhcommon_c.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[11][1]]))
    print ("TXT of rhinocore.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[11][2]]))
    print ("TXT of tl.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[11][3]]))
    print ("DATA of opennurbs.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[12][0]]))
    print ("DATA of rhcommon_c.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[11][1]]))
    print ("DATA of rhinocore.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[11][2]]))
    print ("DATA of tl.dll = " + " to ".join([hex(i) for i in TEMP_LOAD[12][3]]))
    print ("REDRAW BIN Adrss = " + str(hex(TEMP_LOAD[10])))
    print('\033[95m' + "\nLIST OF FLAG\n-------------"+ '\033[0m')
    FLAG_OBJ,FLAG_OBJ_ADRSS,OBJ_TYPE = D_FLAG_OBJ(TEMP_LOAD[4],TEMP_LOAD[11],TEMP_LOAD[12],TEMP_LOAD[1],False,TEMP_LOAD[3])
    [[print("FLAG "  + ("0" + str(j))[-2:] + " OF "+ OBJ_TYPE[i][0] +" = "+ hex(FLAG_OBJ_ADRSS[i][j])) for j in range(len(FLAG_OBJ_ADRSS[i])) ]  for i in range(len(FLAG_OBJ_ADRSS))]
    print('\033[95m' + "\nLIST OF ATTRIBUT FLAG\n---------------------"+ '\033[0m')
    [print("Flag of "+i[1]+" = " +hex(int(i[0])))  for i in TEMP_LOAD[6].tolist()]
def D_SAVE_TEMP (VALUE) :
    TEMP_PATH= os.path.expandvars(r'%TEMP%')+"/onihr/onihr.npy"
    np.save(TEMP_PATH,D_SAVE_TEMP)
    print('\033[91m' +"\nTemporary file save"+ '\033[0m')
    np.save(TEMP_PATH,VALUE)
def D_GET_ON_RD_RH_RD (ppro,RHINO_ADRESS) :
    a,data = [(i.path).split("\\")[-1] for i in ppro.memory_maps(False)],[]
    [data.append(ppro.memory_maps(False)[i]) if a[i] in ["RhinoCore.dll","tl.dll","opennurbs.dll","rhcommon_c.dll"] else None for i in range (0,len(a))]
    data =np.array((sorted(zip(data,[i.path.split("\\")[-1]for i in data]), key = lambda x: x[1])))[:,0]
    data_2 =[[data[0].addr,data[0].rss,data[0].path]]
    [[data_2[-1].append(c) for c in [i.addr,i.rss,i.path]] if i.path == data_2[-1][-1] else data_2.append([i.addr,i.rss,i.path]) in data_2 for i in data[1:]]
    return [i[0] for i in sorted(zip([[int(i[6],16),int(i[6],16) + int(i[7])] for i in data_2], [i[2].split("\\")[-1].upper() for i in data_2]), key = lambda x: x[1])],[i[0] for i in sorted(zip([[int(i[3],16),int(i[3],16) + int(i[4])] for i in data_2], [i[2].split("\\")[-1].upper() for i in data_2]), key = lambda x: x[1])],[i[0] for i in sorted(zip([[int(i[9],16),int(i[9],16) + int(i[10])] for i in data_2], [i[2].split("\\")[-1].upper() for i in data_2]), key = lambda x: x[1])]
def D_REDRAW_BIN_Adrss (ppro,ON_RD_RH_RD) :
    ALL_MAP=[int(i.addr,16) for i in ppro.memory_maps(False)]
    return ALL_MAP[ALL_MAP.index(ON_RD_RH_RD[2][0])+1]+5640
def D_OBJ_MAP_STOR (Head_OBJ) :
    LIST_OF_MAP,LIST_OF_MAP_USE = p.list_mapped_regions(full_mode = True),[]
    for i in Head_OBJ :
        USED = True
        for c in LIST_OF_MAP_USE :
            if c[0]<i and i<c[1] :
                USED = False
                break
        if USED :
            for c in LIST_OF_MAP :
                if c[0]<i and i<c[1] :
                    LIST_OF_MAP_USE.append(c)
                    break
    return LIST_OF_MAP_USE
def D_OBJ_MAP (ADRESS) :
    for i in p.list_mapped_regions(full_mode = True) :
        if i[0]<ADRESS and ADRESS<i[1] :
            OBJ_MAP = [i]
            break
    return OBJ_MAP
def D_OPEN_MALLOC (LIST_MAP_OBJ_STOR,p) :
    print('\033[95m' +"\nMALLOC LOOK\n-----------"+ '\033[0m')
    j=np.zeros((),dtype=np.int64)
    print(LIST_MAP_OBJ_STOR)
    for t in LIST_MAP_OBJ_STOR :
        m,dump = (p.search_all_memory(ctypes.c_byte(int("3D",16)),Memory_map=[t],array_way=True,offset=[7,8],bit_found=True,bit_want=True,bit_offset=0,dump_out=True))
        m=np.array((m))
        m=m[m<int(len(dump[0])+t[0]-800)].tolist()
        m = np.array((m))[np.array(([any([bool(int(dump[0][int((i-t[0])/8)+800])),bool(int(dump[0][int((i-t[0])/8)+784]))]) for i in m]))]
        j = np.append(j,m)
    print ("OBJ MALLOC NUMBR ADRESS = " +  str(np.size(j))+" from " + str(hex(j[1]))+" to "+ str(hex(j[-1])))
    m_open = [x for x in [i if 0==p.read_memory(i+8,ctypes.c_longlong()).value else np.nan for i in j[1:].tolist()] if str(x) != 'nan']
    print ("OBJ MALLOC NUMBR ADRESS = " +  str(np.size(m_open))+" from " + str(hex(m_open[0]))+" to "+ str(hex(m_open[-1])))
    return(m_open,j[1:],p)
def D_FIND_UNDER_OBJ (RHN_DOC_Adrss,p): #OBJECT_LIST_FLAG[4,2]
    ADRESS_STRD_LIGHT = RHN_DOC_Adrss+11224
    return np.array(([p.read_memory(ADRESS_STRD_LIGHT+i,ctypes.c_longlong()).value for i in [152,288,376,440,464,688,720]], ["ON_ModelComponent","ON_Visual","CRhino_ObjectAttribut","ON_MaterialRef","ON_MappingRef","ON_SimpleArray","ON_SimpleArray_DisplayMaterialRef"])).T
def D_REALLOC (OBJ_MAP,pm,SIZE):
    print('\033[95m' +"\nREALLOC LOOK\n------------"+ '\033[0m')
    REALLOC_ADRESS=pm.allocate(SIZE)
    pm.write_bytes(REALLOC_ADRESS,p.dump_map(ctypes.c_byte(),OBJ_MAP,bin_out=False)[0],SIZE)
    print("REALOC DO FROM OLD ADRESS = " + str(hex(ADRESS))+ " TO NEW = " + str(hex(REALLOC_ADRESS)))
    return REALLOC_ADRESS
def D_LINE_TYPE (RHN_DOC_Adrss):
    LINE_TYPE = p.read_memory(RHN_DOC_Adrss+1696,ctypes.c_longlong()).value
    LINE_TYPE = ["".join([chr(y) for y in u.split(b'\x00\x00')[0][::2]]) for u in p.dump_map(ctypes.c_byte(),[(h,h+80) for h in [p.read_memory(j+136,ctypes.c_longlong()).value for j in [p.read_memory(i*8+LINE_TYPE,ctypes.c_longlong()).value for i in range(0,p.read_memory(RHN_DOC_Adrss+1704,ctypes.c_long()).value)]]],bin_out=False)]
    LINE_TYPE = [["Continious","ByParent","ByLayer"]+LINE_TYPE] + [[-1,-1,-1]+list(range(0,len(LINE_TYPE)))]
    print('\033[95m' +"\nLINE TYPE\n---------"+ '\033[0m')
    [print("Line Type "+ LINE_TYPE[0][i]+" = "+ str(LINE_TYPE[1][i])) for i in range(0,len(LINE_TYPE[0]))]
    return LINE_TYPE
def D_LAYER_TABLE (RHN_DOC_Adrss):
    LAYER_TABLE = p.read_memory(RHN_DOC_Adrss+2480,ctypes.c_longlong()).value
    LAYER_TABLE = (np.array(([[k[z[0]:z[1]] for z in [[64,80],[80,96],[136,144],[0,0]] for k in m] for m in [p.dump_map(ctypes.c_byte(),[(h,h+280) for h in [p.read_memory(i*8+LAYER_TABLE,ctypes.c_longlong()).value for i in range(0,p.read_memory(RHN_DOC_Adrss+2488,ctypes.c_long()).value)]],bin_out=False)]])).reshape(4,-1)).tolist()
    LAYER_TABLE[2] = ["".join([chr(o) for o in m.split(b'\x00\x00')[0][::2]]) for m in [p.dump_map(ctypes.c_byte(),[(h,h+200) for h in [int.from_bytes(i,byteorder='little') for i in LAYER_TABLE[2]]],bin_out=False)][0]]
    for i in range(0,len(LAYER_TABLE[2])) :
        c,d = LAYER_TABLE[1][i],LAYER_TABLE[2][i]
        while True :
            if c == b'' :
                LAYER_TABLE[3][i]=d
                break
            else:
                c=LAYER_TABLE[0].index(c)
                c,d=LAYER_TABLE[1][c],LAYER_TABLE[2][c]+"::"+d
    LAYER_TABLE=list(sorted(zip(LAYER_TABLE[0],LAYER_TABLE[2],LAYER_TABLE[3],np.arange(0,len(LAYER_TABLE[0]))), key = lambda x: x[2]))
    print('\033[95m' +"\nLAYER TABLE\n-----------"+ '\033[0m')
    [print(i[2] + " | index = "+ str(i[3])+"\nGUID = " +hex(int.from_bytes(i[0],byteorder='little')))   for i in LAYER_TABLE]
    return LAYER_TABLE
def D_HATCH_PATTERN (RHN_DOC_Adrss):
    HATCH_PATTERN = p.read_memory(RHN_DOC_Adrss+12288,ctypes.c_longlong()).value
    HATCH_PATTERN = np.array((["".join([chr(e) for e in z.split(b'\x00\x00')[0]][::2]) for z in [p.dump_map(ctypes.c_byte(),[(h,h+80) for h in [p.read_memory(t+136,ctypes.c_longlong()).value for t in [p.read_memory(i*8+HATCH_PATTERN,ctypes.c_longlong()).value for i in range(0,p.read_memory(RHN_DOC_Adrss+12296,ctypes.c_long()).value)]]],bin_out=False)][0]]))
    HATCH_PATTERN = np.concatenate((HATCH_PATTERN[np.newaxis,:],np.arange(0,np.size(HATCH_PATTERN))[np.newaxis,:])).tolist()
    print('\033[95m' +"\nHATCH PATTERN\n-------------"+ '\033[0m')
    [print("Hatch Pattern "+ HATCH_PATTERN[0][i]+" = "+ str(HATCH_PATTERN[1][i])) for i in range(0,len(HATCH_PATTERN[0]))]
    return HATCH_PATTERN
def D_CLEAN_OBJ (OBJECT_ARY) :
    if OBJECT_ARY [2] != 'nan' and len(OBJECT_ARY) != 16:
            OBJECT_ARY [2]= np.random.bytes(16)
    if OBJECT_ARY [4] == 'nan' :
        OBJECT_ARY [4] =0
    return OBJECT_ARY
def D_FLAG_OBJ (ON_TXT_RH_TXT,ON_RD_RH_RD,ON_DT_RH_DT,RHN_DOC_Adrss,REALOAD_FLAG,OBJ_LIST_Adrss) :
    RD_TXT_DT_ON_RH = ON_TXT_RH_TXT + ON_RD_RH_RD + ON_DT_RH_DT
    OBJ_TYPE = np.array((["CIR",[[0]]],["CRV",[[0]]],["DIA",[[0],[184,0],[184,56],[184,208]]],["DIL",[[0],[184,0],[184,56],[184,208],[192,440],[192,1096]]],["DIO",[[0],[184,0],[184,56],[184,208]]],["DIR",[[0],[184,0],[184,56],[184,208]]],["DOT",[[0]]],["HAT",[[0],[176],[184,0,8,0]]],["LED",[[0],[184,0],[184,56],[184,208]]],["LIN",[[0]]],["MSH",[[0],[16],[40],[64],[88],[112],[304],[328],[504],[528],[552],[664],[840],[864],[912],[936],[960],[984]]],["PLI",[[0],[24],[48]]],["PLS",[[0],[24],[32,0,0],[48],[56,0,0],[72],[80,0,0],[80,8,0],[80,8,0],[96],[104,0],[104,56],[104,88],[120],[128,0],[128,24,0],[128,104,0],[144],[152,0],[152,24,0],[152,136],[168],[176,0],[176,32],[192],[200,0],[200,24,0],[200,56]]],["PTC",[[0],[16],[40],[64],[88],[112]]],["PTS",[[0]]],["SUB",[[0]]],["TEX",[[0],[184,0],[184,56],[184,208]]]))
    if os.path.exists(os.path.expandvars(r'%TEMP%')+"\\onihr\\"+hex(uuid.getnode())+".npy") and REALOAD_FLAG == False :
        FLAG_GEO_VALUE = np.load(os.path.expandvars(r'%TEMP%')+"\\onihr\\"+hex(uuid.getnode())+".npy",allow_pickle=True)
    else :
        NOTES = p.read_memory(RHN_DOC_Adrss+14672,ctypes.c_longlong()).value
        if "".join([chr(p.read_memory(NOTES+i,ctypes.c_byte()).value) for i in range (0,20,2)])=="ONIHRSETUP" :
            OBJ_LIST_Adrss = [p.read_memory(int(OBJ_LIST_Adrss),ctypes.c_longlong()).value]
            Head_Obj = p.read_memory(int(OBJ_LIST_Adrss[0])+128,ctypes.c_longlong()).value
            while int(Head_Obj) != 0 :
                OBJ_LIST_Adrss.append(Head_Obj)
                Head_Obj = p.read_memory(Head_Obj+128,ctypes.c_longlong()).value
            k = ["".join([chr(int(p.read_memory(j+m,ctypes.c_byte()).value)) for m in range (0,6,2)]) for j in [p.read_memory(int(i)+288,ctypes.c_longlong()).value for i in OBJ_LIST_Adrss]]
            k,FLAG_GEO,FLAG_GEO_VALUE=sorted(zip(k,OBJ_LIST_Adrss),key=lambda x: x[0]),[],[]
            for i in k :
                if i[0] not in FLAG_GEO :
                    FLAG_GEO.append(i[0])
                    FLAG_GEO_OBJ = [p.read_memory(int(i[1]),ctypes.c_longlong()).value]
                    GEO_PTR = p.read_memory(int(i[1])+64,ctypes.c_longlong()).value
                    for y in OBJ_TYPE[np.where(OBJ_TYPE[:,0]==i[0])[0][0]][1] :
                        FLAG_MOVE = GEO_PTR
                        for g in y :
                            FLAG_MOVE=p.read_memory(FLAG_MOVE+g,ctypes.c_longlong()).value
                        FLAG_GEO_OBJ.append(FLAG_MOVE)
                    FLAG_GEO_VALUE.append(FLAG_GEO_OBJ)
            FLAG_GEO_VALUE=np.array(([[[[e,j-RD_TXT_DT_ON_RH[e][0]] for e in [[np.sort([m if RD_TXT_DT_ON_RH[m][0]<j<RD_TXT_DT_ON_RH[m][1] else j for m in range(0,len(RD_TXT_DT_ON_RH))])[0]][0]]] for j in i]for i in FLAG_GEO_VALUE]))
            if os.path.exists(os.path.expandvars(r'%TEMP%')+"/onihr/") == False :
                os.makedirs(os.path.expandvars(r'%TEMP%')+"/onihr/")
            else :
                np.save((os.path.expandvars(r'%TEMP%')+"\\onihr\\"+hex(uuid.getnode())+".npy"),FLAG_GEO_VALUE)
        else :
            raise NameError('\033[91m' +"OPEN SETUP FILE"+ '\033[0m')
    return FLAG_GEO_VALUE ,[[RD_TXT_DT_ON_RH[j[0][0]][0]+j[0][1] for j in i] for i in FLAG_GEO_VALUE.tolist()],OBJ_TYPE
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||OBJ ATTRIBUT
def D_GET_VISUAL_LIST(Adress) :
    for a in [0,320,8] :
        Adress = p.read_memory(Adress+a,ctypes.c_longlong()).value
    return Adress
def D_CREATE_VISUAL_LIST(VISUAL_FLAG,pm) :
    Adress=pm.allocate(56)
    Value = b''.join([D_TO_BY(i) for i in [b'\x05\x00\x00\x00\x00\x00\x00\x00\x90\x00\x00\x00\x00\x00\x00\x00\xfc\xff\x03\x00\x01\x00\x00\x00',VISUAL_FLAG,pm.allocate(320),0,0]])
    pm.write_bytes(Adress,Value,56)
    return Adress
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||GEO ATTRIBUT
def D_POINT_2_OBJ_POINT (Point,FLAG) :
    return b''.join([D_TO_BY(i) for i in [FLAG[14][0],0,Point[1].astype(float)]]),np.array((5)),np.array((5))
def D_PTC_2_OBJ_PTC_CALL (PTC) :
    return np.cumsum([17]+[i for i in [len(PTC[1][0])*3 if len(PTC[1][c]) !=0 else None for c in range(0,3)] if i is not None])
def D_PTC_2_OBJ_PTC (PTC,FLAG) :
    for i in range (1,2) :
        if  len(PTC[1][i]) != len(PTC[1][0]) :
            PTC[1][i] = [] 
    PROC_D_PTC_2_OBJ_PTC_CALL = executor.submit(D_PTC_2_OBJ_PTC_CALL,PTC)
    a= FLAG[13][1]+D_TO_BY(0)+b''.join([(FLAG_DISPACHEUR(FLAG[13],i) if i[1]!=0 else FLAG_DISPACHEUR_NO_CALL(FLAG[13],i)) for i in [[2,len(PTC[1][0])],[3,len(PTC[1][1])],[4,len(PTC[1][2])],[5,0],[6,0]]])
    a=a+b''.join([b''.join([D_TO_BY(np.array((i),dtype=np.float64)) for i in PTC[1][c]]) if len(PTC[1][c])!=0 else b'' for c  in range(0,3)])
    CALL = PROC_D_PTC_2_OBJ_PTC_CALL.result()
    print(a,CALL,CALL[-1])
    return a,CALL,CALL[-1]
def D_CIRCLE_2_OBJ_CIRCLE (Cicle,FLAG) :
    Cicle=np.array(([i.astype(float) if type(i)==np.ndarray else float(i) for i in Cicle.tolist()]))
    return b''.join([D_TO_BY(i) for i in [FLAG[0][1],0,0,Cicle[2],Cicle[0],Cicle[1],np.tile(np.cross(Cicle[0],Cicle[1]).T,(2,1)).flatten(),0,Cicle[3],0,Cicle[4]*np.pi,0,Cicle[4]*np.pi*Cicle[3],0]]),np.array((26)),np.array((26))
def D_CONT_MESH (MESH) :
    CONT=np.insert(np.cumsum(np.array(([227,len(MESH[0])*3,math.ceil(len(MESH[0])*3/2),len(MESH[1])*2,math.ceil(len(MESH[0])*3/2),math.ceil(len(MESH[1])*3/2)]))),-1,0)
    return CONT, CONT[-1]
def D_MESH_2_OBJ_MESH_SPEED (MESH) :
    return np.sort(MESH[0].reshape(-1,3),axis=0)[[0,-1]].flatten().astype(float)
def D_MESH_2_OBJ_MESH (MESH,FLAG) :
    PROC_D_CONT_MESH = executor.submit(D_CONT_MESH,MESH)
    PROC_D_MESH_2_OBJ_MESH_SPEED = executor.submit(D_MESH_2_OBJ_MESH_SPEED,MESH)
    VECT_FACE=np.array(([((d**2/np.abs(np.sum(d**2)))**0.5*np.sign(d))for d in [np.cross(i[0],i[1])  for i in (np.diff(MESH[0][MESH[1]].reshape(-1,3)[np.tile(np.array((1,0,2,0)),len(MESH[1])).reshape(-1,4)+(np.arange(0,len(MESH[1]))*3).reshape(-1,1)].reshape(-1,2,3),axis=1).reshape(-1,2,3)).tolist()]]))
    VECT_VEX = np.array(([np.average(VECT_FACE[c],axis=1).tolist() for c in [np.floor((np.array((np.where(i==MESH[1].flatten())))/3)).astype(int) for i in range(0,np.max(MESH[1])+1)]]))
    VECT_FACE_BYTES,VECT_VEX_BYTES = [d if len(d)%8==0 else d+b'\x00\x00\x00\x00' for d in [b''.join([bytes(ctypes.c_float(i))[::-1] for i in c.flatten().tolist()]) for c in [VECT_FACE,VECT_VEX]]]
    MESH_DOMAIN = PROC_D_MESH_2_OBJ_MESH_SPEED.result()
    a = b''.join([D_TO_BY(i) for i in [FLAG[10][1],0,FLAG[10][2],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT(len(MESH[0])),FLAG[10][3],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT(len(MESH[0])),FLAG[10][4],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT(len(MESH[1])),FLAG[10][5],0,0,FLAG[10][6],np.zeros((23)),FLAG[10][7],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',len(VECT_VEX.tolist()),FLAG[10][8],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',len(VECT_FACE.tolist())]])
    a = a + b''.join(([D_TO_BY(i) for i in [0]*4+[1.0]]*4)[1:])
    a = a + b''.join(([D_TO_BY(i) for i in [FLAG[10][9],0,0,FLAG[10][10],0,0,FLAG[10][11],0,0,b''.join([b'\xd2\x1d3\x9e\xbd\xf8\xe5\xff']*4),0,0,b''.join([b'\xd2\x1d3\x9e\xbd\xf8\xe5\xff']*4),b'@\x15\xfd\xbb\xe9\xbb\xa7\x00',FLAG[10][12],0]]))
    a = a + b''.join(([D_TO_BY(i) for i in [0]*4+[1.0]]*4))
    a = a + b''.join([D_TO_BY(i) for i in [FLAG[10][13],0,0,FLAG[10][14],np.zeros(4),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',FLAG[10][15],0,0,FLAG[10][16],0,0,FLAG[10][17],0,0,FLAG[10][18],np.zeros((8))]])
    a = a + b''.join([D_TO_BY(i) for i in [MESH_DOMAIN,b''.join([bytes(ctypes.c_float(i)) for i in [1,1,1,-1,-1,-1,-1,-1,0,0]])]])+b''.join(([b'\xd2\x1d3\x9e\xbd\xf8\xe5\xff']*6+[b'\xda9\xa3\xee^kK\r2U\xbf\xef\x95`\x18\x90\xaf\xd8\x07\t\x00\x00\x00\x00'])*8) + D_TO_BY(np.zeros((24)).astype(int))
    a = a + b''.join([b''.join([D_TO_BY(i.astype(float)) for i in c]) for c in MESH[0]])
    
    a = a + b''.join([b''.join([bytes(ctypes.c_float(i)) for i in c]) for c in MESH[0]]) + (b'' if len(MESH[0])*3%2 == 0 else b'\x00\x00\x00\x00')
    a = a + b''.join([b''.join([D_ARRAY_COUNT_MULTI(i) for i in np.array((c+[c[-1]])).reshape(2,2).tolist()]) for c in MESH[1].astype(int).tolist()])
    a = a + VECT_VEX_BYTES + VECT_FACE_BYTES
    #a = a + b''.join([D_ARRAY_COUNT_MULTI(i) for i in (MESH[1].reshape(-1,2).astype(int).tolist() if np.size(MESH[1])%2 == 1 else np.concatenate((MESH[1].flatten(),np.zeros(0))).reshape(-1,2).astype(int).tolist())])
    CONT_MESH,SIZE = PROC_D_CONT_MESH.result()
    return a,CONT_MESH,SIZE
def D_LINE_2_OBJ_LINE (Line,FLAG) :
    D_PRINT_FROM(b''.join([D_TO_BY(i) for i in [FLAG[9][1],0,0,Line[0][:3].astype(float),Line[0][3:].astype(float),D_CUM_DOM(Line[0][::-1]).astype(float),0]]))
    return b''.join([D_TO_BY(i) for i in [FLAG[9][1],0,0,Line[0][:3].astype(float),Line[0][3:].astype(float),D_CUM_DOM(Line[0])[::-1].astype(float),3,0]]),np.array((13)),np.array((13))
def D_PLINE_2_OBJ_PLINE(PLine,FLAG,STD_BYTES_ARRAY,CALL_POST,S_SIZE):
    PROC_D_ARRAY_COUNT = executor.submit(D_ARRAY_COUNT,int(np.size(PLine)/3))
    PROC_D_ARRAY_POINT = executor.submit(D_NP_TO_BY,PLine)
    DOM_CUM = D_NP_TO_BY(D_CUM_DOM(PLine))
    ARRAY_POINT = PROC_D_ARRAY_POINT.result()
    ARRAY_COUNT= PROC_D_ARRAY_COUNT.result()
    PLine = FLAG+STD_BYTES_ARRAY[0]+STD_BYTES_ARRAY[0]+FLAG+STD_BYTES_ARRAY[0]+ARRAY_COUNT+FLAG+STD_BYTES_ARRAY[0]+STD_BYTES_ARRAY[0]+ARRAY_COUNT+STD_BYTES_ARRAY[1]+STD_BYTES_ARRAY[0]+ARRAY_POINT+DOM_CUM
    CALL_POST.append((np.array((5,8,8,3+np.size(PLine))))+S_SIZE)
    return PLine, S_SIZE, CALL_POST
def D_CRVARY_2_OBJ_CRVARY (Curve) :
    return D_NP_TO_BY(np.append((Curve,0)))
def D_NURBS_2_OBJ_NURBS (FLAG,d) :
    d=np.array(d).flatten()
    while (np.size(d)-1)%3 != 0 :
        d = np.concatenate((d,np.ones((1)))) 
    S_SIZE = np.array(((np.size(d)-1)*4+9))
    CALL_POST=np.array((7,4,9,(np.size(d)-1)/3+1))
    DELTA_DOM = np.concatenate((D_CUM_DOM(d[:-1])[np.repeat(np.arange(0,int(len(d[:-1])/3)),2)[1:-1]],np.zeros((1)))).astype(float)
    return b''.join([D_TO_BY(i) for i in [FLAG[1][1],0,0,bytes(ctypes.c_longlong(3)),b''.join([D_ARRAY_COUNT_MULTI(i) for i in [[int(d[-1]+1),int((np.size(d)-1)/3)],[int((np.size(d)-1)/3+d[-1]-1),0]]]),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT_MULTI([3,int((np.size(d)-1)/3)*3]),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',DELTA_DOM,b''.join([D_TO_BY(k) for k in (d[:-1].astype(float).reshape(-1,3)).tolist()])]]),CALL_POST,S_SIZE
def D_CONT_BREP (ALL_SPLIT_CRV,ALL_EDGE,ALL_VERTEX,ALL_VERTEX_CONECTION,BREP,ALL_SPLIT_CRV_INDEX) :
    a,STEP=[39,len(ALL_SPLIT_CRV.tolist())]+[j for m in [[9,int(len(i)/3*2-1),len(i)] for i in ALL_SPLIT_CRV.tolist()] for j in m],[0]  #CRVARRAY_0
    a,STEP = a+[len(ALL_EDGE.tolist())]+[j for m in [[9,int(len(i)/3*2-1),len(i)] for i in ALL_EDGE.tolist()] for j in m],STEP+[len(a)-1]#CRVARRAY_1
    a,STEP = a+[len(BREP[0])]+np.repeat(27,len(BREP[0])).tolist(),STEP+[len(a)-1]#SFRARRAY
    a,STEP = a+[np.shape(ALL_VERTEX)[0]*11]+[len(i) for i in ALL_VERTEX_CONECTION],STEP+[len(a)-1]#BVERTXARREY
    a,STEP = a+[len(ALL_EDGE)*18]+np.ones((len(ALL_EDGE))).tolist(),STEP+[len(a)-1]#BEDGEARRAY #*16
    a,STEP = a+[len(ALL_SPLIT_CRV)*30] + np.repeat(17,len(ALL_SPLIT_CRV)).tolist(),STEP+[len(a)-1] #BTRIMARRAY #29
    a,STEP= a+[len(BREP[0])*15]+ [len(i) for i in ALL_SPLIT_CRV_INDEX],STEP+[len(a)-1] #BLOOPARRAY#14
    a,STEP =a+ [len(BREP[0])*28]+np.ones(len(BREP[0])).tolist(),STEP+[len(a)-1] #BFACEARRAY
    a = np.cumsum(a)
    a =  np.concatenate((a[STEP],np.delete(a,STEP)))#FIRSTLAYER
    a[8:8+len(ALL_SPLIT_CRV)*3] = a[8:8+len(ALL_SPLIT_CRV)*3][np.concatenate((np.arange(0,len(ALL_SPLIT_CRV))*3,np.arange(len(ALL_SPLIT_CRV)*3).reshape(-1,3).T[1:].T.flatten()))] #CRVARRAY_REBUILD
    a[7+STEP[1]:6+STEP[2]] = a[7+STEP[1]:6+STEP[2]][np.concatenate((np.arange(0,int((STEP[2]-STEP[1]-1)/3))*3,(np.arange((STEP[2]-STEP[1]-1)).reshape(-1,3).T[1:].T.flatten()))).astype(int)] #CRVARRAY_REBUILD
    a=np.insert(a, (np.repeat((-(np.arange(len(BREP[0])+1))[::-1]),2)[1:-1])-1,np.concatenate((a[STEP[2]+6:STEP[2]+6+len(BREP[0])],np.zeros((len(BREP[0]))))).reshape(2,-1).T.flatten())#BFACEARRAY
    a=np.insert(a,(STEP[7]+2)-(np.arange(len(BREP[0]))+1)[::-1],np.zeros((len(BREP[0]))))
    a=np.insert(a,np.repeat(((STEP[6]+2)-(np.arange(len(ALL_SPLIT_CRV)+1)))[::-1],2)[1:-1],np.concatenate((a[STEP[0]+8:STEP[0]+8+len(ALL_SPLIT_CRV)],np.zeros((len(ALL_SPLIT_CRV))))).reshape(2,-1).T.flatten()) #BEDGETRIM
    a=np.insert(a,np.repeat(((STEP[5]+3)-(np.arange(len(ALL_EDGE)+1)))[::-1],2)[1:-1],np.concatenate((a[STEP[1]+7:STEP[1]+7+len(ALL_EDGE)],np.zeros((len(ALL_EDGE))))).reshape(2,-1).T.flatten())#BEDGEARRAY
    return a
def D_BREP_2_OBJ_BREP_NO_MANIFOLD(ALL_CRV_COUNT) :
    return np.concatenate((ALL_CRV_COUNT[:,np.newaxis], np.arange(len(ALL_CRV_COUNT))[:,np.newaxis]),axis=1).tolist()
def D_BREP_2_OBJ_BREP_ARRAY_REFIND_LINE_INDEX (ALL_SPLIT_CRV_C,ALL_SPLIT_CRV_UNFLAT) :
    return [np.array((h)).reshape(-1,2).tolist() if len(h)%2 == 0 else np.array((h+[0])).reshape(-1,2).tolist() for h in [[np.where(np.all(j==ALL_SPLIT_CRV_C,axis=1))[0][0] for j in i] for i in ALL_SPLIT_CRV_UNFLAT]]
def D_BREP_2_OBJ_BREP_ALL_EDGE_LINK_VERTEX (ALL_EDGE,ALL_VERTEX) :
    return [[np.where(np.all(ALL_EDGE[l][i:i+3]==ALL_VERTEX,axis=1))[0][0]  for i in range(0,np.size(ALL_EDGE[l]),3)] for l in range(0,len(ALL_EDGE.tolist()))]
def D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_VERTEX (ALL_SPLIT_CRV,ALL_VERTEX) :
    return np.array(([[np.where(np.all(i==ALL_VERTEX,axis=1))[0][0] for i in (np.array((l)).reshape(-1,3)[[0,-1]]).tolist()] for l in ALL_SPLIT_CRV.tolist()]))
def D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_FACE (ALL_SPLIT_CRV_UNFLAT) :
    return np.repeat(np.arange(len(ALL_SPLIT_CRV_UNFLAT)),[len(i) for i in ALL_SPLIT_CRV_UNFLAT])
def D_BREP_2_OBJ_BREP_ALL_SPLIT_NO_MANIFOLD (ALL_SPLIT_CRV_UNFLAT,FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV) :
    COPY_FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV,CONT_DUP = [np.copy(FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV).tolist()[c[0]:c[1]] for c in (np.repeat(np.cumsum(([0]+[len(i) for i in ALL_SPLIT_CRV_UNFLAT])),2)[1:-1]).reshape(-1,2).tolist()],[]
    for l in range(0,len(COPY_FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV)) :
        for i in COPY_FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV[l] :
            DUP = np.size([np.where(np.all(i == FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV,axis=1))])
            CONT_DUP =CONT_DUP + [[l*(DUP!=1),DUP]]
    return CONT_DUP
def D_BREP_2_OBJ_BREP_LOOP_ARRAY (ALL_SPLIT_CRV_UNFLAT,P_BASE) :
    LOOP_ARRAY,SHAPE_LIST = [],([0]+[len(i) for i in ALL_SPLIT_CRV_UNFLAT])[:-1]
    for c in range(0,len(ALL_SPLIT_CRV_UNFLAT)) :
        locals()["D_BREP_2_OBJ_BREP_LOOP_ARRAY_SPEED" + str(c)] = executor.submit(D_BREP_2_OBJ_BREP_LOOP_ARRAY_SPEED,ALL_SPLIT_CRV_UNFLAT[c],SHAPE_LIST[c],P_BASE)
    for c in range(0,len(ALL_SPLIT_CRV_UNFLAT)) :
        LOOP_ARRAY = LOOP_ARRAY + locals()["D_BREP_2_OBJ_BREP_LOOP_ARRAY_SPEED" + str(c)].result()
    return LOOP_ARRAY
def D_BREP_2_OBJ_BREP_LOOP_ARRAY_SPEED (i,d,g) :
    Sort_list,Start_pt = [0],np.array(([c[:3] for c in i]))
    while True :
        Sort_list = Sort_list + [np.where(np.all(i[Sort_list[-1]][3:]==Start_pt,axis=1))[0][0]]
        if len(i) == len(Sort_list) :
            Sort_list = np.array((Sort_list))+d
            break
    return (Sort_list.reshape(-1,2)).tolist() if np.size(Sort_list)%2 == 0 else (np.array((Sort_list.tolist()+[0])).reshape(-1,2)).tolist()
def D_VERTEX_NORMAL(NORMAL,ALL_SPLIT_CRV_UNFLAT,ALL_VERTEX):
    ALL_SPLIT_CRV_UN_UNFLAT =np.array(([i for l in (([(np.array([np.concatenate((np.array(i).reshape(-1,3),(np.ones((np.shape(np.array(i).reshape(-1,3))[0]))*l)[:,np.newaxis]),axis=1).tolist() for i in ALL_SPLIT_CRV_UNFLAT[l]])).flatten().tolist() for l in range(0,len(ALL_SPLIT_CRV_UNFLAT))])) for i in l ])).reshape(-1,4)
    return [np.average(np.array(([g for j in np.array((NORMAL))[np.unique(ALL_SPLIT_CRV_UN_UNFLAT[np.where(np.all(c == ALL_SPLIT_CRV_UN_UNFLAT[:,:3],axis=1)),3]).astype(int).tolist()].tolist() for g in j])).reshape(-1,3),axis=0) for c in ALL_VERTEX]
def D_NORMAL_REFLIP (ALL_SPLIT_CRV_UNFLAT,NORMAL):
    ALL_FACE_VERTEX=([np.unique(np.array(([np.roll(l[:3*math.floor(len(l)/3)],3)[:6] for l in i])).reshape(-1,3),axis=0) for i in ALL_SPLIT_CRV_UNFLAT])
    ALL_FACE_VERTEX= np.array(([i for j in [np.concatenate((ALL_FACE_VERTEX[i],(np.ones(np.shape(ALL_FACE_VERTEX[i])[0])*i)[:,np.newaxis]),axis=1).flatten() for i in range(0,len(ALL_FACE_VERTEX))] for i in j])).reshape(-1,4)
    ALL_FACE_CONNECTION=np.unique([i for i in [q for p in [t for u in [[[None if k == ALL_FACE_VERTEX[i,3] else np.sort([int(ALL_FACE_VERTEX[i,3]),k]).tolist()] for k in (ALL_FACE_VERTEX[(np.where(np.all(ALL_FACE_VERTEX[i,:3]==ALL_FACE_VERTEX[:,:3],axis=1))[0].tolist()),3].astype(int).tolist())] for i in range(0,len(ALL_FACE_VERTEX.tolist()))] for t in u] for q in p] if i is not None],axis=0)
    ALL_FACE_CONNECTION,i,b=np.asarray(sorted(zip(ALL_FACE_CONNECTION.tolist(),np.array((ALL_FACE_CONNECTION))[:,0]), key = lambda x: x[1]))[:,0].flatten().tolist(),0,[]
    a=ALL_FACE_CONNECTION
    while True :
        b = b+[a[i]]
        del a[i]
        if len(a) == 0 :
            break
        for o in [1,0] :
            for c in b[::-1] :
                d=np.where(c[o]==np.array((a))[:,0])[0]
                if np.size(d)!=0 :
                    L,d=True,np.sort(d)[0]
                    break
            if L :
                break
    NORMAL=np.array((NORMAL))
    for i in ALL_FACE_CONNECTION :
        if np.sum(NORMAL[i[0]]*NORMAL[i[1]]) <= 0 :
            NORMAL[i[1]] = -NORMAL[i[1]]
    return NORMAL,ALL_FACE_CONNECTION,ALL_FACE_VERTEX
def D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV (ALL_SPLIT_CRV_UNFLAT,ALL_SPLIT_CRV_VEC,VECTEUR):
    ALL_SPLIT_CRV_VEC_UNFLAT = [np.copy(ALL_SPLIT_CRV_VEC)[c[0]:c[1]] for c in np.repeat(np.cumsum([0]+[len(i) for i in ALL_SPLIT_CRV_UNFLAT]),2)[1:-1].reshape(-1,2).tolist()]
    return [f for q in [[R.apply(R.from_matrix(D_rotation_matrix_from_vectors(np.array((1,0,0)),VECTEUR[j][0])),i).astype(float) if np.all(np.array((1,0,0))!=VECTEUR[j][0]) else i.astype(float)  for i in ALL_SPLIT_CRV_VEC_UNFLAT[j]] for j in range(0,len(ALL_SPLIT_CRV_VEC_UNFLAT))] for f in q]
def D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV_FROM_P_BASE (ALL_SPLIT_CRV_UNFLAT,ALL_SPLIT_CRV_VEC,VECTEUR,P_BASE):
    ALL_SPLIT_CRV_FROM_P_BASE = [[i[:math.floor(len(i)/3)*3][-3:]-P_BASE[l] for i in ALL_SPLIT_CRV_UNFLAT[l]]  for l in range (0,len(ALL_SPLIT_CRV_UNFLAT))]
    return [f for q in [[R.apply(R.from_matrix(D_rotation_matrix_from_vectors(np.array((1,0,0)),VECTEUR[j][0])),i).astype(float) if np.all(np.array((1,0,0))!=VECTEUR[j][0]) else i.astype(float) for i in ALL_SPLIT_CRV_FROM_P_BASE[j]] for j in range(0,len(ALL_SPLIT_CRV_FROM_P_BASE))] for f in q]
def D_BREP_2_OBJ_BREP(BREP,FLAG,STD_BYTES_ARRAY,CALL_POST,S_SIZE) :
    DOMAIN = [[np.cumsum([0]+[((p[0+i]-p[3+i])**2+(p[1+i]-p[4+i])**2+(p[2+i]-p[5+i])**2)**0.5 for i in range(0,np.size(p)-3,3)]) for p in m] for m in BREP[0]]
    VECTEUR,VECTEUR_AMP=  [np.array(([[[((i**2/np.abs(np.sum(i**2)))**0.5*np.sign(i)),np.sum(i**2)**0.5] for i in l] for l in [(i[0]-i[2],i[1]-i[3]) for i in [[np.sum((BREP[0][l][i][np.arange(6)+((np.nanargmin((DOMAIN[l][i]-(DOMAIN[l][i][-1]/2))**0.5)-1)*3)]).reshape(-1,3),axis=0)/2 for i in range(0,len(DOMAIN[l]))] for l in range(0,len(DOMAIN))]]]))[:,:,m] for m in range (0,2)]
    END_PT = [np.concatenate(i).reshape(-1,3) for i in [[BREP[0][l][i][np.arange(-3,3)] for i in range(0,len(BREP[0][l]))] for l in range(0,len(BREP[0]))]]
    P_BASE = [np.array(((END_PT[i][np.argsort(np.sum(((VECTEUR[i][0]+VECTEUR[i][1])*END_PT[i]),axis=1))[-1]]))).astype(float) for i in range(0,len(BREP[0]))]
    P_END = [np.array(((END_PT[i][np.argsort(np.sum(((VECTEUR[i][0]+VECTEUR[i][1])*END_PT[i]),axis=1))[0]]))).astype(float) for i in range(0,len(BREP[0]))]
    NORMAL = [np.cross(i[0],i[1]) for i in VECTEUR]
    ALL_PT = np.array(([np.unique(np.array(([f for q in [np.array((c)).flatten().tolist() for c in i.tolist()]for f in q])).reshape(-1,3),axis = 0).tolist() for i in np.array(np.array(BREP[0]))]))
    ALL_START = [[j[0:3] for j in i] for i in BREP[0]]
    ALL_LINE =  [np.concatenate((np.array((i)),np.roll(i,-1,axis=0)),axis=1).tolist() for i in ALL_START]
    ALL_SPLIT_CRV,ALL_SPLIT_CRV_UNFLAT,ALL_SPLIT_CRV_CONT_SPLIT = D_BREP_ALL_SPLIT_CRV (ALL_LINE,ALL_PT) 
    PROC_D_NORMAL_REFLIP=executor.submit(D_NORMAL_REFLIP,ALL_SPLIT_CRV_UNFLAT,NORMAL)
    PROC_D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_FACE = executor.submit(D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_FACE,ALL_SPLIT_CRV_UNFLAT)
    PROC_D_BREP_2_OBJ_BREP_LOOP_ARRAY = executor.submit(D_BREP_2_OBJ_BREP_LOOP_ARRAY,ALL_SPLIT_CRV_UNFLAT,P_BASE)    
    ALL_SPLIT_CRV_INDEX = D_BREP_2_OBJ_BREP_ARRAY_REFIND_LINE_INDEX(ALL_SPLIT_CRV,ALL_SPLIT_CRV_UNFLAT)
    ALL_SPLIT_CRV_VEC = [[i[0]-i[-3],i[1]-i[-2],i[2]-i[-1]]for i in ALL_SPLIT_CRV]
    ALL_SPLIT_CRV_VEC_AMP = [np.sum(np.array((i),dtype=np.float64)**2)**0.5 for i in ALL_SPLIT_CRV_VEC]
    PROC_D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV= executor.submit(D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV,ALL_SPLIT_CRV_UNFLAT,ALL_SPLIT_CRV_VEC,VECTEUR)
    PROC_D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV_FROM_P_BASE = executor.submit(D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV_FROM_P_BASE ,ALL_SPLIT_CRV_UNFLAT,ALL_SPLIT_CRV_VEC,VECTEUR,P_BASE)
    ALL_SPLIT_CRV_DOM = [np.cumsum([0]+[((p[0+i]-p[3+i])**2+(p[1+i]-p[4+i])**2+(p[2+i]-p[5+i])**2)**0.5 for i in range(0,np.size(p)-3,3)]) for p in ALL_SPLIT_CRV]
    ALL_CRV_DOM = np.sum(np.diff(ALL_SPLIT_CRV.reshape(-1,2,3),axis=1)**2,axis=2)**0.5
    ALL_EDGE_sort,ALL_EDGE = np.argsort(np.sum(np.argsort(np.copy(ALL_SPLIT_CRV).reshape(-1,2,3),axis=1)*np.array(([10,5,1])),axis=2),axis=1),np.copy(ALL_SPLIT_CRV).reshape(-1,2,3)
    ALL_EDGE[ALL_EDGE_sort[:,0]==1] = np.roll(ALL_EDGE,1,axis=1)[ALL_EDGE_sort[:,0]==1]
    FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV =ALL_EDGE.reshape(-1,6)
    PROC_D_BREP_2_OBJ_BREP_ALL_SPLIT_NO_MANIFOLD=executor.submit(D_BREP_2_OBJ_BREP_ALL_SPLIT_NO_MANIFOLD,ALL_SPLIT_CRV_UNFLAT,FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV)
    ALL_EDGE,ALL_SPLIT_CRV_TO_EDGE,ALL_CRV_COUNT = np.unique(np.copy((FOR_LINK_ALL_EDGE_ALL_SPLIT_CRV)).reshape(-1,6),axis=0, return_index=True, return_inverse=True,return_counts=True)[1:]
    ALL_EDGE,ALL_VERTEX_CONECTION= np.copy(ALL_SPLIT_CRV)[ALL_EDGE],[]
    PROC_D_BREP_2_OBJ_BREP_NO_MANIFOLD = executor.submit(D_BREP_2_OBJ_BREP_NO_MANIFOLD,ALL_CRV_COUNT)
    ALL_EDGE_DOM = np.sum(np.diff(ALL_EDGE.reshape(-1,2,3),axis=1)**2,axis=2)**0.5
    ALL_EDGE_VEC = [[i[0]-i[-3],i[1]-i[-2],i[2]-i[-1]]for i in ALL_EDGE]
    ALL_EDGE_VEC_AMP = [np.sum(np.array((i),dtype=np.float64)**2)**0.5 for i in ALL_EDGE_VEC]
    ALL_VERTEX,ALL_VERTEX_inv= np.unique(ALL_EDGE.reshape(-1,3), return_inverse=True, axis = 0)
    NORMAL,ALL_FACE_CONNECTION,ALL_FACE_VERTEX = PROC_D_NORMAL_REFLIP.result()
    BACKFACE = [-i for i in NORMAL]
    PROC_D_VERTEX_NORMAL = executor.submit(D_VERTEX_NORMAL,NORMAL,ALL_SPLIT_CRV_UNFLAT,ALL_VERTEX)
    [ALL_VERTEX_CONECTION[i[0]].append(i[1]) if len(ALL_VERTEX_CONECTION)> i[0] else [ALL_VERTEX_CONECTION.append(c) for c in ([[] for r in range (len(ALL_VERTEX_CONECTION),i[0])]+[[i[1]]])] for i in np.roll(np.repeat(ALL_VERTEX_inv,2).reshape(-1,4),-1,axis=1).reshape(-1,2).tolist()]
    B_BOX = np.array(([np.min(ALL_PT,axis=0), np.max(ALL_PT,axis=0)]),dtype=np.float64)
    B_BOX =(np.sort(np.array(([j for i in  [[j for i in  [k[:math.floor(len(k))] for k in t] for j in i] for t in BREP[0].tolist()] for j in i])).reshape(-1,3),axis=0)[[0,-1]]).astype(float)
    PROC_D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_VERTEX = executor.submit(D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_VERTEX, ALL_SPLIT_CRV,ALL_VERTEX)
    PROC_D_BREP_2_OBJ_BREP_ALL_EDGE_LINK_VERTEX =executor.submit(D_BREP_2_OBJ_BREP_ALL_EDGE_LINK_VERTEX,ALL_EDGE,ALL_VERTEX)
    ALL_SPLIT_CRV_VEC_UV = PROC_D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV.result()
    ALL_SPLIT_CRV_VEC_UV_FROM_P_BASE = PROC_D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV_FROM_P_BASE.result()
    ALL_SPLIT_CRV_TO_FACE =  PROC_D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_FACE.result()
    ALL_SPLIT_CRV_NO_MANIFOLD = PROC_D_BREP_2_OBJ_BREP_ALL_SPLIT_NO_MANIFOLD.result()
    ALL_EDGE_MANIFOLD  = PROC_D_BREP_2_OBJ_BREP_NO_MANIFOLD.result()
    ALL_EDGE_TO_VERTEX = PROC_D_BREP_2_OBJ_BREP_ALL_EDGE_LINK_VERTEX.result()
    ALL_SPLIT_CRV_TO_VERTEX = PROC_D_BREP_2_OBJ_BREP_ALL_SPLIT_CRV_LINK_VERTEX.result()
    ALL_VERTEX_NORMAL = PROC_D_VERTEX_NORMAL.result()+ALL_VERTEX
    CALL = D_CONT_BREP (ALL_SPLIT_CRV,ALL_EDGE,ALL_VERTEX,ALL_VERTEX_CONECTION,BREP,ALL_SPLIT_CRV_INDEX)
    LOOP_ARRAY = PROC_D_BREP_2_OBJ_BREP_LOOP_ARRAY.result()
    a= FLAG_DISPACHEUR_NO_CALL(FLAG[12],[1,len(BREP[0])])
    a= a+b''.join([(FLAG_DISPACHEUR(FLAG[12],i)) for i in [[2,np.size(ALL_SPLIT_CRV)/6],[2,len(ALL_EDGE)],[6,len(BREP[0])],[10,np.size(ALL_VERTEX)/3],[14,len(ALL_EDGE)],[18,len(ALL_SPLIT_CRV)],[22,len(BREP[0])],[25,len(BREP[0])]]])
    a=a+b''.join(D_TO_BY(i) for i in [B_BOX.flatten(),np.zeros((6),dtype=np.float64)])
    a=a+D_FAKE_FLAG(int(np.size(ALL_SPLIT_CRV)/6))
    a=a+b''.join([D_NURBS_2_OBJ_NURBS(FLAG,i)[0] for i in ALL_SPLIT_CRV.tolist()])
    a=a+D_FAKE_FLAG(len(ALL_EDGE))
    a=a+b''.join([D_NURBS_2_OBJ_NURBS(FLAG,i)[0] for i in ALL_EDGE ])
    a=a+D_FAKE_FLAG(len(BREP[0]))
    a=a+b''.join([b''.join([D_TO_BY(k) for k in l]) for l in ([[FLAG[12][7],0,0,P_END[i],VECTEUR[i][0],VECTEUR[i][1],NORMAL[i],NORMAL[i],np.zeros((2)),VECTEUR_AMP[i][0],0,VECTEUR_AMP[i][1],0,VECTEUR_AMP[i][0],0,VECTEUR_AMP[i][1]] for i in range(0,len(BREP[0]))])])#SFRARRAY
    a=a+b''.join([b''.join([b''.join([D_TO_BY(j) for j in[FLAG[12][11],0,ALL_VERTEX[l].astype(float),0,D_ARRAY_COUNT_MULTI([0,l]),FLAG[12][12],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT_MULTI([len(ALL_VERTEX_CONECTION[l]),math.ceil(len(ALL_VERTEX_CONECTION[l])/2)*2]),0]])])  for l in range(len(ALL_VERTEX))])+b''.join([D_ARRAY_COUNT_MULTI(i) for i in (np.array(([i for j in [[i]*len(ALL_VERTEX_CONECTION[i]) for i in range(0,len(ALL_VERTEX_CONECTION))] for i in j],[i for j in ALL_VERTEX_CONECTION for i in j])).T.tolist())])#BVERTXARREY
    a=a+b''.join([b''.join(D_TO_BY(g) for g in [FLAG[12][15],0,0,b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',0,0,ALL_EDGE_DOM[i],0,ALL_EDGE_DOM[i],0,b''.join([D_ARRAY_COUNT_MULTI(k) for k in [[0,i],[i,ALL_EDGE_TO_VERTEX[i][0]],[ALL_EDGE_TO_VERTEX[i][-1],0]]]),FLAG[12][12],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT(ALL_EDGE_MANIFOLD[i][0]),0,b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0']) for i in range(0,np.shape(ALL_EDGE)[0])])+b''.join([D_ARRAY_COUNT_MULTI(i) for i in ALL_EDGE_MANIFOLD])
    a=a+b''.join([b''.join(D_TO_BY(g) for g in [FLAG[12][19],0,0,b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',0,0,ALL_SPLIT_CRV_DOM[l][-1],0,ALL_SPLIT_CRV_DOM[l][-1],b''.join([D_ARRAY_COUNT_MULTI(k) for k in [[1,0],[0,l],[l,ALL_SPLIT_CRV_TO_EDGE[l]],ALL_SPLIT_CRV_TO_VERTEX[l],ALL_SPLIT_CRV_NO_MANIFOLD[l],[ALL_SPLIT_CRV_CONT_SPLIT[l],ALL_SPLIT_CRV_TO_FACE[l]]]]),0,0,FLAG[12][21],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',D_ARRAY_COUNT_MULTI([2,4]),ALL_SPLIT_CRV_VEC_UV[l],ALL_SPLIT_CRV_VEC_UV_FROM_P_BASE[l],np.zeros((3)),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0']) for l in range(0,len(ALL_SPLIT_CRV))])+ b''.join([b''.join([D_TO_BY(g) for g in [np.zeros((2)),ALL_SPLIT_CRV_VEC_AMP[l],b'\xd2\x1d3\x9e\xbd\xf8\xe5\xff',np.array(ALL_SPLIT_CRV_VEC[l],dtype=np.float64),b'\xd2\x1d3\x9e\xbd\xf8\xe5\xff',np.zeros((9))]]) for l in range(0,len(ALL_SPLIT_CRV))])
    a=a+b''.join([b''.join(D_TO_BY(g) for g in[u for o in  [[FLAG[12][23],np.zeros((3)),FLAG[12][24],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',[D_ARRAY_COUNT_MULTI(k) for k in [[1,i],[len(ALL_SPLIT_CRV_INDEX[i])*2,len(ALL_SPLIT_CRV_INDEX[i])*2]]],np.zeros((3)),VECTEUR_AMP[i][0],VECTEUR_AMP[i][1],0,b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0'] for i in range(0,len(BREP[0]))] for u in o])])+ b''.join([D_ARRAY_COUNT_MULTI(k) for k in LOOP_ARRAY])
    a=a+b''.join([D_TO_BY(g) for g in[u for o in  [[FLAG[12][26], np.zeros((2)),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0', 0,b''.join(D_ARRAY_COUNT_MULTI(i) for i in [[1,0],[0,i]]),FLAG[12][28],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',b''.join([D_ARRAY_COUNT_MULTI(y) for y in [[1,1],[i,0]]]),np.zeros((2)),b'\x00\x00\x00\x00\xff\xff\xff\xff',np.array((NORMAL[i])).astype(float),np.array((BACKFACE[i])).astype(float),b''.join([b'\xd2\x1d3\x9e\xbd\xf8\xe5\xff']*4), np.zeros((3)),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0'] for i in range (0,len(BREP[0]))] for u in o]]) + b''.join([D_TO_BY(i) for i in range (0,len(BREP[0]))])
    
    print(D_ALL_SPLIT_CRV_2_ALL_SPLIT_CRV_VEC_UV (ALL_SPLIT_CRV_UNFLAT,ALL_SPLIT_CRV_VEC,VECTEUR))
    return a, CALL , CALL[-1]
def D_BREP_ALL_SPLIT_CRV (all_crv,a) :
    all_crv = np.array((all_crv))
    print(all_crv)
    all_crv_split,ALL_SPLIT_CRV_CONT_SPLIT = [],[]
    for i in range(np.shape(all_crv)[0]) :
        all_crv_split_sub,ALL_SPLIT_CRV_CONT_SPLIT_SUB = [],[]
        all_expt_pt = np.concatenate((np.array((a[:i])).flatten().reshape(-1,3),np.array((a[1+i:])).flatten().reshape(-1,3))).reshape(-1,3)
        for j in range(np.shape(all_crv[i])[0]) :
            vect = np.repeat((np.array((all_crv[i][j][0:3]))-np.array((all_crv[i][j][3:6])))[np.newaxis,:],np.size(all_expt_pt)/3,axis=0)
            vect_n = all_expt_pt-np.array((all_crv[i][j][3:6]))
            pt = (all_expt_pt[np.all((np.cross(vect,vect_n))==0,axis=1)])
            if np.size(pt)==0 :
                all_crv_split_sub .append(all_crv[i,j].tolist())
                ALL_SPLIT_CRV_CONT_SPLIT_SUB.append(j)
            else :
                pt = pt.tolist()
                [pt.append(h) for h in [all_crv[i][j][0:3],all_crv[i][j][3:6]]]
                line_split = sorted(zip(pt,np.sum(pt*vect[0],axis=1),([0]*(len(pt)-2))+[1,1]), key = lambda x: x[1])
                cut_poly_split =np.where(np.array((line_split))[:,2]==1)[0]
                line_split =np.array(((np.array((line_split))[:,0][cut_poly_split[0]:cut_poly_split[1]+1]).tolist()))
                [all_crv_split_sub .append(r) for r in np.concatenate((line_split,np.roll(line_split,1,axis=0)),axis=1)[1:].tolist()],[ALL_SPLIT_CRV_CONT_SPLIT_SUB.append(j) for r in range(0,np.shape(line_split)[0]-1)]
        all_crv_split = all_crv_split + [np.array(all_crv_split_sub).tolist()]
        ALL_SPLIT_CRV_CONT_SPLIT=ALL_SPLIT_CRV_CONT_SPLIT+ALL_SPLIT_CRV_CONT_SPLIT_SUB
    return np.array(([i for j in all_crv_split for i in j])),all_crv_split,ALL_SPLIT_CRV_CONT_SPLIT
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||INPUT
OBJECT_LIST_FLAG_IPT=["CIR","CRV","DIA","DIL","DIO","DIR","DOT","HAT","LED","LIN","MSH","PLI","PLS","PTC","PTS","SUB","TEX"]
OBJECT_LIST_OFFSET_ATTRIBUT=[9,9,9,9,9,9,9,9,9,9,9,9,9,6,9,9,9]
SAVE_FLAG = True
ASK_LOAD = True
REALOAD_FLAG = True
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||OBJ ATTRIBUT
def D_OBJ_NAME (NAME) :
    if NAME==NAME : 
        p = b'\x00\x00\x00\x00'+b''.join([bytes(ctypes.c_wchar(i)) for i in NAME])
        p = p+b''.join([b'\x00']*(8-len(p)%8))
        return  p,len(p)
    else :
        return b'',0
def D_CONT_FLAG_ATTRIBUT (OBJECT_ARY_S,SIZE) :
    COUNT = np.tile(np.array((-117,82)),len(OBJECT_ARY_S)).reshape(-1,2)+(np.arange(len(OBJECT_ARY_S))*100)
    COUNT[:,1] = COUNT[:,1] + np.array((SIZE))[:,1]
    COUNT[:,0] = COUNT[:,0] - np.roll((np.array((SIZE)))[:,1],1)
    COUNT = np.concatenate((COUNT,np.tile(((np.ones_like((SIZE))[:,1])*94.5*(np.array((SIZE))[:,1]!=0)+np.arange(len(OBJECT_ARY_S))*100*(np.array((SIZE))[:,1]!=0))[:,np.newaxis],2)),axis = 1).astype(float)
    for i in [COUNT[:,2] == 0,(0,0),(-1,1)] :
        COUNT[i] = np.nan
    COUNT=np.roll(np.concatenate((np.roll(np.array((COUNT)),1), np.concatenate((np.zeros((1)),np.cumsum((np.sum((SIZE),axis = 1).flatten()))))[:-1][:,np.newaxis]),axis=1),-1)
    return np.array((COUNT))
def D_REDRAW_SET (REDRAW_BIN_Adrss,value):
    pm.write_bytes(REDRAW_BIN_Adrss,bytes(ctypes.c_bool(value)),1)
def D_CONT_OBJ_LINE (i) :
    return int(np.sum([np.size(c)/3+2 for c in i[1][0]])) if i[0] == 12 else int(np.sum([np.size(c)/3 for c in i[1][0]])) if i[0] != 14 else 1
def D_OBJ_ATTRIBUT (OBJECT_ARY_S,RHN_OBJ_Adrss,p,FLAG,ATTRIBUT_FLAG,RHN_CONST_Adrss,OBJECT_LIST_OFFSET_ATTRIBUT) :
    NO_OBJ_DRAW = p.read_memory(RHN_OBJ_Adrss,ctypes.c_longlong()).value != 0
    OBJ_ATTRIBUT,SIZE,GUID_LIST_ADD,NAME_BYTES_LIST = b'',[],[],[[],[]]
    STD_BYTES_ARRAY_OBJ_ATT = [b'?\xf0\x00\x00\x00\x00\x00\x00', b'\xbf\xf0\x00\x00\x00\x00\x00\x00', b'\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x1e\x00.\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x80\x00\x00\x00\x00', b'\x00\x00\x00\x00\xda9\xa3\xee^kK\r2U\xbf\xef\x95`\x18\x90\xaf\xd8\x07\t', b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xce\xaa@\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00', b'\x00\x00\x00\x00\x00\x00\xf8\x7f']
    for i in range(0,len(OBJECT_ARY_S)) :
        PROC_D_CONT_OBJ_LINE = executor.submit(D_CONT_OBJ_LINE,OBJECT_ARY_S[i])
        PROC_D_OBJ_NAME,NAME_CALL = [executor.submit(D_OBJ_NAME,OBJECT_ARY_S[i][3]),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0']
        OBJECT_ARY_S[i]=D_CLEAN_OBJ (OBJECT_ARY_S[i])
        NUM = [p.read_memory(p.read_memory((RHN_OBJ_Adrss),ctypes.c_ulonglong()).value+i,ctypes.c_ulonglong()).value for i in [16,168,320]]if NO_OBJ_DRAW else [0,0,0]
        NUM_LINE = PROC_D_CONT_OBJ_LINE.result()
        if OBJECT_ARY_S[i][0] == 14 : 
            DOMAINE =np.tile(OBJECT_ARY_S[i][1],2).astype(float)
        elif OBJECT_ARY_S[i][0] ==12 :
            DOMAINE = (np.sort(np.array(([j for i in  [[j for i in  [k[:math.floor(len(k))] for k in t] for j in i] for t in (OBJECT_ARY_S[i][1][0]).tolist()] for j in i])).reshape(-1,3),axis=0)[[0,-1]]).astype(float) 
        elif OBJECT_ARY_S[i][0] ==0 :
            DOMAINE = np.sort(((np.tile(np.array(([i.tolist() for i in OBJECT_ARY_S[i][1][1:3]])),(2,1)).reshape(-1,6)*[[1],[-1]]).reshape(-1,3)*OBJECT_ARY_S[i][1][3])+OBJECT_ARY_S[i][1][0],axis=0)[[0,-1]].astype(float)
        else :
            DOMAINE = (np.sort(np.array(([j for i in  [[j for i in  [k[:math.floor(len(k)/3)*3] for k in OBJECT_ARY_S[i][1]] for j in i]] for j in i])).reshape(-1,3),axis=0)[[0,-1]]).astype(float)
        OBJ_ATTRIBUT=OBJ_ATTRIBUT +b''.join([D_TO_BY(i) for i in [FLAG[OBJECT_ARY_S[i][0]][0],0,NUM[0]+1,np.zeros((5)),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',DOMAINE.flatten()[::-1],np.zeros((1)),p.read_memory(RHN_OBJ_Adrss,ctypes.c_longlong()).value if NO_OBJ_DRAW else b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',(0 if i == len(OBJECT_ARY_S)-1 else b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0'),RHN_DOC_Adrss,ATTRIBUT_FLAG[0],0,NUM[1]+NUM_LINE,STD_BYTES_ARRAY_OBJ_ATT[2],OBJECT_ARY_S[i][2],np.zeros((2)),STD_BYTES_ARRAY_OBJ_ATT[3],np.zeros((2)),NAME_CALL,np.zeros((7)),b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',np.zeros((2)),ATTRIBUT_FLAG[2],0,OBJECT_ARY_S[i][2],b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0',ATTRIBUT_FLAG[1],D_ARRAY_COUNT_MULTI([OBJECT_ARY_S[i][6],OBJECT_ARY_S[i][7]]),D_ARRAY_COUNT_MULTI([0,-1]),ATTRIBUT_FLAG[3],np.zeros((2)),ATTRIBUT_FLAG[4],STD_BYTES_ARRAY_OBJ_ATT[4],b''.join([STD_BYTES_ARRAY_OBJ_ATT[5] for i in range(0,16)]),ATTRIBUT_FLAG[5],np.zeros((3)),np.zeros((OBJECT_LIST_OFFSET_ATTRIBUT[OBJECT_ARY_S[0][0]]))]])
        GUID_LIST_ADD = GUID_LIST_ADD + [[OBJECT_ARY_S[i][2],NUM[0]+1,i]]
        NAME_BYTES = PROC_D_OBJ_NAME.result()
        [NAME_BYTES_LIST[i].append(NAME_BYTES[i]) for i in range(0,2)]
        SIZE.append([96,2])
        NO_OBJ_DRAW =False
    NAME_BYTES_LIST = [i[::-1] for i in NAME_BYTES_LIST]
    NAME_BYTES_LIST = [k for l in [[NAME_BYTES_LIST[0][i],NAME_BYTES_LIST[1][i]] for i in range(0,len(NAME_BYTES_LIST[0]))] for k in l]
    CONT_FLAG_ATTRIBUT = np.array(D_CONT_FLAG_ATTRIBUT(OBJECT_ARY_S,SIZE)).flatten()
    return OBJ_ATTRIBUT,CONT_FLAG_ATTRIBUT,int(len(OBJ_ATTRIBUT)/8),GUID_LIST_ADD,NAME_BYTES_LIST
def D_OBJ_TO_BYTES_SPEED (i,FLAG,STD_BYTES_ARRAY,CALL_POST,S_SIZE) :
    if i[0] == 9 :
        return D_LINE_2_OBJ_LINE (i[1],FLAG)
    elif i[0] == 10:
        return D_MESH_2_OBJ_MESH(i[1],FLAG)
    elif i[0] == 0 :
        return D_CIRCLE_2_OBJ_CIRCLE (i[1],FLAG)
    elif i[0] == 1 :
        return D_NURBS_2_OBJ_NURBS (FLAG,i[1])
    elif i[0] ==12 :
        return D_BREP_2_OBJ_BREP (i[1],FLAG,STD_BYTES_ARRAY,CALL_POST,S_SIZE)
    elif i[0] ==13 :
        return D_PTC_2_OBJ_PTC (i,FLAG)
    elif i[0] == 14 :
        return D_POINT_2_OBJ_POINT (i[1],FLAG,S_SIZE)

def D_OBJ_TO_BYTES (OBJECT_ARY_S,RHN_OBJ_Adrss,p,FLAG,ATTRIBUT_FLAG,STD_BYTES_ARRAY,RHN_CONST_Adrss,REDRAW_BIN_Adrss, GUID_LIST_ADRSS,OBJECT_LIST_OFFSET_ATTRIBUT) :
    CALL_POST,S_SIZE =[],0
    PROC_D_OBJ_ATTRIBUT = executor.submit(D_OBJ_ATTRIBUT,OBJECT_ARY_S,RHN_OBJ_Adrss,p,FLAG,ATTRIBUT_FLAG,RHN_CONST_Adrss,OBJECT_LIST_OFFSET_ATTRIBUT)
    OBJ_GEO_BYTES,SIZE,CONT_FLAG,o = b'',0,[],0
    for i in OBJECT_ARY_S :
        locals()["PROC_D_OBJ_TO_BYTES_SPEED" + str(o)] = executor.submit(D_OBJ_TO_BYTES_SPEED,i,FLAG,STD_BYTES_ARRAY,CALL_POST,S_SIZE)
        o+=1
    for i in range (0,o) :
        OBJECT,FLAG_OBJ,SIZE_FLAG = locals()["PROC_D_OBJ_TO_BYTES_SPEED" + str(o-1)].result()
        CONT_FLAG,SIZE,OBJ_GEO_BYTES=CONT_FLAG+(SIZE+FLAG_OBJ).tolist() if np.size((FLAG_OBJ)) > 1 else CONT_FLAG+[(SIZE+FLAG_OBJ).tolist()] ,SIZE+SIZE_FLAG,OBJ_GEO_BYTES+OBJECT
    First_ADRESS=pm.allocate(20000)
    OBJ_ATTRIBUT,CONT_FLAG_ATTRIBUT,SIZE_ATTRIBUTE,GUID_LIST_ADD,NAME_BYTES = PROC_D_OBJ_ATTRIBUT.result()
    PROC_D_GUID_BUILD = executor.submit(D_GUID_BUILD,GUID_LIST_ADD,First_ADRESS)
    CONT_FLAG_ATTRIBUT[np.tile(np.array((2,4)),(len(OBJECT_ARY_S))).reshape(-1,2)+(np.arange(len(OBJECT_ARY_S))*5)] = np.repeat(np.array(([0]+np.cumsum(np.array(([int(NAME_BYTES[i+1]/8) for i in range(0,len(NAME_BYTES),2)]))+1).tolist()))[:-1]+SIZE+SIZE_ATTRIBUTE+0.5,2)
    CONT_FLAG_ATTRIBUT = np.concatenate((np.array(([0]+[SIZE_FLAG] if np.size(np.array((SIZE_FLAG)))==1 else SIZE_FLAG),dtype=np.uint)[:-1].reshape(-1,1)+SIZE_ATTRIBUTE,CONT_FLAG_ATTRIBUT.reshape(1,-1)),axis =1)
    CONT_FLAG_ATTRIBUT = (CONT_FLAG_ATTRIBUT.flatten()[CONT_FLAG_ATTRIBUT.flatten()==CONT_FLAG_ATTRIBUT.flatten()]).tolist()
    NAME_BYTES_BYTES = b''.join([c for t in [[i,b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'] for i in NAME_BYTES[::-1][1::2]] for c in t][:-1])
    OBJ_GEO_BYTES,CONT_FLAG,SIZE = OBJ_ATTRIBUT+OBJ_GEO_BYTES+NAME_BYTES_BYTES,CONT_FLAG_ATTRIBUT+(np.array((CONT_FLAG))+SIZE_ATTRIBUTE).tolist(),SIZE+SIZE_ATTRIBUTE
    TO_REBUILD=[OBJ_GEO_BYTES[i:i+8] for i in range(0,len(OBJ_GEO_BYTES),8)]
    FIND_FLAG =np.where(np.array((TO_REBUILD)) == b'\xf0\xf0\xf0\xf0\xf0\xf0\xf0\xf0')[0]
    for i in range(0,len(FIND_FLAG)) :
        TO_REBUILD[int(FIND_FLAG[i])] = bytes(ctypes.c_longlong(int(CONT_FLAG[i]*8+First_ADRESS)))
    TO_REBUILD = b''.join(TO_REBUILD)
    pm.write_bytes(First_ADRESS,TO_REBUILD,len(TO_REBUILD))
    GUID_LIST_ADD = PROC_D_GUID_BUILD.result()
    D_UPDATE_LIST(First_ADRESS,p,RHN_OBJ_Adrss,REDRAW_BIN_Adrss,len(OBJECT_ARY_S),GUID_LIST_ADD, GUID_LIST_ADRSS,pm)
    print(hex(First_ADRESS))
    return OBJ_ATTRIBUT
def D_UPDATE_LIST(ADRESS,p,RHINO_OBJ_Adrss,REDRAW_BIN_Adrss,NBR_NEZ_OBJ,GUID_LIST_ADD, GUID_LIST_ADRSS,pm) :
    D_REDRAW_SET(REDRAW_BIN_Adrss,False)
    PROC_D_UPDATE_GUID = executor.submit (D_UPDATE_GUID,GUID_LIST_ADD, GUID_LIST_ADRSS,NBR_NEZ_OBJ,pm,p)
    if pm.read_longlong(RHINO_OBJ_Adrss) != 0 :
        pm.write_bytes(pm.read_longlong(RHINO_OBJ_Adrss)+136,bytes(ctypes.c_longlong(ADRESS)),8)
    pm.write_bytes(RHINO_OBJ_Adrss,bytes(ctypes.c_longlong(ADRESS)),8)
    COUN_OBJ = pm.read_longlong(RHINO_OBJ_Adrss-8)
    pm.write_bytes(COUN_OBJ,bytes(ctypes.c_longlong(pm.read_longlong(COUN_OBJ)+NBR_NEZ_OBJ)),8)
    D_REDRAW_SET(REDRAW_BIN_Adrss,True)
    PROC_D_UPDATE_GUID.result()
def D_UPDATE_GUID (GUID_LIST_ADD, GUID_LIST_ADRSS,NUM_OBJ,pm,p) :
    HEAD, FEET, FREE, LAST,LIST_START, SIZE = D_GUID_LIST_INFO(p,pm,GUID_LIST_ADRSS)
    if len(str(GUID_LIST_ADD)) > FREE :
        NEW_START =  D_RELOC_GUID_LIST (SIZE, GUID_LIST_ADRSS,pm )
        LAST,LIST_START = LAST - LIST_START + NEW_START, NEW_START
    PROC_D_UPDATE_GUID_SPEED = executor.submit(D_UPDATE_GUID_SPEED,GUID_LIST_ADRSS,NUM_OBJ,LIST_START,pm)
    pm.write_bytes(LAST,GUID_LIST_ADD,len(GUID_LIST_ADD))
    PROC_D_UPDATE_GUID_SPEED.result()
def D_UPDATE_GUID_SPEED (GUID_LIST_ADRSS,NUM_OBJ,LIST_START,pm) :
    PROC_D_UPDATE_GUID_SPEED_SPEED = executor.submit(D_UPDATE_GUID_SPEED_SPEED,LIST_START,NUM_OBJ)
    SUB_LIST = pm.read_longlong(GUID_LIST_ADRSS)
    [pm.write_bytes(SUB_LIST+i[1],i[0],8) for i in [[bytes(ctypes.c_longlong(pm.read_longlong(GUID_LIST_ADRSS)+NUM_OBJ*3)),0],[bytes(ctypes.c_longlong(pm.read_longlong(GUID_LIST_ADRSS+8)+NUM_OBJ)),8],[bytes(ctypes.c_longlong(pm.read_longlong(GUID_LIST_ADRSS+8)+NUM_OBJ)),88]]]
    PROC_D_UPDATE_GUID_SPEED_SPEED.result()
def D_UPDATE_GUID_SPEED_SPEED (LIST_START,NUM_OBJ) :
    [pm.write_bytes(LIST_START+i[1],i[0],8) for i in [[bytes(ctypes.c_longlong(pm.read_longlong(LIST_START)+NUM_OBJ)),0],[bytes(ctypes.c_longlong(pm.read_longlong(LIST_START+24)+NUM_OBJ)),24]]]
def D_RELOC_GUID_LIST (SIZE, GUID_LIST_ADRSS,pm ) :
    LIST_START = pm.read_longlong(GUID_LIST_ADRSS)
    map =  pm.read_longlong(pm.read_longlong(LIST_START+56)-8)
    for i in [458752] :
        if SIZE < i :
            break
    REALLOC_ADRESS = D_REALLOC([map,map+SIZE],pm,i)
    pm.write_bytes(LIST_START+56,bytes(ctypes.c_longlong(REALLOC_ADRESS),8))
    pm.write_bytes(LIST_START+64,bytes(ctypes.c_longlong(4294967296),8))
    return REALLOC_ADRESS
def D_GUID_BUILD (GUID_LIST_ADD,First_ADRESS) :
    a =  b''.join([b''.join([D_TO_BY(j) for j in [GUID_LIST_ADD[i][0],GUID_LIST_ADD[i][1],bytes(ctypes.c_long(257))+np.random.bytes(4),0,1,int(GUID_LIST_ADD[i][2]+(i*100)+First_ADRESS)]]) for i in range(0,len(GUID_LIST_ADD))])
    return a
def D_GUID_LIST_INFO (p,pm,GUID_LIST_ADRSS) :
    LIST_START = pm.read_longlong(pm.read_longlong(GUID_LIST_ADRSS)+56)
    HEAD = np.array((list(bytes(ctypes.c_longlong(pm.read_longlong(LIST_START-8))))))[[0,2,4,7]]
    LIST_GUID_STATE = pm.read_long(pm.read_longlong(GUID_LIST_ADRSS)+64)
    for i in [[16777216,458752],[16777216,4444444]] :
        if LIST_GUID_STATE == i[0] :
            FEET = i[1]+LIST_START+40
    LAST = pm.read_long(LIST_START)*56+32
    print('\033[95m' +"\nGUID LIST INFO\n----------------------"+ '\033[0m')
    print("LIST START = " + hex(LIST_START))
    print("LIST FEET = " + hex(FEET))
    print("LIST STATE = " + hex(i[0]))
    print("LIST GUID = " + hex(LAST+LIST_START-1))
    print("ACTUAL SIZE = " + (hex(i[1])))
    print("FREE SIZE = " + hex(i[1]-LAST))
    print("USE SIZE = " + hex(LAST))
    return HEAD, FEET, i[1]-LAST ,LAST+LIST_START ,LIST_START, i[1]
#||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||EXECUTION
PID = D_PID_LOAD()
with concurrent.futures.ThreadPoolExecutor() as executor:
    with Process.open_process(PID) as p:
        STD_BYTES_ARRAY = [bytes(ctypes.c_longlong(0)),bytes(ctypes.c_longlong(3))[::-1]]
        ppro = psutil.Process(PID)
        pm = pymem.Pymem(PID) #il faut changer process id dans le truc a la con la
        if SAVE_FLAG :
            RHN_DOC_Adrss, RHN_CONST_Adrss, RHN_OBJ_Adrss, ON_RD_RH_RD,OBJECT_LIST_FLAG,ATTRIBUT_FLAG,OBJECT_LIST_FLAG_BYTES,ATTRIBUT_FLAG_BYTES,VISUAL_FLAG,REDRAW_BIN_Adrss,ON_TXT_RH_TXT,ON_DT_RH_DT,GUID_LIST,ASK_LOAD=D_LOAD_TEMP(PID)
        if ASK_LOAD or SAVE_FLAG == False:
            RHN_DOC_Adrss, RHN_CONST_Adrss, RHN_OBJ_Adrss, ON_RD_RH_RD,OBJECT_LIST_FLAG,ATTRIBUT_FLAG,OBJECT_LIST_FLAG_BYTES,ATTRIBUT_FLAG_BYTES,VISUAL_FLAG,REDRAW_BIN_Adrss,ON_TXT_RH_TXT,ON_DT_RH_DT,OBJ_TYPE,GUID_LIST= D_Init_point(ppro,REALOAD_FLAG)
            D_SAVE_TEMP([PID,RHN_DOC_Adrss, RHN_CONST_Adrss, RHN_OBJ_Adrss, ON_RD_RH_RD,OBJECT_LIST_FLAG,ATTRIBUT_FLAG,OBJECT_LIST_FLAG_BYTES,ATTRIBUT_FLAG_BYTES,VISUAL_FLAG,REDRAW_BIN_Adrss,ON_TXT_RH_TXT,ON_DT_RH_DT,GUID_LIST])
        OBJ_LIST_Adrss,LIST_MAP_OBJ_STOR = D_LIST_OBJ(RHN_OBJ_Adrss,OBJECT_LIST_FLAG,OBJECT_LIST_FLAG_IPT)
        LINE_TYPE= D_LINE_TYPE(RHN_DOC_Adrss)
        LAYER_TABLE = D_LAYER_TABLE(RHN_DOC_Adrss)
        HATCH_PATTERN = D_HATCH_PATTERN(RHN_DOC_Adrss)
        OBJECT_ARY_S = [OBJECT_ARY]
        BYTES  = D_OBJ_TO_BYTES(OBJECT_ARY_S, RHN_OBJ_Adrss,p,OBJECT_LIST_FLAG_BYTES,ATTRIBUT_FLAG_BYTES,STD_BYTES_ARRAY,RHN_CONST_Adrss,REDRAW_BIN_Adrss,GUID_LIST,OBJECT_LIST_OFFSET_ATTRIBUT)
