import serial
import time
import paho.mqtt.client as mqtt
import LPi.GPIO as GPIO

ser = serial.Serial("/dev/ttyS1",9600)

GPIO.setmode(GPIO.LS2K)
GPIO.setup(1,GPIO.OUT)
GPIO.setup(4,GPIO.OUT)
GPIO.setup(5,GPIO.OUT)
GPIO.setup(6,GPIO.OUT)
GPIO.setup(10,GPIO.OUT)
GPIO.setup(2,GPIO.OUT)
host = ''
password = ''
username = ''

if not ser.isOpen():
    print("open failed")
else:
    pass

#设置用电器识别标准
Electrical_assembly={'吹风筒1':{'Voltage':224.948,'Current':1560,'Pow_ace':353.771,'Pactive_pow':1},
                     '吹风筒3':{'Voltage':224.948,'Current':3234,'Pow_ace':725.516,'Pactive_pow':1},
                     '电池充电':{'Voltage':224.311,'Current':162,'Pow_ace':20.112,'Pactive_pow':0.5482},
                     '电烙铁':{'Voltage':224.051,'Current':123,'Pow_ace':27.5,'Pactive_pow':1},
                     '手机充电':{'Voltage':224.409,'Current':66,'Pow_ace':7.858,'Pactive_pow':0.5931},
                     '风扇':{'Voltage':224.45,'Current':202,'Pow_ace':44.472,'Pactive_pow':0.9867},
                     }

def Electrical(Voltage,Current,Ep,Pf):
    global a
    global app
    a=''
    for app,value in Electrical_assembly.items():
            i = 0
            if ((list(value.values())[i])-(list(value.values())[i])*0.08)<= Voltage <=((list(value.values())[i])+(list(value.values())[i])*0.08):
                i = i+1
                if ((list(value.values())[i])-(list(value.values())[i])*0.15)<= Current <=((list(value.values())[i])+(list(value.values())[i])*0.15):
                    i = i+1
                    if ((list(value.values())[i]) - (list(value.values())[i]) * 0.15) <= Ep<= ((list(value.values())[i]) + (list(value.values())[i]) * 0.15):
                        i = i+1
                        if ((list(value.values())[i]) - (list(value.values())[i]) * 0.08) <= Pf <= ((list(value.values())[i]) + (list(value.values())[i]) * 0.08):
                            a = app
                            print(app)
            else:
                continue

def tty():
    count = ser.inWaiting()
    lis = [0x55,0x55,0x01,0x02,0x00,0x00,0xAD]
    ser.write(lis)
    if count > 0:
        recv = ser.read(count)
        while lis[0]+lis[1]+lis[2]+lis[3]+lis[4]+lis[5] != lis[6]:  #计算输出校验和
            continue
        Send = recv[0:6]
        Voltage = recv[6:10]
        Current = recv[10:14]
        EP = recv[14:18]
        PF = recv[18:22]
        Frequency = recv[22:26]
        All_Power = recv[26:30]
        Last = recv[30:]

        a = Send[0] + Send[1] + Send[2] + Send[3] + Send[4] + Send[5]
        b = Voltage[0] + Voltage[1] + Voltage[2] + Voltage[3]
        c = Current[0] + Current[1] + Current[2] + Current[3]
        d = EP[0] + EP[1] + EP[2] + EP[3]
        e = PF[0] + PF[1] + PF[2] + PF[3]
        f = Frequency[0] + Frequency[1] + Frequency[2] + Frequency[3]
        g = All_Power[0] + All_Power[1] + All_Power[2] + All_Power[3]
        add = a + b + c + d + e + f + g
        Verify = bin(add)
        Verify_bin = Verify[4:]
        Last_bin = bin(int(Last.hex(), 16))
        Last_Verify = Last_bin[2:]
        Verify_V = int(Verify_bin, 2)
        Last_V = int(Last_Verify, 2)
        while Verify_V != Last_V:  #计算输入校验和
            continue

        Voltage_10 = int(str(Voltage.hex()),16)/1000
        Current_10 = int(str(Current.hex()),16)
        EP_10 = int(str(EP.hex()),16)/1000
        PF_10 = int(str(PF.hex()),16)/10000
        Frequency_10 = int(str(Frequency.hex()),16)/1000
        All_Power_10 = int(str(All_Power.hex()),16)/10000
        if EP_10 >= 600:
            GPIO.output(1,GPIO.LOW)
            GPIO.output(4,GPIO.LOW)
            GPIO.output(5,GPIO.LOW)
            GPIO.output(6,GPIO.LOW)
            GPIO.output(10,GPIO.LOW)
            GPIO.output(2,GPIO.LOW)
        else:
            pass

        def fun():
            global Voltage_10
            global Current_10
            global EP_10
            global PF_10
            Voltage_10 = int(str(Voltage.hex()),16)/1000
            Current_10 = int(str(Current.hex()),16)
            EP_10 = int(str(EP.hex()),16)/1000
            PF_10 = int(str(PF.hex()),16)/10000
            Frequency_10 = int(str(Frequency.hex()),16)/1000
            All_Power_10 = int(str(All_Power.hex()),16)/10000
            Electrical(Voltage_10,Current_10,EP_10,PF_10)   
            print("电压:",Voltage_10,"V")  #10进制的数据
            print("电流:",Current_10,"mA")
            print("有效功率:",EP_10,"W")
            print("功率因素:",PF_10)
            #print("频率:",Frequency_10,"Hz")
            #print("累计电量",All_Power_10,"KW*h")
            print("\n")
            time.sleep(0.5)
        fun()

def paho_mqtt():
    def on_connect(client, userdata, flags, rc):
        client.subscribe(topic1,1)
        client.subscribe(topic2,1)
        client.subscribe(topic3,1)
        client.subscribe(topic4,1)
        client.subscribe(topic5,1)
        client.subscribe(topic6,1)
        tty()
        client.publish(topic7,Current_10,1)
        client.publish(topic8,EP_10,1)
        client.publish(topic9,Voltage_10,1)
        client.publish(topic10,PF_10,1)
        client.publish(topic11,a,1)

    def on_message(client, userdata, msg):
        if len(msg.payload) == 2:
            GPIO.output(1,GPIO.HIGH)
        elif len(msg.payload) == 1:
            GPIO.output(1,GPIO.LOW)
        if len(msg.payload) == 4:
            GPIO.output(4,GPIO.HIGH)
        elif len(msg.payload) == 3:
            GPIO.output(4,GPIO.LOW)
        if len(msg.payload) == 6:
            GPIO.output(5,GPIO.HIGH)
        elif len(msg.payload) == 5:
            GPIO.output(5,GPIO.LOW)
        if len(msg.payload) == 8:
            GPIO.output(6,GPIO.HIGH)
        elif len(msg.payload) == 7:
            GPIO.output(6,GPIO.LOW)
        if len(msg.payload) == 10:
            GPIO.output(10,GPIO.HIGH)
        elif len(msg.payload) == 9:
            GPIO.output(10,GPIO.LOW)
        if len(msg.payload) == 12:
            GPIO.output(2,GPIO.HIGH)
        elif len(msg.payload) == 11:
            GPIO.output(2,GPIO.LOW)
            
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, 1883,60)
    client.username_pw_set(username,password)
    client.loop(timeout=1,max_packets=1)
    client.loop_start()
if __name__=='__main__':
    while True:
        tty()
        paho_mqtt()
