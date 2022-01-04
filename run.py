# _*_ coding: utf-8 _*_
import threading
import tkinter
import tkinter.filedialog
import tkinter.messagebox
from scipy.io import wavfile
import sounddevice as sd
import soundfile

import paddle
from paddlespeech.cli import ASRExecutor, TextExecutor

root = tkinter.Tk()
root.title('Recorder')
root.geometry('870x500')
root.resizable(False, False)

allowRecording = False #录音状态

#播放音频
#myarray = np.arange(fs * length)
#myarray = np.sin(2 * np.pi * f / fs * myarray)
#sd.play(myarray, fs)

#查看录音设备
#sd.query_devices()
#print(sd.default.device[1])
#print(sd.query_devices())
#2,8内录
sd.default.device[0] = 2
fs = 16000 # Hz
length = 15 # s


asr_executor = ASRExecutor()
text_executor = TextExecutor()

def predict(recfile):
	text = asr_executor(
		audio_file=recfile,
		device=paddle.get_device())
	result = text_executor(
		text=text,
		task='punc',
		model='ernie_linear_p3_wudao',
		device=paddle.get_device())
	txt_text.insert('1.0', format(result))

def record():
	global allowRecording
	recmark = 0 #录音线程标记
	while allowRecording:
		recording = sd.rec(frames=fs * length, samplerate=fs, blocking=True, channels=1)
		#wavfile.write('recording'+str(recmark)+'.wav', fs, recording)
		soundfile.write('recording'+str(recmark)+'.wav', recording, fs, subtype="PCM_16")
		txt_text.insert('1.0', 'saved file:recording'+str(recmark)+'.wav\n')
		predict('recording'+str(recmark)+'.wav')
		recmark = recmark+1
		#lbStatus['text'] = 'Ready'
		#allowRecording = False	

def start():
	global allowRecording
	allowRecording = True
	lbStatus['text'] = 'Recording...'
	threading.Thread(target=record).start()

def stop():
	global allowRecording
	allowRecording = False
	lbStatus['text'] = 'Ready'
    
# 关闭程序时检查是否正在录制
def closeWindow():
	if allowRecording:
		tkinter.messagebox.showerror('Recording', 'Please stop recording before close the window.')
		return
	root.destroy()

btnStart = tkinter.Button(root, text='Start', command=start)
btnStart.place(x=30, y=20, width=100, height=20)
btnStop = tkinter.Button(root, text='Stop', command=stop)
btnStop.place(x=140, y=20, width=100, height=20)
lbStatus = tkinter.Label(root, text='Ready', anchor='w', fg='green')    #靠左显示绿色状态字
lbStatus.place(x=30, y=50, width=200, height=20)
txt_label = tkinter.Label(root, text="输出：")
txt_label.place(x=10, y=70)

txt_text = tkinter.Text(root, width=120, height=30)
scroll = tkinter.Scrollbar()
# 放到窗口的右侧, 填充Y竖直方向
scroll.pack(side=tkinter.RIGHT,fill=tkinter.Y)
# 两个控件关联
scroll.config(command=txt_text.yview)
txt_text.config(yscrollcommand=scroll.set)
txt_text.place(x=10, y=100)
txt_text.insert('1.0', 'app start!')

root.protocol('WM_DELETE_WINDOW', closeWindow)

root.mainloop()
