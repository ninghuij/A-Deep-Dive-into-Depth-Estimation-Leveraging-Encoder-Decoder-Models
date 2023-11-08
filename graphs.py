import os
from tensorboard import program
import threading
import webbrowser

def open_webbroser(url):
    webbrowser.open_new_tab(url)

def launchTensorBoard():
    os.system('tensorboard --logdir=' +  'runs')

if __name__ == "__main__":
    
    path = 'runs'

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', path])
    url = tb.launch()
    t1 = threading.Thread(target=launchTensorBoard, args=([]))
    t1.start()

    t2 = threading.Thread(target=open_webbroser(url), args=([]))
    t2.start()