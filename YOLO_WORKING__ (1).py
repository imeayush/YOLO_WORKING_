#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')


# In[2]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[3]:


pip install pyqt5==5.12 pyqtwebengine==5.12


# In[4]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt')


# In[5]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[6]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[7]:


model


# In[45]:


img = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQkwt_7BO_JsnXqw6SaGecxoFCcRx9znWnqJg&s'


# In[46]:


results = model(img)
results.print()


# In[47]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()


# In[48]:


results.show()


# In[12]:


results.render()


# In[ ]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

