#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2


# In[3]:


from skimage.metrics import structural_similarity


# In[4]:


import numpy
numpy.version.version


# In[5]:


import imutils


# In[6]:


from PIL import Image


# In[7]:


import requests


# In[8]:


# !mkdir pan_card_tampering
get_ipython().system('mkdir pan_card_tampering\\image')


# In[9]:


original = Image.open(requests.get('https://www.thestatesman.com/wp-content/uploads/2019/07/pan-card.jpg',stream=True).raw)
tampered = Image.open(requests.get('https://assets1.cleartax-cdn.com/s/img/20170526124335/Pan4.png',stream=True).raw)


# In[10]:


print("Original image format :",original.format)
print("Tampered image format :",tampered.format)


# In[11]:


print("Original image size :",original.size)
print("Tampered image size :",tampered.size)


# In[12]:


original = original.resize((250,160))
print(original.size)
original.save('pan_card_tampering/image/original.png')

tampered = tampered.resize((250,160))
print(tampered.size)
tampered.save('pan_card_tampering/image/tampered.png')


# In[13]:


tampered = Image.open('pan_card_tampering/image/tampered.png')
tampered.save('pan_card_tampering/image/tampered.png')


# In[14]:


original


# In[15]:


tampered


# In[16]:


original = cv2.imread('pan_card_tampering/image/original.png')
tampered = cv2.imread('pan_card_tampering/image/tampered.png')


# In[17]:


original_gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered,cv2.COLOR_BGR2GRAY)


# In[18]:


(score,diff) = structural_similarity(original_gray,tampered_gray, full=True)


# In[19]:


score


# In[20]:


diff


# In[21]:


diff = (diff * 255).astype("uint8")


# In[22]:


diff


# In[23]:


print("SSIM: {}".format(score))


# In[25]:


thresh = cv2.threshold(diff, 0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


# In[26]:


thresh


# In[30]:


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts


# In[31]:


cnts = imutils.grab_contours(cnts)


# In[32]:


cnts


# In[33]:


for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x,y),(x + w, y +h),(0,0,255),2)
    cv2.rectangle(tampered, (x,y),(x + w, y +h),(0,0,255),2)


# In[34]:


Image.fromarray(original)


# In[35]:


Image.fromarray(tampered)


# In[36]:


Image.fromarray(diff)


# In[37]:


Image.fromarray(thresh)


# In[ ]:




