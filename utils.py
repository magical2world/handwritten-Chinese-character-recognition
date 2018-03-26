import os
import numpy as np
import skimage.color
import skimage.transform
import matplotlib.pyplot as plt

class data_process():
	def __init__(self,folder_path):
		self.folder_path=folder_path

	def hex2image(self,hex_data):#transform hexadecimal data into binary image
		length=len(hex_data)
		bin_data=np.zeros((length,4))
		for i in range(length):
                        hex=int(hex_data[i],16)
			for j in range(4):
				bin_data[i,3-j]=hex%2
				hex=hex//2
		return np.float32(bin_data.reshape([64,64,1]))

	def load_file(self):# read hcl file to image
		for file in os.listdir(self.folder_path):
                        images=[]
                        labels=[]
			filename=self.folder_path+'/'+file
			if file[-3:]=='hcl':
				#num+=1
				with open(filename,'rb') as f:
					images_hex=f.read()[512:]
					num_chinese=0
					for begin_index in range(0,len(images_hex),512):
						end_index=begin_index+512
						image_hexcode=images_hex[begin_index:end_index]
						# print image_hexcode.encode('hex')
						image=self.hex2image(image_hexcode.encode('hex'))
						images.append(image)
						labels.append(num_chinese)
						num_chinese+=1
			yield np.array(images),np.array(labels)

# dataset=data_process('data').load_file()
# image,label=dataset.next()
# print image