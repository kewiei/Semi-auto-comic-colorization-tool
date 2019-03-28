import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch.nn.init as init
import torch.optim as optim
import os
import random
from tqdm import tqdm
import cv2

class Colorant(nn.Module):
    def __init__(self, input_nc):
        super(Colorant, self).__init__()
        use_bias = True
        # block0
        block0 = []
        block0 += [nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias)]
        block0 += [nn.ReLU(True),]
        block0 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block0 += [nn.ReLU(True),]
        block0 += [nn.BatchNorm2d(64),]
        
        # block1
        block1 = []
        block1 += [nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),]
        block1 += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block1 += [nn.ReLU(True),]
        block1 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block1 += [nn.ReLU(True),]
        block1 += [nn.BatchNorm2d(128),]

        # block2
        block2 = []
        block2 += [nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),]
        block2 += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block2 += [nn.ReLU(True),]
        block2 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block2 += [nn.ReLU(True),]
        block2 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block2 += [nn.ReLU(True),]
        block2 += [nn.BatchNorm2d(256),]
        
        # block3
        block3 = []
        block3+=[nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        block3+=[nn.ReLU(True),]
        block3+=[nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        block3+=[nn.ReLU(True),]
        block3+=[nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        block3+=[nn.ReLU(True),]
        block3+=[nn.BatchNorm2d(256),]

        # block4
        block4=[]
        block4+=[nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        block4+=[nn.ReLU(True),]
        block4+=[nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        block4+=[nn.ReLU(True),]
        block4+=[nn.Conv2d(256, 256, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        block4+=[nn.ReLU(True),]
        block4+=[nn.BatchNorm2d(256),]
        
        # block5
        block5up=[nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=use_bias),]
        block1to5=[nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=1,bias=use_bias),]
        # add the two feature maps above
        block5=[nn.ReLU(True),]
        block5+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        block5+=[nn.ReLU(True),]
        block5+=[nn.BatchNorm2d(128),]

        # block6
        block6up=[nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1, bias=use_bias),]
        block0to6=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above
        block6=[nn.ReLU(True),]
        block6+=[nn.Conv2d(64, 64, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        block6+=[nn.LeakyReLU(negative_slope=.2),]
        
        to_Lab=[nn.Conv2d(64, 3, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        
        
        self.block0 = nn.Sequential(*block0)
        self.block1 = nn.Sequential(*block1)
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(*block3)
        self.block4 = nn.Sequential(*block4)
        self.block5up = nn.Sequential(*block5up)
        self.block1to5 = nn.Sequential(*block1to5)
        self.block5 = nn.Sequential(*block5)
        self.block6up = nn.Sequential(*block6up)
        self.block0to6 = nn.Sequential(*block0to6)
        self.block6 = nn.Sequential(*block6)
        
        self.to_Lab = nn.Sequential(*to_Lab)
        # self.to_class == nn.Sequential(*to_class)
    """  
    def forward(self, gray_sketch, user_input):
        total_input = torch.cat((gray_sketch,user_input),dim=1)
        """
    def forward(self, total_input):
        #print('total_input',total_input.size())
        result0 = self.block0(total_input)
        #print('result0', result0.size())
        result1 = self.block1(result0)
        #print('result1', result1.size())
        result2 = self.block2(result1)
        #print('result2', result2.size())
        result3 = self.block3(result2)
        #print('result3', result3.size())
        result4 = self.block4(result3)
        #print('result4', result4.size())
        result5up = self.block5up(result4)
        #print('result5up', result5up.size())
        result1to5 = self.block1to5(result1)
        #print('result1to5', result1to5.size())
        
        result5up = F.interpolate(input=result5up,size=(result1to5.size()[2],result1to5.size()[3]),mode="nearest")
        
        result5 = self.block5(result5up+result1to5)
        result6up = self.block6up(result5)
        #print('result6up', result6up.size())
        result0to6 = self.block0to6(result0)
        #print('result0to6', result0to6.size())
        
        result6up = F.interpolate(input=result6up,size=(result0to6.size()[2],result0to6.size()[3]),mode="nearest")
        
        result6 = self.block6(result6up+result0to6)
        #print('result6', result6.size())
        
        result = self.to_Lab(result6)
        return result
    
    def load_param(self, filename = "default.pth"):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)
        
    def save_param(self, filename = "default.pth"):
        torch.save(self.state_dict(), filename)
    
    def init_weights(self, gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)
        self.apply(init_func)
		
def image_iter_from_folder(file_loc):
    materiallist = os.listdir(file_loc)
    while True:
        choice = random.choice(materiallist)
        #print("choice",choice)
        lab = read2lab(material_loc+'\\'+choice)
        yield lab

def image_iter_from_video(video_loc):
    cap = cv2.VideoCapture(video_loc)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while(cap.isOpened()):
        cap.set(1,random.randint(0,length))
        ret, frame = cap.read()
        lab = rgb2lab(frame)
        yield lab
    cap.release()
    cv2.destroyAllWindows()
    #pip install imageio-ffmpeg

def image_iter_from_video_folder(folder_name,color_space='lab',random_amount=10):
    materiallist = os.listdir(folder_name)
    while True:
        cap = cv2.VideoCapture(folder_name+"/"+random.choice(materiallist))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i=0
        while(cap.isOpened() and i<random_amount):
            cap.set(1,random.randint(0,length))
            ret, frame = cap.read()
            if not ret:
                print("skip invalid image")
                continue
            if color_space=='lab':
                frame = rgb2lab(frame)
            else:
                frame = torch.as_tensor(frame,dtype=torch.float).permute(2,0,1).unsqueeze_(dim=0)
            yield frame
            i+=1
        cap.release()
        cv2.destroyAllWindows()
    
    
def read2lab(filename):
    rgb = cv2.imread(filename)
    lab = rgb2lab(rgb)
    return lab

def readrgb(filename):
    rgb = cv2.imread(filename)
    return rgb

def rgb2lab(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB)
    lab = torch.as_tensor(lab/255,dtype=torch.float)
    lab = lab.permute(2,0,1)
    lab.unsqueeze_(dim=0)
    return lab

def lab2rgb_output(lab):
    lab = lab.squeeze()
    lab = lab.permute(1,2,0)
    lab = lab*255
    """
    print(l.max(),a.max(),b.max())
    print(l.min(),a.min(),b.min())
    """
    result = np.asarray(lab.numpy(),dtype=np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
    
def tosketch(lab):
    edge_filter = torch.Tensor([[-1.,-1.,-1.],[-1.,9.,-1.],[-1.,-1.,-1.]])
    edge_filter.unsqueeze_(dim = 0).unsqueeze_(dim = 0)
    gaussian_size = 7
    sigma = 1.
    
    # Gaussian
    def gaussiankernel(l=5, sig=1.):
        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.sum(kernel)
    
    L = lab[:,0,:,:]
    L = torch.as_tensor(L,dtype=torch.float32)
    L.unsqueeze_(dim=1)
    #L.unsqueeze_(dim = 0).unsqueeze_(dim = 0)
    conv1 = nn.Conv2d(1,1,3,padding = 1,bias = False)
    conv1.weight.data = edge_filter
    #print(gaussian_size//2)
    gaussian = nn.Conv2d(1,1,gaussian_size,padding = gaussian_size//2,bias = False)
    gaussiankernel = torch.as_tensor(gaussiankernel(l=gaussian_size, sig=sigma),dtype=torch.float)
    gaussian.weight.data = gaussiankernel.unsqueeze_(dim=0).unsqueeze_(dim=0)
    
    with torch.no_grad():
        edge = conv1(L)
    edge = edge>0.5
    edge = torch.as_tensor(edge,dtype=torch.float)
    #print("after bineary",edge.size())
    with torch.no_grad():
        edge = gaussian(edge)
    #print("after gaussian",edge.size())
    return edge

def simulate_user_input(lab):
    h,w = lab.size()[2],lab.size()[3]
    strokes = []
    for i in range(np.random.randint(20)):
        stroke_loc = np.random.normal(loc=0.5,scale=0.25,size=2)
        stroke_loc = np.array((stroke_loc[0]*h,stroke_loc[1]*w))
        stroke_loc = np.clip(stroke_loc,a_min=(0,0),a_max=(h-1,w-1))
        stroke_h,stroke_w = int(stroke_loc[0]),int(stroke_loc[1])
        stroke_lab = lab[:,:,stroke_h,stroke_w]
        stroke_lab.squeeze_()
        strokes.append((stroke_lab,stroke_h,stroke_w))
    user_input = torch.zeros((1,4,h,w))
    for stroke_lab,stroke_h,stroke_w in strokes:
        user_input[0,0,stroke_h,stroke_w] = stroke_lab[0]
        user_input[0,1,stroke_h,stroke_w] = stroke_lab[1]
        user_input[0,2,stroke_h,stroke_w] = stroke_lab[2]
        user_input[0,3,stroke_h,stroke_w] = 1
    return user_input

def get_input_target(target_lab=None,limit_size=None,use_cuda=False):
    if target_lab==None:
        target_lab = next(image_iter)
    if limit_size!=None:
        scale = (limited_size/target_lab.numel())**0.5
        if scale<1:
            target_lab = F.interpolate(input=target_lab,scale_factor=scale,mode="bilinear",align_corners=True)
    
    sketch = tosketch(target_lab)
    simulated_user_input = simulate_user_input(target_lab)
    input_lab = torch.cat((sketch,simulated_user_input),dim=1)
    if use_cuda:
        return input_lab.cuda(),target_lab.cuda()
    else:
        return input_lab,target_lab
		
material_loc = "material"
default_filename = "rgb_latest.pth"
log_filename = "train log rgb.txt"
limited_size = 3000000
epoch_amount = 5000*0
save_freq = 50
print_freq = 50
zerograd_freq = 10
use_cuda = torch.cuda.is_available()


#train
net = Colorant(5)
try:
    net.load_param(filename=default_filename)
except Exception as e:
    print("load param failed.")
    print(e)
else:
    net.init_weights()
    print("load param successful")

if use_cuda:
    net=net.cuda()

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
smoothl1loss = nn.SmoothL1Loss()

image_iter = image_iter_from_video_folder("material3",'rgb')
#print("target_lab",target_lab.size())

for i in tqdm(range(epoch_amount)):
    input_lab,target_lab = get_input_target(limit_size=limited_size,use_cuda=use_cuda)
    result = net(input_lab)
    loss = smoothl1loss(result, target_lab)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i%save_freq==0:
        net.save_param(filename=default_filename)
    if i%print_freq==0:
        print_loss = loss.detach().item()
        print("current loss",print_loss)
        with open(log_filename,'a') as f:
            f.write("current loss ")
            f.write(str(print_loss))
            f.write("\n")

"""
#test
input_lab,target_lab = get_input_target(limit_size=limited_size,use_cuda=use_cuda)
with torch.no_grad():
    result = net(input_lab)
result = torch.as_tensor(result.detach().cpu().squeeze().permute(1,2,0),dtype=torch.uint8).numpy()

cv2.imshow("image",result)
cv2.waitKey()
cv2.destroyAllWindows()
"""