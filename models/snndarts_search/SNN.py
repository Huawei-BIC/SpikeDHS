import torch
import torch.nn as nn
import torch.nn.functional as F
# import fitlog
from torch.nn.utils.fusion import *
from torch.autograd import Variable
import copy
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
b_delta = 1
# fitlog.add_other(name='b',value=b.item())

N_diffb = 2
N_diffb_mode = [i+(10-N_diffb) for i in range(N_diffb*2+1)]
N_diffb_delta = [(i-10)*0.2 for i in N_diffb_mode]

class ActFun_b5(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
        # BUG TypeError: tanh(): argument 'input' (position 1) must be Tensor, not float
        # BUG legacy constructor expects device type: cpubut device type: cuda was passed
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            # temp = abs(input - thresh) < lens
            b = torch.tensor(5,device=device)
            temp = (1-torch.tanh(b*(input-0.5))**2)*b/2/(torch.tanh(b/2))
            temp[input<=0]=0
            temp[input>=1]=0
        return grad_input * temp.float()


class ActFun_b3(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        ctx.b = 1
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
        # BUG TypeError: tanh(): argument 'input' (position 1) must be Tensor, not float
        # BUG legacy constructor expects device type: cpubut device type: cuda was passed
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            # temp = abs(input - thresh) < lens
            b = torch.tensor(3,device=device)
            temp = (1-torch.tanh(b*(input-0.5))**2)*b/2/(torch.tanh(b/2))
            temp[input<=0]=0
            temp[input>=1]=0
        return grad_input * temp.float()


class ActFun_changeable(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input)
        ctx.b = b
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            # temp = abs(input - thresh) < lens
            b = torch.tensor(ctx.b,device=device)
            temp = (1-torch.tanh(b*(input-0.5))**2)*b/2/(torch.tanh(b/2))
            temp[input<=0]=0
            temp[input>=1]=0
        return grad_input * temp.float(), None


class ActFun_012(torch.autograd.Function):
        
    @staticmethod
    def forward(ctx, input, b):
        ctx.save_for_backward(input)
        ctx.b = b
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        original_bp = False
        if original_bp:
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            temp = abs(input - thresh) < lens
        else:
            input, = ctx.saved_tensors
            device = input.device
            grad_input = grad_output.clone()
            # temp = abs(input - thresh) < lens
            b = torch.tensor(ctx.b,device=device)
            temp = (abs(input - 1) < 2)

        return grad_input * temp.float(), None

class SNN_2d(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d, self).__init__()
        
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        #self.alpha_diffb = nn.Parameter(1e-3*torch.ones(3).cuda(),requires_grad=True)
        self.bn = nn.BatchNorm2d(output_c)

    def forward(self, input, param): #20


        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        if param['mixed_at_mem']:
            return mem_this

        device = input.device
        if param['is_first']:
            self.mem = torch.zeros_like(self.conv1(input), device=device)

        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b) 
        self.mem = self.mem * decay * (1. - spike) 
        return spike

class SNN_2d_Super(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_2d_Super, self).__init__()
        self.super_index = N_diffb_mode
        self.super_diffb = N_diffb_delta
        self.alpha_diffb = nn.Parameter(1e-3*torch.ones(len(self.super_index)).cuda(),requires_grad=True)
        self.snn_optimal = SNN_2d(input_c, output_c, kernel_size=kernel_size, stride=stride, b=b, padding=padding)
        for i in self.super_index:
            exec('self.snn_%s = SNN_2d(input_c, output_c, kernel_size=kernel_size, stride=stride, b=b, padding=padding)'%i)
        #self.named_parameters = [i[0] for i in self.snn_optimal.named_parameters()]
        self.mem = None
        self.act_fun = ActFun_changeable().apply
        
    def forward(self, input, param):
        
        # print(param)
        mode = param['mode']
        if mode == 'optimal':
            spike = self.snn_optimal(input, param)
            
        elif mode == 'combine':
            param['mixed_at_mem'] = True
            n_a = F.softmax(self.alpha_diffb,dim=0)
            # mem_combine = torch.zeros_like(self.snn_10(input, param),device=input.device)
            if param['is_first']:
                self.mem = torch.zeros_like(self.snn_10(input, param), device=input.device)

            for i in range(len(self.super_index)):
                mem_this = eval('self.snn_%s(input, param)'%self.super_index[i])
                exec('self.mem = self.mem + mem_this*n_a[i]')
            spike = self.act_fun(self.mem, self.snn_optimal.b)
            param['mixed_at_mem'] = False
        else:
            if param['is_first']:
                exec('self.snn_%s.b = self.snn_optimal.b + self.super_diffb[%d]'%(mode,int(mode)-(10-N_diffb)))
            spike = eval('self.snn_%s(input, param)'%mode)
            
        return spike

class SNN_Avgpooling(nn.Module):

    def __init__(self, kernel_size,stride,padding,b=3):
        super(SNN_Avgpooling, self).__init__()

        # self.fc1 = nn.Linear(input_c, output_c)
        # self.conv2 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=1, padding=padding)
        # self.bn = nn.BatchNorm2d(output_c)
        self.pooling = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None

    # def forward(self, input, mem=None, b_mid=5): #20
    def forward(self, input, b_mid=5): #20

        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(self.pooling(input), device=device)
        self.mem = self.mem.clone().detach() + self.pooling(input)
        # mem = mem.clone().detach() + self.bn(self.conv1(input.clone()))
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        # mem = self.bn(mem)
        return spike

class SNN_Adaptivepooling(nn.Module):

    def __init__(self, dimension, b=3):
        super(SNN_Adaptivepooling, self).__init__()

        # self.fc1 = nn.Linear(input_c, output_c)
        # self.conv2 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=1, padding=padding)
        # self.bn = nn.BatchNorm2d(output_c)
        self.mem = None
        self.pooling = nn.AdaptiveAvgPool2d(dimension)
        self.act_fun = ActFun_changeable().apply
        self.b = b

    def forward(self, input, b_mid=5): #20

        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(self.pooling(input), device=device)
        self.mem = self.mem.clone().detach() + self.pooling(input)
        # mem = mem.clone().detach() + self.bn(self.conv1(input.clone()))
        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        # mem = self.bn(mem)
        return spike

class SNN_2d_fc(nn.Module):

    def __init__(self, input_c, output_c, b=3):
        super(SNN_2d_fc, self).__init__()

        self.fc1 = nn.Linear(input_c, output_c)
        self.mem = None
        # self.conv2 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm1d(output_c)
        self.act_fun = ActFun_changeable().apply
        self.b = b

    def forward(self, input, b_mid=5): #20
        
        if not self.bn.training:
            linear_bn = fuse_linear_bn_eval(self.fc1,self.bn)
            output = linear_bn(input)
        else:
            output = self.bn(self.fc1(input))
        
        return output

        device = input.device
        if self.mem is None:
            self.mem = torch.zeros_like(self.fc1(input), device=device)
            
        self.mem = self.mem.clone().detach() + output

        spike = self.act_fun(self.mem, self.b)
        self.mem = self.mem * decay * (1. - spike)
        return spike

class SNN_3d(nn.Module):

    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_3d, self).__init__()
        self.act_fun = ActFun_changeable().apply
        self.b = b
        self.mem = None
        self.bn = nn.BatchNorm3d(output_c)
        self.conv1 = nn.Conv3d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, input, param): #20
        device = input.device
        if param['is_first'] and not param['mixed_at_mem']:
            self.mem = torch.zeros_like(self.conv1(input), device=device)

        if not self.bn.training:
            conv_bn = fuse_conv_bn_eval(self.conv1,self.bn)
            mem_this = conv_bn(input)
        else:
            mem_this = self.bn(self.conv1(input))

        if param['mixed_at_mem']:
            # print('mem out!!!!!')
            return mem_this
        # print('normal out!!!')
        self.mem = self.mem + mem_this
        spike = self.act_fun(self.mem, self.b) 
        self.mem = self.mem * decay * (1. - spike) 
        return spike


# for name in self.named_parameters:

    # print(eval('self.snn_optimal.%s'%name).requires_grad)
    # exec('self.snn_s.%s = self.snn_optimal.%s.clone()'%(name, name))
    # # optimal = eval('self.snn_optimal.%s'%name)
    # # print(optimal)
    # # exec('self.snn_s.%s = optimal.data.clone()'%name) 
    # eval('self.snn_s.%s'%name).requires_grad_(True)
    # print(eval('self.snn_s.%s'%name).requires_grad)
    # print(eval('self.snn_optimal.%s'%name).requires_grad)
    # a=cc
    # self.snn_s.load_state_dict(self.snn_optimal.state_dict())

class SNN_3d_Super(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1, padding=1, b=3):
        super(SNN_3d_Super, self).__init__()
        self.super_index = N_diffb_mode
        self.super_diffb = N_diffb_delta
        self.alpha_diffb = nn.Parameter(1e-3*torch.ones(len(self.super_index)).cuda(),requires_grad=True)
        self.snn_optimal = SNN_3d(input_c, output_c, kernel_size=kernel_size, stride=stride, b=b, padding=padding)
        for i in self.super_index:
            exec('self.snn_%s = SNN_3d(input_c, output_c, kernel_size=kernel_size, stride=stride, b=b, padding=padding)'%i)
        self.named_parameters = [i[0] for i in self.snn_optimal.named_parameters()]

    def forward(self, input, param):
        
        # print(param)
        mode = param['mode']
        param['mixed_at_mem'] = True
        if mode == 'optimal':
            spike = self.snn_optimal(input, param)
        elif mode == 'combine':
            n_a = F.softmax(self.alpha_diffb,dim=0)
            spike_combine = torch.zeros_like(self.snn_10(input, param),device=input.device)
            for i in range(len(self.super_index)):
                spike = eval('self.snn_%s(input, param)'%self.super_index[i])
                exec('spike_combine+=spike*n_a[i]')
            return spike_combine
        else:
            if param['is_first']:
                exec('self.snn_%s.b = self.snn_optimal.b + self.super_diffb[%d]'%(mode,int(mode)-(10-N_diffb)))
            spike = eval('self.snn_%s(input, param)'%mode)
            
        return spike

class SCNN(nn.Module):
    def __init__(self, input_c, output_c):
        super(SCNN, self).__init__()

        self.cfg_cnn = [(input_c, 32, 1, 1, 3),
                (32, output_c, 1, 1, 3),]
        # kernel size
        self.cfg_kernel = [28, 14, 7]
        # fc layer
        # self.cfg_fc = [128, 10]

        in_planes, out_planes, stride, padding, kernel_size = self.cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = self.cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        # self.fc1 = nn.Linear(self.cfg_kernel[-1] * self.cfg_kernel[-1] * self.cfg_cnn[-1][1], self.cfg_fc[0])
        # self.fc2 = nn.Linear(self.cfg_fc[0], self.cfg_fc[1])

    def forward(self, input, time_window = 10): #20
        device = input.device
        c1_mem = c1_spike = c1_sumspike = torch.zeros(input.shape[0], self.cfg_cnn[0][1], input.shape[-2], input.shape[-1], device=device)
        c2_mem = c2_spike = c2_sumspike = torch.zeros(input.shape[0], self.cfg_cnn[1][1], int(input.shape[-2]/2), int(input.shape[-1]/2), device=device)

        # c1_mem = c1_spike = c1_sumspike = torch.zeros(batch_size, self.cfg_cnn[0][1], self.cfg_kernel[0], self.cfg_kernel[0], device=device)
        # c2_mem = c2_spike = c2_sumspike = torch.zeros(batch_size, self.cfg_cnn[1][1], self.cfg_kernel[1], self.cfg_kernel[1], device=device)

        # h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, self.cfg_fc[0], device=device)
        # h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, self.cfg_fc[1], device=device)

        # print('c1_mem',c1_mem.shape)
        # print('c2_mem',c2_mem.shape)
        
        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)
            c1_sumspike += c1_spike

            x = F.avg_pool2d(c1_spike, 2)
            # print('x',x.shape)
            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike)
            c2_sumspike += c2_spike

            # print('c1_mem',c1_mem.shape)
            # print('c2_mem',c2_mem.shape)
            # x = F.avg_pool2d(c2_spike, 2)
            

            # x = x.view(batch_size, -1)

            # h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            # h1_sumspike += h1_spike
            # h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            # h2_sumspike += h2_spike

        outputs = c2_sumspike / time_window
        return outputs, 



class SCNN_single_conv(nn.Module):
    def __init__(self, input_c, output_c):
        super(SCNN_single_conv, self).__init__()

        self.cfg_cnn = [(input_c, output_c, 1, 1, 3)]

        in_planes, out_planes, stride, padding, kernel_size = self.cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)


    def forward(self, input, time_window = 16): #20
        device = input.device
        c1_mem = c1_spike = c1_sumspike = torch.zeros(input.shape, device=device)
        x = input > torch.rand(input.size(), device=device) # prob. firing

        # print('c1_mem',c1_mem.shape)
        
        for step in range(time_window): # simulation time steps
            step_input = x[:,step]

            c1_mem[:,step], c1_spike[:,step] = mem_update(self.conv1, step_input.float(), c1_mem[:,step], c1_spike[:,step])
            # c1_sumspike[:,step] += c1_spike[:,step]

            # print('c1_mem[:,step]',c1_mem[:,step].shape)

        # outputs = c1_sumspike / time_window
        return c1_spike 

class SCNN_frontend(nn.Module):
    def __init__(self, input_c, output_c):
        super(SCNN_frontend, self).__init__()

        self.cfg_cnn = [(input_c, output_c, 1, 1, 3)]

        in_planes, out_planes, stride, padding, kernel_size = self.cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_b5().apply
        # self.decay = nn.Parameter((0.5-0.8)*torch.rand(16,1,1)+0.8)
        # self.decay.requires_grad = True
        self.decay = 0.2
        self.con1x1 = nn.Conv2d(input_c,output_c,kernel_size=1)
    def mem_update(self, ops, x, mem, spike):
        # mem = mem.clone() * self.decay * (1. - spike) + ops(x)
        mem = mem * self.decay * (1. - spike) + ops(x)
        spike = self.act_fun(mem) 
        return mem, spike

    def forward(self, input): #20
        device = input.device
        # self.decay = self.decay
        c1_mem = c1_spike = torch.zeros([input.shape[0], input.shape[1], self.cfg_cnn[0][1], input.shape[-2], input.shape[-1]], device=device)
        # print('c1_mem',c1_mem.shape)
        #x = input > torch.rand(input.size(), device=device) # prob. firing

        S = input.shape[1]
        for step in range(S): # simulation time steps
            #x = input[:,step] > torch.rand(input[:,step].size(), device=device) # prob. firing
            
            use_input_replace_mem = True
            use_abs_input = False
            if use_input_replace_mem:
                x = input[:,step]
                c1_mem[:,step] = self.con1x1(x.clone())
                # print('after c1_mem',c1_mem.shape)
            elif use_abs_input:
                x = torch.abs(input[:,step])
            
            # print('x',x.shape)
            c1_mem[:,step], c1_spike[:,step] = self.mem_update(self.conv1, x.float(), c1_mem[:,step], c1_spike[:,step])

            # BUG 
        return c1_mem

class SCNN_single_conv_big_forloop_b3(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1):
        super(SCNN_single_conv_big_forloop_b3, self).__init__()
        padding = 1
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_b3().apply

    def forward(self, input, mem=None): #20
        device = input.device
        # print('input',input.shape)
        # print('mem',type(mem))
        if mem is None:
            mem = torch.zeros_like(self.conv1(input), device=device)


        # print('c1_mem',mem.shape)
        # x = input > torch.rand(input.size(), device=device) # prob. firing

        mem = mem + self.conv1(input)
        spike = self.act_fun(mem) 
        mem = mem * decay * (1. - spike)

        return spike, mem

class SCNN_single_conv_big_forloop_b5(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1):
        super(SCNN_single_conv_big_forloop_b5, self).__init__()
        padding = 1
        self.conv1 = nn.Conv2d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_b5().apply

    def forward(self, input, mem=None): #20
        device = input.device
        # print('input',input.shape)
        # print('mem',type(mem))
        if mem is None:
            mem = torch.zeros_like(self.conv1(input), device=device)


        # print('c1_mem',mem.shape)
        # x = input > torch.rand(input.size(), device=device) # prob. firing

        mem = mem + self.conv1(input)
        spike = self.act_fun(mem) 
        mem = mem * decay * (1. - spike)

        return spike, mem

class SCNN_single_conv_big_forloop_3d_b3(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1):
        super(SCNN_single_conv_big_forloop_3d_b3, self).__init__()
        padding = 1
        self.conv1 = nn.Conv3d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_b3().apply


    def forward(self, input, mem=None): #20
        device = input.device
        if mem is None:
            mem = torch.zeros_like(self.conv1(input), device=device)

        mem = mem + self.conv1(input)
        spike = self.act_fun(mem) 
        mem = mem * decay * (1. - spike)

        return spike, mem

class SCNN_single_conv_big_forloop_3d_b5(nn.Module):
    def __init__(self, input_c, output_c, kernel_size=3, stride=1):
        super(SCNN_single_conv_big_forloop_3d_b5, self).__init__()
        padding = 1
        self.conv1 = nn.Conv3d(input_c, output_c, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act_fun = ActFun_b5().apply


    def forward(self, input, mem=None): #20
        device = input.device
        if mem is None:
            mem = torch.zeros_like(self.conv1(input), device=device)

        mem = mem + self.conv1(input)
        spike = self.act_fun(mem) 
        mem = mem * decay * (1. - spike)

        return spike, mem