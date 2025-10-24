import torch
from torch import nn



class GroupExtractor(nn.Module):

    def __init__(self,
                 in_channel:int=4,
                 type:str='L'):
        super().__init__()
        self.in_channel = in_channel
        if type.lower() in ['l', 'l_8']:
            widths = (64, 128, 256, 512)
            num_groups = 8
            init_down = False
        elif type.lower()=='l_4':
            widths = (64, 128, 256, 512)
            num_groups = 4
            init_down = False
        elif type.lower() in ['m', 'm_16']:
            widths = (32, 64, 128, 256)
            num_groups = 16
            init_down = False
        elif type.lower() in ['m_8']:
            widths = (32, 64, 128, 256)
            num_groups = 8
            init_down = False
        elif type.lower()=='s' or type.lower()=='s_32':
            widths = (16, 32, 64, 128)
            num_groups = 32
            init_down = False
        elif type.lower()=='s_16':
            widths = (16, 32, 64, 128)
            num_groups = 16
            init_down = False
        elif type.lower()=='xs':
            widths = (16, 32, 64, 128)
            num_groups = 16
            init_down = False
        elif type.lower() in ['xxs', 's_8']:
            widths = (16, 32, 64, 128)
            num_groups = 8
            init_down = False
        elif type.lower()=='conv3_8':
            widths = (128, 128, 128)
            num_groups = 8
            init_down = True
        elif type.lower()=='conv3_16':
            widths = (128, 128, 128)
            num_groups = 16
            init_down = True
        elif type.lower()=='single':
            widths = (64, 128, 256, 512)
            num_groups = 1
            init_down = False
        elif type.lower()=='one':
            widths = (16*16,)
            num_groups = 1
            init_down = False
        elif type.lower()=='two':
            widths = (16, 32)
            num_groups = 16
            init_down = False
        elif type.lower()=='three':
            widths = (16, 32, 64)
            num_groups = 16
            init_down = False
        elif type.lower()=='flatten':
            widths = None
            num_groups = None
            init_down = False
        elif type.lower()=='none':
            widths = None
            num_groups = 1
            init_down = False
        elif type.lower()=='same_1':
            widths = (16, )
            num_groups = 1
            init_down = True
        elif type.lower()=='same_2':
            widths = (16, 64)
            num_groups = 1
            init_down = True
        elif type.lower()=='same_3':
            widths = (16, 64, 256)
            num_groups = 1
            init_down = True
        else:
            raise NotImplementedError
        self.name = type
        self.num_groups = num_groups
        if type.lower() not in ['flatten', 'one', 'none', 'same_1']:
            if init_down:
                self.projector = nn.Conv2d(self.in_channel, widths[0]*num_groups, 4, 2, 1)
            else:
                self.projector = nn.Conv2d(self.in_channel, widths[0]*num_groups, 3, 1, 1)
            self.act0 = nn.LeakyReLU()
            bulk_lst = []
            for wd_idx in range(len(widths)-1):
                bulk_lst.append(nn.Conv2d(widths[wd_idx]*num_groups, widths[wd_idx+1]*num_groups, 4, 2, 1, groups=num_groups))
                if wd_idx != len(widths)-2:
                    bulk_lst.append(nn.LeakyReLU())
            self.bulk = nn.Sequential(*bulk_lst)
        elif type.lower() == 'flatten':
            self.projector = nn.Flatten()
            self.act0 = nn.Sequential()
            self.bulk = nn.Sequential()
        elif type.lower()=='one':
            self.projector = nn.Conv2d(self.in_channel, widths[0]*num_groups, 3, 1, 1)
            self.act0 = nn.Sequential()
            self.bulk = nn.Sequential()
        elif type.lower()=='none':
            self.projector = nn.Sequential()
            self.act0 = nn.Sequential()
            self.bulk = nn.Sequential()
        elif type.lower()=='same_1':
            self.projector = nn.Conv2d(self.in_channel, widths[0]*num_groups, 3, 1, 1)
            self.act0 = nn.Sequential()
            self.bulk = nn.Sequential()
        return None
    
    def forward(self, x):
        if self.num_groups is None:
            return self.projector(x)[:,:,None,None]
        else:
            return self.bulk(self.act0(self.projector(x)))
        

        

        


class Projector(nn.Module):

    def __init__(self):
        self.in_channel = 4
        widths = (16, 32, 64, 128)
        self.num_groups = 16
        self.projector = nn.Conv2d(self.in_channel, widths[0]*self.num_groups, 3, 1, 1)
        self.act0 = nn.LeakyReLU()
        bulk_lst = []
        for wd_idx in range(len(widths)-1):
            bulk_lst.append(nn.Conv2d(widths[wd_idx]*self.num_groups, widths[wd_idx+1]*self.num_groups, 4, 2, 1, groups=self.num_groups))
            if wd_idx != len(widths)-2:
                bulk_lst.append(nn.LeakyReLU())
        self.bulk = nn.Sequential(*bulk_lst)
    
    def forward(self, x):
        return self.bulk(self.act0(self.projector(x)))