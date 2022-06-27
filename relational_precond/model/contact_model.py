import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from relational_precond.model.pointconv_util_groupnorm import PointConvDensitySetAbstraction

class SpatialClassifier(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifier, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        self.fc_spatial = nn.Linear(inp_emb_size, 2)
        self.fc_right = nn.Linear(inp_emb_size, 1)
        self.fc_left = nn.Linear(inp_emb_size, 1)
        self.fc_front = nn.Linear(inp_emb_size, 1)
        self.fc_behind = nn.Linear(inp_emb_size, 1)

        # self.model = nn.Sequential(
        #     nn.Linear(self.inp_emb_size + self.action_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, self.inp_emb_size)
        # )

    def forward(self, inp):
        x = inp
        out = self.model(x)
        return out

    def forward_image_right(self, z):

        #prediction = F.softmax(self.fc_right(z), dim=1)
        prediction = F.relu(self.fc_right(z))
        return prediction

    def forward_image_left(self, z):

        #prediction = F.softmax(self.fc_left(z), dim=1)
        prediction = F.relu(self.fc_left(z))
        return prediction

    def forward_image_front(self, z):

        #prediction = F.softmax(self.fc_front(z), dim=1)
        prediction = F.relu(self.fc_front(z))
        return prediction

    def forward_image_behind(self, z):

        #prediction = F.softmax(self.fc_behind(z), dim=1)
        prediction = F.relu(self.fc_behind(z))
        return prediction

class SpatialClassifierHorizon(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierHorizon, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_right = nn.Linear(inp_emb_size, 2)
        self.fc_horizon = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_horizon = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )
        # self.fc_horizon_1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)
        # )
        # self.fc_right = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # )
        # self.fc1 = nn.Linear(inp_emb_size, 128)
        # self.bn1 = nn.GroupNorm(1, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.GroupNorm(1, 64)
        # self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.GroupNorm(1, 32)
        # self.fc5 = nn.Linear(32, 2)

    def forward_image_horizon(self, z, MSE = False):
        if not MSE:
            prediction = F.softmax(self.fc_horizon(z), dim = 1)
        else:
            prediction = F.relu(self.fc_horizon_1(z))
        return prediction


class SpatialClassifierRight(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierRight, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_right = nn.Linear(inp_emb_size, 2)
        self.fc_right = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_right = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )
        # self.fc1 = nn.Linear(inp_emb_size, 128)
        # self.bn1 = nn.GroupNorm(1, 128)
        # self.fc3 = nn.Linear(128, 64)
        # self.bn3 = nn.GroupNorm(1, 64)
        # self.fc4 = nn.Linear(64, 32)
        # self.bn4 = nn.GroupNorm(1, 32)
        # self.fc5 = nn.Linear(32, 2)

    def forward_image_right(self, z, MSE = False):
        if not MSE:
            prediction = F.softmax(self.fc_right(z), dim = 1)
        else:
            prediction = F.relu(self.fc_right_1(z))
        return prediction

class SpatialClassifierLeft(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierLeft, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_left = nn.Linear(inp_emb_size, 2)
        self.fc_left = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_left = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )
        # self.fc_left_1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)
        # )


    def forward_image_left(self, z, MSE = False):
        if not MSE:
            prediction = F.softmax(self.fc_left(z), dim = 1)
        else:
            prediction = F.relu(self.fc_left_1(z))
        return prediction

class SpatialClassifierVertical(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierVertical, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_front = nn.Linear(inp_emb_size, 2)
        self.fc_vertical = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_vertical = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )
        # self.fc_vertical_1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)
        # )


    def forward_image_vertical(self, z, MSE = False):
        if not MSE:
            prediction = F.softmax(self.fc_vertical(z), dim = 1)
        else:
            prediction = F.relu(self.fc_vertical_1(z))
        return prediction


class SpatialClassifierFront(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierFront, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_front = nn.Linear(inp_emb_size, 2)
        self.fc_front = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_front = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )
        # self.fc_front_1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)
        # )


    def forward_image_front(self, z, MSE = False):
        if not MSE:
            prediction = F.softmax(self.fc_front(z), dim = 1)
        else:
            prediction = F.relu(self.fc_front_1(z))
        return prediction

class SpatialClassifierBehind(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierBehind, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_behind = nn.Linear(inp_emb_size, 2)
        self.fc_behind = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_behind = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )
        # self.fc_behind_1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 1)
        # )

    def forward_image_behind(self, z, MSE = False):
        if not MSE:
            prediction = F.softmax(self.fc_behind(z), dim = 1)
        else:
            prediction = F.relu(self.fc_behind_1(z))
        return prediction


class SpatialClassifierStack(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierStack, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_stack = nn.Linear(inp_emb_size, 2)
        self.fc_stack = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_stack = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )

    def forward_image_stack(self, z):

        prediction = F.softmax(self.fc_stack(z), dim = 1)
        #prediction = F.relu(self.fc_stack(z))
        return prediction

class SpatialClassifierTop(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierTop, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_stack = nn.Linear(inp_emb_size, 2)
        self.fc_top = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_top = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )

    def forward_image_top(self, z):

        prediction = F.softmax(self.fc_top(z), dim = 1)
        #prediction = F.relu(self.fc_stack(z))
        return prediction

class SpatialClassifierBelow(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SpatialClassifierBelow, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_stack = nn.Linear(inp_emb_size, 2)
        self.fc_below = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)
        )
        # self.fc_below = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 2)
        # )

    def forward_image_below(self, z):

        prediction = F.softmax(self.fc_below(z), dim = 1)
        #prediction = F.relu(self.fc_stack(z))
        return prediction

class SigmoidRelations(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(SigmoidRelations, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_stack = nn.Linear(inp_emb_size, 2)
        
        self.fc_relations = nn.Sequential(
            nn.Linear(256, 128),
            nn.GroupNorm(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        ) # version 4-8
        
        # self.fc_relations = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.GroupNorm(1, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 2),
        #     nn.Sigmoid()
        # ) # version 4-8
        
        # self.fc_relations = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.GroupNorm(1, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 32),
        #     nn.GroupNorm(1, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 6),
        #     nn.Sigmoid()
        # ) # version 3-5

        # self.fc_relations = nn.Sequential(
        #     nn.Linear(32, 32),
        #     nn.GroupNorm(1, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 6),
        #     nn.Sigmoid()
        # ) # version Feb


    def forward_image_relations(self, z):

        prediction = self.fc_relations(z)
        #prediction = F.relu(self.fc_stack(z))
        return prediction

class SigmoidRelations_1(nn.Module):
    def __init__(self):
        super(SigmoidRelations_1, self).__init__()
        
        self.fc_relations = nn.Sequential(
            nn.Linear(256, 128),
            nn.GroupNorm(1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
            nn.Sigmoid()
        ) # version 4-8
        
        # self.fc_relations = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.GroupNorm(1, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 2),
        #     nn.Sigmoid()
        # ) # version 4-8
        
        # self.fc_relations = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.GroupNorm(1, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(64, 32),
        #     nn.GroupNorm(1, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 6),
        #     nn.Sigmoid()
        # ) # version 3-5

        # self.fc_relations = nn.Sequential(
        #     nn.Linear(32, 32),
        #     nn.GroupNorm(1, 32),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(32, 6),
        #     nn.Sigmoid()
        # ) # version Feb


    def forward_image_relations(self, z):

        prediction = self.fc_relations(z)
        #prediction = F.relu(self.fc_stack(z))
        return prediction


class Contrasive(nn.Module):
    def __init__(self, inp_emb_size, args):
        super(Contrasive, self).__init__()
        self.inp_emb_size = inp_emb_size
        self.args = args
        #self.fc_stack = nn.Linear(inp_emb_size, 2)
        self.fc_contrasive = nn.Sequential(
            nn.Linear(32, 32),
            nn.GroupNorm(1, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 128)
        )
        # self.fc_contrasive = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(1, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 128)
        # )

    def forward_image_contrasive(self, z):

        prediction = self.fc_contrasive(z)
        #prediction = F.relu(self.fc_stack(z))
        return prediction


class PointConv(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        # self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[512], bandwidth = 0.4, group_all=True)
        
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[128], bandwidth = 0.4, group_all=True) # version 3-5

        # self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[16], bandwidth = 0.2, group_all=True) # version feb
        
        
        #self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[128], bandwidth = 0.4, group_all=True)  

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))



        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        # x = self.fc5(x)

        return x


class PointConv_planar(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv_planar, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        # self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        # self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        # self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[512], bandwidth = 0.4, group_all=True)
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=64 + 3, mlp=[16], bandwidth = 0.2, group_all=True)
        #self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[128], bandwidth = 0.4, group_all=True)  

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        #l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l2_points.view(B, 16)
        # x = F.relu(self.bn1(self.fc1(x)))
        # x = F.relu(self.bn2(self.fc2(x)))



        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        # x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        # x = self.fc5(x)

        return x
