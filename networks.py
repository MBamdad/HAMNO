import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
import config
def get_grid_2d(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def get_grid_3d(shape, device):
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# Complex multiplication
def compl_mul2d(inp, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", inp, weights)


def compl_mul3d(inp, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients --  1. Compute Fourier Transform
        # Now, instead of working with raw pixel/grid values, we work with frequency components\
        # Suppose the input x has shape (batch, in_channels, H, W)
        # x_ft has shape: batch×in_channels×H×(W/2+1)
        x_ft = torch.fft.rfft2(x) # 2D Fast Fourier Transform (FFT) --> v0 = F(v0)

        # Multiply relevant Fourier modes -- 2. Apply Spectral Convolution
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        # v1(k1,k2)= W(k1,k2).v0(k1,k2) --> W(k1,k2) are learnable parameters that control how much each frequency mode contributes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1))) # Apply Inverse Fourier Transform (iFFT) --> v1(x,y) = F^(-1)[v1(k1,k2)]
        return x


class SpectralConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]

        # Compute Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Safe effective modes for current tensor size.
        # m1/m2 use //2 to avoid overlap between positive and negative modes.
        # m3 uses rFFT size directly because the last dimension is one-sided.
        m1 = min(self.modes1, x_ft.size(-3) // 2, self.weights1.size(-3))
        m2 = min(self.modes2, x_ft.size(-2) // 2, self.weights1.size(-2))
        m3 = min(self.modes3, x_ft.size(-1), self.weights1.size(-1))

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        out_ft[:, :, :m1, :m2, :m3] = \
            compl_mul3d(
                x_ft[:, :, :m1, :m2, :m3],
                self.weights1[:, :, :m1, :m2, :m3]
            )

        out_ft[:, :, -m1:, :m2, :m3] = \
            compl_mul3d(
                x_ft[:, :, -m1:, :m2, :m3],
                self.weights2[:, :, :m1, :m2, :m3]
            )

        out_ft[:, :, :m1, -m2:, :m3] = \
            compl_mul3d(
                x_ft[:, :, :m1, -m2:, :m3],
                self.weights3[:, :, :m1, :m2, :m3]
            )

        out_ft[:, :, -m1:, -m2:, :m3] = \
            compl_mul3d(
                x_ft[:, :, -m1:, -m2:, :m3],
                self.weights4[:, :, :m1, :m2, :m3]
            )

        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, T=1, num_layers=2):
        """
        Initialize the MLP2d class.
        Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - mid_channels: Number of intermediate channels.
        - T: Number of blocks (default=1).
        - num_layers: Number of layers in each block (default=2).
        """
        super(MLP2d, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(T):
            self.layers.append(nn.Conv2d(in_channels, mid_channels, 1))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Conv2d(mid_channels, mid_channels, 1))
            self.layers.append(nn.Conv2d(mid_channels, out_channels, 1))

    def forward(self, x, t=0):
        start = t * self.num_layers
        end = start + self.num_layers
        for i in range(start, end - 1):
            x = F.gelu(self.layers[i](x))
        x = self.layers[end - 1](x)
        return x


class MLP3d(MLP2d):
    def __init__(self, in_channels, out_channels, mid_channels, T=1, num_layers=2):
        super(MLP3d, self).__init__(in_channels, out_channels, mid_channels, T, num_layers)

        self.layers = nn.ModuleList()
        for _ in range(T):
            self.layers.append(nn.Conv3d(in_channels, mid_channels, 1))
            # After (3x3x3 kernel)
            #self.layers.append(nn.Conv3d(in_channels, mid_channels, 3, padding=1))  ## Changed from 1*1*1 to 3*3*3
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Conv3d(mid_channels, mid_channels, 1))
                #self.layers.append(nn.Conv3d(mid_channels, mid_channels, 3, padding=1))  ## Changed from 1 to 3
            self.layers.append(nn.Conv3d(mid_channels, out_channels, 1))
            #self.layers.append(nn.Conv3d(mid_channels, out_channels, 3, padding=1))  ## Changed from 1 to 3


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, width_q, T_in, T_out, n_layers):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 8  # pad the domain if input is non-periodic
        self.n_layers = n_layers

        self.p = nn.Linear(T_in + 2, self.width)  # We start with an input x == u(x,y) of shape (batch,x,y,c), We lift it to a higher-dimensional space using a linear layer
        # v0(x,y) = p(x=u)
        self.convs = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(n_layers)]) # 2D Fast Fourier Transform (FFT) --> v0 = F(v0)
        self.mlps = nn.ModuleList([MLP2d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)]) # Pointwise convolution layers
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP2d(self.width, 1, self.width_q)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = get_grid_2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            '''
            x1 = self.mlps[i](x1): Local Mixing Using MLP: Since the Fourier convolution captures global dependencies,
             we still need local interactions --> v_i+1 = sigma(W.vi  +  b), 
             which W and b are learnable parameters, and σ is the activation function.
            '''
            x2 = self.ws[i](x)
            '''
             x2 = self.ws[i](x): applies a pointwise convolution (1×1 convolution) to the input tensor x.
                self.ws is a list (nn.ModuleList) of 1×1 convolutional layers.
                Each self.ws[i] is a 2D convolution layer (nn.Conv2d) with a kernel size of 1x1.
                The purpose of these layers is to perform a linear transformation of the feature maps 
                without mixing spatial locations.

            '''
            x = x1 + x2 #  Merge Global and Local Representations
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        '''
         Output Projection back to the desired shape using another MLP
        v_out = Q.v_final(x,y)
        Q is a learnable projection.

        '''
        x = x.permute(0, 2, 3, 1)
        #The final shape of x is (batch,x,y,1), which represents the predicted function value at each spatial location.
        return x


class TNO2d(FNO2d):
    def __init__(self, modes1, modes2, width, width_q, width_h, T_in, T_out, n_layers, n_layers_q=2, n_layers_h=4):
        super(TNO2d, self).__init__(modes1, modes2, width, width_q, T_in, T_out, n_layers)
        '''
         TNO2d extends FNO2d. It introduces temporal modeling by adding two MLP layers:
        self.q → projects the Fourier features to output over time.
        self.h → handles temporal dependencies between consecutive time steps.
        New parameters added:
        width_h → controls temporal memory features.
        n_layers_q → depth of self.q (output MLP).
        n_layers_h → depth of self.h (temporal evolution MLP).
        '''
        self.width_h = width_h
        #self.q = MLP2d(self.width, 1, self.width, T_out) # for AC
        #self.q2 = MLP2d(1, 1, self.width // 4, T_out - 1)
        #self.q = MLP2d(self.width, 1, 2 * self.width, T_out)  # for CH3D
        #self.q2 = MLP2d(1, 1, self.width, T_out - 1)
        self.q = MLP2d(self.width, 1, self.width_q, T_out, n_layers_q)  # for CHNL
        self.h = MLP2d(1, 1, self.width_h, T_out - 1, n_layers_h)

    def forward(self, x):
        grid = get_grid_2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1) # a(x) or= x : Input function (e.g., initial condition for a PDE)
        x = self.p(x) # 	Lifts input to a high-dimensional space
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x # x′=GELU(FourierConv(x)+MLP(x)+PointwiseConv(x)

        # x = x[..., :-self.padding, :-self.padding]
        '''
         Temporal Evolution Loop
        Initial time step prediction:
        Uses self.q(x) to generate the first time step.
        Stores result in X[..., 0]
        '''
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)
        xt = self.q(x)
        X[..., 0] = xt.permute(0, 2, 3, 1).squeeze(-1)

        for t in range(1, self.T_out):
            x1 = self.q(x, t) # Predicts the next step using Fourier features.  # Q_n∘(W_L+ K_L )∘...∘P(a(x)), Projects final Fourier features to outpu
            x2 = self.h(xt, t - 1) # Uses previous output (xt) to refine the next state. # H_n∘G_θ (x,t_(n-1) )(a(x)), Models dependency on past states
            xt = x1 + x2 #  Solution at time t_n : x_t = G_θ (x,t_n )(a(x))
            X[..., t] = xt.permute(0, 2, 3, 1).squeeze(-1)
            '''
             Uses previous output (xt) to refine the next state.
            Combines both predictions --> x_t=MLP_q(x)+MLP_h[(x t−1)]
            Stores result in X[..., t]
            '''
        return X


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, width_q, T_in, T_out, n_layers, n_layers_q=2, n_layers_h=2):
        super(FNO3d, self).__init__()

        """
        The FNO3d class is a deep learning model designed for solving spatiotemporal problems. 
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 time_steps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 time_steps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 6  # pad the domain if input is non-periodic
        self.n_layers = n_layers

        self.p = nn.Linear(self.T_in + 3, self.width)  # Lifting Layer: input channel is 12: the solution of the first 10 time_steps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.convs = nn.ModuleList(
            [SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([MLP3d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        #self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 3, padding=1) for _ in range(n_layers)])  ## kernel changed
        #self.q = MLP3d(self.width, 1, self.width)  # output channel is 1: u(x, y)
        self.q = MLP3d(self.width, 1, self.width_q)  # output channel is 1: u(x, y)

    def forward(self, x):
        #x = x.unsqueeze(3).repeat([1, 1, 1, self.T_out, 1])
        grid = get_grid_3d(x.shape, x.device)
        #print(' x shape:', x.shape)
        x = torch.cat((x, grid), dim=-1)
        #print(' x shape after cat:', x.shape)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        #x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        #print(' x shape after permute:', x.shape)
        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        #x = x[..., :-self.padding]
        x = self.q(x)
        #x = x.permute(0, 2, 3, 4, 1)[..., 0]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 4, 1)
        #print('FNO3d return x shape:', x.shape)
        return x


class TNO3d(FNO3d):
    def __init__(self, modes1, modes2, modes3, width, width_q, width_h, T_in, T_out, n_layers):
        super(TNO3d, self).__init__(modes1, modes2, modes3, width, width_q, T_in, T_out, n_layers)
        """
        The super() function calls the parent class (FNO3d) constructor to initialize 
        the parameters that are inherited from the parent class.
        input: the initial condition and locations (a(x, y, z), x, y, z)
        input shape: (batchsize, x=s, y=s, z=s, c=4)
        output: the solution 
        output shape: (batchsize, x=s, y=s, z=s, t=T)
        """
        self.width_h = width_h

        #self.q = MLP3d(self.width, 1, self.width, T_out)
        #self.q2 = MLP3d(1, 1, self.width // 4, T_out - 1)
        self.q = MLP3d(self.width, 1, self.width_q, T_out)
        self.h = MLP3d(1, 1, self.width_h, T_out - 1)

    def forward(self, x):
        grid = get_grid_3d(x.shape, x.device)
        #print('x shape: ',x.shape)
        #print('grid shape: ', grid.shape)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding]
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)
        xt = self.q(x)
        X[..., 0] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)
        for t in range(1, self.T_out):
            x1 = self.q(x, t)
            x2 = self.h(xt, t - 1)
            xt = x1 + x2
            X[..., t] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)

        #print('shape X for model TNO: ', X.shape)
        return X


def get_grid_3D(shape, device):
    batchsize, size_x, size_y, size_z, _ = shape  # Note: last dim is channels, not time
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)  # Returns (batch, x, y, z, 3)


class FNO4d(nn.Module):
    def __init__(self, modes1, modes2, modes3, modes4_internal, width, width_q, T_in_channels, n_layers):
        super(FNO4d, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4_internal
        self.width = width
        self.width_q = width_q
        self.T_in = T_in_channels
        self.n_layers = n_layers
        self.padding = 6

        # Input is (x,y,z) + time channels (t_in_channels) + 3 spatial coordinates
        self.p = nn.Linear(self.T_in + 3, self.width)  # +3 for (x,y,z) coordinates

        self.convs = nn.ModuleList([
            SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
            for _ in range(n_layers)
        ])
        self.mlps = nn.ModuleList([MLP3d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        self.q = MLP3d(self.width, 1, self.width_q)  # Output channel is 1

    def forward(self, x):
        # Input shape: (batch, x, y, z, t_in_channels)
        grid = get_grid_3D(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  # Now both are 5D: (batch, x, y, z, t_in_channels + 3)
        x = self.p(x)  # Lift to higher dimension
        x = x.permute(0, 4, 1, 2, 3)  # (batch, channels, x, y, z)

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        x = self.q(x)
        x = x.permute(0, 2, 3, 4, 1)  # (batch, x, y, z, 1)
        return x

#### FFNO4d    #####
####
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Helpers for complex multiplication (einsum)
# -----------------------------
def compl_mul1d_x(inp, weights):
    # inp: (B, Cin, Kx, Y, Z), w: (Cin, Cout, Kx) -> (B, Cout, Kx, Y, Z)
    return torch.einsum("bcxyz,cok->boxyz", inp, weights)

def compl_mul1d_y(inp, weights):
    # inp: (B, Cin, X, Ky, Z), w: (Cin, Cout, Ky) -> (B, Cout, X, Ky, Z)
    return torch.einsum("bcxyz,coy->boxyz", inp, weights)

def compl_mul1d_z(inp, weights):
    # inp: (B, Cin, X, Y, Kz), w: (Cin, Cout, Kz) -> (B, Cout, X, Y, Kz)
    return torch.einsum("bcxyz,coz->boxyz", inp, weights)

def compl_mul2d_xy(inp, weights):
    # inp: (B, Cin, Kx, Ky, Z)
    # w  : (Cin, Cout, Kx, Ky)
    # out: (B, Cout, Kx, Ky, Z)
    return torch.einsum("b c x y z, c o x y -> b o x y z", inp, weights)

def compl_mul2d_xz(inp, weights):
    # inp: (B, Cin, Kx, Y, Kz)
    # w  : (Cin, Cout, Kx, Kz)
    # out: (B, Cout, Kx, Y, Kz)
    return torch.einsum("b c x y z, c o x z -> b o x y z", inp, weights)

def compl_mul2d_yz(inp, weights):
    # inp: (B, Cin, X, Ky, Kz)
    # w  : (Cin, Cout, Ky, Kz)
    # out: (B, Cout, X, Ky, Kz)
    return torch.einsum("b c x y z, c o y z -> b o x y z", inp, weights)


def complex_weight(shape, scale):
    real = torch.randn(*shape) * scale
    imag = torch.randn(*shape) * scale
    return torch.complex(real, imag)


def complex_weight_pairwise(shape, scale, decay_power=0.5):
    """
    shape: (Cin, Cout, M1, M2)

    Initialize pairwise spectral weights with mode-dependent decay:
    lower-frequency modes start larger, higher-frequency modes smaller.
    """
    cin, cout, m1, m2 = shape

    real = torch.randn(cin, cout, m1, m2) * scale
    imag = torch.randn(cin, cout, m1, m2) * scale

    w = torch.complex(real, imag)

    idx1 = torch.arange(m1, dtype=torch.float32).view(1, 1, m1, 1)
    idx2 = torch.arange(m2, dtype=torch.float32).view(1, 1, 1, m2)

    # mode decay: 1 / ((1+i)(1+j))^(decay_power/2)
    decay = 1.0 / torch.pow((1.0 + idx1) * (1.0 + idx2), decay_power / 2.0)

    return w * decay


class EnhancedFactorizedSpectralConv3d(nn.Module):
    """
    Enhanced FFNO-style operator in 3D:
      K(x) = Kx + Ky + Kz + Kxy + Kxz + Kyz

    - Kx,Ky,Kz: 1D factorized spectral mixing per axis
    - Kxy,Kxz,Kyz: pairwise spectral couplings
    """

    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z, use_pairwise=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_pairwise = use_pairwise

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        pair_scale = 0.5 * scale  # small, practical change

        self.weights_x = nn.Parameter(complex_weight((in_channels, out_channels, modes_x), scale))
        self.weights_y = nn.Parameter(complex_weight((in_channels, out_channels, modes_y), scale))
        self.weights_z = nn.Parameter(complex_weight((in_channels, out_channels, modes_z), scale))

        if self.use_pairwise:
            self.weights_xy = nn.Parameter(complex_weight((in_channels, out_channels, modes_x, modes_y), pair_scale))
            self.weights_xz = nn.Parameter(complex_weight((in_channels, out_channels, modes_x, modes_z), pair_scale))
            self.weights_yz = nn.Parameter(complex_weight((in_channels, out_channels, modes_y, modes_z), pair_scale))

    def forward(self, x):
        """
        x: (B, Cin, X, Y, Z) real
        returns dict of components
        """
        B, Cin, X, Y, Z = x.shape
        device = x.device
        comps = {}

        # ---- 1D: X ----
        x_ft = torch.fft.rfft(x, dim=-3)
        out_ft = torch.zeros(B, self.out_channels, X // 2 + 1, Y, Z, dtype=torch.cfloat, device=device)
        mx = min(self.modes_x, X // 2 + 1)
        out_ft[:, :, :mx, :, :] = compl_mul1d_x(x_ft[:, :, :mx, :, :], self.weights_x[:, :, :mx])
        comps["x"] = torch.fft.irfft(out_ft, n=X, dim=-3)

        # ---- 1D: Y ----
        y_ft = torch.fft.rfft(x, dim=-2)
        out_ft = torch.zeros(B, self.out_channels, X, Y // 2 + 1, Z, dtype=torch.cfloat, device=device)
        my = min(self.modes_y, Y // 2 + 1)
        out_ft[:, :, :, :my, :] = compl_mul1d_y(y_ft[:, :, :, :my, :], self.weights_y[:, :, :my])
        comps["y"] = torch.fft.irfft(out_ft, n=Y, dim=-2)

        # ---- 1D: Z ----
        z_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=device)
        mz = min(self.modes_z, Z // 2 + 1)
        out_ft[:, :, :, :, :mz] = compl_mul1d_z(z_ft[:, :, :, :, :mz], self.weights_z[:, :, :mz])
        comps["z"] = torch.fft.irfft(out_ft, n=Z, dim=-1)

        if not self.use_pairwise:
            return comps

        # ---- 2D: XY ----
        xy_ft = torch.fft.rfftn(x, dim=[-3, -2])
        out_ft = torch.zeros(B, self.out_channels, X, Y // 2 + 1, Z, dtype=torch.cfloat, device=device)

        mx2 = min(self.modes_x, X)
        my2 = min(self.modes_y, Y // 2 + 1)
        out_ft[:, :, :mx2, :my2, :] = compl_mul2d_xy(
            xy_ft[:, :, :mx2, :my2, :], self.weights_xy[:, :, :mx2, :my2]
        )
        out_ft[:, :, -mx2:, :my2, :] = compl_mul2d_xy(
            xy_ft[:, :, -mx2:, :my2, :], self.weights_xy[:, :, :mx2, :my2]
        )
        comps["xy"] = torch.fft.irfftn(out_ft, s=(X, Y), dim=[-3, -2])

        # ---- 2D: XZ ----
        xz_ft = torch.fft.rfftn(x, dim=[-3, -1])
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=device)

        mx2 = min(self.modes_x, X)
        mz2 = min(self.modes_z, Z // 2 + 1)
        out_ft[:, :, :mx2, :, :mz2] = compl_mul2d_xz(
            xz_ft[:, :, :mx2, :, :mz2], self.weights_xz[:, :, :mx2, :mz2]
        )
        out_ft[:, :, -mx2:, :, :mz2] = compl_mul2d_xz(
            xz_ft[:, :, -mx2:, :, :mz2], self.weights_xz[:, :, :mx2, :mz2]
        )
        comps["xz"] = torch.fft.irfftn(out_ft, s=(X, Z), dim=[-3, -1])

        # ---- 2D: YZ ----
        yz_ft = torch.fft.rfftn(x, dim=[-2, -1])
        out_ft = torch.zeros(B, self.out_channels, X, Y, Z // 2 + 1, dtype=torch.cfloat, device=device)

        my2 = min(self.modes_y, Y)
        mz2 = min(self.modes_z, Z // 2 + 1)
        out_ft[:, :, :, :my2, :mz2] = compl_mul2d_yz(
            yz_ft[:, :, :, :my2, :mz2], self.weights_yz[:, :, :my2, :mz2]
        )
        out_ft[:, :, :, -my2:, :mz2] = compl_mul2d_yz(
            yz_ft[:, :, :, -my2:, :mz2], self.weights_yz[:, :, :my2, :mz2]
        )
        comps["yz"] = torch.fft.irfftn(out_ft, s=(Y, Z), dim=[-2, -1])

        return comps

class FFNOBlock3d(nn.Module):
    """
    Step 3: small enrichment
      - separate spectral weighting into:
          axis terms   : x + y + z
          pairwise terms: xy + xz + yz
      - keep learned weights for point and local branches
      - keep residual exactly as: return x + y
    """

    def __init__(self, width, modes, expansion=2, use_pairwise=True, use_local_conv=True, dropout=0.0):
        super().__init__()

        self.use_pairwise = use_pairwise
        self.use_local_conv = use_local_conv

        self.spec_op = EnhancedFactorizedSpectralConv3d(
            in_channels=width,
            out_channels=width,
            modes_x=modes,
            modes_y=modes,
            modes_z=modes,
            use_pairwise=use_pairwise
        )

        self.point = nn.Conv3d(width, width, kernel_size=1)

        if use_local_conv:
            # small new change: boundary-aware local branch
            self.local = nn.Conv3d(
                width, width,
                kernel_size=3,
                padding=1,
                padding_mode='replicate'
            )
        else:
            self.local = None

        self.alpha_axis = nn.Parameter(torch.tensor(1.0))
        self.alpha_pair = nn.Parameter(torch.tensor(1.0)) if use_pairwise else None
        self.alpha_point = nn.Parameter(torch.tensor(1.0))
        self.alpha_local = nn.Parameter(torch.tensor(1.0)) if use_local_conv else None

        self.norm = nn.GroupNorm(1, width)

        hidden = expansion * width
        self.mlp = nn.Sequential(
            nn.Conv3d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, width, kernel_size=1),
        )

    def spec(self, x):
        comps = self.spec_op(x)

        axis = comps["x"] + comps["y"] + comps["z"]
        y = self.alpha_axis * axis

        if self.use_pairwise:
            pair = comps["xy"] + comps["xz"] + comps["yz"]
            y = y + self.alpha_pair * pair

        return y

    def forward(self, x):
        y = self.spec(x) + self.alpha_point * self.point(x)

        if self.local is not None:
            y = y + self.alpha_local * self.local(x)

        y = self.norm(y)
        y = self.mlp(y)
        y = F.gelu(y)
        return x + y


class FFNO4d(nn.Module):
    """
    Input : (B, X, Y, Z, T_in_channels)
    Output: (B, X, Y, Z, 1)

    Small new change:
      - learnable late fusion of lifted input features with final FFNO features
    """

    def __init__(self, modes, width, width_q, T_in_channels, n_layers,
                 expansion=2, use_pairwise=True, use_local_conv=True, dropout=0.0):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in_channels
        self.n_layers = n_layers

        self.p = nn.Linear(self.T_in + 3, self.width)

        self.blocks = nn.ModuleList([
            FFNOBlock3d(
                width=self.width,
                modes=self.modes,
                expansion=expansion,
                use_pairwise=use_pairwise,
                use_local_conv=use_local_conv,
                dropout=dropout
            )
            for _ in range(self.n_layers)
        ])

        # changed: head now sees both final features and lifted input features
        self.proj1 = nn.Conv3d(2 * self.width, self.width_q, kernel_size=1)
        self.proj2 = nn.Conv3d(self.width_q, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in_channels)
        grid = get_grid_3D(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)   # (B, width, X, Y, Z)

        # save lifted features
        x0 = x

        for blk in self.blocks:
            x = blk(x)

        # learnable late fusion instead of hard addition
        x = torch.cat([x, x0], dim=1)  # (B, 2*width, X, Y, Z)

        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        x = x.permute(0, 2, 3, 4, 1)
        return x


##################################
############# MHNO with FFNO backbone

class FFNO3d(nn.Module):
    """
    FFNO backbone (like FNO3d but operator is FFNO blocks).
    This returns a single-step prediction (B, X, Y, Z, 1) like FNO3d does.

    Input:  (B, X, Y, Z, T_in)
    Output: (B, X, Y, Z, 1)
    """
    def __init__(self, modes, width, width_q, T_in, n_layers,
                 expansion=2, use_pairwise=True, use_local_conv=True):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.n_layers = n_layers

        # Lift (T_in + 3 coords) -> width
        self.p = nn.Linear(self.T_in + 3, self.width)

        # FFNO operator blocks
        self.blocks = nn.ModuleList([
            FFNOBlock3d(
                width=self.width,
                modes=self.modes,
                expansion=expansion,
                use_pairwise=use_pairwise,
                use_local_conv=use_local_conv
            )
            for _ in range(self.n_layers)
        ])

        # Projection head to 1 channel (single-step)
        self.proj1 = nn.Conv3d(self.width, self.width_q, kernel_size=1)
        self.proj2 = nn.Conv3d(self.width_q, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        grid = get_grid_3d(x.shape, x.device)     # (B,X,Y,Z,3)
        x = torch.cat((x, grid), dim=-1)          # (B,X,Y,Z,T_in+3)

        x = self.p(x)                             # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3)              # (B,width,X,Y,Z)

        for blk in self.blocks:
            x = blk(x)

        y = F.gelu(self.proj1(x))
        y = self.proj2(y)                         # (B,1,X,Y,Z)
        y = y.permute(0, 2, 3, 4, 1)              # (B,X,Y,Z,1)
        return y


class MHNO_FFNO(FFNO3d):
    """
    MHNO / TNO-style model but with FFNO backbone.
    Mirrors your TNO3d logic:
      - backbone features from FFNO blocks
      - q(x,t): time-dependent projection from features to state
      - h(xt, t-1): history correction using previous predicted state

    Input:  (B, X, Y, Z, T_in)   (e.g., T_in=4 past frames)
    Output: (B, X, Y, Z, T_out)  (autoregressive rollout length)
    """
    def __init__(self, modes, width, width_q, width_h, T_in, T_out, n_layers,
                 expansion=2, use_pairwise=True, use_local_conv=True,
                 n_layers_q=2, n_layers_h=2):
        super().__init__(modes=modes, width=width, width_q=width_q, T_in=T_in, n_layers=n_layers,
                         expansion=expansion, use_pairwise=use_pairwise, use_local_conv=use_local_conv)

        self.T_out = T_out
        self.width_h = width_h

        # Replace the single-step proj head with time-conditioned heads like TNO3d:
        # q: maps backbone features -> next state, conditioned on time index t
        # h: maps previous predicted state -> correction, conditioned on (t-1)

        # IMPORTANT: your existing MLP3d supports "T blocks" via forward(x, t=...)
        # We'll use it exactly like your TNO3d does.
        self.q_time = MLP3d(self.width, 1, self.width_q, T_out, num_layers=n_layers_q)   # (B,1,X,Y,Z) per t
        self.h_time = MLP3d(1, 1, self.width_h, max(T_out - 1, 1), num_layers=n_layers_h)

        # Keep the backbone blocks from FFNO3d (self.blocks) and lifting layer self.p.

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        grid = get_grid_3d(x.shape, x.device)     # (B,X,Y,Z,3)
        x = torch.cat((x, grid), dim=-1)          # (B,X,Y,Z,T_in+3)

        x = self.p(x)                             # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3)              # (B,width,X,Y,Z)

        # Backbone operator stack (FFNO)
        for blk in self.blocks:
            x = blk(x)

        # Autoregressive rollout like TNO3d
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)  # (B,X,Y,Z,T_out)

        # First step uses q at t=0
        xt = self.q_time(x, t=0)                                  # (B,1,X,Y,Z)
        X[..., 0] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)          # (B,X,Y,Z)

        # Next steps
        for t in range(1, self.T_out):
            x1 = self.q_time(x, t=t)                              # (B,1,X,Y,Z)
            x2 = self.h_time(xt, t=t-1)                            # (B,1,X,Y,Z)
            xt = x1 + x2
            X[..., t] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)

        return X

################3
################  DeepONet

class _MLP(nn.Module):
    """
    Simple MLP that works on the last dimension of an arbitrary-shaped tensor.
    Input : (..., in_dim)
    Output: (..., out_dim)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, depth=4, act=nn.GELU):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"
        layers = [nn.Linear(in_dim, hidden_dim), act()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

        # stable init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
class CoordFiLM(nn.Module):
    """Trunk: coords -> (gamma, beta) for FiLM conditioning."""
    def __init__(self, hidden=64, depth=4, channels=64):
        super().__init__()
        self.mlp = _MLP(in_dim=3, hidden_dim=hidden, out_dim=2*channels, depth=depth)

    def forward(self, coords):
        # coords: (B,X,Y,Z,3)
        gb = self.mlp(coords)  # (B,X,Y,Z,2C)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class BranchEncoder3D(nn.Module):
    """
    Global branch: encodes full spatiotemporal input field into feature map.
    Uses Conv3D blocks (local inductive bias) + residuals for stability.
    """
    def __init__(self, in_ch, width=64, depth=4, use_groupnorm=True):
        super().__init__()
        self.in_proj = nn.Conv3d(in_ch, width, kernel_size=1)

        blocks = []
        for _ in range(depth):
            blocks.append(nn.Conv3d(width, width, kernel_size=3, padding=1))
            if use_groupnorm:
                blocks.append(nn.GroupNorm(1, width))
            blocks.append(nn.GELU())
            blocks.append(nn.Conv3d(width, width, kernel_size=3, padding=1))
            if use_groupnorm:
                blocks.append(nn.GroupNorm(1, width))
            blocks.append(nn.GELU())
        self.blocks = nn.Sequential(*blocks)

        # small init helps
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.depth = depth
        self.width = width

    def forward(self, x):
        # x: (B,T_in,X,Y,Z) OR (B,in_ch,X,Y,Z)
        if x.dim() == 5 and x.shape[1] != self.width:
            pass
        y = self.in_proj(x)
        # residual-style unrolling
        # blocks is a flat sequential; we apply residual every 2 convs (per "block")
        idx = 0
        for _ in range(self.depth):
            y0 = y
            y = self.blocks[idx](y); idx += 1
            y = self.blocks[idx](y); idx += 1
            y = self.blocks[idx](y); idx += 1
            y = self.blocks[idx](y); idx += 1
            y = self.blocks[idx](y); idx += 1
            y = self.blocks[idx](y); idx += 1
            y = y + y0
        return y


class DeepONet3D_Robust(nn.Module):
    """
    Robust DeepONet-style operator net:
      - Branch: 3D CNN encoder on full history field
      - Trunk: coord MLP -> FiLM modulation
      - Head: 1x1x1 conv projection to scalar field

    Input : (B,X,Y,Z,T_in)
    Output: (B,X,Y,Z,1)
    """
    def __init__(
        self,
        T_in_channels: int,
        width: int = 64,
        branch_depth: int = 4,
        trunk_hidden: int = 64,
        trunk_depth: int = 4,
        head_width: int = 64,
        use_coords_in_input: bool = False,   # optional concat coords to branch input
    ):
        super().__init__()
        self.T_in = T_in_channels
        self.width = width
        self.use_coords_in_input = use_coords_in_input

        branch_in = T_in_channels + (3 if use_coords_in_input else 0)

        self.branch = BranchEncoder3D(in_ch=branch_in, width=width, depth=branch_depth)
        self.trunk  = CoordFiLM(hidden=trunk_hidden, depth=trunk_depth, channels=width)

        self.head = nn.Sequential(
            nn.Conv3d(width, head_width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(head_width, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: (B,X,Y,Z,T_in)
        B, X, Y, Z, Cin = x.shape
        assert Cin >= self.T_in

        grid = get_grid_3D(x.shape, x.device)  # (B,X,Y,Z,3)

        # branch input -> (B, C, X,Y,Z)
        u = x[..., :self.T_in]
        if self.use_coords_in_input:
            u = torch.cat([u, grid], dim=-1)

        u = u.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,X,Y,Z)

        feat = self.branch(u)  # (B,width,X,Y,Z)

        # FiLM from coords: coords are (B,X,Y,Z,3)
        gamma, beta = self.trunk(grid)  # each (B,X,Y,Z,width)
        gamma = gamma.permute(0, 4, 1, 2, 3).contiguous()
        beta  = beta.permute(0, 4, 1, 2, 3).contiguous()

        # FiLM modulation
        feat = feat * (1.0 + gamma) + beta

        y = self.head(feat)  # (B,1,X,Y,Z)
        y = y.permute(0, 2, 3, 4, 1).contiguous()  # (B,X,Y,Z,1)
        return y


#####

# ============================================================
# Pure Robust Transformer Neural Operator for 3D one-step prediction
# General-purpose, PDE-agnostic, one-step only
# ============================================================

class TNOLinearAttention3D(nn.Module):
    """
    Linear multi-head attention over flattened 3D grid tokens.

    Input:
        x      : (B, N, C)
        coords : (B, N, 3)

    Output:
        y      : (B, N, C)
    """
    def __init__(self, dim, heads=4, dim_head=16, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.eps = eps

        self.norm = nn.LayerNorm(dim)
        self.coord_proj = nn.Linear(3, dim)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

    def _phi(self, x):
        return F.elu(x) + 1.0

    def forward(self, x, coords):
        x = self.norm(x + self.coord_proj(coords))  # (B,N,C)

        B, N, _ = x.shape
        q = self.to_q(x).view(B, N, self.heads, self.dim_head)
        k = self.to_k(x).view(B, N, self.heads, self.dim_head)
        v = self.to_v(x).view(B, N, self.heads, self.dim_head)

        q = self._phi(q)
        k = self._phi(k)

        k_sum = k.sum(dim=1)  # (B,H,D)
        z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, k_sum) + self.eps)  # (B,N,H)

        kv = torch.einsum("bnhd,bnhe->bhde", k, v)                      # (B,H,D,D)
        out = torch.einsum("bnhd,bhde,bnh->bnhe", q, kv, z)            # (B,N,H,D)

        out = out.reshape(B, N, self.inner_dim)
        out = self.to_out(out)
        return out


class TNOLocalSpectralMixer3D(nn.Module):
    """
    Grid-space regressor:
      spectral branch + pointwise branch + local conv branch
    """
    def __init__(self, width, modes, mlp_ratio=2, dropout=0.0):
        super().__init__()

        self.spec = SpectralConv3d(width, width, modes, modes, modes)
        self.point = nn.Conv3d(width, width, kernel_size=1)

        # local branch for robustness on non-periodic / mixed problems
        self.local = nn.Conv3d(
            width, width,
            kernel_size=3,
            padding=1,
            padding_mode='replicate'
        )

        self.gate_spec = nn.Parameter(torch.tensor(1.0))
        self.gate_point = nn.Parameter(torch.tensor(1.0))
        self.gate_local = nn.Parameter(torch.tensor(1.0))

        self.norm = nn.GroupNorm(1, width)

        hidden = mlp_ratio * width
        self.mlp = nn.Sequential(
            nn.Conv3d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv3d(hidden, width, kernel_size=1),
        )

    def forward(self, x):
        y = (
            self.gate_spec * self.spec(x)
            + self.gate_point * self.point(x)
            + self.gate_local * self.local(x)
        )
        y = self.norm(y)
        y = F.gelu(y)
        y = y + self.mlp(y)
        return x + y


class TNOBlock3D_Pure(nn.Module):
    """
    One pure TNO block:
      sequence attention -> grid reshape -> spectral/local mixer -> residual
    """
    def __init__(self, width, modes, heads=4, dim_head=16, mlp_ratio=2, dropout=0.0):
        super().__init__()

        self.attn = TNOLinearAttention3D(
            dim=width,
            heads=heads,
            dim_head=dim_head,
        )
        self.seq_norm = nn.LayerNorm(width)

        self.mixer = TNOLocalSpectralMixer3D(
            width=width,
            modes=modes,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x, coords):
        # x: (B,C,X,Y,Z)
        B, C, X, Y, Z = x.shape

        # sequence reshape
        seq = x.permute(0, 2, 3, 4, 1).reshape(B, X * Y * Z, C)

        # attention
        seq = seq + self.attn(seq, coords)
        seq = self.seq_norm(seq)

        # back to grid
        grid = seq.view(B, X, Y, Z, C).permute(0, 4, 1, 2, 3).contiguous()

        # spectral + local mixing
        grid = self.mixer(grid)
        return grid


class TNO3d_PureRobust(nn.Module):
    """
    Pure robust Transformer Neural Operator for 3D one-step prediction.

    Input:
        x : (B, X, Y, Z, T_in)

    Output:
        y : (B, X, Y, Z, 1)

    Properties:
    - PDE-agnostic
    - one-step only
    - pure data-driven
    - robust via:
        * coordinate lifting
        * linear attention
        * spectral regression
        * local conv branch
        * residual blocks
        * residual output update from last frame
    """
    def __init__(
        self,
        modes,
        width,
        width_q,
        T_in,
        n_layers=3,
        heads=4,
        dim_head=16,
        mlp_ratio=2,
        dropout=0.0,
        use_coords_in_lift=True,
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.use_coords_in_lift = use_coords_in_lift

        in_dim = T_in + (3 if use_coords_in_lift else 0)
        self.p = nn.Linear(in_dim, width)

        self.blocks = nn.ModuleList([
            TNOBlock3D_Pure(
                width=width,
                modes=modes,
                heads=heads,
                dim_head=dim_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Conv3d(width, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, 1, kernel_size=1),
        )

        self.out_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (B,X,Y,Z,T_in)
        B, X, Y, Z, _ = x.shape

        grid = get_grid_3D(x.shape, x.device)   # (B,X,Y,Z,3)
        u_last = x[..., -1:]                    # (B,X,Y,Z,1)

        if self.use_coords_in_lift:
            x_in = torch.cat([x, grid], dim=-1)
        else:
            x_in = x

        feat = self.p(x_in)                                 # (B,X,Y,Z,width)
        feat = feat.permute(0, 4, 1, 2, 3).contiguous()     # (B,width,X,Y,Z)

        coords = grid.reshape(B, X * Y * Z, 3)

        for blk in self.blocks:
            feat = blk(feat, coords)

        delta = self.head(feat)                             # (B,1,X,Y,Z)
        delta = delta.permute(0, 2, 3, 4, 1).contiguous()   # (B,X,Y,Z,1)

        # generic residual one-step update
        y = u_last + self.out_scale * delta
        return y




##### HAMNO

# ============================================================
# HAMNO
# Hierarchical Adaptive Multi-scale Neural Operator for 3D fields
# General-purpose, no PDE-specific architectural bias
# Input : (B, X, Y, Z, T_in)
# Output: (B, X, Y, Z, 1)
# ============================================================

class HAMNOLocalGlobalBlock3d(nn.Module):
    """
    General operator block:
      - local 3D branch
      - global spectral branch
      - parallel gated fusion
      - residual MLP
    """
    def __init__(self, width, modes, expansion=2):
        super().__init__()

        hidden = expansion * width

        self.norm1 = nn.GroupNorm(1, width)

        self.local = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
        )

        self.spec = SpectralConv3d(width, width, modes, modes, modes)
        self.spec_pw = nn.Conv3d(width, width, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv3d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 2 * width, kernel_size=1),
        )

        self.mix = nn.Conv3d(width, width, kernel_size=1)

        self.norm2 = nn.GroupNorm(1, width)

        self.mlp = nn.Sequential(
            nn.Conv3d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, width, kernel_size=1),
        )

    def forward(self, x):
        h = self.norm1(x) # The input feature v is first normalized before being sent to the two branches.

        h_local = self.local(h) # The local branch is a pair of 3D convolutions, so it captures nearby spatial structure.
        h_spec = self.spec_pw(F.gelu(self.spec(h))) # The global branch applies a spectral operator followed by a nonlinear activation and a pointwise projection.

        gates = self.gate(h)
        g_local, g_spec = gates.chunk(2, dim=1)
        g_local = torch.sigmoid(g_local)
        g_spec = torch.sigmoid(g_spec)

        h = g_local * h_local + g_spec * h_spec
        h = self.mix(h)

        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class HAMNOStage3d(nn.Module):
    """
    A stack of local-global operator blocks at one spatial scale.
    """
    def __init__(self, width, modes, depth, expansion=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            HAMNOLocalGlobalBlock3d(width=width, modes=modes, expansion=expansion)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class HAMNO3d(nn.Module):
    """
    Hierarchical multi-scale neural operator for one-step prediction.

    Design:
      1) lift input + coordinates
      2) fine stage
      3) downsample -> coarse stage
      4) downsample -> bottleneck stage
      5) transpose-conv upsample with skip fusion
      6) output head
    """
    def __init__(
        self,
        modes,
        width,
        width_q,
        T_in,
        n_layers,
        expansion=2,
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.n_layers = n_layers

        self.p = nn.Linear(T_in + 3, width)

        self.stem = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1),
        )

        self.stage1 = HAMNOStage3d(
            width=width,
            modes=min(modes, 8),
            depth=max(1, n_layers),
            expansion=expansion,
        )

        self.down1 = nn.Conv3d(width, 2 * width, kernel_size=2, stride=2)

        self.stage2 = HAMNOStage3d(
            width=2 * width,
            modes=min(modes, 6),
            depth=max(1, n_layers),
            expansion=expansion,
        )

        self.down2 = nn.Conv3d(2 * width, 4 * width, kernel_size=2, stride=2)

        self.stage3 = HAMNOStage3d(
            width=4 * width,
            modes=min(modes, 4),
            depth=max(1, n_layers),
            expansion=expansion,
        )

        # deterministic upsampling
        self.up2 = nn.ConvTranspose3d(4 * width, 2 * width, kernel_size=2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv3d(4 * width, 2 * width, kernel_size=1),
            nn.GELU(),
        )
        self.refine2 = HAMNOStage3d(
            width=2 * width,
            modes=min(modes, 6),
            depth=1,
            expansion=expansion,
        )

        self.up1 = nn.ConvTranspose3d(2 * width, width, kernel_size=2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv3d(2 * width, width, kernel_size=1),
            nn.GELU(),
        )
        self.refine1 = HAMNOStage3d(
            width=width,
            modes=min(modes, 8),
            depth=1,
            expansion=expansion,
        )

        self.late_fuse = nn.Sequential(
            nn.Conv3d(2 * width, width, kernel_size=1),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Conv3d(width, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, 1, kernel_size=1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p.weight)
        nn.init.zeros_(self.p.bias)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)               # (B,X,Y,Z,T_in+3)

        x = self.p(x)                                  # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()      # (B,width,X,Y,Z)

        x0 = self.stem(x) # gives an early refined feature map

        e1 = self.stage1(x0)                           # (B,w,32,32,32)
        e2 = self.stage2(self.down1(e1))               # (B,2w,16,16,16)
        b  = self.stage3(self.down2(e2))               # (B,4w,8,8,8)

        d2 = self.up2(b)                               # (B,2w,16,16,16)
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))
        d2 = self.refine2(d2)

        d1 = self.up1(d2)                              # (B,w,32,32,32)
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))
        d1 = self.refine1(d1)

        out = self.late_fuse(torch.cat([d1, x0], dim=1))
        out = self.head(out)                           # (B,1,X,Y,Z)

        out = out.permute(0, 2, 3, 4, 1).contiguous()  # (B,X,Y,Z,1)
        return out



####### UNO3d
import torch
import torch.nn as nn

from neuralop.models import UNO
'''

class UNO3d(nn.Module):
    """
    3D wrapper around the official NeuralOperator UNO.

    Input:
        x: (B, X, Y, Z, T_in)

    Output:
        out: (B, X, Y, Z, 1)

    This keeps the original UNO structure:
        - official UNO class
        - official FNOBlocks
        - official ChannelMLP
        - official uno_scalings / resample
        - configurable horizontal skips
    """

    def __init__(
        self,
        modes,
        width,
        width_q,
        T_in,
        n_layers,
        expansion=2,
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.n_layers = n_layers

        # For a 5-layer UNO:
        # layer 0: fine
        # layer 1: downsample
        # layer 2: bottleneck
        # layer 3: same/coarse
        # layer 4: upsample
        #
        # This is close to the original UNO style.
        self.uno_n_layers = 5

        self.uno_out_channels = [
            width,
            2 * width,
            4 * width,
            2 * width,
            width,
        ]

        self.uno_n_modes = [
            [min(modes, 12), min(modes, 12), min(modes, 12)],
            [min(modes, 6), min(modes, 6), min(modes, 6)],
            [min(modes, 4), min(modes, 4), min(modes, 4)],
            [min(modes, 6), min(modes, 6), min(modes, 6)],
            [min(modes, 12), min(modes, 12), min(modes, 12)],
        ]

        self.uno_scalings = [
            [1.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 4.0],
        ]

        self.horizontal_skips_map = {
            4: 0,
            3: 1,
        }

        self.uno = UNO(
            in_channels=T_in,
            out_channels=1,
            hidden_channels=width,

            n_layers=self.uno_n_layers,
            uno_out_channels=self.uno_out_channels,
            uno_n_modes=self.uno_n_modes,
            uno_scalings=self.uno_scalings,
            horizontal_skips_map=self.horizontal_skips_map,

            lifting_channels=width_q,
            projection_channels=width_q,

            positional_embedding="grid",

            channel_mlp_expansion=0.5,
            channel_mlp_dropout=0.0,

            norm=None,
            fno_skip="linear",
            horizontal_skip="linear",
            channel_mlp_skip="linear",

            domain_padding=None,
        )

    def forward(self, x):
        # Your data format:
        # x: (B, X, Y, Z, T_in)

        # Official UNO expects:
        # x: (B, T_in, X, Y, Z)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        out = self.uno(x)

        # Convert back:
        # out: (B, 1, X, Y, Z) -> (B, X, Y, Z, 1)
        out = out.permute(0, 2, 3, 4, 1).contiguous()

        return out


'''
## Modified UNO3d which is more accurate compare to the above UNO

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNOBlock3d(nn.Module):
    """
    Original-UNO-style operator block:
      spectral convolution + pointwise convolution + channel MLP
    No local branch, no gating.
    """
    def __init__(self, in_channels, out_channels, modes, expansion=2):
        super().__init__()

        hidden = expansion * out_channels

        self.spec = SpectralConv3d(
            in_channels, out_channels,
            modes, modes, modes
        )

        self.w = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.mlp = nn.Sequential(
            nn.Conv3d(out_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, out_channels, kernel_size=1),
        )

        self.norm = nn.GroupNorm(1, out_channels)

    def forward(self, x, output_shape=None):
        # resize BEFORE FFT
        if output_shape is not None and tuple(x.shape[-3:]) != tuple(output_shape):
            x = F.interpolate(
                x,
                size=output_shape,
                mode='nearest',
            )

        x_spec = self.spec(x)
        x_pw = self.w(x)

        x = x_spec + x_pw
        x = F.gelu(x)

        x = x + self.mlp(self.norm(x))

        return x


class UNO3d(nn.Module):
    """
    Standalone 3D UNO baseline.

    Input:
        x: (B, X, Y, Z, T_in)

    Output:
        out: (B, X, Y, Z, 1)

    This follows the UNO idea:
        lift + grid
        encoder spectral blocks
        lower-resolution bottleneck
        decoder spectral blocks
        U-shaped skip connections
        projection head
    """
    def __init__(
        self,
        modes,
        width,
        width_q,
        T_in,
        n_layers,
        expansion=2,
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.n_layers = n_layers

        # SAFE MODES FOR MULTI-SCALE FFT
        m1 = min(modes, 8)
        m2 = min(modes, 4)
        m3 = min(modes, 2)


        # Lift input + 3D coordinates
        self.p = nn.Linear(T_in + 3, width)

        # Encoder
        # Encoder
        self.enc1 = nn.ModuleList([
            UNOBlock3d(
                in_channels=width,
                out_channels=width,
                modes=m1,
                expansion=expansion,
            )
            for _ in range(max(1, n_layers))
        ])

        self.down1 = UNOBlock3d(
            in_channels=width,
            out_channels=2 * width,
            modes=m2,
            expansion=expansion,
        )

        self.enc2 = nn.ModuleList([
            UNOBlock3d(
                in_channels=2 * width,
                out_channels=2 * width,
                modes=m2,
                expansion=expansion,
            )
            for _ in range(max(1, n_layers))
        ])

        self.down2 = UNOBlock3d(
            in_channels=2 * width,
            out_channels=4 * width,
            modes=m3,
            expansion=expansion,
        )

        # Bottleneck
        self.bottleneck = nn.ModuleList([
            UNOBlock3d(
                in_channels=4 * width,
                out_channels=4 * width,
                modes=m3,
                expansion=expansion,
            )
            for _ in range(max(1, n_layers))
        ])

        # Decoder
        self.up2 = UNOBlock3d(
            in_channels=4 * width,
            out_channels=2 * width,
            modes=m2,
            expansion=expansion,
        )

        self.fuse2 = nn.Conv3d(4 * width, 2 * width, kernel_size=1)

        self.dec2 = UNOBlock3d(
            in_channels=2 * width,
            out_channels=2 * width,
            modes=m2,
            expansion=expansion,
        )

        self.up1 = UNOBlock3d(
            in_channels=2 * width,
            out_channels=width,
            modes=m1,
            expansion=expansion,
        )

        self.fuse1 = nn.Conv3d(2 * width, width, kernel_size=1)

        self.dec1 = UNOBlock3d(
            in_channels=width,
            out_channels=width,
            modes=m1,
            expansion=expansion,
        )

        # Projection
        self.q = nn.Sequential(
            nn.Conv3d(width, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, 1, kernel_size=1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p.weight)
        nn.init.zeros_(self.p.bias)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)

        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        size1 = x.shape[-3:]
        size2 = tuple(s // 2 for s in size1)
        size3 = tuple(s // 4 for s in size1)

        # Encoder level 1
        e1 = x
        for blk in self.enc1:
            e1 = blk(e1, output_shape=size1)

        # Encoder level 2
        e2 = self.down1(e1, output_shape=size2)
        for blk in self.enc2:
            e2 = blk(e2, output_shape=size2)

        # Bottleneck
        b = self.down2(e2, output_shape=size3)
        for blk in self.bottleneck:
            b = blk(b, output_shape=size3)

        # Decoder level 2
        d2 = self.up2(b, output_shape=e2.shape[-3:])
        d2 = torch.cat([d2, e2], dim=1)
        d2 = F.gelu(self.fuse2(d2))
        d2 = self.dec2(d2, output_shape=e2.shape[-3:])

        # Decoder level 1
        d1 = self.up1(d2, output_shape=e1.shape[-3:])
        d1 = torch.cat([d1, e1], dim=1)
        d1 = F.gelu(self.fuse1(d1))
        d1 = self.dec1(d1, output_shape=size1)

        out = self.q(d1)

        out = out.permute(0, 2, 3, 4, 1).contiguous()
        return out



### U-Net3d

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """
    Original U-Net block converted to 3D:
    Conv3d -> ReLU -> Conv3d -> ReLU
    No BatchNorm, no residuals, no attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet3d(nn.Module):
    """
    3D version of the original 2015 U-Net.

    Input:
        x: (B, X, Y, Z, T_in)

    Output:
        out: (B, X, Y, Z, 1)
    """
    def __init__(
        self,
        T_in,
        width=32,
        out_channels=1,
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv3d(T_in, width)
        self.pool1 = nn.Conv3d(width, width, kernel_size=2, stride=2)

        self.enc2 = DoubleConv3d(width, 2 * width)
        self.pool2 = nn.Conv3d(2 * width, 2 * width, kernel_size=2, stride=2)

        self.enc3 = DoubleConv3d(2 * width, 4 * width)
        self.pool3 = nn.Conv3d(4 * width, 4 * width, kernel_size=2, stride=2)

        self.enc4 = DoubleConv3d(4 * width, 8 * width)
        self.pool4 = nn.Conv3d(8 * width, 8 * width, kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv3d(8 * width, 16 * width)

        # Decoder
        self.up4 = nn.ConvTranspose3d(16 * width, 8 * width, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3d(16 * width, 8 * width)

        self.up3 = nn.ConvTranspose3d(8 * width, 4 * width, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3d(8 * width, 4 * width)

        self.up2 = nn.ConvTranspose3d(4 * width, 2 * width, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3d(4 * width, 2 * width)

        self.up1 = nn.ConvTranspose3d(2 * width, width, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3d(2 * width, width)

        # Final 1x1 convolution
        self.out_conv = nn.Conv3d(width, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)

        out = out.permute(0, 2, 3, 4, 1).contiguous()
        return out




### U-FNO3d

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator
from functools import reduce


class UFNO_SpectralConv3d(nn.Module):
    """
    Original U-FNO 3D Fourier layer.
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels,
                modes1, modes2, modes3,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels,
                modes1, modes2, modes3,
                dtype=torch.cfloat
            )
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels,
                modes1, modes2, modes3,
                dtype=torch.cfloat
            )
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels,
                modes1, modes2, modes3,
                dtype=torch.cfloat
            )
        )

    def compl_mul3d(self, input, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, x_ft.shape[-3])
        m2 = min(self.modes2, x_ft.shape[-2])
        m3 = min(self.modes3, x_ft.shape[-1])

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3],
            self.weights1[:, :, :m1, :m2, :m3],
        )

        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3],
            self.weights2[:, :, :m1, :m2, :m3],
        )

        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3],
            self.weights3[:, :, :m1, :m2, :m3],
        )

        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3],
            self.weights4[:, :, :m1, :m2, :m3],
        )

        x = torch.fft.irfftn(
            out_ft,
            s=(x.size(-3), x.size(-2), x.size(-1))
        )

        return x


class UFNO_U_net(nn.Module):
    """
    Original mini U-Net block used inside U-FNO.
    """
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super().__init__()

        self.input_channels = input_channels

        self.conv1 = self.conv(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.conv2 = self.conv(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.conv2_1 = self.conv(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.conv3 = self.conv(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.conv3_1 = self.conv(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels * 2, output_channels)
        self.deconv0 = self.deconv(input_channels * 2, output_channels)

        self.output_layer = self.output(
            input_channels * 2,
            output_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))

        out_deconv2 = self.deconv2(out_conv3)
        concat2 = torch.cat((out_conv2, out_deconv2), dim=1)

        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), dim=1)

        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), dim=1)

        out = self.output_layer(concat0)

        return out

    def conv(self, in_planes, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv3d(
                in_planes,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm3d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(
                input_channels,
                output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )


class UFNO_SimpleBlock3d(nn.Module):
    """
    Original U-FNO block:
        3 Fourier layers
        3 U-Fourier layers
    """
    def __init__(self, modes1, modes2, modes3, width, T_in):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width

        # T_in physical channels + 3 coordinate channels
        self.fc0 = nn.Linear(T_in + 3, width)

        self.conv0 = UFNO_SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv1 = UFNO_SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv2 = UFNO_SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv3 = UFNO_SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv4 = UFNO_SpectralConv3d(width, width, modes1, modes2, modes3)
        self.conv5 = UFNO_SpectralConv3d(width, width, modes1, modes2, modes3)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)
        self.w4 = nn.Conv1d(width, width, 1)
        self.w5 = nn.Conv1d(width, width, 1)

        self.unet3 = UFNO_U_net(width, width, 3, 0.0)
        self.unet4 = UFNO_U_net(width, width, 3, 0.0)
        self.unet5 = UFNO_U_net(width, width, 3, 0.0)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        x1 = self.conv0(x)
        x2 = self.w0(x.reshape(batchsize, self.width, -1)).reshape(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = F.relu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.w1(x.reshape(batchsize, self.width, -1)).reshape(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = F.relu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.w2(x.reshape(batchsize, self.width, -1)).reshape(
            batchsize, self.width, size_x, size_y, size_z
        )
        x = F.relu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.w3(x.reshape(batchsize, self.width, -1)).reshape(
            batchsize, self.width, size_x, size_y, size_z
        )
        x3 = self.unet3(x)
        x = F.relu(x1 + x2 + x3)

        x1 = self.conv4(x)
        x2 = self.w4(x.reshape(batchsize, self.width, -1)).reshape(
            batchsize, self.width, size_x, size_y, size_z
        )
        x3 = self.unet4(x)
        x = F.relu(x1 + x2 + x3)

        x1 = self.conv5(x)
        x2 = self.w5(x.reshape(batchsize, self.width, -1)).reshape(
            batchsize, self.width, size_x, size_y, size_z
        )
        x3 = self.unet5(x)
        x = F.relu(x1 + x2 + x3)

        x = x.permute(0, 2, 3, 4, 1).contiguous()

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class UFNO3d(nn.Module):
    """
    Original U-FNO 3D architecture adapted to your interface.

    Input:
        x: (B, X, Y, Z, T_in)

    Output:
        out: (B, X, Y, Z, 1)
    """
    def __init__(
        self,
        modes,
        width,
        width_q=None,
        T_in=1,
        n_layers=None,
        expansion=None,
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.T_in = T_in

        self.block = UFNO_SimpleBlock3d(
            modes1=modes,
            modes2=modes,
            modes3=modes,
            width=width,
            T_in=T_in,
        )

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        # Original U-FNO uses padding of 8
        x = F.pad(
            F.pad(x, (0, 0, 0, 8, 0, 8), mode="replicate"),
            (0, 0, 0, 0, 0, 0, 0, 8),
            mode="constant",
            value=0,
        )

        x = self.block(x)

        x = x.view(batchsize, size_x + 8, size_y + 8, size_z + 8, 1)
        x = x[:, :size_x, :size_y, :size_z, :]

        return x

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c





# ============================================================
# HMNO3dPlus
# Improved Hierarchical Multi-scale Neural Operator for 3D fields
# General-purpose, no PDE-specific architectural bias
# Input : (B, X, Y, Z, T_in)
# Output: (B, X, Y, Z, 1)
# ============================================================

class SEBlock3d(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w


class HMNOPlusBlock3d(nn.Module):
    """
    Improved operator block:
      - local 3D branch
      - spectral branch
      - axial directional branch
      - gated fusion
      - channel attention
      - residual MLP
    """
    def __init__(self, width, modes, expansion=2):
        super().__init__()

        hidden = expansion * width
        self.norm1 = nn.GroupNorm(1, width)

        # local branch
        self.local = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
        )

        # spectral branch
        self.spec = SpectralConv3d(width, width, modes, modes, modes)
        self.spec_pw = nn.Conv3d(width, width, kernel_size=1)

        # axial directional branch (fully parallel, GPU-friendly)
        self.ax_x = nn.Conv3d(
            width, width,
            kernel_size=(5, 1, 1),
            padding=(2, 0, 0),
            groups=width,
            padding_mode='replicate'
        )
        self.ax_y = nn.Conv3d(
            width, width,
            kernel_size=(1, 5, 1),
            padding=(0, 2, 0),
            groups=width,
            padding_mode='replicate'
        )
        self.ax_z = nn.Conv3d(
            width, width,
            kernel_size=(1, 1, 5),
            padding=(0, 0, 2),
            groups=width,
            padding_mode='replicate'
        )
        self.ax_pw = nn.Conv3d(width, width, kernel_size=1)

        # gated fusion among 3 branches
        self.gate = nn.Sequential(
            nn.Conv3d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, 3 * width, kernel_size=1),
        )

        self.mix = nn.Conv3d(width, width, kernel_size=1)
        self.se = SEBlock3d(width, reduction=4)

        self.norm2 = nn.GroupNorm(1, width)
        self.mlp = nn.Sequential(
            nn.Conv3d(width, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden, width, kernel_size=1),
        )

    def forward(self, x):
        h = self.norm1(x)

        h_local = self.local(h)
        h_spec = self.spec_pw(F.gelu(self.spec(h)))

        h_ax = self.ax_x(h) + self.ax_y(h) + self.ax_z(h)
        h_ax = self.ax_pw(F.gelu(h_ax))

        gates = self.gate(h)
        g_local, g_spec, g_ax = gates.chunk(3, dim=1)
        g_local = torch.sigmoid(g_local)
        g_spec = torch.sigmoid(g_spec)
        g_ax = torch.sigmoid(g_ax)

        h = g_local * h_local + g_spec * h_spec + g_ax * h_ax
        h = self.mix(h)
        h = self.se(h)

        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class HMNOPlusStage3d(nn.Module):
    def __init__(self, width, modes, depth, expansion=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            HMNOPlusBlock3d(width=width, modes=modes, expansion=expansion)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class HMNO3dPlus(nn.Module):
    """
    Improved hierarchical multi-scale neural operator for one-step prediction.

    Upgrades over HMNO3d:
      - axial branch for directional propagation
      - channel attention
      - dual prediction head: initial prediction + correction
      - stronger late refinement
    """
    def __init__(
        self,
        modes,
        width,
        width_q,
        T_in,
        n_layers,
        expansion=2,
    ):
        super().__init__()

        self.modes = modes
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.n_layers = n_layers

        self.p = nn.Linear(T_in + 3, width)

        self.stem = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=1),
        )

        # encoder
        self.stage1 = HMNOPlusStage3d(
            width=width,
            modes=min(modes, 8),
            depth=max(1, n_layers),
            expansion=expansion,
        )
        self.down1 = nn.Conv3d(width, 2 * width, kernel_size=2, stride=2)

        self.stage2 = HMNOPlusStage3d(
            width=2 * width,
            modes=min(modes, 6),
            depth=max(1, n_layers),
            expansion=expansion,
        )
        self.down2 = nn.Conv3d(2 * width, 4 * width, kernel_size=2, stride=2)

        self.stage3 = HMNOPlusStage3d(
            width=4 * width,
            modes=min(modes, 4),
            depth=max(1, n_layers),
            expansion=expansion,
        )

        # decoder
        self.up2 = nn.ConvTranspose3d(4 * width, 2 * width, kernel_size=2, stride=2)
        self.fuse2 = nn.Sequential(
            nn.Conv3d(4 * width, 2 * width, kernel_size=1),
            nn.GELU(),
        )
        self.refine2 = HMNOPlusStage3d(
            width=2 * width,
            modes=min(modes, 6),
            depth=1,
            expansion=expansion,
        )

        self.up1 = nn.ConvTranspose3d(2 * width, width, kernel_size=2, stride=2)
        self.fuse1 = nn.Sequential(
            nn.Conv3d(2 * width, width, kernel_size=1),
            nn.GELU(),
        )
        self.refine1 = HMNOPlusStage3d(
            width=width,
            modes=min(modes, 8),
            depth=1,
            expansion=expansion,
        )

        # late fusion
        self.late_fuse = nn.Sequential(
            nn.Conv3d(2 * width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )

        # first prediction head
        self.head0 = nn.Sequential(
            nn.Conv3d(width, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, 1, kernel_size=1),
        )

        # correction head
        self.corr_in = nn.Sequential(
            nn.Conv3d(width + 1, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GELU(),
        )
        self.corr_head = nn.Sequential(
            nn.Conv3d(width, width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width_q, 1, kernel_size=1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.p.weight)
        nn.init.zeros_(self.p.bias)

    def forward(self, x):
        # x: (B, X, Y, Z, T_in)
        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)               # (B,X,Y,Z,T_in+3)

        x = self.p(x)                                  # (B,X,Y,Z,width)
        x = x.permute(0, 4, 1, 2, 3).contiguous()      # (B,width,X,Y,Z)

        x0 = self.stem(x)

        # encoder
        e1 = self.stage1(x0)                           # (B,w,32,32,32)
        e2 = self.stage2(self.down1(e1))               # (B,2w,16,16,16)
        b  = self.stage3(self.down2(e2))               # (B,4w,8,8,8)

        # decoder
        d2 = self.up2(b)                               # (B,2w,16,16,16)
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))
        d2 = self.refine2(d2)

        d1 = self.up1(d2)                              # (B,w,32,32,32)
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))
        d1 = self.refine1(d1)

        feat = self.late_fuse(torch.cat([d1, x0], dim=1))

        # initial prediction
        y0 = self.head0(feat)                          # (B,1,X,Y,Z)

        # correction prediction
        corr_feat = self.corr_in(torch.cat([feat, y0], dim=1))
        dy = self.corr_head(corr_feat)                 # (B,1,X,Y,Z)

        y = y0 + dy
        y = y.permute(0, 2, 3, 4, 1).contiguous()      # (B,X,Y,Z,1)
        return y












##################
##################


class PMNO_DeepONet(nn.Module):
    """
    PMNO-style wrapper for DeepONet3D_Robust.

    Base network predicts one step:
        (B,X,Y,Z,T_in) -> (B,X,Y,Z,1)

    Wrapper rolls out T_out steps autoregressively and adds a small
    temporal correction from the recent history window.

    Output:
        (B,X,Y,Z,T_out)
    """
    def __init__(self, deeponet, k, T_out=1, pm_width=16):
        super().__init__()
        self.deeponet = deeponet
        self.k = k
        self.T_out = T_out

        # small temporal mixer over the last k states
        # input per voxel: k values
        self.pm1 = nn.Linear(k, pm_width)
        self.pm2 = nn.Linear(pm_width, 1)

        # small gate so base DeepONet remains dominant initially
        self.alpha = nn.Parameter(torch.tensor(-2.0))

    def temporal_predictor(self, x):
        # x: (B,X,Y,Z,k)
        z = F.gelu(self.pm1(x))
        z = self.pm2(z)   # (B,X,Y,Z,1)
        return z

    def forward(self, x, steps=None):
        # x: (B,X,Y,Z,k)
        T = self.T_out if steps is None else steps

        preds = []
        x_cur = x

        gate = torch.sigmoid(self.alpha)

        for _ in range(T):
            # one-step DeepONet prediction
            u_base = self.deeponet(x_cur)              # (B,X,Y,Z,1)

            # PMNO-style temporal correction from history
            u_pm = self.temporal_predictor(x_cur)      # (B,X,Y,Z,1)

            # blended prediction
            u_next = (1.0 - gate) * u_base + gate * u_pm

            preds.append(u_next)

            # shift window
            x_cur = torch.cat([x_cur[..., 1:], u_next], dim=-1)

        return torch.cat(preds, dim=-1)                # (B,X,Y,Z,T_out)




class LocalPhysicsCorrection3D(nn.Module):
    """
    Small local correction block for physics-friendly rollout.

    Input channels can include:
      - DeepONet feature map
      - current raw state prediction
      - previous state
      - predicted increment

    Output:
      - local correction field (B,1,X,Y,Z)
    """
    def __init__(self, in_channels, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class DeepONet3D_TNOPhysics(nn.Module):
    """
    DeepONet backbone + TNO-style temporal heads + small local physics-friendly correction.

    Input : (B,X,Y,Z,T_in)
    Output: (B,X,Y,Z,T_out)

    Design:
      feat = DeepONet branch/trunk features
      x_t_raw = q_t(feat) + h_{t-1}(x_{t-1})
      x_t = x_t_raw + gate_t * corr_t(feat, x_t_raw, x_{t-1}, x_t_raw - x_{t-1})
    """
    def __init__(
        self,
        T_in_channels: int,
        T_out: int,
        width: int = 64,
        branch_depth: int = 4,
        trunk_hidden: int = 64,
        trunk_depth: int = 4,
        head_width: int = 64,
        use_coords_in_input: bool = False,
        q_width: int = 64,
        h_width: int = 32,
        corr_width: int = 32,
        n_layers_q: int = 2,
        n_layers_h: int = 2,
    ):
        super().__init__()

        self.T_in = T_in_channels
        self.T_out = T_out
        self.width = width
        self.use_coords_in_input = use_coords_in_input

        branch_in = T_in_channels + (3 if use_coords_in_input else 0)

        # same strong DeepONet backbone
        self.branch = BranchEncoder3D(
            in_ch=branch_in,
            width=width,
            depth=branch_depth
        )
        self.trunk = CoordFiLM(
            hidden=trunk_hidden,
            depth=trunk_depth,
            channels=width
        )

        # TNO-style time-dependent heads
        self.q_time = MLP3d(
            in_channels=width,
            out_channels=1,
            mid_channels=q_width,
            T=T_out,
            num_layers=n_layers_q
        )

        self.h_time = MLP3d(
            in_channels=1,
            out_channels=1,
            mid_channels=h_width,
            T=max(T_out - 1, 1),
            num_layers=n_layers_h
        )

        # physics-friendly local correction
        # channels = feat(width) + xt_raw(1) + xt_prev(1) + delta(1)
        self.corr = LocalPhysicsCorrection3D(
            in_channels=width + 3,
            hidden_channels=corr_width
        )

        # projection before q if needed (usually useful)
        self.pre_q = nn.Sequential(
            nn.Conv3d(width, head_width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(head_width, width, kernel_size=1)
        )

        # small learnable gate per horizon step
        self.corr_gate = nn.Parameter(torch.full((T_out,), -2.0))

    def forward_features(self, x):
        # x: (B,X,Y,Z,T_in)
        B, X, Y, Z, Cin = x.shape
        assert Cin >= self.T_in

        grid = get_grid_3D(x.shape, x.device)  # (B,X,Y,Z,3)

        u = x[..., :self.T_in]
        if self.use_coords_in_input:
            u = torch.cat([u, grid], dim=-1)

        u = u.permute(0, 4, 1, 2, 3).contiguous()  # (B,C,X,Y,Z)

        feat = self.branch(u)  # (B,width,X,Y,Z)

        gamma, beta = self.trunk(grid)  # each (B,X,Y,Z,width)
        gamma = gamma.permute(0, 4, 1, 2, 3).contiguous()
        beta  = beta.permute(0, 4, 1, 2, 3).contiguous()

        feat = feat * (1.0 + gamma) + beta
        feat = self.pre_q(feat)

        return feat

    def forward(self, x):
        # x: (B,X,Y,Z,T_in)
        feat = self.forward_features(x)   # (B,width,X,Y,Z)

        B, _, X, Y, Z = feat.shape
        out = torch.zeros(B, X, Y, Z, self.T_out, device=feat.device, dtype=feat.dtype)

        # previous observed state from input history
        x_prev = x[..., -1:].permute(0, 4, 1, 2, 3).contiguous()   # (B,1,X,Y,Z)

        # t = 0
        xt_raw = self.q_time(feat, t=0)  # (B,1,X,Y,Z)
        delta = xt_raw - x_prev

        corr_in = torch.cat([feat, xt_raw, x_prev, delta], dim=1)
        corr = self.corr(corr_in)

        gate = torch.sigmoid(self.corr_gate[0])
        xt = xt_raw + gate * corr

        out[..., 0] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)

        # t >= 1
        for t in range(1, self.T_out):
            x1 = self.q_time(feat, t=t)       # feature-conditioned prediction
            x2 = self.h_time(xt, t=t-1)       # temporal memory correction
            xt_raw = x1 + x2

            delta = xt_raw - xt
            corr_in = torch.cat([feat, xt_raw, xt, delta], dim=1)
            corr = self.corr(corr_in)

            gate = torch.sigmoid(self.corr_gate[t])
            xt = xt_raw + gate * corr

            out[..., t] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)

        return out

####



# ============================================================
# PMNO-style FNO for AC
# full replacement block for networks.py
# keeps names unchanged:
#   SpectralConv3d_PMNO
#   FNO3D_PMNO
#   PMNO_AC
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SpectralConv3d_PMNO(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2=None, modes3=None):
        super().__init__()

        if modes2 is None:
            modes2 = modes1
        if modes3 is None:
            modes3 = modes1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1.0 / (in_channels * out_channels)

        self.weights1 = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                dtype=torch.cfloat
            )
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                dtype=torch.cfloat
            )
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                dtype=torch.cfloat
            )
        )

    @staticmethod
    def compl_mul3d(inp, weights):
        return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)

    def forward(self, x):
        # x: (B,C,X,Y,Z)
        B, C, X, Y, Z = x.shape
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        out_ft = torch.zeros(
            B, self.out_channels, X, Y, Z // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        m1 = min(self.modes1, X)
        m2 = min(self.modes2, Y)
        m3 = min(self.modes3, Z // 2 + 1)

        out_ft[:, :, :m1, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, :m2, :m3], self.weights1[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, :m2, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, :m2, :m3], self.weights2[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, :m1, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, :m1, -m2:, :m3], self.weights3[:, :, :m1, :m2, :m3]
        )
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(
            x_ft[:, :, -m1:, -m2:, :m3], self.weights4[:, :, :m1, :m2, :m3]
        )

        x = torch.fft.irfftn(out_ft, s=(X, Y, Z))
        return x
class PMNOBlock3d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()

        self.spec = SpectralConv3d_PMNO(width, width, modes, modes, modes)
        self.point = nn.Conv3d(width, width, kernel_size=1)

        self.local = nn.Sequential(
            nn.Conv3d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, kernel_size=3, padding=1),
        )

        self.norm = nn.GroupNorm(1, width)

        self.mlp = nn.Sequential(
            nn.Conv3d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(2 * width, width, kernel_size=1),
        )

    def forward(self, x):
        y = self.spec(x) + self.point(x) + self.local(x)
        y = self.norm(y)
        y = self.mlp(y)
        y = F.gelu(y)
        return x + y
class FNO3D_PMNO(nn.Module):
    """
    Slightly stronger PMNO-compatible one-step backbone.

    Input : (B,S,S,S,T_in)
    Output: (B,S,S,S,1)
    """
    def __init__(self, modes=8, width=32, T_in=None, width_q=None, n_layers=4):
        super().__init__()

        if T_in is None:
            T_in = config.T_IN_CHANNELS
        if width_q is None:
            width_q = width

        self.modes = modes
        self.width = width
        self.T_in = T_in
        self.width_q = width_q
        self.n_layers = n_layers

        # compress history before lifting
        self.history_proj = nn.Linear(self.T_in, self.T_in)

        # use full history + coordinates
        self.p = nn.Linear(self.T_in + 3, self.width)

        self.blocks = nn.ModuleList([
            PMNOBlock3d(self.width, self.modes) for _ in range(self.n_layers)
        ])

        self.q = nn.Sequential(
            nn.Conv3d(self.width, self.width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.width_q, self.width_q, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.width_q, 1, kernel_size=1),
        )

    def forward(self, x):
        # x: (B,S,S,S,T_in)
        grid = get_grid_3d(x.shape, x.device)

        x_hist = self.history_proj(x)
        x = torch.cat((x_hist, grid), dim=-1)      # (B,S,S,S,T_in+3)

        x = self.p(x)                              # (B,S,S,S,width)
        x = x.permute(0, 4, 1, 2, 3)              # (B,width,S,S,S)

        for blk in self.blocks:
            x = blk(x)

        x = self.q(x)                              # (B,1,S,S,S)
        x = x.permute(0, 2, 3, 4, 1)              # (B,S,S,S,1)
        return x
class PMNO_AC(nn.Module):
    """
    PMNO autoregressive rollout.
    Keeps the same interface:
        model(x, steps=T_out)
    """
    def __init__(self, fno, k=3):
        super().__init__()
        self.fno = fno
        self.k = k

    def forward(self, u_hist, steps=1):
        """
        u_hist: (B,S,S,S,k)
        steps : rollout length
        """
        x = u_hist
        preds = []

        for _ in range(steps):
            u_next = self.fno(x)                   # (B,S,S,S,1)
            preds.append(u_next)
            x = torch.cat([x[..., 1:], u_next], dim=-1)

        return torch.cat(preds, dim=-1)           # (B,S,S,S,steps)




