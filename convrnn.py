import torch
import torch.nn as nn

class ConvGRUCell(nn.Module):
    ''' Initialize ConvGRU cell '''
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        self.reset_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size+hidden_size, hidden_size, kernel_size, padding=padding)

        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_tensor, hidden_state):
        if hidden_state is None:
            B, C, *spatial_dim = input_tensor.size()
            hidden_state = torch.zeros([B,self.hidden_size,*spatial_dim]).cuda()
        # [B, C, H, W]
        combined = torch.cat([input_tensor, hidden_state], dim=1) #concat in C
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        out = torch.tanh(self.out_gate(torch.cat([input_tensor, hidden_state * reset], dim=1)))
        new_state = hidden_state * (1 - update) + out * update
        return new_state


class ConvGRU(nn.Module):
    ''' Initialize a multi-layer Conv GRU '''
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, dropout=0.1):
        super(ConvGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        
        self.conv1 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2), padding=(1, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_size * 8 * 8, 5)
        
        cell_list = []
        for i in range(self.num_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_size
            cell = ConvGRUCell(input_dim, self.hidden_size, self.kernel_size)
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cell_list.append(getattr(self, name))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(p=dropout)

        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, x, hidden_state=None):
        [B, seq_len, *_] = x.size()

        if hidden_state is None:
            hidden_state = [None] * self.num_layers
        # input: image sequences [B, T, C, H, W]
        current_layer_input = x 
        del x

        last_state_list = []

        for idx in range(self.num_layers):
            cell_hidden = hidden_state[idx]
            output_inner = []
            for t in range(seq_len):
                cell_hidden = self.cell_list[idx](current_layer_input[:,t,:], cell_hidden)
                cell_hidden = self.dropout_layer(cell_hidden) # dropout in each time step
                output_inner.append(cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)
            current_layer_input = layer_output

            last_state_list.append(cell_hidden)

        # last_state_list shape = [16, 1, HIDDEN_SIZE, 112, 112]
        last_state_list = torch.stack(last_state_list, dim=1)
        h = self.relu(self.conv1(last_state_list[:,-1,:]))
        h = self.maxpool1(h)
        h = self.relu(self.conv2(h))
        h = self.maxpool2(h)
        out = self.fc1(h.view(h.size(0), -1))
        probs = self.softmax(out)

        return probs
#        return layer_output, last_state_list

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
            
        self.conv1 = nn.Conv2d(self.hidden_channels[-1], self.hidden_channels[-1], kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.conv2 = nn.Conv2d(self.hidden_channels[-1], self.hidden_channels[-1], kernel_size=(3, 3), stride=2, padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_channels[-1] * 2 * 2, 5)
        if torch.__version__.split('.')[0]=='1':
            self.softmax = nn.Softmax(dim = 1)
        else:
            self.softmax = nn.Softmax()     # PyTorch 0.2 Operates on 2D arrays

    def forward(self, input):
        internal_state = []
        outputs = []
        for idx in range(input.size(2)):
            inp = input[:,:,idx,:]
            
            for step in range(self.step):
                x = inp
                for i in range(self.num_layers):
                    # all cells are initialized in the first step
                    name = 'cell{}'.format(i)
                    if step == 0:
                        bsize, _, height, width = x.size()
                        (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                                 shape=(height, width))
                        internal_state.append((h, c))
    
                    # do forward
                    (h, c) = internal_state[i]
                    x, new_c = getattr(self, name)(x, h, c)
                    internal_state[i] = (x, new_c)
                # only record effective steps
                if step in self.effective_step:
                    outputs.append(x)
                    
        h = self.relu(self.conv1(outputs[-1]))
        h = self.maxpool1(h)
        h = self.relu(self.conv2(h))
        h = self.maxpool2(h)
        out = self.fc1(h.view(h.size(0), -1))
        probs = self.softmax(out)

        return probs

#if __name__ == '__main__':
#    crnn = ConvGRU(input_size=10, hidden_size=20, kernel_size=3, num_layers=2)
#    data = torch.randn(4, 5, 10, 6, 6) # [B, seq_len, C, H, W], temporal axis=1
#    crnn = crnn.to(torch.device("cuda:0"))
#    data = data.cuda()
#    output, hn = crnn(data)
#    #import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # gradient check
    convlstm = ConvLSTM(input_channels=512, hidden_channels=[128, 64, 64, 32, 32], kernel_size=3, step=5,
                        effective_step=[4]).cuda()
    loss_fn = torch.nn.MSELoss()

#    input = torch.randn(1, 512, 64, 32).cuda()
#    target = torch.randn(1, 32, 64, 32).double().cuda()
    input = torch.randn(1, 512, 8, 64, 64).cuda()
    target = torch.randn(1, 32, 8, 64, 64).double().cuda()

    output = convlstm(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(res)
    
    
    
class Conv3DLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1]).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda())


class Conv3DLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x)

        return outputs, (x, new_c)
    