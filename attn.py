import torch
import torch.nn.functional as F
import itertools

class BaseHOLinear(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 bias=True, 
                 nblock=2,
                 order=2,
                 set_order_weights=None, 
                 softmax_order_weights=False):
        '''
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        bias: bool, whether to use bias
        nblock: int, number of blocks to split the input channels into
        order: int, order of the operator
        set_order_weights: list or None, list of order weights
        softmax_order_weights: bool, whether to apply softmax to the order weights
        '''
        super(BaseHOLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nblock = nblock
        self.order = order
        assert self.order <= self.nblock, "order should be less than or equal to nblock"

        self.dim_split = [in_channels // nblock * i for i in range(nblock)] + [in_channels]

        if set_order_weights is None:
            self.order_weights = torch.nn.Parameter(torch.Tensor([0] + [1.0] + [0] * (order-1)))
        else: 
            # fix the order weights
            assert len(set_order_weights) == order + 1, "set_order_weights should have length order + 1"
            self.register_buffer('order_weights', torch.Tensor(set_order_weights))

        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels))
        if bias: self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else: self.bias = None

        self.softmax_order_weights = softmax_order_weights

        self.reset_parameters()

    def reset_parameters(self):
        #TODO: initialization can be modified according to the specific task
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None: torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.order == 1:
            return F.linear(x, self.weight, self.bias)
        
        ips = []
        for i in range(self.nblock):
            indices = torch.arange(self.dim_split[i], self.dim_split[i+1], dtype=torch.long).to(x.device)
            ips.append(F.linear(
                torch.index_select(x, -1, indices), 
                torch.index_select(self.weight, -1, indices)))
        # ips: [nblock, :, out_channels]
        if self.softmax_order_weights: order_weights = F.softmax(self.order_weights, dim=0)
        else: order_weights = self.order_weights
        result = torch.sum(torch.stack(ips), 0) * order_weights[1] + order_weights[0]
        for cur_order in range(2, self.order+1):
            for combination in itertools.combinations(ips, cur_order):
                ips_comb = torch.stack(combination)
                result += torch.prod(ips_comb, 0) * order_weights[cur_order]
        if self.bias is not None: return result + self.bias
        else: return result