import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(
        self, input_size, input_dim, hidden_dim, kernel_size, bias,
    ):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int) if images (conv2d) -> Height and width of input tensor as (height, width).
                    int if sequence (conv1d) -> Width of window (width)
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int) if images (conv2d)
                     int if sequence (conv1d)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.conv2d = isinstance(input_size, tuple)
        if self.conv2d:
            self.height, self.width = input_size
        else:
            self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        if self.conv2d:
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        else:
            self.padding = kernel_size // 2
        self.bias = bias

        if self.conv2d:
            self.conv = nn.Conv2d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=self.input_dim + self.hidden_dim,
                out_channels=4 * self.hidden_dim,
                kernel_size=self.kernel_size,
                padding=self.padding,
                bias=self.bias,
            )

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, device):
        d = (self.height, self.width) if self.conv2d else (self.width,)
        return (
            torch.zeros(batch_size, self.hidden_dim, *d).to(device),
            torch.zeros(batch_size, self.hidden_dim, *d).to(device),
        )


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        self.conv2d = isinstance(input_size, tuple)
        self._check_kernel_size_consistency(kernel_size, tuple if self.conv2d else int)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        if self.conv2d:
            self.height, self.width = input_size
        else:
            self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_size=(self.height, self.width) if self.conv2d else self.width,
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w) -> images (conv2d)
            4-D Tensor either of shape (t, b, c, w) or (b, t, c, w) -> sequence of windows (conv1d)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            if self.conv2d:
                # (t, b, c, h, w) -> (b, t, c, h, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
            else:
                # (t, b, c, w) -> (b, t, c, w)
                input_tensor = input_tensor.permute(1, 0, 2, 3)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(
                batch_size=input_tensor.size(0), device=input_tensor.get_device()
            )

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    cur_layer_input[:, t, ...], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size, t):
        if not (
            isinstance(kernel_size, t)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, t) for elem in kernel_size])
            )
        ):
            raise ValueError(f"kernel_size must be {t} or list of {t}")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

