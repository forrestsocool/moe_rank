import torch.nn as nn
#from dsf.python.layers.activations import Dice


def get_activation(act, hidden_size=None, dice_dim=2):
    """Construct activation layers

    :param act: str or nn.Module, name of activation function
    :param hidden_size: int, used for Dice activation
    :param dice_dim: int, used for Dice activation
    :return: activation layer
    """
    if isinstance(act, str):
        if act.lower() == "sigmoid":
            act_layer = nn.Sigmoid()
        elif act.lower() == "linear":
            act_layer = nn.Identity()
        elif act.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act.lower() == 'prelu':
            act_layer = nn.PReLU()
        # elif act.lower() == 'dice':
        #     assert dice_dim
        #     act_layer = Dice(hidden_size, dice_dim)
        elif act.lower() == 'tanh':
            act_layer = nn.Tanh()
        else:
            raise ValueError('not supported type of act: %s' % act)
    elif isinstance(act, nn.Module):
        act_layer = act()
    else:
        raise ValueError('not supported type of act: %s' % act)

    return act_layer
