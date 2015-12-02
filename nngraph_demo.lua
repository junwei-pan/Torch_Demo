require 'nngraph'
require 'nn'

function simple_feedforward_nn()
    -- it is common style to mark inputs with identity nodes for clarity.
    input = nn.Identity()()

    -- each hidden layer is achieved by connecting the previous one
    -- here we define a single hidden layer network
    h1 = nn.Tanh()(nn.Linear(20, 10)(input))
    output = nn.Linear(10, 1)(h1)
    mlp = nn.gModule({input}, {output})

    x = torch.rand(20)
    dx = torch.rand(1)
    mlp:updateOutput(x)
    mlp:updateGradInput(x, dx)
    mlp:accGradParameters(x, dx)

    -- draw graph (the forward graph, '.fg')
    -- this will produce an SVG in the runtime directory
    graph.dot(mlp.fg, 'MLP', 'MLP')
end

simple_feedforward_nn()
