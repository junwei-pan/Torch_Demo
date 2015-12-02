-- Based on https://github.com/soumith/cvpr2015/blob/master/Deep%20Learning%20with%20Torch.ipynb

require 'itorch'
require 'nn';
function download_data ()
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

function shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

function init_net ()
    net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 20, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
    net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialConvolution(20, 16, 5, 5))
    net:add(nn.SpatialMaxPooling(2,2,2,2))
    net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.Linear(120, 84))
    net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems
    return net
end

function evaluate_cunn ()
    
    correct = 0
    d_wc = {}
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i]:cuda()):float()
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end
    
    print(correct, 100*correct/10000 .. ' % ')

    class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i]:cuda()):float()
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            class_performance[groundtruth] = class_performance[groundtruth] + 1
        end
    end

    for i=1,#classes do
        print(classes[i], 100*class_performance[i]/1000 .. ' %')
    end
end


function evaluate ()
    
    correct = 0
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
    end

    print(correct, 100*correct/10000 .. ' % ')

    class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    for i=1,10000 do
        local groundtruth = testset.label[i]
        local prediction = net:forward(testset.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            class_performance[groundtruth] = class_performance[groundtruth] + 1
        end
    end

    for i=1,#classes do
        print(classes[i], 100*class_performance[i]/1000 .. ' %')
    end
end

trainset = torch.load('data/cifar10-train.t7')
testset = torch.load('data/cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

-- ignore setmetatable for now, it is a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
        return {t.data[i], t.label[i]} 
    end}
);

trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size() 
    return self.data:size(1) 
end

print(trainset:size()) -- just to test

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize Test Data
testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Set up network
net = init_net()
maxIteration = 1

criterion = nn.ClassNLLCriterion()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = maxIteration -- just do 5 epochs of training
t_start = os.time()
trainer:train(trainset)
print("----->Time<-----")
print(os.difftime(os.time(), t_start))

evaluate(testset)
require 'cunn'
net = init_net()
net = net:cuda()
criterion = criterion:cuda()
trainset.data = trainset.data:cuda()
trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = maxIteration -- just do 5 epochs of training.
t_start = os.time()
trainer:train(trainset)
print("----->Time<-----")
print(os.difftime(os.time(), t_start))

evaluate_cunn(testset)
