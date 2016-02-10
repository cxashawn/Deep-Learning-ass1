-- result scripts for assignments 1
require 'torch'
require 'nn'

tsize = 10000
data_path = 'mnist.t7'
test_file = paths.concat(data_path, 'test_32x32.t7')
result_model = torch.load('model.net')
loaded = torch.load(test_file, 'ascii')
testData = {
    data = loaded.data,
    labels = loaded.labels,
    size = function() return tsize end
}
for t = 1, testData:size() / 500 do
    local input = testData.data[t]
    input = input:double()
    pred = result_model:forward(input)
    local max = torch.max(pred)
    local j
    for i = 1, 10 do 
        if pred[i] == max then j = i
        end
    end
    print(t ..' ' .. j .. ' '.. testData.labels[t])
end
