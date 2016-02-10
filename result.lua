-- result scripts for assignments 1
require 'torch'
require 'nn'
require 'xlua'

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

testData.data = testData.data:float()
mean = testData.data[{ {},1,{},{} }]:mean()
std = testData.data[{ {},1,{},{} }]:std()
testData.data[{ {},1,{},{} }]:add(-mean)
testData.data[{ {},1,{},{} }]:div(std)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
for i = 1,testData:size() do
   testData.data[{ i,{1},{},{} }] = normalization:forward(testData.data[{ i,{1},{},{} }])
end
f = io.open('prediction.csv', 'w')
f:write('Id' .. ',' .. 'Prediction' .. '\n')
for t = 1, testData:size() / 500 do
    xlua.progress(t, testData:size())
    local input = testData.data[t]
    input = input:double()
    pred = result_model:forward(input)
    local _,i = torch.max(pred,1)
    --print(pred)
    --print(t ..' ' .. i:select(1,1) .. ' '.. testData.labels[t])
    f:write(t .. ',' .. i:select(1,1) .. '\n')
end
f:close()
