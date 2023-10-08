import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import torch_tensorrt
import tensorrt as trt
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules, tensor_quant
# quant_modules.initialize()
# quant_nn.TensorQuantizer.use_fb_fake_quant = True

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

class FeatureWeightNet(nn.Module):
    def __init__(self, num_feature, neighbors=9, G=8, stage=3):
        super(FeatureWeightNet, self).__init__()
        self.neighbors = neighbors
        self.G = G
        self.stage = stage

        self.conv0 = ConvBnReLU3D(G, 16, 1, 1, 0)
        self.conv1 = ConvBnReLU3D(16, 8, 1, 1, 0)
        self.similarity = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        self.output = nn.Sigmoid()

    def forward(self, x):  # ref_feature:torch.Tensor, grid:torch.Tensor
        # ref_feature: reference feature map
        # grid: position of sampling points in adaptive spatial cost aggregation

        # [B,Neighbor,H,W]
        x = self.similarity(self.conv1(self.conv0(x))).squeeze(1)

        return self.output(x)

def func1():
    data = torch.rand((1, 8, 9, 150, 200), dtype=torch.float)
    testing_dataset = torch.utils.data.TensorDataset(data)

    testing_dataloader = torch.utils.data.DataLoader(
        dataset=testing_dataset, batch_size=1, shuffle=False, num_workers=1
    )
    # Test cases can assume using GPU id: 0
    calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
        testing_dataloader,
        cache_file="./calibration.cache",
        use_cache=False,
        algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )

    model = FeatureWeightNet(64, 9, 4, 3).cuda().eval()
    inp = torch.rand((1, 4, 9, 600, 800), dtype=torch.float).cuda()
    jit_model = torch.jit.trace(model, inp).cuda().eval()
    out1 = model(inp)
    out2 = jit_model(inp)
    inputs = [
        torch_tensorrt.Input(
            shape=[1, 4, 9, 600, 800],
            dtype=torch.float,
        )
    ]
    enabled_precisions = { torch.int8}
    trt_ts_module = torch_tensorrt.ts.compile(module=jit_model,
                                              inputs=inputs,

                                              calibrator=calibrator,
                                              enabled_precisions=enabled_precisions,
                                              truncate_long_and_double=True,
                                              )
    torch.jit.save(trt_ts_module,"./test_ts.ts")
    print("end____________-----")
def func2():
   model1 = torch.jit.load("./test_ts.ts").cuda()
   model2 = FeatureWeightNet(64, 9, 4, 3).cuda().eval()
   inp = torch.rand((1, 4, 9, 600, 800), dtype=torch.float).cuda()
   while(1):
       torch.cuda.synchronize(0)
       time1 = time.time()
       out = model1(inp)
       torch.cuda.synchronize(0)
       time2 = time.time()
       print("time:{:.4f}".format((time2-time1)*1000))

if __name__ == '__main__':
    mat = torch.tensor( [[[ 5.0352e+02,  3.9128e+02, -2.8931e+02, -1.9251e+04],
         [ 9.6465e+01, -2.7994e+02, -5.8070e+02, -4.7713e+04],
         [-1.5722e-01,  5.8735e-01, -7.9391e-01, -4.7776e+01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 6.9795e+02, -1.8480e+02,  1.3974e+02, -5.2560e+03],
         [ 9.2594e+01, -5.4519e+02, -4.1725e+02,  2.1105e+04],
         [ 7.1166e-01,  1.2795e-01, -6.9077e-01,  3.8619e+01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 2.9043e+02,  8.5177e+02, -6.5300e+01, -3.1712e+02],
         [-7.4520e+02,  3.6753e+02, -2.4351e+02,  3.0985e+01],
         [-3.2066e-01,  6.1795e-01,  7.1786e-01,  9.5131e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]])

    mat = torch.inverse(mat)
    print("------------------------")