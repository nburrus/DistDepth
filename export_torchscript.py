# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import os
import torch
from networks.depth_decoder import DepthDecoder
from networks.resnet_encoder import ResnetEncoder
from utils import output_to_depth

torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistDepth(torch.nn.Module):
    def __init__(self, dir_prefix):
        super().__init__()
        print("Loading the pretrained network")
        self.encoder = ResnetEncoder(152, False)
        loaded_dict_enc = torch.load(
            dir_prefix + "ckpts/encoder.pth",
            map_location=device,
        )

        filtered_dict_enc = {
            k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(device)
        self.encoder.eval()

        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(
            dir_prefix + "ckpts/depth.pth",
            map_location=device,
        )
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(device)
        self.depth_decoder.eval()

    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.depth_decoder(features)

        out = outputs[("out", 0)]
        depth = output_to_depth(out, 0.1, 10)
        return depth

if __name__ == "__main__":

    dir_prefix = "./"
    file_list = open(dir_prefix + "demo_list.txt", "r")
    files = file_list.readlines()
    output_path = dir_prefix + "results"

    with torch.no_grad():

        dd = DistDepth(dir_prefix)

        sample_input = torch.zeros(1, 3, 256, 256).to(device)
        scripted_module = torch.jit.trace(dd, sample_input)
        torch.jit.save (scripted_module, f"dist-depth-{256}x{256}.scripted.pt")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for idx, file in enumerate(files):

            raw_img = np.transpose(
                cv2.imread(dir_prefix + file[:-1], -1)[:, :, :3], (2, 0, 1)
            )
            input_image = torch.from_numpy(raw_img).float().to(device)
            input_image = (input_image / 255.0).unsqueeze(0)
            input_image = torch.nn.functional.interpolate(
                input_image, (256, 256), mode="bilinear", align_corners=False
            )

            depth = dd (input_image)
            
            depth = torch.nn.functional.interpolate(
                depth, (raw_img.shape[2], raw_img.shape[1]), mode="bilinear", align_corners=False
            )
            metric_depth = depth.cpu().numpy().squeeze()

            normalizer = mpl.colors.Normalize(vmin=0.1, vmax=5.0)
            mapper = cm.ScalarMappable(norm=normalizer, cmap="viridis_r")
            colormapped_im = (mapper.to_rgba(metric_depth)[:, :, :3] * 255).astype(
                np.uint8
            )

            cv2.imwrite(os.path.join(output_path, f"{idx:02d}.png"), colormapped_im)
