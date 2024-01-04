import ffmpeg
import numpy as np
from torchvision.transforms import Resize
import torch
from transformers import BertTokenizer
import tqdm
import os
import pandas as pd

def video_decode(path):
    input_args = {
        "hwaccel": "nvdec",
        #"vcodec": "h264_cuvid",
        "c:v": "h264_cuvid",
        "loglevel":"quiet"
    }
    
    out, err = (
        ffmpeg
        .input(path,**input_args)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24', vf='scale=320x180')
        .run(capture_stdout=True)
    )
    video = np.frombuffer(out, np.uint8).reshape([-1, 180, 320, 3])
    del out
    video = torch.tensor(video)
    video = video.float()
    video = video.permute(0, 3, 1, 2)
    video = video[::12]
    max_frames = 20
    if video.size(0) > max_frames:
        video = video[:max_frames]  # Trim the video
    elif video.size(0) < max_frames:
        padding = torch.zeros((max_frames - video.size(0),) + video.size()[1:])
        video = torch.cat((video, padding), dim=0)  # Pad the video
    return video

def write_to_path(video, label, path):
    with open(path, 'wb+') as f:
        np.save(f, video)
        np.save(f, label)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def translate_all(csv_path, image_path, output_path):
    df = pd.read_csv(csv_path, sep="\t")
    data = df["SENTENCE_NAME"]
    labels = df["SENTENCE"]
    max_frames = 20
    for idx in tqdm.tqdm(range(len(data))):
        if not os.path.exists(image_path+data.iloc[idx]+".mp4"):
            continue
        video = video_decode(image_path+data.iloc[idx]+".mp4")
        label = labels.iloc[idx]
        label = torch.tensor(tokenizer.encode(label, add_special_tokens=False))
        if label.size(0) > max_frames:
            label = label[:max_frames]
        elif label.size(0) < max_frames:
            padding = torch.zeros((max_frames - label.size(0),))
            label = torch.cat((label, padding), dim=0)
        label = label.long()
        write_to_path(video, label, output_path+data.iloc[idx]+".npy")

type = "train"
translate_all("data/"+type+".csv", "data/"+type+"_mp4/", "data/"+type+"_npy/")