import os
import pandas as pd
from pytube import YouTube
from pytube.exceptions import VideoUnavailable


def try_video(reference):
    try:
        video = YouTube(f"https://www.youtube.com/watch?v={reference}")
    except VideoUnavailable:
        video = "Видео недоступно."
    return video

def cleardata(directory):
    df = pd.DataFrame(columns=['id', 'reference', 'frame', 'x', 'y', 'w', 'h'])
    for identity in os.listdir(directory):
        if identity == '.DS_Store':
              continue
        identity = os.path.join(directory, identity)
        for reference in os.listdir(identity):
            if reference == '.DS_Store':
                continue
            if try_video(reference) != "Видео недоступно.":
                break
        reference = os.path.join(identity, reference)
        file = os.path.join(reference, os.listdir(reference)[0])
        with open(file, 'r', encoding="utf-8") as f:
            text = f.readlines()
            row = [text[0].split()[-1]]
            row.append(text[1].split()[-1])
            for r in list(map(float, (text[7].split()))):
                row.append(r)
            df.loc[-1] = row
            df.index = df.index + 1
            df = df.sort_index()
        f.close()
    df = df.sort_values('id').reset_index(drop=True)
    df.to_csv('all_vox.csv', sep='\t', encoding='utf-8')


cleardata('./txt')