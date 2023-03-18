import cv2
from speechbrain.pretrained import EncoderClassifier
import onnxruntime
import torch
import torchaudio
import json
from pytube.exceptions import VideoUnavailable
import pandas as pd


def audio_to_features(classifier, signal, signal_len, device='cpu'):
    """
    """
    wavs = signal
    wav_lens = signal_len

    # Manage single waveforms in input
    if len(wavs.shape) == 1:
        wavs = wavs.unsqueeze(0)
    # Assign full length if wav_lens is not assigned
    if wav_lens is None:
        wav_lens = torch.ones(wavs.shape[0], device=device)
    # Storing waveform in the specified device
    wavs, wav_lens = wavs.to(device), wav_lens.to(device)
    wavs = wavs.float()
    # Computing features and embeddings
    feats = classifier.mods.compute_features(wavs)
    feats = classifier.mods.mean_var_norm(feats, wav_lens)

    # for tg bot
    feats = feats.cpu().numpy()

    return feats, wav_lens


def classify_batch_speechbrain(classifier, sound_embedding, emb_lens):
    voice_embedding = classifier.encode_batch(sound_embedding, emb_lens)
    class_probabilities = classifier.mods.classifier(voice_embedding).squeeze(1)
    return class_probabilities


def post_processing(labels, class_scores, top_k=2):
    values, indices = torch.topk(class_scores, top_k, dim=-1)
    top_results = []
    for i in range(top_k):
        index = indices[0][i].item()
        label_id = labels[str(index)]
        top_results.append((label_id, values[0][i].item()))

    return top_results


def build_classifier_speechbrain(device='cpu'):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                                savedir="pretrained_models/spkrec-xvect-voxceleb",
                                                run_opts={"device": device})
    classifier.eval()
    return classifier


def build_label_decoder(labels_path='vox_ids.json'):
    with open(labels_path, 'r') as fp:
        labels = json.load(fp)
    return labels


def get_photo_coords(label_id):
    df = pd.read_csv('all_vox.csv', sep='\t')
    reference = df.loc[df['id'] == label_id, ['reference']].values[0][0]
    frame = df.loc[df['id'] == label_id, ['frame']].values[0][0]

    return str(reference), int(frame)


def get_name(label_id):
    import re
    from pytube import YouTube

    # Получить объект YouTube видео по URL
    reference = get_photo_coords(label_id)[0]
    try:
        video = YouTube(f"https://www.youtube.com/watch?v={reference}")
    except VideoUnavailable:
        print("Видео недоступно.")
    title = video.title

    interviewee_name = re.search(r'Interview(er|ee|ee:|ed by| with|s|ed)\s?(.+?)( about| on| in|,|:|$)', title)
    if interviewee_name:
        print(interviewee_name.group(2))

def get_frame(reference, frame_number):
    import cv2
    import os
    from pytube import YouTube

    try:
        yt = YouTube(f"https://www.youtube.com/watch?v={reference}")
        stream = yt.streams.get_highest_resolution()
        stream.download(filename='video.mp4')
    except VideoUnavailable:
        return None

    cap = cv2.VideoCapture('video.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    os.remove('video.mp4')
    return frame

def crop_image(image, id):
    from PIL import Image

    all_vox = pd.read_csv('./all_vox.csv', sep='\t')
    image = Image.open(image)

    x = all_vox[all_vox.id == id].x
    y = all_vox[all_vox.id == id].y
    w = all_vox[all_vox.id == id].w
    h = all_vox[all_vox.id == id].h
    if w <= 1:
        x, w = x * image.size[0], w * image.size[0]
        y, h = y * image.size[0], h * image.size[0]
    print(x, y, w, h)
    im_crop = image.crop((x, y, x + w, y + h))
    print('2')
    im_crop.save('celeb_image.jpg', quality=95)
    print('3')


def get_signal(filename='./example3.wav'):
    signal, _ = torchaudio.load(filename)
    return signal


def main(classifier, signal, label_decoder, signal_len=None, top_k=2):
    signal.clone().detach()

    emb, emb_len = audio_to_features(classifier, signal, signal_len)

    class_probs = classify_batch_speechbrain(classifier, signal, emb_len)

    top_results = post_processing(label_decoder, class_probs, top_k=top_k)

    return top_results, emb


if __name__ == '__main__':
    signal = get_signal()
    label_decoder = build_label_decoder()
    sp_classifier = build_classifier_speechbrain()
    top_results, _ = main(sp_classifier, signal, label_decoder, top_k=2)
    print(top_results)

    for i, (label_id, score) in enumerate(top_results):
        try:
            reference, frame_num = get_photo_coords(label_id)
            frame = get_frame(reference, frame_num)
            if frame is not None and not frame.size == 0:
                cv2.imwrite(f'frame_{i}.jpg', frame)
                crop_image('/Users/nikolacrnobrnja/Desktop/hse_project/frame_1.jpg', label_id)
                get_name(label_id)
                print(f'Done for result {i}!')
                break
            else:
                raise Exception(f"Empty or invalid frame for result {i}.")
        except (VideoUnavailable, Exception) as e:
            if i == len(top_results) - 1:
                print("All top results exhausted.")