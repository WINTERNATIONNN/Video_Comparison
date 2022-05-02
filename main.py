import glob
import pandas as pd
import cv2
import librosa
import argparse
import errno
import os
from skimage.metrics import structural_similarity as ssim
from moviepy.editor import VideoFileClip, audio
from mfcc import mfcc
from correlation import correlate
from videohash import VideoHash


def compare_ssim(video_a, video_b):
    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)

    total_frame = cap_a.get(cv2.CAP_PROP_FRAME_COUNT)
    ssim_score = 0
    for i in range(1, 11):
        cap_a.set(1, total_frame/i)
        cap_b.set(1, total_frame/i)
        success, image1 = cap_a.read()
        success, image2 = cap_b.read()
        if success:
            image1, image2 = cv2.resize(image1, (1280, 640)), cv2.resize(image2, (1280, 640))
            ssim_score = ssim_score+ssim(image1, image2, channel_axis=2, data_range=255, multichannel=True)
    ssim_score = ssim_score/10
    return ssim_score


def get_hash(video_arr):
    video_dic = {}
    for tem in video_arr:
        video_dic[tem] = VideoHash(tem)
    return video_dic


def compare_duration(va, vb):
    video_1, video_2 = cv2.VideoCapture(va), cv2.VideoCapture(vb)
    duration_1, duration_2 = video_1.get(cv2.CAP_PROP_FRAME_COUNT). video_2.get(cv2.CAP_PROP_FRAME_COUNT)
    video_1.release()
    video_2.release()
    if duration_1 - duration_2 != 0:
        return False
    else:
        return True


def audio_compare(va, vb):
    audio_file_name1, audio_file_name2 = "va.mp3", "vb.mp3"
    VideoFileClip(va).audio.write_audiofile(audio_file_name1)
    VideoFileClip(vb).audio.write_audiofile(audio_file_name2)
    if librosa.get_duration(filename=audio_file_name1) < 5 :
        return mfcc(audio_file_name1, audio_file_name2)
    else:
        return mfcc(audio_file_name1, audio_file_name2) and correlate(audio_file_name1, audio_file_name2)


def video_compare(video_hash1, v_file_list1, video_hash2, v_file_list2):
    same_video_record = []
    for i in range(len(v_file_list1)):
        for j in range(len(v_file_list2)):
            if (video_hash1[v_file_list1[i]] - video_hash2[v_file_list1[j]]) <= 2 and compare_duration(v_file_list1[i], v_file_list2[j]):
                ssim_score = compare_ssim(v_file_list1[i], v_file_list2[j])
                audio_match = audio_compare(v_file_list1[i], v_file_list2[j])
                if ssim_score > 0.54 and (ssim_score > 0.59 or audio_match):
                    same_video_record.append([v_file_list1[i], v_file_list2[j], ssim_score, audio_match])
                    v_file_list2.pop(j)
                    j -= 1
                    break
    return same_video_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i ", "--source-path", help="source file")
    parser.add_argument("-o ", "--target-path", help="target file")
    args = parser.parse_args()

    source = args.source_path if args.source_path else None
    target = args.target_path if args.target_path else None
    if source is None or target is None:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source + ", " + target)

    video_file_list1, video_file_list2 = glob.glob(source + "*.mp4"), glob.glob(target + "*.mp4")
    hash_video1 = get_hash(video_file_list1)
    hash_video2 = get_hash(video_file_list2)
    df = pd.DataFrame(video_compare(hash_video1, video_file_list1, hash_video2, video_file_list2))
    df.to_csv("test.csv")

